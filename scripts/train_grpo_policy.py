#!/usr/bin/env python3
"""
GRPO policy training for UniRAG-Policy.

Initialize from oracle policy warm-start.
Uses RAGEnvironment for reward computation.
Cost annealing: lambda_cost starts at 0, linearly increases to final value.
GRPO with 8 generations per prompt.
Tracks: accuracy, avg cost, action distribution over training.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.rag_environment import (
    ACTION_COSTS, RAGAction, RAGEnvironment, RAGRewardFunction, PolicyNetwork,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_grpo_policy")


def parse_args():
    parser = argparse.ArgumentParser(description="Train RAG policy with GRPO + cost annealing")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument("--oracle_policy_path", type=str, default=None,
                        help="Path to oracle policy checkpoint for warm-start")
    parser.add_argument("--data_dir", type=str, default="./data/datasets")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_policy")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--cost_lambda_start", type=float, default=0.0,
                        help="Initial cost lambda (annealed from this)")
    parser.add_argument("--cost_lambda_final", type=float, default=0.3,
                        help="Final cost lambda (annealed to this)")
    parser.add_argument("--cost_annealing_fraction", type=float, default=0.5,
                        help="Fraction of training to anneal cost over")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


ACTION_NAMES = ["no_retrieve", "retrieve_1", "retrieve_3", "retrieve_5",
                "retrieve_10", "rewrite", "multi_hop"]


def format_policy_prompt(query: str, state_desc: str) -> str:
    return (
        f"<|im_start|>system\nYou are a RAG policy controller. "
        f"Decide the best retrieval action for the given query.\n"
        f"Actions: {', '.join(ACTION_NAMES)}\n"
        f"Consider: query complexity, expected accuracy gain, retrieval cost.<|im_end|>\n"
        f"<|im_start|>user\nQuery: {query}\nState: {state_desc}\n"
        f"Choose the best action.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_training_queries(data_dir, config, max_per_dataset=2000):
    """Load queries from prepared JSONL files or HF datasets."""
    queries = []

    if os.path.exists(data_dir):
        for fname in os.listdir(data_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(data_dir, fname)) as f:
                ds_queries = [json.loads(line.strip()) for line in f]
            if len(ds_queries) > max_per_dataset:
                random.shuffle(ds_queries)
                ds_queries = ds_queries[:max_per_dataset]
            queries.extend(ds_queries)
            logger.info(f"Loaded {len(ds_queries)} from {fname}")
    else:
        for ds_cfg in config["evaluation"]["datasets"]:
            try:
                subset = ds_cfg.get("subset")
                if subset:
                    ds = load_dataset(ds_cfg["dataset_id"], subset, split="train", trust_remote_code=True)
                else:
                    ds = load_dataset(ds_cfg["dataset_id"], split="train", trust_remote_code=True)
                max_s = min(len(ds), max_per_dataset)
                ds = ds.shuffle(seed=42).select(range(max_s))
                for ex in ds:
                    q = ex.get("question", ex.get("claim", ""))
                    if isinstance(q, dict):
                        q = q.get("text", "")
                    a = ex.get("answer", ex.get("label", ""))
                    if isinstance(a, dict):
                        a = a.get("value", str(a))
                    if isinstance(a, list):
                        a = a[0] if a else ""
                    if q and a:
                        queries.append({"query": str(q), "answer": str(a)})
                logger.info(f"Loaded {max_s} from {ds_cfg['name']}")
            except Exception as e:
                logger.warning(f"Failed to load {ds_cfg['name']}: {e}")

    if not queries:
        logger.info("Generating synthetic queries...")
        for i in range(5000):
            queries.append({
                "query": f"What is the capital of country {i % 200}?",
                "answer": f"Capital City {i % 200}",
            })

    logger.info(f"Total training queries: {len(queries)}")
    return queries


def get_annealed_cost_lambda(step, total_steps, start, final, anneal_fraction):
    """Linearly anneal cost lambda from start to final."""
    anneal_steps = int(total_steps * anneal_fraction)
    if step >= anneal_steps:
        return final
    progress = step / max(anneal_steps, 1)
    return start + (final - start) * progress


def grpo_update(policy_model, tokenizer, rollout_buffer, optimizer, config, device):
    """GRPO update with group-relative advantages."""
    grpo_cfg = config["grpo"]
    clip_range = grpo_cfg["clip_range"]
    kl_coeff = grpo_cfg["kl_coeff"]
    entropy_coeff = grpo_cfg["entropy_coeff"]

    if not rollout_buffer:
        return {}

    query_groups = defaultdict(list)
    for item in rollout_buffer:
        query_groups[item["query_id"]].append(item)

    total_policy_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for qid, group in query_groups.items():
        if len(group) < 2:
            continue

        rewards = torch.tensor([g["reward"] for g in group], device=device)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = (rewards - mean_r) / std_r

        for g, adv in zip(group, advantages):
            inputs = tokenizer(
                g["prompt"], return_tensors="pt", truncation=True, max_length=512,
            ).to(device)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = policy_model(**inputs)
                logits = outputs.logits[:, -1, :]

            action_token_ids = [
                tokenizer.encode(name, add_special_tokens=False)[0] for name in ACTION_NAMES
            ]
            action_logits = logits[0, action_token_ids]
            probs = F.softmax(action_logits, dim=-1)
            log_probs = F.log_softmax(action_logits, dim=-1)

            log_prob = log_probs[g["action"]]
            old_log_prob = torch.tensor(g["log_prob"], device=device)

            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -torch.min(ratio * adv, clipped_ratio * adv)

            entropy = -(probs * log_probs).sum()
            kl_penalty = kl_coeff * (old_log_prob - log_prob)

            loss = policy_loss + kl_penalty - entropy_coeff * entropy
            loss.backward()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()
            num_updates += 1

    if num_updates > 0:
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
        "num_updates": num_updates,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device = torch.device(f"cuda:{max(local_rank, 0)}" if torch.cuda.is_available() else "cpu")

    policy_model_name = config["policy"]["model"]
    logger.info("=== UniRAG GRPO Policy Training ===")
    logger.info(f"Policy model: {policy_model_name}")
    logger.info(f"Cost annealing: {args.cost_lambda_start} → {args.cost_lambda_final} "
                f"over {args.cost_annealing_fraction:.0%} of training")

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": max(local_rank, 0)} if torch.cuda.is_available() else "cpu",
    )
    policy_model.config.use_cache = False

    lora_cfg = config["policy"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters()

    queries = load_training_queries(args.data_dir, config)

    train_cfg = config["training"]
    grpo_cfg = config["grpo"]

    num_epochs = args.num_epochs or train_cfg["num_train_epochs"]
    num_generations = args.num_generations
    temperature = args.temperature or grpo_cfg["temperature"]
    batch_size = args.batch_size or train_cfg["per_device_train_batch_size"]
    lr = args.learning_rate or train_cfg["learning_rate"]

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=lr,
    )

    total_steps = (len(queries) // batch_size) * num_epochs
    training_log = []
    global_step = 0

    for epoch in range(num_epochs):
        random.shuffle(queries)
        epoch_rewards = []
        epoch_costs = []
        epoch_accuracies = []
        action_counts = defaultdict(int)

        cost_lambda = get_annealed_cost_lambda(
            epoch * (len(queries) // batch_size),
            total_steps,
            args.cost_lambda_start,
            args.cost_lambda_final,
            args.cost_annealing_fraction,
        )

        reward_fn = RAGRewardFunction(cost_lambda=cost_lambda)
        env = RAGEnvironment(reward_fn=reward_fn, max_steps=3)

        for batch_idx in range(0, len(queries), batch_size):
            batch = queries[batch_idx:batch_idx + batch_size]
            rollout_buffer = []

            for qi, item in enumerate(batch):
                query = item["query"]
                gold = item["answer"]
                query_id = f"{epoch}_{batch_idx}_{qi}"

                for gen_idx in range(num_generations):
                    state = env.reset(query, gold)
                    state_desc = (
                        f"complexity={state.query_complexity:.2f}, "
                        f"retrievals={state.num_retrievals_done}"
                    )
                    prompt = format_policy_prompt(query, state_desc)

                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=512,
                    ).to(device)

                    with torch.no_grad():
                        outputs = policy_model(**inputs)
                        logits = outputs.logits[:, -1, :]

                    action_token_ids = [
                        tokenizer.encode(name, add_special_tokens=False)[0]
                        for name in ACTION_NAMES
                    ]
                    action_logits = logits[0, action_token_ids]
                    probs = F.softmax(action_logits / temperature, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action_idx = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action_idx)).item()

                    action = RAGAction(action_idx)
                    transition = env.step(action)

                    rollout_buffer.append({
                        "query_id": query_id,
                        "prompt": prompt,
                        "action": action_idx,
                        "log_prob": log_prob,
                        "reward": transition.reward,
                        "info": transition.info,
                    })

                    epoch_rewards.append(transition.reward)
                    epoch_costs.append(transition.info.get("cost", 0))
                    epoch_accuracies.append(transition.info.get("accuracy", 0))
                    action_counts[ACTION_NAMES[action_idx]] += 1

            update_info = grpo_update(policy_model, tokenizer, rollout_buffer, optimizer, config, device)
            global_step += 1

            step_cost_lambda = get_annealed_cost_lambda(
                global_step, total_steps,
                args.cost_lambda_start, args.cost_lambda_final,
                args.cost_annealing_fraction,
            )
            reward_fn.cost_lambda = step_cost_lambda

            if global_step % train_cfg["logging_steps"] == 0:
                recent_n = min(200, len(epoch_rewards))
                avg_reward = np.mean(epoch_rewards[-recent_n:])
                avg_cost = np.mean(epoch_costs[-recent_n:])
                avg_acc = np.mean(epoch_accuracies[-recent_n:])
                logger.info(
                    f"Epoch {epoch} Step {global_step} | "
                    f"λ_cost={step_cost_lambda:.3f} | "
                    f"Reward={avg_reward:.4f} Cost={avg_cost:.4f} Acc={avg_acc:.4f} | "
                    f"PLoss={update_info.get('policy_loss', 0):.4f} "
                    f"Ent={update_info.get('entropy', 0):.4f}"
                )

            if global_step % train_cfg["save_steps"] == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                policy_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        total_actions = sum(action_counts.values())
        action_dist = {k: v / max(total_actions, 1) for k, v in action_counts.items()}

        epoch_log = {
            "epoch": epoch,
            "avg_reward": float(np.mean(epoch_rewards)),
            "avg_cost": float(np.mean(epoch_costs)),
            "avg_accuracy": float(np.mean(epoch_accuracies)),
            "cost_lambda": cost_lambda,
            "action_distribution": action_dist,
            "num_rollouts": len(epoch_rewards),
        }
        training_log.append(epoch_log)

        logger.info(
            f"=== Epoch {epoch} complete | Reward={epoch_log['avg_reward']:.4f} "
            f"Cost={epoch_log['avg_cost']:.4f} Acc={epoch_log['avg_accuracy']:.4f} "
            f"λ={cost_lambda:.3f} ==="
        )
        logger.info(f"  Action dist: {action_dist}")

    policy_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    summary = {
        "model": policy_model_name,
        "num_epochs": num_epochs,
        "num_generations": num_generations,
        "cost_lambda_start": args.cost_lambda_start,
        "cost_lambda_final": args.cost_lambda_final,
        "final_avg_reward": training_log[-1]["avg_reward"] if training_log else None,
        "final_avg_cost": training_log[-1]["avg_cost"] if training_log else None,
        "final_avg_accuracy": training_log[-1]["avg_accuracy"] if training_log else None,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"=== Training complete. Model saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
