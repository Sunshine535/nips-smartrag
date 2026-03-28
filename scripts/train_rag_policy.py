#!/usr/bin/env python3
"""RL training: policy decides when/what/how much to retrieve.
GRPO on Qwen/Qwen3.5-4B policy model."""

import argparse
import json
import logging
import os
import sys
from collections import deque

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from collections import defaultdict
from contextlib import nullcontext
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag_environment import (
    ACTION_COSTS,
    RAGAction,
    RAGEnvironment,
    RAGRewardFunction,
    PolicyNetwork,
)


def load_training_queries(config: dict):
    """Load QA datasets for RL environment."""
    queries = []
    for ds_cfg in config["evaluation"]["datasets"]:
        name = ds_cfg["name"]
        try:
            subset = ds_cfg.get("subset")
            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split="train")
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split="train")
            max_s = ds_cfg.get("max_samples", 2000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))

            for ex in ds:
                q = ex.get("question", ex.get("claim", ""))
                if isinstance(q, dict):
                    q = q.get("text", "")
                a = ex.get("answer", ex.get("label", ""))
                if isinstance(a, dict):
                    a = a.get("value", str(a))
                if q and a:
                    queries.append({"query": str(q), "answer": str(a), "source": name})
            logger.info("Loaded %d queries from %s", min(len(ds), max_s), name)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)

    if not queries:
        logger.info("Generating synthetic training queries...")
        for i in range(5000):
            queries.append({
                "query": f"What is the capital of country {i % 200}?",
                "answer": f"Capital City {i % 200}",
                "source": "synthetic",
            })
    logger.info("Total training queries: %d", len(queries))
    return queries


def format_policy_prompt(query: str, state_desc: str) -> str:
    """Format input for the LLM policy."""
    return (
        f"<|im_start|>system\nYou are a RAG policy controller. "
        f"Decide the best retrieval action for the given query.\n"
        f"Actions: no_retrieve, retrieve_1, retrieve_3, retrieve_5, retrieve_10, rewrite, multi_hop\n"
        f"Consider: query complexity, expected accuracy gain, retrieval cost.<|im_end|>\n"
        f"<|im_start|>user\nQuery: {query}\nState: {state_desc}\n"
        f"Choose the best action.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


ACTION_NAMES = ["no_retrieve", "retrieve_1", "retrieve_3", "retrieve_5", "retrieve_10", "rewrite", "multi_hop"]


def grpo_update(policy_model, tokenizer, rollout_buffer, optimizer, config, device):
    """GRPO update step (DDP-compatible with no_sync for gradient accumulation)."""
    grpo_cfg = config["grpo"]
    clip_range = grpo_cfg["clip_range"]
    kl_coeff = grpo_cfg["kl_coeff"]
    entropy_coeff = grpo_cfg["entropy_coeff"]

    if not rollout_buffer:
        return {}

    query_groups = defaultdict(list)
    for item in rollout_buffer:
        query_groups[item["query_id"]].append(item)

    update_items = []
    for qid, group in query_groups.items():
        if len(group) < 2:
            continue
        rewards = torch.tensor([g["reward"] for g in group], device=device)
        mean_r = rewards.mean()
        std_r = rewards.std().clamp(min=1e-8)
        advantages = (rewards - mean_r) / std_r
        for g, adv in zip(group, advantages):
            update_items.append((g, adv))

    if not update_items:
        return {}

    total_policy_loss = 0.0
    total_entropy = 0.0
    n = len(update_items)
    no_sync_fn = getattr(policy_model, "no_sync", None)

    for i, (g, adv) in enumerate(update_items):
        sync_ctx = no_sync_fn() if (i < n - 1 and no_sync_fn) else nullcontext()
        with sync_ctx:
            inputs = tokenizer(
                g["prompt"], return_tensors="pt", truncation=True, max_length=512,
            ).to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = policy_model(**inputs)
                logits = outputs.logits[:, -1, :]

            action_token_ids = [tokenizer.encode(name, add_special_tokens=False)[0] for name in ACTION_NAMES]
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

            loss = (policy_loss + kl_penalty - entropy_coeff * entropy) / n
            loss.backward()

        total_policy_loss += policy_loss.item()
        total_entropy += entropy.item()

    torch.nn.utils.clip_grad_norm_(
        [p for p in policy_model.parameters() if p.requires_grad], 1.0
    )
    optimizer.step()
    optimizer.zero_grad()

    return {
        "policy_loss": total_policy_loss / n,
        "entropy": total_entropy / n,
        "num_updates": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RAG policy with GRPO")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/policy")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{max(local_rank, 0)}")
    is_main = local_rank <= 0

    policy_model_name = config["policy"]["model"]
    if is_main:
        logger.info("=== RAG Policy Training (GRPO) ===")
        logger.info("Policy model: %s", policy_model_name)
        logger.info("World size: %d", world_size)

    tokenizer = AutoTokenizer.from_pretrained(policy_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    policy_model = policy_model.to(device)
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
    if config["training"].get("gradient_checkpointing"):
        policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if is_main:
        policy_model.print_trainable_parameters()

    if local_rank >= 0:
        policy_model = DDP(policy_model, device_ids=[local_rank], find_unused_parameters=True)

    reward_fn = RAGRewardFunction(
        cost_lambda=config["environment"]["cost_lambda"],
        accuracy_weight=config["environment"]["accuracy_weight"],
    )
    env = RAGEnvironment(reward_fn=reward_fn, max_steps=3)

    queries = load_training_queries(config)
    if world_size > 1:
        per_rank = len(queries) // world_size
        start = local_rank * per_rank
        queries = queries[start:start + per_rank]
        if is_main:
            logger.info("Data split: %d queries/rank (%d total)", per_rank, per_rank * world_size)

    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
    )

    train_cfg = config["training"]
    grpo_cfg = config["grpo"]
    num_epochs = train_cfg["num_train_epochs"]
    num_generations = grpo_cfg["num_generations"]
    temperature = grpo_cfg["temperature"]

    training_log = []
    global_step = 0

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_costs = []

        import random
        random.shuffle(queries)

        for batch_idx in range(0, len(queries), train_cfg["per_device_train_batch_size"]):
            batch = queries[batch_idx:batch_idx + train_cfg["per_device_train_batch_size"]]
            rollout_buffer = []

            for qi, item in enumerate(batch):
                query = item["query"]
                gold = item["answer"]
                query_id = f"{epoch}_{batch_idx}_{qi}"

                for gen_idx in range(num_generations):
                    state = env.reset(query, gold)
                    state_desc = f"complexity={state.query_complexity:.2f}, retrievals={state.num_retrievals_done}"
                    prompt = format_policy_prompt(query, state_desc)

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
                    with torch.no_grad():
                        outputs = policy_model(**inputs)
                        logits = outputs.logits[:, -1, :]

                    action_token_ids = [tokenizer.encode(name, add_special_tokens=False)[0] for name in ACTION_NAMES]
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

            update_info = grpo_update(policy_model, tokenizer, rollout_buffer, optimizer, config, device)
            global_step += 1

            if is_main and global_step % train_cfg["logging_steps"] == 0:
                avg_reward = sum(epoch_rewards[-100:]) / max(len(epoch_rewards[-100:]), 1)
                avg_cost = sum(epoch_costs[-100:]) / max(len(epoch_costs[-100:]), 1)
                logger.info(
                    "Epoch %d Step %d | Reward: %.4f | Cost: %.4f | Policy loss: %.4f | Entropy: %.4f",
                    epoch, global_step, avg_reward, avg_cost,
                    update_info.get("policy_loss", 0), update_info.get("entropy", 0),
                )

            if is_main and global_step % train_cfg["save_steps"] == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_model = policy_model.module if hasattr(policy_model, "module") else policy_model
                save_model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

        avg_ep_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_ep_cost = sum(epoch_costs) / max(len(epoch_costs), 1)
        training_log.append({
            "epoch": epoch,
            "avg_reward": avg_ep_reward,
            "avg_cost": avg_ep_cost,
            "num_rollouts": len(epoch_rewards),
        })
        if is_main:
            logger.info("=== Epoch %d complete | Avg reward: %.4f | Avg cost: %.4f ===",
                        epoch, avg_ep_reward, avg_ep_cost)

    if is_main:
        save_model = policy_model.module if hasattr(policy_model, "module") else policy_model
        save_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)
        logger.info("=== Training complete. Model saved to %s ===", args.output_dir)

    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
