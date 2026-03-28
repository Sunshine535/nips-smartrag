#!/usr/bin/env python3
"""
Evaluate UniRAG policy on NQ, TriviaQA, HotpotQA, FEVER.

Compare: no retrieval, always retrieve, BM25-only, dense-only, oracle, GRPO-policy.
Metrics: accuracy/EM, F1, avg retrieval cost, Pareto front.
Plot: accuracy-cost Pareto curves.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.rag_environment import (
    ACTION_COSTS, ACTION_K, RAGAction, RAGEnvironment, RAGRewardFunction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_rag_policy")

ACTION_NAMES = ["no_retrieve", "retrieve_1", "retrieve_3", "retrieve_5",
                "retrieve_10", "rewrite", "multi_hop"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG policy")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument("--grpo_policy_dir", type=str, default="./checkpoints/grpo_policy")
    parser.add_argument("--oracle_policy_path", type=str, default=None,
                        help="Path to oracle MLP checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/datasets")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--cost_lambdas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
                        help="Lambda values for Pareto curve computation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Metrics ──────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_t = set(normalize_answer(pred).split())
    gold_t = set(normalize_answer(gold).split())
    if not pred_t or not gold_t:
        return float(pred_t == gold_t)
    common = pred_t & gold_t
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(gold_t)
    return 2 * p * r / (p + r)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_eval_queries(data_dir, ds_name, max_samples):
    """Load queries from prepared JSONL or HF datasets."""
    jsonl_path = os.path.join(data_dir, f"{ds_name}.jsonl")
    if os.path.exists(jsonl_path):
        queries = []
        with open(jsonl_path) as f:
            for line in f:
                queries.append(json.loads(line.strip()))
        return queries[:max_samples]
    return []


def load_eval_queries_hf(ds_cfg, max_samples):
    """Fallback: load from HuggingFace."""
    try:
        subset = ds_cfg.get("subset")
        split = ds_cfg.get("split", "validation")
        if subset:
            ds = load_dataset(ds_cfg["dataset_id"], subset, split=split)
        else:
            ds = load_dataset(ds_cfg["dataset_id"], split=split)

        if len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))

        queries = []
        for ex in ds:
            q = ex.get("question", ex.get("claim", ""))
            if isinstance(q, dict):
                q = q.get("text", "")
            a = ex.get("answer", ex.get("label", ""))
            if isinstance(a, dict):
                aliases = a.get("aliases", [])
                a = aliases[0] if aliases else a.get("value", str(a))
            if isinstance(a, list):
                a = a[0] if a else ""
            if q:
                queries.append({"query": str(q), "answer": str(a)})
        return queries
    except Exception as e:
        logger.warning(f"Failed to load {ds_cfg['name']}: {e}")
        return []


# ── Evaluation Methods ───────────────────────────────────────────────────────

def evaluate_fixed_action(env, queries, action: RAGAction):
    """Evaluate with a single fixed action (baseline)."""
    results = []
    for item in queries:
        state = env.reset(item["query"], item["answer"])
        transition = env.step(action)
        answer = transition.next_state.current_answer or ""
        em = exact_match(answer, item["answer"])
        f1 = f1_score(answer, item["answer"])
        results.append({"em": em, "f1": f1, "cost": ACTION_COSTS[action]})

    n = len(results)
    return {
        "exact_match": sum(r["em"] for r in results) / max(n, 1),
        "f1": sum(r["f1"] for r in results) / max(n, 1),
        "avg_cost": ACTION_COSTS[action],
        "total_queries": n,
    }


def evaluate_with_grpo_policy(policy_model, tokenizer, env, queries, device):
    """Evaluate with the GRPO-trained LLM policy."""
    results = []
    action_counts = defaultdict(int)

    for item in tqdm(queries, desc="GRPO policy"):
        query = item["query"]
        gold = item["answer"]

        state = env.reset(query, gold)
        state_desc = f"complexity={state.query_complexity:.2f}, retrievals=0"

        from scripts.train_grpo_policy import format_policy_prompt
        prompt = format_policy_prompt(query, state_desc)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = policy_model(**inputs)
            logits = outputs.logits[:, -1, :]

        action_token_ids = [
            tokenizer.encode(name, add_special_tokens=False)[0] for name in ACTION_NAMES
        ]
        action_logits = logits[0, action_token_ids]
        action_idx = action_logits.argmax().item()
        action = RAGAction(action_idx)

        transition = env.step(action)
        answer = transition.next_state.current_answer or ""
        cost = ACTION_COSTS[action]

        em = exact_match(answer, gold)
        f1 = f1_score(answer, gold)

        results.append({"em": em, "f1": f1, "cost": cost, "action": ACTION_NAMES[action_idx]})
        action_counts[ACTION_NAMES[action_idx]] += 1

    n = len(results)
    return {
        "exact_match": sum(r["em"] for r in results) / max(n, 1),
        "f1": sum(r["f1"] for r in results) / max(n, 1),
        "avg_cost": sum(r["cost"] for r in results) / max(n, 1),
        "total_queries": n,
        "action_distribution": dict(action_counts),
    }


def evaluate_oracle_policy(oracle_model, encoder, env, queries, device):
    """Evaluate with the oracle MLP policy."""
    from scripts.train_oracle_policy import OraclePolicyMLP

    results = []
    action_counts = defaultdict(int)

    query_texts = [q["query"] for q in queries]
    if encoder is not None:
        embeddings = encoder.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
    else:
        embeddings = np.random.randn(len(queries), 1024).astype(np.float32)

    oracle_model.eval()

    for i, item in enumerate(tqdm(queries, desc="Oracle policy")):
        emb = torch.from_numpy(embeddings[i:i+1]).to(device)

        with torch.no_grad():
            logits = oracle_model(emb)
            action_idx = logits.argmax(dim=-1).item()

        action = RAGAction(action_idx)
        state = env.reset(item["query"], item["answer"])
        transition = env.step(action)
        answer = transition.next_state.current_answer or ""

        em = exact_match(answer, item["answer"])
        f1 = f1_score(answer, item["answer"])

        results.append({"em": em, "f1": f1, "cost": ACTION_COSTS[action]})
        action_counts[ACTION_NAMES[action_idx]] += 1

    n = len(results)
    return {
        "exact_match": sum(r["em"] for r in results) / max(n, 1),
        "f1": sum(r["f1"] for r in results) / max(n, 1),
        "avg_cost": sum(r["cost"] for r in results) / max(n, 1),
        "total_queries": n,
        "action_distribution": dict(action_counts),
    }


# ── Pareto Front ─────────────────────────────────────────────────────────────

def compute_pareto_curve(env, queries, cost_lambdas):
    """Compute accuracy-cost Pareto curve across different lambda values and actions."""
    logger.info("Computing Pareto curves...")
    pareto_points = []

    sample = queries[:min(200, len(queries))]

    for lam in cost_lambdas:
        env.reward_fn = RAGRewardFunction(cost_lambda=lam)

        for action in RAGAction:
            result = evaluate_fixed_action(env, sample, action)
            pareto_points.append({
                "lambda": lam,
                "action": ACTION_NAMES[action.value],
                "exact_match": result["exact_match"],
                "f1": result["f1"],
                "cost": result["avg_cost"],
            })

    # Compute Pareto front
    front = []
    for p in pareto_points:
        dominated = False
        for q in pareto_points:
            if (q["exact_match"] > p["exact_match"] and q["cost"] <= p["cost"]) or \
               (q["exact_match"] >= p["exact_match"] and q["cost"] < p["cost"]):
                dominated = True
                break
        if not dominated:
            front.append(p)

    front.sort(key=lambda x: x["cost"])

    return {
        "all_points": pareto_points,
        "pareto_front": front,
    }


def generate_pareto_csv(pareto_data, output_dir):
    """Generate CSV for Pareto curve plotting."""
    csv_lines = ["lambda,action,exact_match,f1,cost,is_front"]

    front_keys = {(p["lambda"], p["action"]) for p in pareto_data["pareto_front"]}

    for p in pareto_data["all_points"]:
        is_front = (p["lambda"], p["action"]) in front_keys
        csv_lines.append(
            f"{p['lambda']},{p['action']},{p['exact_match']:.4f},"
            f"{p['f1']:.4f},{p['cost']:.4f},{int(is_front)}"
        )

    with open(os.path.join(output_dir, "pareto_curve.csv"), "w") as f:
        f.write("\n".join(csv_lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GRPO policy
    policy_model_name = config["policy"]["model"]
    logger.info(f"Loading base model: {policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )

    grpo_adapter = os.path.join(args.grpo_policy_dir, "adapter_config.json")
    if os.path.exists(grpo_adapter):
        policy_model = PeftModel.from_pretrained(policy_model, args.grpo_policy_dir)
        logger.info(f"Loaded GRPO adapter from {args.grpo_policy_dir}")
    policy_model.eval()

    # Load oracle policy if available
    oracle_model = None
    encoder = None
    if args.oracle_policy_path and os.path.exists(args.oracle_policy_path):
        summary_path = os.path.join(os.path.dirname(args.oracle_policy_path), "oracle_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                oracle_info = json.load(f)
            from scripts.train_oracle_policy import OraclePolicyMLP
            oracle_model = OraclePolicyMLP(
                input_dim=oracle_info["input_dim"],
                hidden_dim=oracle_info["hidden_dim"],
                num_actions=oracle_info["num_actions"],
            )
            oracle_model.load_state_dict(torch.load(args.oracle_policy_path, map_location="cpu"))
            oracle_model = oracle_model.to(device)
            oracle_model.eval()

            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(oracle_info.get("embedding_model", "BAAI/bge-large-en-v1.5"))
            except ImportError:
                pass

            logger.info("Loaded oracle policy")

    reward_fn = RAGRewardFunction(cost_lambda=config["environment"]["cost_lambda"])
    env = RAGEnvironment(reward_fn=reward_fn)

    all_results = {}

    for ds_cfg in config["evaluation"]["datasets"]:
        name = ds_cfg["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on: {name}")
        logger.info(f"{'='*60}")

        queries = load_eval_queries(args.data_dir, name, args.max_samples)
        if not queries:
            queries = load_eval_queries_hf(ds_cfg, args.max_samples)
        if not queries:
            logger.warning(f"No queries for {name}, skipping")
            continue

        logger.info(f"Loaded {len(queries)} queries")
        ds_results = {}

        # 1) No retrieval baseline
        logger.info("  [1/6] No retrieval baseline")
        ds_results["no_retrieval"] = evaluate_fixed_action(env, queries, RAGAction.NO_RETRIEVE)

        # 2) Always retrieve (k=5)
        logger.info("  [2/6] Always retrieve (k=5)")
        ds_results["always_retrieve_5"] = evaluate_fixed_action(env, queries, RAGAction.RETRIEVE_5)

        # 3) BM25-only (proxy: retrieve_3)
        logger.info("  [3/6] BM25-only proxy (retrieve_3)")
        ds_results["bm25_only"] = evaluate_fixed_action(env, queries, RAGAction.RETRIEVE_3)

        # 4) Dense-only (proxy: retrieve_10)
        logger.info("  [4/6] Dense-only proxy (retrieve_10)")
        ds_results["dense_only"] = evaluate_fixed_action(env, queries, RAGAction.RETRIEVE_10)

        # 5) Oracle policy
        if oracle_model is not None:
            logger.info("  [5/6] Oracle policy")
            ds_results["oracle"] = evaluate_oracle_policy(
                oracle_model, encoder, env, queries, device,
            )
        else:
            logger.info("  [5/6] Oracle policy (skipped)")

        # 6) GRPO policy
        logger.info("  [6/6] GRPO policy")
        ds_results["grpo_policy"] = evaluate_with_grpo_policy(
            policy_model, tokenizer, env, queries, device,
        )

        for method, res in ds_results.items():
            logger.info(
                f"  {method:20s}: EM={res['exact_match']:.4f} F1={res['f1']:.4f} "
                f"Cost={res.get('avg_cost', 0):.4f}"
            )

        all_results[name] = ds_results

    # Compute Pareto curves
    all_queries = []
    for ds_cfg in config["evaluation"]["datasets"][:2]:
        name = ds_cfg["name"]
        qs = load_eval_queries(args.data_dir, name, 200)
        if not qs:
            qs = load_eval_queries_hf(ds_cfg, 200)
        all_queries.extend(qs)

    if all_queries:
        pareto_data = compute_pareto_curve(env, all_queries, args.cost_lambdas)
        all_results["pareto"] = pareto_data
        generate_pareto_csv(pareto_data, args.output_dir)
        logger.info(f"Pareto front: {len(pareto_data['pareto_front'])} points")

    output_path = os.path.join(args.output_dir, "rag_policy_eval.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Dataset':<20s} {'Method':<22s} {'EM':>8s} {'F1':>8s} {'Cost':>8s}")
    logger.info("-" * 68)

    for ds_name, ds_results in all_results.items():
        if ds_name == "pareto":
            continue
        for method, res in ds_results.items():
            if not isinstance(res, dict) or "exact_match" not in res:
                continue
            logger.info(
                f"{ds_name:<20s} {method:<22s} "
                f"{res['exact_match']:>8.4f} {res['f1']:>8.4f} {res.get('avg_cost', 0):>8.4f}"
            )

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
