#!/usr/bin/env python3
"""
Create oracle policy for warm-starting GRPO.

For each question, try all 7 RAG actions, record which gives best
accuracy-cost tradeoff, and train an oracle MLP policy on
(question_embedding → best_action). Provides warm start for GRPO.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag_environment import (
    RAGAction, ACTION_COSTS, RAGEnvironment, RAGRewardFunction, PolicyNetwork,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_oracle_policy")


def parse_args():
    parser = argparse.ArgumentParser(description="Train oracle policy for GRPO warm-start")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/datasets",
                        help="Directory with prepared dataset JSONL files")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/oracle_policy")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--max_queries_per_dataset", type=int, default=2000)
    parser.add_argument("--cost_lambda", type=float, default=0.3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_queries(data_dir, max_per_dataset=2000):
    """Load queries from prepared JSONL files."""
    queries = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".jsonl"):
            continue
        ds_name = fname.replace(".jsonl", "")
        with open(os.path.join(data_dir, fname)) as f:
            ds_queries = [json.loads(line.strip()) for line in f]
        if len(ds_queries) > max_per_dataset:
            np.random.shuffle(ds_queries)
            ds_queries = ds_queries[:max_per_dataset]
        queries.extend(ds_queries)
        logger.info(f"Loaded {len(ds_queries)} queries from {ds_name}")
    return queries


def encode_queries(queries, embedding_model_name, batch_size=128):
    """Encode queries into embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model_name)
    except ImportError:
        logger.warning("sentence-transformers not installed, using random embeddings")
        dim = 1024
        embeddings = np.random.randn(len(queries), dim).astype(np.float32)
        return embeddings, dim

    texts = [q["query"] for q in queries]
    logger.info(f"Encoding {len(texts)} queries...")
    embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                normalize_embeddings=True)
    dim = embeddings.shape[1]
    logger.info(f"Embedding dim: {dim}")
    return embeddings.astype(np.float32), dim


def find_oracle_actions(queries, config, cost_lambda=0.3):
    """For each query, try all 7 actions and find the best one."""
    logger.info("Finding oracle actions for each query...")

    reward_fn = RAGRewardFunction(cost_lambda=cost_lambda)
    env = RAGEnvironment(reward_fn=reward_fn, max_steps=3)

    oracle_actions = []
    action_dist = {a.name: 0 for a in RAGAction}

    for item in tqdm(queries, desc="Finding oracle actions"):
        query = item["query"]
        gold = item["answer"]

        best_action = RAGAction.NO_RETRIEVE
        best_reward = -float("inf")
        action_rewards = {}

        for action in RAGAction:
            state = env.reset(query, gold)
            transition = env.step(action)
            reward = transition.reward
            action_rewards[action.name] = reward

            if reward > best_reward:
                best_reward = reward
                best_action = action

        oracle_actions.append(best_action.value)
        action_dist[best_action.name] += 1

    logger.info("Oracle action distribution:")
    total = len(queries)
    for name, count in sorted(action_dist.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:15s}: {count:5d} ({count/total:.1%})")

    return np.array(oracle_actions, dtype=np.int64), action_dist


class OracleDataset(Dataset):
    def __init__(self, embeddings, actions):
        self.embeddings = torch.from_numpy(embeddings)
        self.actions = torch.from_numpy(actions)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.actions[idx]


class OraclePolicyMLP(nn.Module):
    """MLP that maps query embedding to action distribution."""

    def __init__(self, input_dim, hidden_dim=128, num_actions=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_oracle(dataset, input_dim, args, device):
    """Train the oracle MLP policy."""
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = OraclePolicyMLP(input_dim, args.hidden_dim, num_actions=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    logger.info(f"Oracle MLP: {sum(p.numel() for p in model.parameters()):,} parameters")

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for embs, labels in train_loader:
            embs, labels = embs.to(device), labels.to(device)
            logits = model(embs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * embs.shape[0]
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total += embs.shape[0]

        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for embs, labels in val_loader:
                embs, labels = embs.to(device), labels.to(device)
                logits = model(embs)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += embs.shape[0]

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        epoch_loss = train_loss / max(train_total, 1)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            logger.info(f"Epoch {epoch:3d}: loss={epoch_loss:.4f} "
                        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.data_dir):
        queries = load_queries(args.data_dir, args.max_queries_per_dataset)
    else:
        logger.info("Data dir not found, loading from HF datasets directly...")
        from src.rag_environment import RAGAction
        queries = []
        for ds_cfg in config["evaluation"]["datasets"]:
            try:
                subset = ds_cfg.get("subset")
                split = ds_cfg.get("split", "validation")
                if subset:
                    ds = load_dataset(ds_cfg["dataset_id"], subset, split=split, trust_remote_code=True)
                else:
                    ds = load_dataset(ds_cfg["dataset_id"], split=split, trust_remote_code=True)

                max_s = min(len(ds), args.max_queries_per_dataset)
                ds = ds.shuffle(seed=42).select(range(max_s))
                for ex in ds:
                    q = ex.get("question", ex.get("claim", ""))
                    if isinstance(q, dict):
                        q = q.get("text", "")
                    a = ex.get("answer", ex.get("label", ""))
                    if isinstance(a, dict):
                        a = a.get("aliases", [a.get("value", "")])[0] if a.get("aliases") else a.get("value", str(a))
                    if isinstance(a, list):
                        a = a[0] if a else ""
                    if q:
                        queries.append({"query": str(q), "answer": str(a)})
                logger.info(f"Loaded {max_s} from {ds_cfg['name']}")
            except Exception as e:
                logger.warning(f"Failed to load {ds_cfg['name']}: {e}")

    if not queries:
        logger.error("No queries found. Run setup_rag_infrastructure.py first.")
        return

    logger.info(f"Total queries: {len(queries)}")

    embeddings, embed_dim = encode_queries(queries, args.embedding_model)

    oracle_actions, action_dist = find_oracle_actions(
        queries, config, cost_lambda=args.cost_lambda,
    )

    dataset = OracleDataset(embeddings, oracle_actions)

    logger.info("\nTraining oracle MLP policy...")
    model, history, best_acc = train_oracle(dataset, embed_dim, args, device)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "oracle_policy.pt"))

    summary = {
        "input_dim": embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_actions": 7,
        "best_val_accuracy": best_acc,
        "num_queries": len(queries),
        "oracle_action_distribution": action_dist,
        "cost_lambda": args.cost_lambda,
        "embedding_model": args.embedding_model,
    }
    with open(os.path.join(args.output_dir, "oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nOracle policy trained. Best val accuracy: {best_acc:.4f}")
    logger.info(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
