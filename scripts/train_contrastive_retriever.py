#!/usr/bin/env python3
"""Train graph-contrastive retriever with InfoNCE + synonym loss + polysemy loss + Laplacian reg.
Base encoder: BAAI/bge-large-en-v1.5. Hard negatives via BM25."""

import argparse
import json
import logging
import os
import pickle
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph_retriever import GraphContrastiveRetriever, SynonymGraph


class QARetrievalDataset(Dataset):
    """(query, positive_doc, negative_doc) triplets with optional synonym pairs."""

    def __init__(self, queries, pos_docs, neg_docs, tokenizer, max_length=512):
        self.queries = queries
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "pos_doc": self.pos_docs[idx],
            "neg_doc": self.neg_docs[idx],
        }


def collate_fn(batch, tokenizer, max_length=512):
    queries = [b["query"] for b in batch]
    pos_docs = [b["pos_doc"] for b in batch]
    neg_docs = [b["neg_doc"] for b in batch]
    q_enc = tokenizer(queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    p_enc = tokenizer(pos_docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    n_enc = tokenizer(neg_docs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {"query": q_enc, "pos_doc": p_enc, "neg_doc": n_enc}


def build_bm25_hard_negatives(queries: list, corpus: list, top_k: int = 10) -> list:
    """Build hard negatives using BM25 scoring."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        hard_negs = []
        for i, q in enumerate(queries):
            scores = bm25.get_scores(q.lower().split())
            ranked = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
            neg_idx = ranked[1] if len(ranked) > 1 else 0
            hard_negs.append(corpus[neg_idx])
            if i % 5000 == 0 and i > 0:
                logger.info("  BM25 hard negatives: %d/%d", i, len(queries))
        return hard_negs
    except ImportError:
        logger.warning("rank_bm25 not installed, using random negatives")
        import random
        return [random.choice(corpus) for _ in queries]


def load_training_data(config: dict, tokenizer):
    """Load NQ train set with BM25 hard negatives."""
    logger.info("Loading training data...")
    queries, pos_docs, neg_docs = [], [], []

    try:
        nq = load_dataset("google-research-datasets/natural_questions", "default",
                          split="train", streaming=True, trust_remote_code=True)
        count = 0
        for ex in nq:
            if count >= 50000:
                break
            q = ex.get("question", {})
            question = q.get("text", "") if isinstance(q, dict) else str(q)
            answer = ""
            annotations = ex.get("annotations", {})
            if isinstance(annotations, dict):
                sa = annotations.get("short_answers", [])
                if sa and isinstance(sa, list) and len(sa) > 0:
                    answer = str(sa[0])
            if question and len(question) > 5:
                queries.append(question)
                pos_docs.append(answer if answer else question)
                count += 1
    except Exception as e:
        logger.warning("Failed to load NQ: %s. Using synthetic data.", e)

    if len(queries) < 1000:
        logger.info("Generating synthetic training data...")
        for i in range(20000):
            queries.append(f"What is the answer to question number {i}?")
            pos_docs.append(f"The answer to question {i} involves concept {i % 100} in domain {i % 10}.")

    logger.info("Building BM25 hard negatives for %d queries...", len(queries))
    neg_docs = build_bm25_hard_negatives(queries, pos_docs)

    logger.info("Training data: %d triplets", len(queries))
    return QARetrievalDataset(queries, pos_docs, neg_docs, tokenizer)


def build_synonym_pairs(graph: SynonymGraph, max_pairs: int = 10000) -> list:
    """Extract synonym pairs from graph for synonym contrastive loss."""
    pairs = []
    for src, dst, etype in graph.edges:
        if etype in ("synonym", "embedding_synonym", "paraphrase"):
            if src < len(graph.node_texts) and dst < len(graph.node_texts):
                pairs.append((graph.node_texts[src], graph.node_texts[dst]))
            if len(pairs) >= max_pairs:
                break
    logger.info("Extracted %d synonym pairs for training", len(pairs))
    return pairs


def build_polysemy_groups(graph: SynonymGraph, max_groups: int = 5000) -> list:
    """Extract polysemy groups: (word, [sense1, sense2, ...])."""
    poly_map = {}
    for src, dst, etype in graph.edges:
        if etype == "polysemy" and src < len(graph.node_texts):
            parent = graph.node_texts[src]
            child = graph.node_texts[dst]
            if parent not in poly_map:
                poly_map[parent] = []
            poly_map[parent].append(child)

    groups = [(word, senses) for word, senses in poly_map.items() if len(senses) >= 2]
    logger.info("Extracted %d polysemy groups for training", len(groups))
    return groups[:max_groups]


def synonym_contrastive_loss(encoder, tokenizer, retriever, synonym_pairs: list,
                             device, temperature: float = 0.1, max_batch: int = 64) -> torch.Tensor:
    """Pull synonym embeddings together, push non-synonyms apart."""
    if not synonym_pairs:
        return torch.tensor(0.0, device=device)

    import random
    batch = random.sample(synonym_pairs, min(max_batch, len(synonym_pairs)))
    texts_a = [p[0] for p in batch]
    texts_b = [p[1] for p in batch]

    enc_a = tokenizer(texts_a, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc_b = tokenizer(texts_b, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc_a = {k: v.to(device) for k, v in enc_a.items()}
    enc_b = {k: v.to(device) for k, v in enc_b.items()}

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        emb_a = encoder(**enc_a).last_hidden_state[:, 0, :]
        emb_b = encoder(**enc_b).last_hidden_state[:, 0, :]
        proj_a = F.normalize(retriever.encode_docs(emb_a), p=2, dim=-1)
        proj_b = F.normalize(retriever.encode_docs(emb_b), p=2, dim=-1)

    sim_matrix = proj_a @ proj_b.T / temperature
    labels = torch.arange(sim_matrix.size(0), device=device)
    return (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2


def polysemy_discrimination_loss(encoder, tokenizer, retriever, polysemy_groups: list,
                                 device, margin: float = 0.3, max_batch: int = 32) -> torch.Tensor:
    """Push different senses of polysemous words apart."""
    if not polysemy_groups:
        return torch.tensor(0.0, device=device)

    import random
    batch = random.sample(polysemy_groups, min(max_batch, len(polysemy_groups)))

    loss = torch.tensor(0.0, device=device)
    count = 0
    for word, senses in batch:
        if len(senses) < 2:
            continue
        sense_pair = random.sample(senses, 2)
        texts = [word] + sense_pair
        enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            emb = encoder(**enc).last_hidden_state[:, 0, :]
            proj = F.normalize(retriever.encode_docs(emb), p=2, dim=-1)

        sim_01 = (proj[0] * proj[1]).sum()
        sim_02 = (proj[0] * proj[2]).sum()
        sim_12 = (proj[1] * proj[2]).sum()
        loss += F.relu(margin - (sim_01 - sim_12).abs())
        loss += F.relu(margin - (sim_02 - sim_12).abs())
        count += 2

    return loss / max(count, 1)


def encode_texts(model, inputs, device):
    """Get [CLS] embedding from encoder model."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


def main():
    parser = argparse.ArgumentParser(description="Train graph-contrastive retriever")
    parser.add_argument("--config", type=str, default="configs/graph_config.yaml")
    parser.add_argument("--graph_dir", type=str, default="outputs/graph")
    parser.add_argument("--output_dir", type=str, default="outputs/retriever")
    parser.add_argument("--synonym_weight", type=float, default=0.3)
    parser.add_argument("--polysemy_weight", type=float, default=0.2)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank <= 0
    os.makedirs(args.output_dir, exist_ok=True)

    ret_cfg = config["retriever"]
    train_cfg = config["training"]

    encoder_name = ret_cfg["base_model"]
    logger.info("Loading encoder: %s", encoder_name)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float32,
                                        trust_remote_code=True).to(device)

    retriever = GraphContrastiveRetriever(
        encoder_dim=ret_cfg["hidden_dim"],
        gnn_hidden=ret_cfg["hidden_dim"],
        gnn_layers=ret_cfg["graph_gnn_layers"],
        gnn_heads=ret_cfg["gnn_heads"],
        dropout=ret_cfg["gnn_dropout"],
    ).to(device)

    # Load graph
    laplacian, node_embeds, adj = None, None, None
    graph = None
    graph_path = os.path.join(args.graph_dir, "synonym_graph.pkl")
    if os.path.exists(graph_path):
        logger.info("Loading graph from %s", args.graph_dir)
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        laplacian = torch.load(os.path.join(args.graph_dir, "laplacian.pt"), map_location=device)
        adj = torch.load(os.path.join(args.graph_dir, "adjacency.pt"), map_location=device)
        emb_path = os.path.join(args.graph_dir, "node_embeddings.pt")
        if os.path.exists(emb_path):
            node_embeds = torch.load(emb_path, map_location=device)
    else:
        logger.warning("No graph found at %s, training without graph losses", graph_path)

    synonym_pairs = build_synonym_pairs(graph) if graph else []
    polysemy_groups = build_polysemy_groups(graph) if graph else []

    if local_rank >= 0:
        retriever = DDP(retriever, device_ids=[local_rank], find_unused_parameters=True)

    dataset = load_training_data(config, tokenizer)
    sampler = DistributedSampler(dataset) if local_rank >= 0 else None
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_train_batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=train_cfg["dataloader_num_workers"],
        collate_fn=lambda batch: collate_fn(batch, tokenizer, train_cfg["max_seq_length"]),
        pin_memory=True,
    )

    all_params = list(retriever.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=train_cfg["learning_rate"],
                                  weight_decay=train_cfg["weight_decay"])
    total_steps = len(dataloader) * train_cfg["num_train_epochs"] // train_cfg["gradient_accumulation_steps"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler()
    retriever_module = retriever.module if hasattr(retriever, "module") else retriever

    logger.info("Starting training: %d epochs, %d steps/epoch", train_cfg["num_train_epochs"], len(dataloader))
    logger.info("Loss weights: synonym=%.2f, polysemy=%.2f, laplacian=%.2f",
                args.synonym_weight, args.polysemy_weight, ret_cfg["graph_laplacian_weight"])

    global_step = 0
    for epoch in range(train_cfg["num_train_epochs"]):
        if sampler:
            sampler.set_epoch(epoch)
        retriever.train()
        encoder.train()
        epoch_losses = {"total": 0.0, "info_nce": 0.0, "synonym": 0.0,
                        "polysemy": 0.0, "laplacian": 0.0}

        for step, batch in enumerate(dataloader):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                q_emb = encode_texts(encoder, batch["query"], device)
                p_emb = encode_texts(encoder, batch["pos_doc"], device)
                n_emb = encode_texts(encoder, batch["neg_doc"], device)

                graph_sub = min(1000, node_embeds.shape[0]) if node_embeds is not None else 0
                losses = retriever_module.forward(
                    query_embeds=q_emb,
                    pos_doc_embeds=p_emb,
                    neg_doc_embeds=n_emb,
                    node_embeds=node_embeds[:graph_sub] if node_embeds is not None else None,
                    adj=adj[:graph_sub, :graph_sub] if adj is not None else None,
                    laplacian=laplacian[:graph_sub, :graph_sub] if laplacian is not None else None,
                    temperature=ret_cfg["contrastive_temperature"],
                    laplacian_weight=ret_cfg["graph_laplacian_weight"],
                )

                syn_loss = synonym_contrastive_loss(
                    encoder, tokenizer, retriever_module, synonym_pairs, device
                )
                poly_loss = polysemy_discrimination_loss(
                    encoder, tokenizer, retriever_module, polysemy_groups, device
                )

                total_loss = (losses["loss"]
                              + args.synonym_weight * syn_loss
                              + args.polysemy_weight * poly_loss)
                total_loss = total_loss / train_cfg["gradient_accumulation_steps"]

            scaler.scale(total_loss).backward()

            epoch_losses["total"] += total_loss.item()
            epoch_losses["info_nce"] += losses["info_nce"].item()
            epoch_losses["synonym"] += syn_loss.item()
            epoch_losses["polysemy"] += poly_loss.item()
            epoch_losses["laplacian"] += losses["laplacian_loss"].item()

            if (step + 1) % train_cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if is_main and global_step % train_cfg["logging_steps"] == 0:
                    logger.info(
                        "Epoch %d Step %d | Loss: %.4f | InfoNCE: %.4f | Syn: %.4f | Poly: %.4f | Lap: %.4f | LR: %.2e",
                        epoch, step, total_loss.item() * train_cfg["gradient_accumulation_steps"],
                        losses["info_nce"].item(), syn_loss.item(), poly_loss.item(),
                        losses["laplacian_loss"].item(), scheduler.get_last_lr()[0],
                    )

                if is_main and global_step % train_cfg["save_steps"] == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(retriever_module.state_dict(), os.path.join(ckpt_dir, "retriever.pt"))
                    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder.pt"))

        if is_main:
            n = max(len(dataloader), 1)
            logger.info("Epoch %d complete. Avg losses: total=%.4f info_nce=%.4f syn=%.4f poly=%.4f lap=%.4f",
                        epoch, epoch_losses["total"] / n, epoch_losses["info_nce"] / n,
                        epoch_losses["synonym"] / n, epoch_losses["polysemy"] / n,
                        epoch_losses["laplacian"] / n)

    if is_main:
        logger.info("Saving final model to %s", args.output_dir)
        torch.save(retriever_module.state_dict(), os.path.join(args.output_dir, "retriever_final.pt"))
        torch.save(encoder.state_dict(), os.path.join(args.output_dir, "encoder_final.pt"))
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
            json.dump({"retriever": config["retriever"],
                       "synonym_weight": args.synonym_weight,
                       "polysemy_weight": args.polysemy_weight}, f, indent=2)
        logger.info("=== Training complete ===")

    if local_rank >= 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
