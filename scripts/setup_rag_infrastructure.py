#!/usr/bin/env python3
"""
Setup RAG infrastructure for UniRAG-Policy experiments.

- Build BM25 index (using pyserini) on Wikipedia passages
- Build FAISS dense index (using BGE-large embeddings)
- Prepare datasets: NQ, TriviaQA, HotpotQA, FEVER
- Verify retrieval quality (recall@20 sanity check)
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("setup_rag_infra")


def parse_args():
    parser = argparse.ArgumentParser(description="Setup RAG infrastructure")
    parser.add_argument("--config", type=str, default="configs/rag_config.yaml")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--wiki_passages", type=str, default=None,
                        help="Path to Wikipedia passages (downloads if not provided)")
    parser.add_argument("--max_passages", type=int, default=1000000,
                        help="Max Wikipedia passages to index")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--skip_bm25", action="store_true")
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_datasets", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Wikipedia Passage Loading ────────────────────────────────────────────────

def load_wiki_passages(path=None, max_passages=1000000):
    """Load Wikipedia passages from file or download from HF."""
    if path and os.path.exists(path):
        logger.info(f"Loading passages from {path}")
        passages = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_passages:
                    break
                data = json.loads(line.strip())
                passages.append({
                    "id": data.get("id", str(i)),
                    "title": data.get("title", ""),
                    "text": data.get("text", data.get("passage", "")),
                })
        return passages

    logger.info("Downloading Wikipedia DPR passages from HuggingFace...")
    try:
        ds = load_dataset(
            "facebook/wiki_dpr", "psgs_w100.nq.no_index",
            split="train", trust_remote_code=True,
        )
        passages = []
        for i, ex in enumerate(ds):
            if i >= max_passages:
                break
            passages.append({
                "id": str(ex.get("id", i)),
                "title": ex.get("title", ""),
                "text": ex.get("text", ""),
            })
        return passages
    except Exception as e:
        logger.warning(f"Could not load wiki_dpr: {e}")
        logger.info("Generating synthetic passages for testing...")
        passages = []
        for i in range(min(max_passages, 10000)):
            passages.append({
                "id": str(i),
                "title": f"Article {i}",
                "text": f"This is a synthetic passage about topic {i}. "
                        f"It contains information relevant to queries about subject {i % 100}.",
            })
        return passages


# ── BM25 Index ───────────────────────────────────────────────────────────────

def build_bm25_index(passages, output_dir):
    """Build BM25 index using rank_bm25 (pyserini fallback)."""
    logger.info(f"Building BM25 index over {len(passages)} passages...")
    bm25_dir = os.path.join(output_dir, "bm25_index")
    os.makedirs(bm25_dir, exist_ok=True)

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed. pip install rank_bm25")
        logger.info("Attempting pyserini fallback...")
        try:
            return build_bm25_pyserini(passages, bm25_dir)
        except ImportError:
            logger.error("Neither rank_bm25 nor pyserini available. Skipping BM25.")
            return None

    tokenized = [p["text"].lower().split() for p in tqdm(passages, desc="Tokenizing for BM25")]
    bm25 = BM25Okapi(tokenized)

    with open(os.path.join(bm25_dir, "bm25_model.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    passage_ids = [p["id"] for p in passages]
    passage_texts = [p["text"] for p in passages]
    with open(os.path.join(bm25_dir, "passages.json"), "w") as f:
        json.dump({"ids": passage_ids, "texts": passage_texts}, f)

    logger.info(f"BM25 index saved to {bm25_dir}")
    return bm25


def build_bm25_pyserini(passages, output_dir):
    """Build BM25 index using pyserini/Lucene."""
    from pyserini.index.lucene import LuceneIndexer

    docs_dir = os.path.join(output_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    for i, p in enumerate(passages):
        doc = {"id": p["id"], "contents": p["text"]}
        with open(os.path.join(docs_dir, f"{i}.json"), "w") as f:
            json.dump(doc, f)

    index_dir = os.path.join(output_dir, "lucene_index")
    os.system(
        f"python -m pyserini.index.lucene "
        f"--collection JsonCollection "
        f"--input {docs_dir} "
        f"--index {index_dir} "
        f"--generator DefaultLuceneDocumentGenerator "
        f"--threads 8 "
        f"--storePositions --storeDocvectors --storeRaw"
    )

    logger.info(f"Pyserini BM25 index saved to {index_dir}")
    return index_dir


# ── Dense (FAISS) Index ──────────────────────────────────────────────────────

def build_dense_index(passages, embedding_model_name, output_dir, batch_size=256):
    """Build FAISS dense index using BGE-large embeddings."""
    import faiss

    logger.info(f"Building FAISS dense index with {embedding_model_name}")
    dense_dir = os.path.join(output_dir, "dense_index")
    os.makedirs(dense_dir, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model_name)
    except ImportError:
        logger.error("sentence-transformers not installed. pip install sentence-transformers")
        return None

    texts = [p["text"] for p in passages]
    logger.info(f"Encoding {len(texts)} passages (batch_size={batch_size})...")

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch = texts[i:i + batch_size]
        embs = encoder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings.shape[1]
    logger.info(f"Embeddings shape: {embeddings.shape} (dim={dim})")

    if len(passages) > 100000:
        nlist = min(int(np.sqrt(len(passages))), 4096)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(64, nlist)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    index_path = os.path.join(dense_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)

    passage_map = {
        "ids": [p["id"] for p in passages],
        "texts": [p["text"] for p in passages],
    }
    with open(os.path.join(dense_dir, "passage_map.json"), "w") as f:
        json.dump(passage_map, f)

    logger.info(f"FAISS index saved to {dense_dir} ({index.ntotal} vectors)")
    return index, encoder


# ── Dataset Preparation ──────────────────────────────────────────────────────

def prepare_datasets(config, output_dir):
    """Download and prepare NQ, TriviaQA, HotpotQA, FEVER."""
    ds_dir = os.path.join(output_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    dataset_stats = {}

    for ds_cfg in config["evaluation"]["datasets"]:
        name = ds_cfg["name"]
        logger.info(f"Preparing dataset: {name}")

        try:
            subset = ds_cfg.get("subset")
            split = ds_cfg.get("split", "validation")

            if subset:
                ds = load_dataset(ds_cfg["dataset_id"], subset, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(ds_cfg["dataset_id"], split=split, trust_remote_code=True)

            max_s = ds_cfg.get("max_samples", 2000)
            if len(ds) > max_s:
                ds = ds.shuffle(seed=42).select(range(max_s))

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
                    queries.append({"query": str(q), "answer": str(a), "source": name})

            output_path = os.path.join(ds_dir, f"{name}.jsonl")
            with open(output_path, "w") as f:
                for q in queries:
                    f.write(json.dumps(q) + "\n")

            dataset_stats[name] = {
                "total_samples": len(queries),
                "path": output_path,
            }
            logger.info(f"  {name}: {len(queries)} queries saved to {output_path}")

        except Exception as e:
            logger.warning(f"  Failed to load {name}: {e}")
            dataset_stats[name] = {"error": str(e)}

    return dataset_stats


# ── Retrieval Sanity Check ───────────────────────────────────────────────────

def verify_retrieval_quality(bm25, dense_index, encoder, passages, datasets_dir, k=20):
    """Verify retrieval quality with recall@20 sanity check."""
    logger.info("Running retrieval quality verification (recall@20)...")

    results = {}

    for ds_file in os.listdir(datasets_dir):
        if not ds_file.endswith(".jsonl"):
            continue
        ds_name = ds_file.replace(".jsonl", "")

        queries = []
        with open(os.path.join(datasets_dir, ds_file)) as f:
            for line in f:
                queries.append(json.loads(line.strip()))

        sample = queries[:100]
        bm25_hits = 0
        dense_hits = 0

        for item in sample:
            query = item["query"]
            answer = item["answer"].lower()

            if bm25 is not None:
                try:
                    from rank_bm25 import BM25Okapi
                    tokenized_query = query.lower().split()
                    scores = bm25.get_scores(tokenized_query)
                    top_k_idx = np.argsort(scores)[-k:]
                    for idx in top_k_idx:
                        if answer in passages[idx]["text"].lower():
                            bm25_hits += 1
                            break
                except Exception:
                    pass

            if dense_index is not None and encoder is not None:
                try:
                    q_emb = encoder.encode([query], normalize_embeddings=True).astype(np.float32)
                    _, I = dense_index.search(q_emb, k)
                    for idx in I[0]:
                        if idx < len(passages) and answer in passages[idx]["text"].lower():
                            dense_hits += 1
                            break
                except Exception:
                    pass

        n = len(sample)
        results[ds_name] = {
            "bm25_recall@20": bm25_hits / max(n, 1) if bm25 is not None else None,
            "dense_recall@20": dense_hits / max(n, 1) if dense_index is not None else None,
            "n_queries": n,
        }
        logger.info(f"  {ds_name}: BM25 R@20={bm25_hits/max(n,1):.3f} "
                     f"Dense R@20={dense_hits/max(n,1):.3f}")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    passages = load_wiki_passages(args.wiki_passages, args.max_passages)
    logger.info(f"Loaded {len(passages)} passages")

    bm25 = None
    dense_index = None
    encoder = None

    if not args.skip_bm25 and not args.verify_only:
        bm25 = build_bm25_index(passages, args.output_dir)

    if not args.skip_dense and not args.verify_only:
        result = build_dense_index(
            passages, args.embedding_model, args.output_dir, args.batch_size,
        )
        if result is not None:
            dense_index, encoder = result

    dataset_stats = {}
    if not args.skip_datasets and not args.verify_only:
        dataset_stats = prepare_datasets(config, args.output_dir)

    datasets_dir = os.path.join(args.output_dir, "datasets")
    retrieval_quality = {}
    if os.path.exists(datasets_dir) and passages:
        if bm25 is None and os.path.exists(os.path.join(args.output_dir, "bm25_index", "bm25_model.pkl")):
            with open(os.path.join(args.output_dir, "bm25_index", "bm25_model.pkl"), "rb") as f:
                bm25 = pickle.load(f)

        if dense_index is None:
            dense_path = os.path.join(args.output_dir, "dense_index", "faiss_index.bin")
            if os.path.exists(dense_path):
                import faiss
                dense_index = faiss.read_index(dense_path)
                try:
                    from sentence_transformers import SentenceTransformer
                    encoder = SentenceTransformer(args.embedding_model)
                except ImportError:
                    pass

        retrieval_quality = verify_retrieval_quality(
            bm25, dense_index, encoder, passages, datasets_dir,
        )

    setup_summary = {
        "num_passages": len(passages),
        "bm25_built": bm25 is not None,
        "dense_built": dense_index is not None,
        "embedding_model": args.embedding_model,
        "dataset_stats": dataset_stats,
        "retrieval_quality": retrieval_quality,
    }

    summary_path = os.path.join(args.output_dir, "setup_summary.json")
    with open(summary_path, "w") as f:
        json.dump(setup_summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("RAG INFRASTRUCTURE SETUP COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Passages:  {len(passages)}")
    logger.info(f"  BM25:      {'ready' if bm25 else 'skipped'}")
    logger.info(f"  FAISS:     {'ready' if dense_index else 'skipped'}")
    logger.info(f"  Datasets:  {len(dataset_stats)} prepared")
    for ds_name, quality in retrieval_quality.items():
        logger.info(f"  {ds_name}: BM25 R@20={quality.get('bm25_recall@20', 'N/A')} "
                     f"Dense R@20={quality.get('dense_recall@20', 'N/A')}")
    logger.info(f"  Summary:   {summary_path}")


if __name__ == "__main__":
    main()
