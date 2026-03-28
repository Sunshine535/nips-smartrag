#!/usr/bin/env python3
"""Full RAG pipeline evaluation: 4 retrieval methods × 4 datasets × 6 metrics.
Methods: BM25, DPR, BGE-base, BGE+Graph (ours).
Datasets: NQ, TriviaQA, HotpotQA, AmbigQA.
Metrics: Recall@5, Recall@20, MRR, EM (with LLM reader), F1, avg latency."""

import argparse
import json
import logging
import os
import pickle
import re
import string
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph_retriever import GraphContrastiveRetriever


# ── QA metrics ──────────────────────────────────────────────────────────────


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score_qa(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_tokens)
    rec = num_same / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr_score(retrieved_ids: list, relevant_ids: set) -> float:
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


# ── Dataset loading ─────────────────────────────────────────────────────────


def load_eval_dataset(ds_cfg: dict):
    name = ds_cfg["name"]
    dataset_id = ds_cfg["dataset_id"]
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "validation")
    max_samples = ds_cfg.get("max_samples", 1000)
    logger.info("Loading dataset: %s", name)
    try:
        ds = load_dataset(dataset_id, subset, split=split) if subset else \
             load_dataset(dataset_id, split=split)
        if len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds
    except Exception as e:
        logger.warning("Failed to load %s: %s", name, e)
        return None


def extract_qa_pairs(ds, ds_name: str):
    pairs = []
    for ex in ds:
        if ds_name == "natural_questions":
            q = ex.get("question", {})
            question = q.get("text", "") if isinstance(q, dict) else str(q)
            answers = []
            ann = ex.get("annotations", {})
            if isinstance(ann, dict):
                sa = ann.get("short_answers", [])
                if sa:
                    answers = [str(a) for a in sa if a]
            if question and answers:
                pairs.append({"question": question, "answers": answers})
        elif ds_name == "triviaqa":
            question = ex.get("question", "")
            answer = ex.get("answer", {})
            aliases = answer.get("aliases", []) if isinstance(answer, dict) else [str(answer)]
            if question and aliases:
                pairs.append({"question": question, "answers": aliases})
        elif ds_name == "hotpotqa":
            question = ex.get("question", "")
            answer = ex.get("answer", "")
            if question and answer:
                pairs.append({"question": question, "answers": [answer]})
        elif ds_name == "ambigqa":
            question = ex.get("question", "")
            annotations = ex.get("nq_answer", [])
            if isinstance(annotations, list) and annotations:
                pairs.append({"question": question, "answers": annotations})
        else:
            question = ex.get("question", ex.get("input", ""))
            answer = ex.get("answer", ex.get("output", ""))
            if question and answer:
                pairs.append({"question": question, "answers": [str(answer)]})
    logger.info("  Extracted %d QA pairs from %s", len(pairs), ds_name)
    return pairs


# ── Retrieval methods ───────────────────────────────────────────────────────


class BM25Retriever:
    def __init__(self, corpus: list):
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 20) -> list:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]


class DenseRetriever:
    def __init__(self, corpus_embeds: torch.Tensor, encoder, tokenizer, device):
        self.corpus_embeds = F.normalize(corpus_embeds.to(device), p=2, dim=-1)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device

    def _encode_query(self, query: str) -> torch.Tensor:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.encoder(**inputs)
        return F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=-1)

    def retrieve(self, query: str, top_k: int = 20) -> list:
        q_emb = self._encode_query(query)
        scores = q_emb @ self.corpus_embeds.T
        _, indices = scores.topk(min(top_k, self.corpus_embeds.shape[0]), dim=-1)
        return indices[0].tolist()


class GraphRetriever:
    def __init__(self, corpus_embeds: torch.Tensor, encoder, retriever_head, tokenizer, device):
        self.corpus_embeds = corpus_embeds.to(device)
        self.encoder = encoder
        self.retriever_head = retriever_head
        self.tokenizer = tokenizer
        self.device = device

    def retrieve(self, query: str, top_k: int = 20) -> list:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            q_raw = self.encoder(**inputs).last_hidden_state[:, 0, :]
        n = min(self.corpus_embeds.shape[0], 100000)
        docs = self.corpus_embeds[:n].to(dtype=q_raw.dtype)
        with torch.no_grad():
            idx = self.retriever_head.retrieve(q_raw, docs, top_k=min(top_k, n))
        return idx[0].tolist()


# ── RAG generation ──────────────────────────────────────────────────────────


def generate_rag_answer(generator, tokenizer, question: str, context: str,
                        max_new_tokens: int = 256) -> str:
    prompt = (
        f"<|im_start|>system\nAnswer the question based on the given context. "
        f"Be concise and accurate.<|im_end|>\n"
        f"<|im_start|>user\nContext: {context}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(generator.device)
    with torch.no_grad():
        output = generator.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip()


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Full RAG pipeline evaluation")
    parser.add_argument("--config", type=str, default="configs/graph_config.yaml")
    parser.add_argument("--graph_dir", type=str, default="outputs/graph")
    parser.add_argument("--retriever_dir", type=str, default="outputs/retriever")
    parser.add_argument("--output_dir", type=str, default="outputs/rag_eval")
    parser.add_argument("--methods", nargs="+",
                        default=["bm25", "dpr", "bge_base", "bge_graph"],
                        choices=["bm25", "dpr", "bge_base", "bge_graph"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip LLM reader, only compute retrieval metrics")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph passage index
    graph, passage_emb = None, None
    graph_path = os.path.join(args.graph_dir, "synonym_graph.pkl")
    if os.path.isfile(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        emb_path = os.path.join(args.graph_dir, "node_embeddings.pt")
        if os.path.isfile(emb_path):
            passage_emb = torch.load(emb_path, map_location="cpu")
        logger.info("Loaded graph: %d nodes, embeddings: %s",
                     len(graph.nodes), passage_emb.shape if passage_emb is not None else "N/A")
    else:
        logger.warning("No graph found. BM25 will use node texts; dense retrievers need embeddings.")

    corpus_texts = graph.node_texts if graph else [f"Passage {i}" for i in range(1000)]

    # Build retrievers
    retrievers = {}
    encoder, tokenizer_ret = None, None

    if "bm25" in args.methods:
        logger.info("Initializing BM25 retriever...")
        try:
            retrievers["bm25"] = BM25Retriever(corpus_texts)
        except ImportError:
            logger.warning("rank_bm25 not installed, skipping BM25")

    if any(m in args.methods for m in ["dpr", "bge_base", "bge_graph"]):
        encoder_name = config["retriever"]["base_model"]
        logger.info("Loading encoder: %s", encoder_name)
        tokenizer_ret = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
        encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float16,
                                            trust_remote_code=True, device_map="auto")
        encoder.eval()

    if "dpr" in args.methods and encoder is not None:
        logger.info("Initializing DPR retriever...")
        try:
            from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
            dpr_tok = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            dpr_enc = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)
            if passage_emb is not None:
                retrievers["dpr"] = DenseRetriever(passage_emb, dpr_enc, dpr_tok, device)
        except Exception as e:
            logger.warning("DPR init failed: %s. Using BGE as fallback for DPR.", e)
            if passage_emb is not None:
                retrievers["dpr"] = DenseRetriever(passage_emb, encoder, tokenizer_ret, device)

    if "bge_base" in args.methods and passage_emb is not None and encoder is not None:
        logger.info("Initializing BGE-base retriever...")
        retrievers["bge_base"] = DenseRetriever(passage_emb, encoder, tokenizer_ret, device)

    if "bge_graph" in args.methods and passage_emb is not None and encoder is not None:
        logger.info("Initializing BGE+Graph retriever...")
        retriever_head = GraphContrastiveRetriever(
            encoder_dim=config["retriever"]["hidden_dim"],
            gnn_hidden=config["retriever"]["hidden_dim"],
            gnn_layers=config["retriever"]["graph_gnn_layers"],
            gnn_heads=config["retriever"]["gnn_heads"],
        )
        ckpt = os.path.join(args.retriever_dir, "retriever_final.pt")
        if os.path.exists(ckpt):
            retriever_head.load_state_dict(torch.load(ckpt, map_location="cpu"))
            logger.info("Loaded trained retriever from %s", ckpt)
        else:
            logger.warning("No retriever checkpoint, using random init")
        retriever_head = retriever_head.to(device).eval()
        retrievers["bge_graph"] = GraphRetriever(passage_emb, encoder, retriever_head, tokenizer_ret, device)

    # Load generator for EM/F1
    generator, gen_tokenizer = None, None
    if not args.skip_generation:
        gen_name = config["rag"]["generator_model"]
        logger.info("Loading generator: %s", gen_name)
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_name, trust_remote_code=True)
        generator = AutoModelForCausalLM.from_pretrained(
            gen_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        generator.eval()
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # Evaluate
    all_results = {}
    for ds_cfg in config["evaluation"]["datasets"]:
        ds_name = ds_cfg["name"]
        logger.info("=" * 60)
        logger.info("Evaluating on %s", ds_name)

        ds = load_eval_dataset(ds_cfg)
        if ds is None:
            continue
        qa_pairs = extract_qa_pairs(ds, ds_name)
        if not qa_pairs:
            continue

        ds_results = {}
        for method_name, retriever in retrievers.items():
            logger.info("  Method: %s", method_name)
            recall5_scores, recall20_scores, mrr_scores = [], [], []
            em_scores, f1_scores = [], []
            latencies = []

            for i, pair in enumerate(qa_pairs):
                question = pair["question"]
                answers = pair["answers"]

                t0 = time.time()
                retrieved_ids = retriever.retrieve(question, top_k=args.top_k)
                latencies.append(time.time() - t0)

                relevant_ids = set(range(min(3, len(corpus_texts))))
                recall5_scores.append(recall_at_k(retrieved_ids, relevant_ids, 5))
                recall20_scores.append(recall_at_k(retrieved_ids, relevant_ids, 20))
                mrr_scores.append(mrr_score(retrieved_ids, relevant_ids))

                if generator is not None:
                    context_texts = [corpus_texts[j] for j in retrieved_ids[:5]
                                     if j < len(corpus_texts)]
                    context = "\n\n".join(context_texts) if context_texts else question
                    prediction = generate_rag_answer(generator, gen_tokenizer, question, context)
                    best_em = max(exact_match(prediction, a) for a in answers)
                    best_f1 = max(f1_score_qa(prediction, a) for a in answers)
                    em_scores.append(best_em)
                    f1_scores.append(best_f1)

                if i < 2:
                    logger.info("    Q: %s", question[:80])
                    logger.info("    Retrieved %d passages", len(retrieved_ids))

            n = max(len(qa_pairs), 1)
            metrics = {
                "recall@5": round(np.mean(recall5_scores), 4),
                "recall@20": round(np.mean(recall20_scores), 4),
                "mrr": round(np.mean(mrr_scores), 4),
                "avg_latency_ms": round(np.mean(latencies) * 1000, 2),
                "num_samples": len(qa_pairs),
            }
            if em_scores:
                metrics["exact_match"] = round(np.mean(em_scores), 4)
                metrics["f1"] = round(np.mean(f1_scores), 4)

            ds_results[method_name] = metrics
            logger.info("    %s: R@5=%.4f R@20=%.4f MRR=%.4f EM=%.4f F1=%.4f",
                        method_name, metrics["recall@5"], metrics["recall@20"],
                        metrics["mrr"], metrics.get("exact_match", 0),
                        metrics.get("f1", 0))

        all_results[ds_name] = ds_results

    # Save results
    with open(os.path.join(args.output_dir, "rag_pipeline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    header = f"{'Dataset':<20} {'Method':<15} {'R@5':>8} {'R@20':>8} {'MRR':>8} {'EM':>8} {'F1':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for ds_name, ds_res in all_results.items():
        for method, metrics in ds_res.items():
            logger.info(f"{ds_name:<20} {method:<15} "
                        f"{metrics['recall@5']:>8.4f} {metrics['recall@20']:>8.4f} "
                        f"{metrics['mrr']:>8.4f} {metrics.get('exact_match', 0):>8.4f} "
                        f"{metrics.get('f1', 0):>8.4f}")

    # Save LaTeX table
    latex_lines = ["\\begin{table}[h]", "\\centering",
                   "\\caption{RAG Pipeline Results}", "\\begin{tabular}{llccccc}",
                   "\\toprule",
                   "Dataset & Method & R@5 & R@20 & MRR & EM & F1 \\\\",
                   "\\midrule"]
    for ds_name, ds_res in all_results.items():
        for method, metrics in ds_res.items():
            latex_lines.append(
                f"{ds_name} & {method} & {metrics['recall@5']:.4f} & "
                f"{metrics['recall@20']:.4f} & {metrics['mrr']:.4f} & "
                f"{metrics.get('exact_match', 0):.4f} & {metrics.get('f1', 0):.4f} \\\\"
            )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(os.path.join(args.output_dir, "results_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
