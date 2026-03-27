#!/usr/bin/env python3
"""End-to-end RAG evaluation on NQ/TriviaQA/HotpotQA/AmbigQA."""

import argparse
import json
import logging
import os
import pickle
import re
import string
import sys
from collections import Counter

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.graph_retriever import GraphContrastiveRetriever


def normalize_answer(s: str) -> str:
    """Standard QA answer normalization."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
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


def load_eval_dataset(ds_cfg: dict):
    """Load an evaluation dataset."""
    name = ds_cfg["name"]
    dataset_id = ds_cfg["dataset_id"]
    subset = ds_cfg.get("subset")
    split = ds_cfg.get("split", "validation")
    max_samples = ds_cfg.get("max_samples", 1000)

    logger.info("Loading eval dataset: %s (id=%s)", name, dataset_id)
    try:
        if subset:
            ds = load_dataset(dataset_id, subset, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
        if len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds
    except Exception as e:
        logger.warning("Failed to load %s: %s", name, e)
        return None


def extract_qa_pairs(ds, ds_name: str):
    """Extract (question, answer) pairs from different dataset formats."""
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


def load_graph_passage_index(graph_dir: str):
    """Load synonym-graph nodes + precomputed encoder embeddings for dense retrieval."""
    graph_path = os.path.join(graph_dir, "synonym_graph.pkl")
    emb_path = os.path.join(graph_dir, "node_embeddings.pt")
    if not os.path.isfile(graph_path) or not os.path.isfile(emb_path):
        return None, None
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    doc_emb = torch.load(emb_path, map_location="cpu")
    return graph, doc_emb


def retrieve_context(
    question: str,
    encoder,
    ret_tokenizer,
    retriever: GraphContrastiveRetriever,
    graph,
    doc_emb: torch.Tensor,
    device: torch.device,
    top_k: int,
    max_passages: int = 100_000,
) -> str:
    """Cosine-style retrieval via retriever head over encoded graph node texts."""
    q_inputs = ret_tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
    with torch.no_grad():
        q_raw = encoder(**q_inputs).last_hidden_state[:, 0, :]
    n = min(doc_emb.shape[0], len(graph.node_texts), max_passages)
    if n <= 0:
        return ""
    docs = doc_emb[:n].to(device=device, dtype=q_raw.dtype)
    with torch.no_grad():
        idx = retriever.retrieve(q_raw, docs, top_k=min(top_k, n))
    picked = idx[0].tolist()
    snippets = [graph.node_texts[i] for i in picked if i < len(graph.node_texts)]
    return "\n\n".join(snippets)


def generate_rag_answer(generator, tokenizer, question: str, context: str, max_new_tokens: int = 256) -> str:
    """Generate an answer given question and retrieved context."""
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


def main():
    parser = argparse.ArgumentParser(description="End-to-end GraphConRAG evaluation")
    parser.add_argument("--config", type=str, default="configs/graph_config.yaml")
    parser.add_argument("--graph_dir", type=str, default="outputs/graph")
    parser.add_argument("--retriever_dir", type=str, default="outputs/retriever")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    rag_cfg = config["rag"]

    # Load retriever encoder
    encoder_name = config["retriever"]["base_model"]
    logger.info("Loading retriever encoder: %s", encoder_name)
    ret_tokenizer = AutoTokenizer.from_pretrained(encoder_name, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
    encoder.eval()

    # Load trained retriever head
    retriever = GraphContrastiveRetriever(
        encoder_dim=config["retriever"]["hidden_dim"],
        gnn_hidden=config["retriever"]["hidden_dim"],
        gnn_layers=config["retriever"]["graph_gnn_layers"],
        gnn_heads=config["retriever"]["gnn_heads"],
    )
    ret_ckpt = os.path.join(args.retriever_dir, "retriever_final.pt")
    if os.path.exists(ret_ckpt):
        retriever.load_state_dict(torch.load(ret_ckpt, map_location="cpu"))
        logger.info("Loaded retriever checkpoint from %s", ret_ckpt)
    else:
        logger.warning("No retriever checkpoint found, using random init")
    retriever = retriever.to(encoder.device)
    retriever.eval()

    passage_graph, passage_emb = load_graph_passage_index(args.graph_dir)
    if passage_graph is None or passage_emb is None:
        logger.warning(
            "No passage index under %s (expected synonym_graph.pkl + node_embeddings.pt). "
            "RAG context will be a placeholder; run build_synonym_graph.py to enable retrieval.",
            args.graph_dir,
        )
    elif passage_emb.shape[0] != len(passage_graph.node_texts):
        logger.warning(
            "node_embeddings.pt length (%d) != graph nodes (%d); truncating to min.",
            passage_emb.shape[0],
            len(passage_graph.node_texts),
        )

    # Load generator
    gen_model_name = rag_cfg["generator_model"]
    logger.info("Loading generator: %s", gen_model_name)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, trust_remote_code=True)
    generator = AutoModelForCausalLM.from_pretrained(
        gen_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    generator.eval()
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    all_results = {}
    for ds_cfg in config["evaluation"]["datasets"]:
        ds_name = ds_cfg["name"]
        logger.info("=== Evaluating on %s ===", ds_name)

        ds = load_eval_dataset(ds_cfg)
        if ds is None:
            continue
        qa_pairs = extract_qa_pairs(ds, ds_name)
        if not qa_pairs:
            logger.warning("No QA pairs extracted for %s", ds_name)
            continue

        em_scores, f1_scores = [], []
        top_k = int(rag_cfg.get("top_k", 10))
        for i, pair in enumerate(qa_pairs):
            question = pair["question"]
            answers = pair["answers"]

            if passage_graph is not None and passage_emb is not None:
                context = retrieve_context(
                    question,
                    encoder,
                    ret_tokenizer,
                    retriever,
                    passage_graph,
                    passage_emb,
                    encoder.device,
                    top_k=top_k,
                )
                if not context.strip():
                    context = (
                        "[Retrieval returned no passages] "
                        f"Question context for: {question}"
                    )
            else:
                # TODO: add a Wikipedia / corpus passage index for real open-domain RAG;
                # graph nodes are WordNet-style strings and are only a proxy corpus.
                context = (
                    "[No passage index — retrieval disabled] "
                    f"Question context for: {question}"
                )

            prediction = generate_rag_answer(generator, gen_tokenizer, question, context)

            best_em = max(exact_match(prediction, a) for a in answers)
            best_f1 = max(f1_score(prediction, a) for a in answers)
            em_scores.append(best_em)
            f1_scores.append(best_f1)

            if i < 3:
                logger.info("  Q: %s", question[:100])
                logger.info("  A: %s | Pred: %s", answers[0][:50], prediction[:50])
                logger.info("  EM: %.2f | F1: %.2f", best_em, best_f1)

        avg_em = sum(em_scores) / max(len(em_scores), 1)
        avg_f1 = sum(f1_scores) / max(len(f1_scores), 1)
        all_results[ds_name] = {
            "exact_match": round(avg_em, 4),
            "f1": round(avg_f1, 4),
            "num_samples": len(qa_pairs),
        }
        logger.info("  %s: EM=%.4f, F1=%.4f (%d samples)", ds_name, avg_em, avg_f1, len(qa_pairs))

    output_path = os.path.join(args.output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("\n=== EVALUATION SUMMARY ===")
    for name, res in all_results.items():
        logger.info("  %s: EM=%.4f, F1=%.4f", name, res["exact_match"], res["f1"])
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
