# SmartRAG: Graph-Enhanced Retrieval with Adaptive Policy for Knowledge-Intensive QA

SmartRAG combines two complementary ideas for open-domain, knowledge-intensive question answering:

1. **Graph-supervised contrastive retrieval (GraphConRAG).**  
   Instead of training dense retriever embeddings from query–passage pairs alone, we inject **structure from a lexical / synonym graph** so that the encoder learns representations that respect semantic neighborhoods. Contrastive objectives pull connected concepts together and push hard negatives apart, yielding **more discriminative retrieval embeddings** than standard DPR-style training when the knowledge source is large and lexically diverse.

2. **Cost-aware retrieval policy via reinforcement learning (UniRAG-Policy).**  
   Retrieval is not free: latency, token budget, and noise from irrelevant chunks all affect downstream answer quality. We train a **policy with GRPO (Group Relative Policy Optimization)** to decide **when to retrieve, what query formulation to use, and how many passages to inject**, using rewards that trade answer correctness against retrieval cost. An **oracle policy** (trained with privileged information) provides an upper bound and distillation signal for the learned policy.

Together, SmartRAG improves both **what** is retrieved (graph-contrastive retriever) and **whether and how much** to retrieve (RL policy), reducing unnecessary retrieval while preserving accuracy on hard multi-hop and ambiguous questions.

---

## Repository layout (intended)

| Path | Role |
|------|------|
| `scripts/setup_rag_infrastructure.py` | Build BM25 indexes and FAISS dense indices over corpora |
| `scripts/build_synonym_graph.py` | Construct synonym / lexical graph for contrastive supervision |
| `scripts/train_contrastive_retriever.py` | Train graph-contrastive dense retriever |
| `scripts/eval_rag_pipeline.py` | Compare BM25, DPR, BGE, **BGE+Graph** |
| `scripts/train_oracle_policy.py` | Train oracle (upper-bound) retrieval policy |
| `scripts/train_grpo_policy.py` | Train GRPO retrieval policy |
| `scripts/eval_rag_policy.py` | Compare **no-retrieve**, **always-retrieve**, **oracle**, **learned** |
| `scripts/eval_combined_rag.py` | Joint evaluation: graph retriever + adaptive policy |
| `scripts/run_ablations.py` | Ablations on graph structure, contrastive setup, and policy rewards |
| `scripts/run_all_experiments.sh` | Master pipeline (Phases 0–8) |

---

## Benchmarks

Experiments are designed around standard **knowledge-intensive QA** and verification benchmarks:

- **Natural Questions (NQ)** — real user queries against Wikipedia  
- **TriviaQA** — trivia with broad evidence  
- **HotpotQA** — multi-hop reasoning over disjoint paragraphs  
- **AmbigQA** — disambiguation and multiple valid interpretations  
- **FEVER** — fact verification against evidence passages  

These datasets stress **retrieval precision**, **coverage**, and **policy decisions** (e.g., when extra hops or disambiguation require more context).

---

## Installation

```bash
cd nips-smartrag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if using default English NLP pipeline
```

For GPU training, install a **CUDA-enabled** PyTorch build matching your driver (see [pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Running the full pipeline

From the project root (after activating the venv):

```bash
bash scripts/run_all_experiments.sh
```

**Quick smoke run** (reduced data / steps — implementation-specific):

```bash
bash scripts/run_all_experiments.sh --quick
```

**Skip specific phases** (e.g., reuse existing indices):

```bash
bash scripts/run_all_experiments.sh --skip-phase 0 --skip-phase 1
```

The script sources the shared **`gpu_utils.sh`** from the parent monorepo (`github_repos/_shared/gpu_utils.sh`) for GPU detection, `torchrun` helpers, and common Hugging Face cache settings.

---

## Method summary

| Component | Idea | Benefit |
|-----------|------|---------|
| Synonym graph | Edges link paraphrases / synonyms / related terms | Richer supervision than isolated (q, p+) pairs |
| Graph contrastive loss | Align embeddings along graph edges; separate hard negatives | Stronger dense retrieval than vanilla contrastive DPR |
| Oracle policy | Supervised with access to gold evidence and labels | Upper bound and teacher for RL |
| GRPO policy | Optimizes task reward minus retrieval cost | Adaptive retrieval depth and timing |

---

## Citation

If you use this codebase, please cite the SmartRAG technical report or paper when available, and the original GraphConRAG and UniRAG-Policy works on which this repository is based.

---

## License

See `LICENSE` in the repository root (add as appropriate for your release).
