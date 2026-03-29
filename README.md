# SmartRAG: Graph-Enhanced Retrieval with Adaptive Policy for Knowledge-Intensive QA

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-smartrag.git
cd nips-smartrag

# 2. Install dependencies
bash setup.sh

# 3. Run all experiments
bash run.sh

# 4. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-smartrag_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

## Project Structure

```
nips-smartrag/
├── README.md
├── LICENSE
├── setup.sh                              # One-click environment setup
├── requirements.txt
├── configs/
│   └── rag_config.yaml                   # RAG pipeline configuration
├── scripts/
│   ├── gpu_utils.sh                      # Auto GPU detection utilities
│   ├── run_all_experiments.sh            # Master pipeline (Phase 0–8)
│   ├── setup_rag_infrastructure.py       # Phase 0: BM25 + FAISS setup
│   ├── build_synonym_graph.py            # Phase 1: Synonym graph
│   ├── train_contrastive_retriever.py    # Phase 2: Contrastive retriever (torchrun)
│   ├── eval_rag_pipeline.py             # Phase 3: RAG evaluation
│   ├── train_oracle_policy.py           # Phase 4: Oracle policy
│   ├── train_grpo_policy.py             # Phase 5: GRPO policy (torchrun)
│   ├── eval_rag_policy.py              # Phase 6: Policy evaluation
│   ├── eval_combined_rag.py            # Phase 7: Combined evaluation
│   └── run_ablations.py                # Phase 8: Ablation studies
├── src/                                  # Core modules
├── results/                              # Experiment outputs
└── logs/                                 # Training logs
```

## Experiments

| Phase | Description | Method |
|-------|------------|--------|
| 0 | RAG infrastructure setup | BM25 + FAISS indexing |
| 1 | Synonym graph construction | Graph-based query expansion |
| 2 | Contrastive retriever training | Graph-contrastive learning (torchrun) |
| 3 | RAG pipeline evaluation | End-to-end retrieval + generation |
| 4 | Oracle policy training | Supervised adaptive retrieval |
| 5 | GRPO policy training | RL-based retrieval policy (torchrun) |
| 6 | Policy evaluation | Compare oracle vs GRPO policies |
| 7 | Combined evaluation | Full SmartRAG pipeline assessment |
| 8 | Ablation studies | Component contribution analysis |

## Models

| Component | Model | Role |
|-----------|-------|------|
| Policy | Qwen/Qwen3.5-4B | Adaptive retrieval policy |
| Generator | Qwen/Qwen3.5-9B | Answer generation |
| Embeddings | BAAI/bge-large-en-v1.5 | Dense retrieval |

## Citation

```bibtex
@inproceedings{smartrag2026neurips,
  title={SmartRAG: Graph-Enhanced Retrieval with Adaptive Policy for Knowledge-Intensive QA},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
