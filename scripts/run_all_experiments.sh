#!/usr/bin/env bash
# =============================================================================
# SmartRAG — master experiment pipeline
# Graph-contrastive retrieval (GraphConRAG) + cost-aware RL policy (UniRAG-Policy)
# =============================================================================
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Optional: activate local virtual environment (create with: python -m venv .venv)
# -----------------------------------------------------------------------------
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/venv/bin/activate"
fi

# -----------------------------------------------------------------------------
# Shared GPU utilities (NeurIPS monorepo layout)
# -----------------------------------------------------------------------------
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/../../_shared/gpu_utils.sh" 2>/dev/null || source "${SCRIPT_DIR}/gpu_utils.sh"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
QUICK=0
SKIP_PHASES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      QUICK=1
      shift
      ;;
    --skip-phase)
      [[ $# -ge 2 ]] || { echo "Usage: $0 [--quick] [--skip-phase N] ..."; exit 1; }
      SKIP_PHASES+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: run_all_experiments.sh [options]

  --quick              Use reduced data / fewer steps for smoke tests and debugging.
  --skip-phase N       Skip phase N (0–8). May be repeated.

Phases:
  0  Setup RAG infrastructure (BM25 + FAISS)
  1  Build synonym graph
  2  Train graph-contrastive retriever
  3  Evaluate retriever (BM25, DPR, BGE, BGE+Graph)
  4  Train oracle retrieval policy
  5  Train GRPO retrieval policy
  6  Evaluate policy (no-retrieve, always, oracle, learned)
  7  Combined graph retriever + policy evaluation
  8  Ablations

Requires: NVIDIA GPU(s), Python venv with requirements installed, HF datasets access.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

EXTRA_ARGS=()
QUICK_PHASE0=()
QUICK_PHASE1=()
QUICK_PHASE4=()
if [[ "${QUICK}" -eq 1 ]]; then
  QUICK_PHASE0=(--max_passages 500 --batch_size 32)
  QUICK_PHASE1=(--max_articles 200)
  QUICK_PHASE4=(--max_queries_per_dataset 50)
fi

should_run_phase() {
  local p="$1"
  [[ ${#SKIP_PHASES[@]} -eq 0 ]] && return 0
  for s in "${SKIP_PHASES[@]}"; do
    [[ "$s" == "$p" ]] && return 1
  done
  return 0
}

run_py() {
  local script="$1"
  shift
  echo "============================================================================"
  echo " Running: python ${script} $*"
  echo "============================================================================"
  python "${PROJECT_ROOT}/${script}" "$@"
}

# -----------------------------------------------------------------------------
# Environment & GPUs
# -----------------------------------------------------------------------------
auto_setup

echo ""
echo "============================================"
echo " SmartRAG full pipeline"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Quick mode   : ${QUICK}"
if [[ ${#SKIP_PHASES[@]} -eq 0 ]]; then
  echo "  Skip phases  : none"
else
  echo "  Skip phases  : ${SKIP_PHASES[*]}"
fi
echo "============================================"
echo ""

# -----------------------------------------------------------------------------
# Phase 0: RAG infrastructure — BM25 corpora + dense FAISS index
# -----------------------------------------------------------------------------
if should_run_phase 0; then
  run_py scripts/setup_rag_infrastructure.py "${QUICK_PHASE0[@]}"
fi

# -----------------------------------------------------------------------------
# Phase 1: Synonym / lexical graph for graph-supervised contrastive signals
# -----------------------------------------------------------------------------
if should_run_phase 1; then
  run_py scripts/build_synonym_graph.py "${QUICK_PHASE1[@]}"
fi

# -----------------------------------------------------------------------------
# Phase 2: Graph-contrastive retriever training (GraphConRAG)
# -----------------------------------------------------------------------------
if should_run_phase 2; then
  TORCHRUN="$(get_torchrun_cmd)"
  # shellcheck disable=SC2086
  echo "============================================================================"
  echo " Phase 2: ${TORCHRUN} scripts/train_contrastive_retriever.py ..."
  echo "============================================================================"
  ${TORCHRUN} "${PROJECT_ROOT}/scripts/train_contrastive_retriever.py"
fi

# -----------------------------------------------------------------------------
# Phase 3: Retriever comparison — BM25, DPR, BGE, BGE+Graph
# -----------------------------------------------------------------------------
if should_run_phase 3; then
  run_py scripts/eval_rag_pipeline.py
fi

# -----------------------------------------------------------------------------
# Phase 4: Oracle (upper-bound) retrieval policy for supervised references
# -----------------------------------------------------------------------------
if should_run_phase 4; then
  run_py scripts/train_oracle_policy.py "${QUICK_PHASE4[@]}"
fi

# -----------------------------------------------------------------------------
# Phase 5: GRPO-trained cost-aware policy (UniRAG-Policy)
# -----------------------------------------------------------------------------
if should_run_phase 5; then
  TORCHRUN="$(get_torchrun_cmd)"
  # shellcheck disable=SC2086
  echo "============================================================================"
  echo " Phase 5: ${TORCHRUN} scripts/train_grpo_policy.py ..."
  echo "============================================================================"
  ${TORCHRUN} "${PROJECT_ROOT}/scripts/train_grpo_policy.py"
fi

# -----------------------------------------------------------------------------
# Phase 6: Policy evaluation — no-retrieve, always-retrieve, oracle, learned
# -----------------------------------------------------------------------------
if should_run_phase 6; then
  run_py scripts/eval_rag_policy.py
fi

# -----------------------------------------------------------------------------
# Phase 7: End-to-end — graph-enhanced retriever + adaptive policy
# -----------------------------------------------------------------------------
if should_run_phase 7; then
  run_py scripts/eval_combined_rag.py
fi

# -----------------------------------------------------------------------------
# Phase 8: Ablations (graph edges, contrastive negatives, policy reward shaping, etc.)
# -----------------------------------------------------------------------------
if should_run_phase 8; then
  run_py scripts/run_ablations.py
fi

echo ""
echo "============================================"
echo " SmartRAG pipeline completed successfully."
echo "============================================"

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
