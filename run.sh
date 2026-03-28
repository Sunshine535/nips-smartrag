#!/bin/bash
# =============================================================================
# ONE-COMMAND entry point: run ALL experiments (environment assumed ready)
# Usage: bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo " Starting full experiment pipeline"
echo " Project: $(basename "$PROJ_DIR")"
echo " Time:    $(date)"
echo "============================================================"

# Step 1: Activate venv if present; otherwise use system Python
if [ -f "$PROJ_DIR/.venv/bin/activate" ]; then
    source "$PROJ_DIR/.venv/bin/activate"
    echo "[env] Activated venv: $PROJ_DIR/.venv"
else
    echo "[env] No .venv found, using system Python"
fi

# Step 2: Quick dependency check
python3 -c "import torch, transformers, datasets" 2>/dev/null || {
    echo "[ERROR] Missing dependencies. Run: bash setup.sh"
    exit 1
}

# Step 3: Run all experiments with real-time output + log file
echo ""
echo "[2/2] Running all experiments..."
echo "  Log file: $PROJ_DIR/run.log"
echo ""

bash scripts/run_all_experiments.sh 2>&1 | tee "$PROJ_DIR/run.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " Pipeline completed successfully!"
    echo " Results: $PROJ_DIR/results/"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline failed (exit code: $EXIT_CODE)"
    echo " Check log: $PROJ_DIR/run.log"
    echo "============================================================"
    exit $EXIT_CODE
fi
