#!/bin/bash
# =============================================================================
# ONE-COMMAND entry point: setup environment + run ALL experiments + show progress
# Usage: bash run.sh
# =============================================================================
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# ====== 插入这段 ======
export UV_CACHE_DIR="/tmp/uv-cache-$(hostname)"
mkdir -p "$UV_CACHE_DIR"

if [ -L "$PROJ_DIR/.venv" ]; then
    rm -f "$PROJ_DIR/.venv"
elif [ -d "$PROJ_DIR/.venv" ]; then
    if [ ! -f "$PROJ_DIR/.venv/bin/activate" ] || \
       ! "$PROJ_DIR/.venv/bin/python" --version &>/dev/null; then
        echo "[FIX] .venv is broken, removing..."
        rm -rf "$PROJ_DIR/.venv" 2>/dev/null || \
            mv "$PROJ_DIR/.venv" "/tmp/.venv_dead_$(date +%s)" 2>/dev/null || true
    fi
fi
# ====== 插入结束 ======

echo "============================================================"
echo " Starting full experiment pipeline"
echo " Project: $(basename "$PROJ_DIR")"
echo " Time:    $(date)"
echo "============================================================"

# Step 1: Setup environment 
echo ""
echo "[1/2] Syncing environment (Ensuring up-to-date)..."
bash setup.sh
source "$PROJ_DIR/.venv/bin/activate"

# Step 2: Run all experiments with real-time output + log file
echo ""
echo "[2/2] Running all experiments (full production mode)..."
echo "  Log file: $PROJ_DIR/run.log"
echo "  Progress is shown below in real-time."
echo "  To run in background: nohup bash run.sh > run.log 2>&1 &"
echo ""

bash scripts/run_all_experiments.sh 2>&1 | tee "$PROJ_DIR/run.log"
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo " Pipeline completed successfully!"
    echo " Results: $PROJ_DIR/results/"
    echo " Log:     $PROJ_DIR/run.log"
    echo ""
    echo " To package results: bash collect_results.sh"
    echo "============================================================"
else
    echo "============================================================"
    echo " Pipeline failed (exit code: $EXIT_CODE)"
    echo " Check log: $PROJ_DIR/run.log"
    echo " To resume: bash run.sh (completed phases are skipped)"
    echo "============================================================"
    exit $EXIT_CODE
fi
