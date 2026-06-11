#!/bin/bash
set -e

WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
HEALER="$WORKSPACE/self_healing/healer.py"
PYTHON_TEST="$WORKSPACE/self_healing/tests/python/data_pipeline.py"
NODE_TEST="$WORKSPACE/self_healing/tests/node/index.js"

echo ""
echo "========================================================"
echo "   SELF-HEALING SCRIPT — Cross-Language Demo Runner     "
echo "========================================================"

# --- Python Test ---
echo ""
echo ">>> [Python] Verifying bug exists..."
python "$PYTHON_TEST" 2>&1 && echo "No error (check the bug!)" || echo ">>> Bug confirmed. Handing off to healer..."

echo ""
echo ">>> [Python] Launching healer..."
python "$HEALER" "python $PYTHON_TEST"

echo ""
echo ">>> [Python] Verifying fix..."
python "$PYTHON_TEST" && echo ">>> [Python] Healed successfully!" || echo ">>> [Python] Still failing."

# --- Node.js Test ---
echo ""
echo "========================================================"
echo ""
echo ">>> [Node.js] Verifying bug exists..."
node "$NODE_TEST" 2>&1 && echo "No error (check the bug!)" || echo ">>> Bug confirmed. Handing off to healer..."

echo ""
echo ">>> [Node.js] Launching healer..."
python "$HEALER" "node $NODE_TEST"

echo ""
echo ">>> [Node.js] Verifying fix..."
node "$NODE_TEST" && echo ">>> [Node.js] Healed successfully!" || echo ">>> [Node.js] Still failing."

echo ""
echo "========================================================"
echo "   All scenarios complete. Check git log for commits.  "
echo "========================================================"
echo ""
git -C "$WORKSPACE" log --oneline -5
