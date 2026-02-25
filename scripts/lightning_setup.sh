#!/bin/bash
# Lightning.ai Studio — Setup Script
#
# Run this ONCE when creating a new Lightning.ai Studio.
# After setup, open any notebook in notebooks/ and execute normally.
#
# Usage:
#   bash scripts/lightning_setup.sh
#
# Or from a fresh studio:
#   git clone https://github.com/paulohenriquevn/tcc.git /teamspace/studios/this_studio/TCC
#   cd /teamspace/studios/this_studio/TCC
#   bash scripts/lightning_setup.sh

set -euo pipefail

STUDIO_DIR="/teamspace/studios/this_studio"
REPO_DIR="${STUDIO_DIR}/TCC"
CACHE_DIR="${STUDIO_DIR}/cache"

echo "=== Lightning.ai Studio Setup ==="
echo "Studio: ${STUDIO_DIR}"
echo "Repo:   ${REPO_DIR}"
echo "Cache:  ${CACHE_DIR}"
echo ""

# 1. Verify we're in the right place
if [ ! -f "${REPO_DIR}/requirements.txt" ]; then
    echo "ERROR: ${REPO_DIR}/requirements.txt not found."
    echo "Clone the repo first:"
    echo "  git clone https://github.com/paulohenriquevn/tcc.git ${REPO_DIR}"
    exit 1
fi

cd "${REPO_DIR}"

# 2. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q
pip install huggingface_hub -q

# 3. Create persistent cache directory
mkdir -p "${CACHE_DIR}"
echo "Cache directory: ${CACHE_DIR}"

# 4. Verify GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# 5. Run tests to verify installation
echo ""
echo "Running tests..."
python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5

echo ""
echo "=== Setup Complete ==="
echo "Open any notebook in notebooks/ to start."
echo "The notebooks auto-detect Lightning.ai — no manual path changes needed."
