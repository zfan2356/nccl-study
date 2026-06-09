#!/bin/bash
# Prepare the nccl-study development environment.
#
# This script:
#   1. Installs Python dependencies (pybind11)
#   2. Builds all CUDA pybind11 modules (launcher + primitives)
#   3. Registers Python paths via .pth file
#
# Usage:
#   bash scripts/prepare_env.sh
#
# Prerequisites:
#   - CUDA toolkit installed (/usr/local/cuda)
#   - Python with pip (conda or system)
#   - cmake >= 3.18

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== nccl-study environment setup ==="
echo "Root: ${ROOT_DIR}"

# 1. Install Python dependencies
echo ""
echo "[1/3] Installing Python dependencies..."
pip install pybind11 --quiet

PYBIND11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
echo "  pybind11 cmake dir: ${PYBIND11_DIR}"

# 2. Build CUDA modules
echo ""
echo "[2/3] Building CUDA pybind11 modules..."

echo "  Building launcher/ipc_utils..."
cmake -S "${ROOT_DIR}/launcher" -B "${ROOT_DIR}/build/launcher" \
  -Dpybind11_DIR="${PYBIND11_DIR}" -Wno-dev --quiet 2>/dev/null
cmake --build "${ROOT_DIR}/build/launcher" --quiet

echo "  Building primitives/l20 (ring_buffer_p2p_ipc + standalone)..."
cmake -S "${ROOT_DIR}/primitives/l20" -B "${ROOT_DIR}/build/primitives_l20" \
  -Dpybind11_DIR="${PYBIND11_DIR}" -Wno-dev --quiet 2>/dev/null
cmake --build "${ROOT_DIR}/build/primitives_l20" --quiet

echo "  Done."

# 3. Register Python paths
echo ""
echo "[3/3] Registering Python paths..."

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
PTH_FILE="${SITE_PACKAGES}/nccl-study.pth"

cat > "${PTH_FILE}" << EOF
${ROOT_DIR}/build/launcher
${ROOT_DIR}/build/primitives_l20
${ROOT_DIR}/launcher/python
EOF

echo "  Installed: ${PTH_FILE}"

# Verify
echo ""
echo "=== Verifying imports ==="
python3 -c "import ipc_utils; print('  ipc_utils: OK')"
python3 -c "import ring_buffer_p2p_ipc; print('  ring_buffer_p2p_ipc: OK')"
python3 -c "from launcher import init_dist; print('  launcher: OK')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "You can now run experiments:"
echo "  torchrun --nproc_per_node=2 primitives/l20/python/run_ring_buffer_p2p.py"
