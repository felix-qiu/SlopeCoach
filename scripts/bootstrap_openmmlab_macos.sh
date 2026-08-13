#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$REPO_ROOT/artifacts/openmmlab-venv"
SRC_DIR="$REPO_ROOT/artifacts/openmmlab-src/mmcv-2.1.0"
PYTHON="$ENV_DIR/bin/python"

test "$(uname -s)" = "Darwin" || { echo "macOS is required" >&2; exit 2; }
command -v uv >/dev/null
command -v git >/dev/null
command -v clang++ >/dev/null

if [[ ! -x "$PYTHON" ]]; then
  uv venv "$ENV_DIR" --python 3.11
fi
"$PYTHON" -m pip install "setuptools<81" wheel pip packaging ninja psutil
"$PYTHON" -m pip install -r "$REPO_ROOT/python/openmmlab-requirements.txt" --no-build-isolation
"$PYTHON" -m pip uninstall -y mmcv mmcv-lite

if [[ ! -d "$SRC_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 --branch v2.1.0 https://github.com/open-mmlab/mmcv.git "$SRC_DIR"
fi
test "$(git -C "$SRC_DIR" describe --tags --exact-match)" = "v2.1.0"
test "$(git -C "$SRC_DIR" remote get-url origin)" = "https://github.com/open-mmlab/mmcv.git"

# PyTorch 2.1 headers specialize a libc++ trait that Clang 21 marks invalid.
# This diagnostic-only compatibility flag leaves official third-party source unchanged.
(cd "$SRC_DIR" && MMCV_WITH_OPS=1 \
  CFLAGS="${CFLAGS:-} -Wno-invalid-specialization" \
  CXXFLAGS="${CXXFLAGS:-} -Wno-invalid-specialization" \
  "$PYTHON" -m pip install --no-build-isolation -v -e .)

"$PYTHON" -c "import mmcv._ext; from mmcv.ops import nms; print(mmcv._ext.__file__); print(nms)"
"$PYTHON" "$SRC_DIR/.dev_scripts/check_installation.py"
echo "OpenMMLab runtime ready. Checkpoints were not downloaded."
