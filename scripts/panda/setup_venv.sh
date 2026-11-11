#!/usr/bin/env bash
# Create a venv that runs scripts/panda/train_muse.py (prep and train) and the eval scripts.
# Panda needs transformers==4.40.2 and numpy<2; newer transformers rejects its config.
#
#   bash scripts/panda/setup_venv.sh                  # -> /workspace/venv_panda
#   VENV=~/venv_panda CUDA=cu121 bash scripts/panda/setup_venv.sh
#   FULL=1 bash scripts/panda/setup_venv.sh           # also install the whole panda repo (dysts, gluonts, ...)
set -euo pipefail

VENV=${VENV:-/workspace/venv_panda}
PYTHON=${PYTHON:-python3}
CUDA=${CUDA:-cu124}            # cu118 / cu121 / cu124 / cpu; must not be newer than the driver's CUDA (nvidia-smi)
TORCH=${TORCH:-2.5.1}
REPO=$(cd "$(dirname "$0")/../.." && pwd)

"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip wheel

pip install "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$CUDA"
pip install \
    "numpy<2" \
    "transformers==4.40.2" \
    "huggingface_hub>=0.19,<1.0" \
    "safetensors>=0.4" \
    accelerate \
    scipy \
    "mne<1.9" \
    wandb \
    pandas \
    tqdm

if [ "${FULL:-0}" = 1 ]; then
    pip install -e "$REPO"
fi

python - <<'EOF'
import sys, numpy, torch, transformers, mne
print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| torch", torch.__version__,
      "| transformers", transformers.__version__, "| mne", mne.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert transformers.__version__ == "4.40.2" and numpy.__version__.startswith("1."), "wrong versions"
EOF
cd "$REPO" && python -c "from panda.patchtst.patchtst import PatchTSTForPrediction; print('panda import ok')"

echo
echo "done. activate with:  source $VENV/bin/activate"
