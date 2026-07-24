#!/usr/bin/env bash
# Set up and run this implementation on a local NVIDIA GPU (e.g. RTX PRO
# Blackwell / sm_120) from a VS Code Ubuntu / WSL terminal.
#
# From scratch:
#   git clone https://github.com/Leviathan2006/Model-Misspecification-Repo.git
#   cd Model-Misspecification-Repo
#   bash baseline/run_local.sh
#
# Datasets are generated on demand (or loaded from baseline/data/*.npz if
# already cached at the requested size).
set -euo pipefail

cd "$(dirname "$0")"   # always run relative to this script (baseline/), no
                       # matter where it's invoked from

PYTHON=${PYTHON:-python3}

echo ">>> [1/4] creating virtual environment (.venv)"
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo ">>> [2/4] installing dependencies"
python -m pip install -r requirements.txt matplotlib
# Blackwell (sm_120) needs a CUDA 12.8+ build of PyTorch. If training later errors
# with "no kernel image is available for execution on the device", re-run with the
# nightly channel (printed at the end of this script).
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo ">>> [3/4] GPU sanity check"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "| compute capability", torch.cuda.get_device_capability(0))
else:
    print("WARNING: no GPU visible — will run on CPU")
PY

echo ">>> [4/4] running the paper's method (diffusion-reaction)"
# quick sanity check first, then the full paper-setting run
python run_correction.py --mode all --quick
# known vs misspecified vs corrected at the paper's Table 1 settings
python run_correction.py --mode all

cat <<'MSG'

Done.

Notes:
  * run_correction.py needs no dataset file: it evaluates the manufactured
    diffusion-reaction solution analytically at its own sensor / collocation /
    observation points.
  * Burgers and the hyperelastic beam are also implemented -- run them the
    same way (each regenerates its own data on demand at the paper's sizes):
        python run_burgers.py --case all --mode all
        python run_hyperelastic.py --mode all
    (hyperelastic's FEM data generation alone is CPU-bound -- hours, not
    minutes -- so it's not run by default here.)
  * cavity_flow has a verified data generator (velocity + pressure) but no
    method driver yet.
  * If you saw an sm_120 / "no kernel image" error, reinstall torch with:
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
MSG
