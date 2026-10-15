#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
THESIS_VENV_PATH="${THESIS_VENV_PATH:-$HOME/thesis_t5}"
GEMMA_VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
REQUIREMENTS="$REPO_ROOT/cmu_evaluator_project/scripts/requirements_gemma4.txt"

[[ -x "$THESIS_VENV_PATH/bin/python" ]] || {
  echo "ERROR: dissertation Python not found: $THESIS_VENV_PATH/bin/python" >&2
  exit 2
}
[[ -f "$REQUIREMENTS" ]] || {
  echo "ERROR: missing requirements: $REQUIREMENTS" >&2
  exit 2
}

if [[ ! -x "$GEMMA_VENV_PATH/bin/python" ]]; then
  "$THESIS_VENV_PATH/bin/python" -m venv "$GEMMA_VENV_PATH"
fi

thesis_site="$($THESIS_VENV_PATH/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
gemma_site="$($GEMMA_VENV_PATH/bin/python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

# Reuse the cluster-tested Torch build without modifying thesis_t5. Packages
# installed directly in gemma4_cmu take precedence over this fallback path.
printf '%s\n' "$thesis_site" > "$gemma_site/thesis_t5_base.pth"

"$GEMMA_VENV_PATH/bin/python" -m pip install --upgrade pip
"$GEMMA_VENV_PATH/bin/python" -m pip install --upgrade -r "$REQUIREMENTS"

"$GEMMA_VENV_PATH/bin/python" - <<'PY'
import accelerate
import huggingface_hub
import torch
import transformers

print("Gemma environment ready")
print("torch=", torch.__version__)
print("torch_path=", torch.__file__)
print("transformers=", transformers.__version__)
print("accelerate=", accelerate.__version__)
print("huggingface_hub=", huggingface_hub.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401
    print("gemma_class_import=PASS")
else:
    print("gemma_class_import=DEFERRED_TO_GPU_PREFLIGHT")
PY
