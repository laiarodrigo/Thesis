#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
THESIS_VENV_PATH="${THESIS_VENV_PATH:-$HOME/thesis_t5}"
GEMMA_VENV_PATH="${GEMMA_VENV_PATH:-$HOME/gemma4_cmu}"
MODEL_ID="google/gemma-4-E4B-it"
failures=0

cd "$REPO_ROOT" || exit 2

echo "=== CMU pilot cluster preflight ==="
echo "host=$(hostname)"
echo "repo_root=$(pwd -P)"
echo "thesis_venv=$THESIS_VENV_PATH"
echo "gemma_venv=$GEMMA_VENV_PATH"
echo

check_adapter() {
  local system_id="$1"
  local relative_path="$2"
  local adapter="$REPO_ROOT/$relative_path"
  echo "=== Adapter: $system_id ==="
  if [[ ! -d "$adapter" ]]; then
    echo "STATUS=MISSING"
    echo "EXPECTED_PATH=$adapter"
    failures=$((failures + 1))
    echo
    return
  fi
  echo "STATUS=FOUND"
  echo "REAL_PATH=$(realpath "$adapter")"
  for filename in adapter_config.json adapter_model.safetensors tokenizer.json; do
    if [[ -f "$adapter/$filename" ]]; then
      echo "FOUND $filename bytes=$(stat -c %s "$adapter/$filename")"
    else
      echo "MISSING $filename"
      failures=$((failures + 1))
    fi
  done
  find "$adapter" -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
  echo
}

check_adapter \
  B4-D-G-PTBR \
  outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_decoder_ptbr
check_adapter \
  B4-D-GF-PTBR \
  outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_frmt_decoder_ptbr

echo "=== Frozen source ==="
python3 - <<'PY'
import json
from pathlib import Path

path = Path("cmu_evaluator_project/data/prepared/pilot_01/source/source.json")
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload["sentences"]
print(f"path={path.resolve()}")
print(f"document_id={payload['document_id']}")
print(f"sentence_count={len(rows)}")
print("sentence_ids=" + ",".join(str(row["sentence_id"]) for row in rows))
assert len(rows) == 10
assert [str(row["sentence_id"]) for row in rows] == [str(i) for i in range(3, 13)]
PY
if [[ $? -ne 0 ]]; then
  failures=$((failures + 1))
fi
echo

echo "=== Dissertation environment ==="
if [[ -f "$THESIS_VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$THESIS_VENV_PATH/bin/activate"
  python - <<'PY'
import platform
import accelerate
import peft
import torch
import transformers

print("python=" + platform.python_version())
print("torch=" + torch.__version__)
print("transformers=" + transformers.__version__)
print("peft=" + peft.__version__)
print("accelerate=" + accelerate.__version__)
print("cuda_available=" + str(torch.cuda.is_available()))
PY
  if [[ $? -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  deactivate 2>/dev/null || true
else
  echo "STATUS=MISSING thesis environment: $THESIS_VENV_PATH"
  failures=$((failures + 1))
fi
echo

echo "=== Gemma 4 environment and access (metadata only; no weights) ==="
if [[ -f "$GEMMA_VENV_PATH/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$GEMMA_VENV_PATH/bin/activate"
  python - "$MODEL_ID" <<'PY'
import platform
import re
import sys

import accelerate
import huggingface_hub
import torch
import transformers
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url, model_info

model_id = sys.argv[1]
print("python=" + platform.python_version())
print("torch=" + torch.__version__)
print("transformers=" + transformers.__version__)
print("accelerate=" + accelerate.__version__)
print("huggingface_hub=" + huggingface_hub.__version__)
print("hf_token_available=" + str(bool(get_token())))
parts = tuple(int(value) for value in re.findall(r"\d+", transformers.__version__)[:3])
if parts < (5, 10, 1):
    raise SystemExit("ERROR: Gemma 4 requires transformers>=5.10.1")
from transformers import AutoModelForMultimodalLM, AutoProcessor  # noqa: F401
info = model_info(model_id)
print("model_id=" + info.id)
print("model_revision=" + info.sha)
print("model_gated=" + str(info.gated))
metadata = get_hf_file_metadata(hf_hub_url(model_id, "model.safetensors", revision=info.sha))
print("weights_metadata_access=OK")
print("weights_size_bytes=" + str(metadata.size))
PY
  if [[ $? -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  deactivate 2>/dev/null || true
else
  echo "STATUS=MISSING Gemma environment: $GEMMA_VENV_PATH"
  echo "SETUP_REQUIRED=1"
  failures=$((failures + 1))
fi
echo

if [[ "$failures" -eq 0 ]]; then
  echo "PREFLIGHT_STATUS=PASS"
else
  echo "PREFLIGHT_STATUS=BLOCKED"
  echo "PREFLIGHT_FAILURE_COUNT=$failures"
fi
exit "$failures"
