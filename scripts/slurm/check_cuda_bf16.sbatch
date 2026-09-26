#!/usr/bin/env bash
#SBATCH --job-name=bf16probe
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -n "${VENV_PATH:-}" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV_NAME"
fi

echo "[$(date)] host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi || true

python3 - <<'PY'
import os
import sys

import torch

print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in this job.")
    sys.exit(1)

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print("device_name:", props.name)
print("compute_capability:", f"{props.major}.{props.minor}")

bf16_flag = getattr(torch.cuda, "is_bf16_supported", None)
if callable(bf16_flag):
    try:
        print("torch_cuda_is_bf16_supported:", bf16_flag())
    except Exception as exc:
        print("torch_cuda_is_bf16_supported_error:", repr(exc))
else:
    print("torch_cuda_is_bf16_supported: unavailable")

try:
    a = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device=device, dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    print("bf16_matmul_ok:", True)
    print("bf16_matmul_dtype:", c.dtype)
    print("bf16_matmul_mean:", float(c.float().mean().item()))
except Exception as exc:
    print("bf16_matmul_ok:", False)
    print("bf16_matmul_error:", repr(exc))

try:
    from transformers import Seq2SeqTrainingArguments

    probe_dir = os.path.join("/tmp", f"bf16_probe_{os.getpid()}")
    _ = Seq2SeqTrainingArguments(
        output_dir=probe_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        bf16=True,
        fp16=False,
        do_eval=False,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )
    print("transformers_bf16_args_ok:", True)
except Exception as exc:
    print("transformers_bf16_args_ok:", False)
    print("transformers_bf16_args_error:", repr(exc))
PY
