#!/usr/bin/env bash
#SBATCH --job-name=cmu_preflight
#SBATCH --partition=hlt_msc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --output=cmu_evaluator_project/logs/slurm/%x-%j.out
#SBATCH --error=cmu_evaluator_project/logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-/cfs/home/u036584/repos/Thesis}}"
cd "$REPO_ROOT"

echo "[$(date)] GPU preflight host=$(hostname) job_id=${SLURM_JOB_ID:-na}"
nvidia-smi
bash cmu_evaluator_project/scripts/slurm/preflight_cmu_pilot.sh
