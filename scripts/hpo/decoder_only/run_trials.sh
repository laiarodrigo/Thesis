#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${1:-hpo/decoder_only/qwen3_trials_manifest.csv}"
RUN_STATUS_PATH="${2:-hpo/decoder_only/qwen3_run_status.csv}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH"
  echo "Generate trials first with:"
  echo "  python scripts/hpo/decoder_only/generate_axolotl_trials.py"
  exit 1
fi

mkdir -p "$(dirname "$RUN_STATUS_PATH")"
echo "trial_id,trial_name,status,started_at,ended_at,config_path,output_dir" > "$RUN_STATUS_PATH"

while IFS=, read -r trial_id trial_name config_path output_dir dataset_prepared_path params_path; do
  if [[ "$trial_id" == "trial_id" ]]; then
    continue
  fi

  started_at="$(date -Iseconds)"
  status="ok"
  mkdir -p "$output_dir"

  if ! axolotl preprocess "$config_path" --debug > "$output_dir/preprocess.log" 2>&1; then
    status="preprocess_failed"
  fi

  if [[ "$status" == "ok" ]]; then
    if ! axolotl train "$config_path" > "$output_dir/train.log" 2>&1; then
      status="train_failed"
    fi
  fi

  ended_at="$(date -Iseconds)"
  echo "${trial_id},${trial_name},${status},${started_at},${ended_at},${config_path},${output_dir}" >> "$RUN_STATUS_PATH"
  echo "[${trial_name}] ${status}"
done < "$MANIFEST_PATH"

echo "Run status written to: $RUN_STATUS_PATH"
