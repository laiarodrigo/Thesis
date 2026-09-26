#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:?set MODEL_ID}"
DATASET_PATH="${DATASET_PATH:?set DATASET_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --model-id "$MODEL_ID"
  --dataset-path "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-32}"
  --concurrency "${CONCURRENCY:-8}"
  --max-retries "${MAX_RETRIES:-4}"
  --retry-backoff-seconds "${RETRY_BACKOFF_SECONDS:-3}"
  --env-file "${IAEDU_ENV_FILE:-bla.env}"
  --request-timeout "${REQUEST_TIMEOUT:-180}"
)

if [[ -n "${CLASSIFICATION_CANDIDATES:-}" ]]; then
  # shellcheck disable=SC2206
  CANDIDATES=( $CLASSIFICATION_CANDIDATES )
  ARGS+=(--classification-candidates "${CANDIDATES[@]}")
fi

if [[ -n "${IAEDU_ENDPOINT:-}" ]]; then
  ARGS+=(--endpoint "$IAEDU_ENDPOINT")
fi
if [[ -n "${IAEDU_API_KEY:-}" ]]; then
  ARGS+=(--api-key "$IAEDU_API_KEY")
fi
if [[ -n "${IAEDU_CHANNEL_ID:-}" ]]; then
  ARGS+=(--channel-id "$IAEDU_CHANNEL_ID")
fi
if [[ -n "${IAEDU_THREAD_ID:-}" ]]; then
  ARGS+=(--thread-id "$IAEDU_THREAD_ID")
fi
if [[ -n "${IAEDU_SHORT_THREAD_ID:-}" ]]; then
  ARGS+=(--short-thread-id "$IAEDU_SHORT_THREAD_ID")
fi
if [[ -n "${IAEDU_USER_INFO:-}" ]]; then
  ARGS+=(--user-info "$IAEDU_USER_INFO")
fi
if [[ -n "${IAEDU_USER_ID:-}" ]]; then
  ARGS+=(--user-id "$IAEDU_USER_ID")
fi
if [[ -n "${IAEDU_USER_CONTEXT:-}" ]]; then
  ARGS+=(--user-context "$IAEDU_USER_CONTEXT")
fi

"$PYTHON_BIN" scripts/decoder_only/axolotl/evaluate_classification_iaedu_chat_model.py "${ARGS[@]}" "$@"
