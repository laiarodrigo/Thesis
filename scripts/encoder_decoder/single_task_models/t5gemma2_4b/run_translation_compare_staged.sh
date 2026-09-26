#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <stageA|stageAonly|stageAonlycls|stageAonlylabelfirst|stageAonlylabelfirstcls|stageAplus|stageApluslabelfirst|stageB|stageBwiki|stageBmix|stageBwikimix|stageBwikimixcls|stageBwikimixlabelfirst|stageBwikimixlabelfirstcls|stageC|stageCwer> <r16|r24|r48> [extra args for launcher]" >&2
  exit 2
fi

STAGE="$1"
RANK="$2"
shift 2
NOTE=""

case "$STAGE" in
  stageA)
    case "$RANK" in
      r16) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageA_opensubs_frmt.yaml" ;;
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_frmt.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_frmt.yaml" ;;
      *) echo "Invalid rank for stageA: $RANK (use r16, r24 or r48)" >&2; exit 2 ;;
    esac
    NOTE="stageA uses the historical OpenSubs-only stage-A configs. For OpenSubs+FRMT stage A, use: $0 stageAplus r24"
    ;;
  stageAonly)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_only.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_only.yaml" ;;
      *) echo "Invalid rank for stageAonly: $RANK (currently r24 and r48 are configured)" >&2; exit 2 ;;
    esac
    ;;
  stageAonlycls)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_only_with_cls.yaml" ;;
      *) echo "Invalid rank for stageAonlycls: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageAonlylabelfirst)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_only_label_first.yaml" ;;
      *) echo "Invalid rank for stageAonlylabelfirst: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageAonlylabelfirstcls)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_only_label_first_with_cls.yaml" ;;
      *) echo "Invalid rank for stageAonlylabelfirstcls: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageAplus)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageA_opensubs_plus_frmt.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_frmt.yaml" ;;
      *) echo "Invalid rank for stageAplus: $RANK (currently r24 and r48 are configured)" >&2; exit 2 ;;
    esac
    ;;
  stageApluslabelfirst)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageA_opensubs_plus_frmt_label_first.yaml" ;;
      *) echo "Invalid rank for stageApluslabelfirst: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageB)
    case "$RANK" in
      r16) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r16_stageB_gpt_adapt.yaml" ;;
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_adapt.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_adapt.yaml" ;;
      *) echo "Invalid rank for stageB: $RANK (use r16, r24 or r48)" >&2; exit 2 ;;
    esac
    NOTE="stageB uses the GPT+FRMT stage-B mix. For GPT-only Wikipedia stage B, use: $0 stageBwiki r24"
    ;;
  stageBwiki)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_wikipedia_only.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_only.yaml" ;;
      *) echo "Invalid rank for stageBwiki: $RANK (currently r24 and r48 are configured)" >&2; exit 2 ;;
    esac
    ;;
  stageBmix)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_refresh2_frmt.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_refresh2_frmt.yaml" ;;
      *) echo "Invalid rank for stageBmix: $RANK (currently r24 and r48 are configured)" >&2; exit 2 ;;
    esac
    ;;
  stageBwikimix)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageB_gpt_wikipedia_plus_frmt.yaml" ;;
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt.yaml" ;;
      *) echo "Invalid rank for stageBwikimix: $RANK (currently r24 and r48 are configured)" >&2; exit 2 ;;
    esac
    ;;
  stageBwikimixcls)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_with_cls.yaml" ;;
      *) echo "Invalid rank for stageBwikimixcls: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageBwikimixlabelfirst)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first.yaml" ;;
      *) echo "Invalid rank for stageBwikimixlabelfirst: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageBwikimixlabelfirstcls)
    case "$RANK" in
      r48) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml" ;;
      *) echo "Invalid rank for stageBwikimixlabelfirstcls: $RANK (currently only r48 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageC)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gptrefresh2.yaml" ;;
      *) echo "Invalid rank for stageC: $RANK (currently only r24 is configured)" >&2; exit 2 ;;
    esac
    ;;
  stageCwer)
    case "$RANK" in
      r24) CONFIG="configs/encoder_decoder/t5gemma2_4b/comparison_staged/translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer.yaml" ;;
      *) echo "Invalid rank for stageCwer: $RANK (currently only r24 is configured)" >&2; exit 2 ;;
    esac
    ;;
  *)
    echo "Invalid stage: $STAGE (use stageA, stageAonly, stageAonlycls, stageAonlylabelfirst, stageAonlylabelfirstcls, stageAplus, stageApluslabelfirst, stageB, stageBwiki, stageBmix, stageBwikimix, stageBwikimixcls, stageBwikimixlabelfirst, stageBwikimixlabelfirstcls, stageC or stageCwer)" >&2
    exit 2
    ;;
esac

if [[ -n "$NOTE" ]]; then
  echo "NOTE: $NOTE" >&2
fi

if [[ "$STAGE" == "stageC" || "$STAGE" == "stageCwer" ]]; then
  python3 scripts/encoder_decoder/stage_c/train_stage_c_seq2seq_grpo.py --config "$CONFIG" "$@"
else
  python3 scripts/encoder_decoder/single_task_models/t5gemma2/step4a_translation_train_boilerplate.py --config "$CONFIG" "$@"
fi
