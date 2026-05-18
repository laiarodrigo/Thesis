#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from upsert_translation_report_row import FIELDNAMES, build_rows, load_summary
except ModuleNotFoundError:
    from scripts.encoder_decoder.eval.upsert_translation_report_row import (
        FIELDNAMES,
        build_rows,
        load_summary,
    )


MODEL_TO_PREDICTIONS = {
    "4B Two-Staged LoRA r24 - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/stageB_r24_checkpoint700_adaptive_beam4/20260306_094232_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - Golden Collection (GPT Refresh 2026/03/09)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/gpt_refresh_20260309/4b_translation_stageB_r24/20260311_135403_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - Golden Collection (2nd GPT Refresh 2026/03/12)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/gpt_refresh_2st_20260312_2109/4b_translation_stageB_r24/20260315_215450_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/translation_270m_stageB_full_golden/20260228_222837_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT Refresh 2026/03/09)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/gpt_refresh_20260309/270m_translation_stageB_fullft/20260309_175605_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/270m_two_staged_old/20260311_184804_translation_predictions.jsonl",
    ),
    "4B LoRA baseline (r=8) - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/gemma4b/golden_collection/translation/checkpoint-1050/20260225_105839_translation_predictions.jsonl",
    ),
    "4B LoRA r24 - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/gemma4b/golden_collection/translation_lora_r24_adaptive_beam4/20260227_104005_translation_predictions.jsonl",
    ),
    "4B LoRA r24 (Train: GPT+FRMT) - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/gemma4b/golden_collection/translation_lora_r24_gpt_frmt_adaptive_beam4/20260301_171150_translation_predictions.jsonl",
    ),
    "270M Full-FT - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/golden collection/translation/20260224_163511_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (2nd GPT Refresh 2026/03/12)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/gpt_refresh_2st_20260312_2109/270m_translation_stageB_fullft_golden/20260313_185424_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (2nd GPT Refresh 2026/03/12)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/gpt_refresh_2st_20260312_2109/270m_translation_stageB_fullft/20260313_185424_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - Golden Collection (GPT+FRMT Stage B)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/gpt_refresh2_frmt/4b_translation_stageB_r24_golden/20260317_200500_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - Golden Collection": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/stageC_drgrpo_frmt_gptrefresh2/4b_translation_stageC_r24_golden/20260318_175733_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - FRMT Eval (old Stage B no FRMT)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/gpt_refresh_20260309/4b_translation_stageB_r24/20260322_222516_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - FRMT Eval": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/stageC_drgrpo_frmt_gptrefresh2/4b_translation_stageC_r24_fullfrmt/20260322_222516_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - FRMT Eval (GPT+FRMT Stage B)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/gpt_refresh2_frmt/4b_translation_stageB_r24_fullfrmt/20260322_224206_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - Golden Collection (GPT-Wiki Stage B)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageB_gpt_wiki/20260330_170813_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - Golden Collection (GPT-Wiki Stage B)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki/20260331_124432_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki Stage B)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki/20260401_101835_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - Golden Collection (GPT-Wiki Stage B-WER Reward)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer/20260402_104555_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - Golden Collection (GPT-Wiki+FRMT Stage B)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageB_gpt_wiki_frmt/20260402_185848_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r24 - FRMT Eval (GPT-Wiki Stage B)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageB_gpt_wiki/20260330_170813_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - FRMT Eval (GPT-Wiki Stage B)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki/20260331_151506_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - FRMT Eval (GPT-Wiki Stage B)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r48_stageB_gpt_wiki/20260401_102910_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - FRMT Eval (GPT-Wiki Stage B-WER Reward)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_wer/20260402_104555_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki+FRMT Stage B with cls)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_with_cls/20260411_154621_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - FRMT Eval (GPT-Wiki+FRMT Stage B with cls)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_with_cls/20260411_154621_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first_with_cls/20260412_115025_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first_with_cls/20260412_115025_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - Golden Collection (GPT-Wiki+FRMT Stage B label-first translation-only)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first/20260413_115428_translation_predictions.jsonl",
    ),
    "4B Two-Staged LoRA r48 - FRMT Eval (GPT-Wiki+FRMT Stage B label-first translation-only)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/t5gemma2_4b_translation_r48_stageB_gpt_wiki_frmt_label_first/20260413_115428_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - Golden Collection (GPT-Wiki Stage B-BLEU+WER Reward)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_bleu_wer/20260412_034835_translation_predictions.jsonl",
    ),
    "4B Stage C Dr. GRPO r24 - FRMT Eval (GPT-Wiki Stage B-BLEU+WER Reward)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt_eval/t5gemma2_4b_translation_r24_stageC_drgrpo_frmt_gpt_wiki_bleu_wer/20260412_034835_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B with cls)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B with cls)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls/20260422_182519_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls/20260422_182519_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B with cls PtBrVId Stage A)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls_from_stageA_opensubs_plus_ptbrvarid/20260422_220504_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B with cls PtBrVId Stage A)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_with_cls_from_stageA_opensubs_plus_ptbrvarid/20260422_220504_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls PtBrVId Stage A)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid/20260422_220504_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls PtBrVId Stage A)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls_from_stageA_opensubs_plus_ptbrvarid/20260422_220504_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - Golden Collection (GPT-Wiki+FRMT Stage B label-first with cls equal PtBrVId Stage A)": (
        "golden_collection",
        "eval_results/encoder_decoder/compare_staged/golden_collection/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls_equal/20260424_032042_translation_predictions.jsonl",
    ),
    "270M Two-Staged Full-FT - FRMT Eval (GPT-Wiki+FRMT Stage B label-first with cls equal PtBrVId Stage A)": (
        "frmt",
        "eval_results/encoder_decoder/compare_staged/frmt/t5gemma2_270m_fullft_stageB_gpt_wiki_frmt_label_first_with_cls_equal/20260424_032042_translation_predictions.jsonl",
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the translation report CSVs with overall and per-direction rows, "
            "using TER instead of WER."
        )
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_and_copy_scores_report.csv"),
    )
    parser.add_argument(
        "--output-all",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_wer_scores_report_all.csv"),
    )
    parser.add_argument(
        "--output-golden",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_wer_scores_report_golden_collection.csv"),
    )
    parser.add_argument(
        "--output-frmt",
        type=Path,
        default=Path("eval_results/encoder_decoder/bleu_copy_wer_scores_report_frmt.csv"),
    )
    return parser.parse_args()


def derive_summary_path(predictions_path: Path) -> Path:
    name = predictions_path.name
    if name.endswith("_predictions.jsonl"):
        return predictions_path.with_name(name.replace("_predictions.jsonl", "_summary.json"))
    raise ValueError(f"Unexpected predictions filename: {predictions_path}")


def main() -> None:
    args = parse_args()
    with args.input_report.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    enriched_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for row in rows:
        model_name = row["model"]
        mapping = MODEL_TO_PREDICTIONS.get(model_name)
        if mapping is None:
            missing.append(model_name)
            continue
        eval_set, predictions_rel = mapping
        predictions_path = Path(predictions_rel).resolve()
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file for {model_name}: {predictions_path}")
        summary_path = derive_summary_path(predictions_path)
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file for {model_name}: {summary_path}")
        summary = load_summary(summary_path)
        enriched_rows.extend(
            build_rows(
                model_name=model_name,
                eval_set=eval_set,
                summary=summary,
                predictions_path=predictions_path,
            )
        )

    if missing:
        raise SystemExit(
            "Missing model->predictions mappings for:\n- " + "\n- ".join(sorted(missing))
        )

    args.output_all.parent.mkdir(parents=True, exist_ok=True)
    args.output_golden.parent.mkdir(parents=True, exist_ok=True)
    args.output_frmt.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, subset: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(subset)

    write_csv(args.output_all, enriched_rows)
    write_csv(
        args.output_golden,
        [row for row in enriched_rows if row["eval_set"] == "golden_collection"],
    )
    write_csv(
        args.output_frmt,
        [row for row in enriched_rows if row["eval_set"] == "frmt"],
    )

    print(f"Wrote {len(enriched_rows)} rows -> {args.output_all}")
    print(f"Wrote {sum(row['eval_set'] == 'golden_collection' for row in enriched_rows)} rows -> {args.output_golden}")
    print(f"Wrote {sum(row['eval_set'] == 'frmt' for row in enriched_rows)} rows -> {args.output_frmt}")


if __name__ == "__main__":
    main()
