# scripts/encoder_decoder/multitask/build_multitask_jsonl.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from task_protocol import (
    CLS, build_decoder_target, map_encoder_prefix_to_decoder_task,
    normalize_class_label, strip_encoder_prefix
)

def convert_translation(in_path: Path):
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prefix, clean_input = strip_encoder_prefix(ex["input_text"])
            task = map_encoder_prefix_to_decoder_task(prefix or "")
            if task == CLS:
                continue
            yield {
                "input_text": clean_input,
                "target_text": build_decoder_target(task, ex["target_text"]),
                "task": "translation",
            }

def convert_classification(in_path: Path):
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            _, clean_input = strip_encoder_prefix(ex["input_text"])
            label = normalize_class_label(ex["target_text"])
            yield {
                "input_text": clean_input,
                "target_text": build_decoder_target(CLS, label),
                "task": "classification",
            }

def write_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translation-in", type=Path, required=True)
    ap.add_argument("--classification-in", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(convert_translation(args.translation_in))
    rows.extend(convert_classification(args.classification_in))
    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows -> {args.out}")

if __name__ == "__main__":
    main()
