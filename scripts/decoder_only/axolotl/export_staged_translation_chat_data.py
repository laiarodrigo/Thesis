#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM_TRANSLATION = (
    "Es um assistente especialista em portugues europeu e portugues do Brasil. "
    "A tua tarefa e converter frases entre as duas variantes, mantendo o significado, "
    "o registo e um estilo natural. Responde apenas com a traducao final, "
    "sem explicacoes, sem comentarios e sem texto adicional."
)

USER_TRANSL_BR2PT = (
    "Converte o seguinte texto de portugues do Brasil para portugues europeu, "
    "mantendo o sentido e soando natural em portugues europeu. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

USER_TRANSL_PT2BR = (
    "Converte o seguinte texto de portugues europeu para portugues do Brasil, "
    "mantendo o sentido e soando natural em portugues do Brasil. "
    "Responde apenas com a frase convertida.\n\n"
    "Texto: {source}"
)

TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Export staged translation JSONL files into decoder-only chat JSONL "
            "for instruction tuning."
        )
    )
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--valid-input", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for chat JSONL files.",
    )
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--valid-file", default="valid.jsonl")
    parser.add_argument(
        "--drop-equal-pairs",
        action="store_true",
        help="Skip examples where normalized source text equals target text.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def strip_task_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def infer_direction(row: dict[str, Any]) -> str | None:
    value = normalize_space(str(row.get("direction") or row.get("task") or "")).casefold()
    if value in {"translate_br2pt", "br2pt"}:
        return "br2pt"
    if value in {"translate_pt2br", "pt2br"}:
        return "pt2br"

    input_text = normalize_space(str(row.get("input_text") or ""))
    if input_text.startswith("<br-pt>"):
        return "br2pt"
    if input_text.startswith("<pt-br>"):
        return "pt2br"
    return None


def extract_source(row: dict[str, Any]) -> str:
    source = normalize_space(str(row.get("source_text") or ""))
    if source:
        return source
    return strip_task_prefix(str(row.get("input_text") or ""))


def build_chat_record(source: str, target: str, direction: str) -> dict[str, Any]:
    if direction == "br2pt":
        user_content = USER_TRANSL_BR2PT.format(source=source)
    elif direction == "pt2br":
        user_content = USER_TRANSL_PT2BR.format(source=source)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_TRANSLATION},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ]
    }


def export_split(
    input_path: Path,
    output_path: Path,
    *,
    drop_equal_pairs: bool,
) -> dict[str, Any]:
    counts = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            counts["rows_read"] += 1
            row = json.loads(line)
            direction = infer_direction(row)
            source = extract_source(row)
            target = normalize_space(str(row.get("target_text") or ""))
            if direction is None or not source or not target:
                counts["rows_skipped_invalid"] += 1
                continue

            is_equal = source == target
            if drop_equal_pairs and is_equal:
                counts["rows_skipped_equal"] += 1
                continue

            record = build_chat_record(source, target, direction)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            counts["rows_written"] += 1
            counts[f"direction:{direction}"] += 1
            counts[f"dataset:{normalize_space(str(row.get('dataset') or 'UNKNOWN')) or 'UNKNOWN'}"] += 1
            counts["equal_pairs"] += int(is_equal)

    return dict(counts)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_out = args.out_dir / args.train_file
    valid_out = args.out_dir / args.valid_file

    train_stats = export_split(
        args.train_input,
        train_out,
        drop_equal_pairs=bool(args.drop_equal_pairs),
    )
    valid_stats = export_split(
        args.valid_input,
        valid_out,
        drop_equal_pairs=bool(args.drop_equal_pairs),
    )

    report = {
        "train": train_stats,
        "valid": valid_stats,
        "drop_equal_pairs": bool(args.drop_equal_pairs),
        "train_input": args.train_input.as_posix(),
        "valid_input": args.valid_input.as_posix(),
        "train_output": train_out.as_posix(),
        "valid_output": valid_out.as_posix(),
    }
    report_path = args.out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
