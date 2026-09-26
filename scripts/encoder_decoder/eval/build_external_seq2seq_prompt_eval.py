#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


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

SYSTEM_CLASSIFICATION = (
    "Es um linguista especialista em portugues europeu e portugues do Brasil. "
    "A tua tarefa e identificar a variante correta do texto."
)

USER_CLASSIFICATION = (
    "Classifica a variante do texto como uma destas etiquetas: BR, PT, equal. "
    "Usa 'equal' apenas quando o texto e igual nas duas variantes. "
    "Responde apenas com uma etiqueta.\n\n"
    "Texto: {source}"
)

ENCODER_TASK_PREFIX_RE = re.compile(
    r"^\s*(?:<(br-pt|pt-br|pt-pt|id|cls)>|((?:BR|PT|CLS)\b))(?:\s*:\s*|\s+)",
    flags=re.IGNORECASE,
)
DECODER_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:<(?:pt-br|pt-pt)>\s*:?\s*|(?:BR|PT|pt-br|pt-pt)\b(?:\s*:\s*|\s+))",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build natural-language prompt eval JSONLs for external encoder-decoder "
            "seq2seq baselines from the PT/BR control-string eval views."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/control_string_ptbr_eval"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/encoder_decoder/t5gemma2/external_seq2seq_prompt_eval"),
    )
    parser.add_argument("--source-view", default="encoder_unified")
    parser.add_argument("--output-view", default="prompt_unified")
    parser.add_argument("--datasets", nargs="+", default=["frmt", "golden"])
    return parser.parse_args()


def normalize_space(text: object) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def strip_encoder_task_prefix(text: object) -> str:
    raw = str(text or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def strip_decoder_label_prefix(text: object) -> str:
    raw = normalize_space(text)
    match = DECODER_LABEL_PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def normalize_label(text: object) -> str | None:
    raw = normalize_space(text).lower()
    if not raw:
        return None
    first = raw.split(" ", 1)[0].strip(",:;.-_")
    if first == "br":
        return "BR"
    if first == "pt":
        return "PT"
    if "equal" in raw or "shared" in raw or raw == "same" or first == "igual":
        return "equal"
    if "pt-br" in raw or "ptbr" in raw or "brasil" in raw:
        return "BR"
    if "pt-pt" in raw or "ptpt" in raw or "europeu" in raw or "portugal" in raw:
        return "PT"
    return None


def infer_direction(row: dict[str, Any]) -> str | None:
    raw_direction = normalize_space(row.get("direction") or row.get("task")).lower()
    if raw_direction in {"translate_br2pt", "br2pt", "br-pt", "<br-pt>"}:
        return "br2pt"
    if raw_direction in {"translate_pt2br", "pt2br", "pt-br", "<pt-br>"}:
        return "pt2br"

    raw_input = str(row.get("input_text") or "")
    match = ENCODER_TASK_PREFIX_RE.match(raw_input)
    if not match:
        return None
    prefix = (match.group(1) or match.group(2) or "").lower()
    if prefix in {"br-pt", "br"}:
        return "br2pt"
    if prefix in {"pt-br", "pt"}:
        return "pt2br"
    return None


def extract_source(row: dict[str, Any]) -> str:
    source = normalize_space(row.get("source_text"))
    if source:
        return source
    text = normalize_space(row.get("text"))
    if text:
        return text
    return strip_encoder_task_prefix(row.get("input_text"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def build_translation_prompt(source: str, direction: str) -> str:
    if direction == "br2pt":
        user = USER_TRANSL_BR2PT.format(source=source)
    elif direction == "pt2br":
        user = USER_TRANSL_PT2BR.format(source=source)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return f"System: {SYSTEM_TRANSLATION}\n\nUser: {user}\n\nAssistant:"


def build_classification_prompt(source: str) -> str:
    user = USER_CLASSIFICATION.format(source=source)
    return f"System: {SYSTEM_CLASSIFICATION}\n\nUser: {user}\n\nAssistant:"


def copy_metadata(row: dict[str, Any], *, dataset_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": row.get("dataset") or dataset_name}
    for key in ("id", "source_id", "bucket", "direction"):
        if key in row:
            out[key] = row[key]
    return out


def build_translation_file(src_path: Path, dst_path: Path, *, dataset_name: str) -> dict[str, int]:
    counts = {"read": 0, "written": 0, "missing_direction": 0, "missing_source_or_target": 0}
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(src_path):
            counts["read"] += 1
            direction = infer_direction(row)
            if direction is None:
                counts["missing_direction"] += 1
                continue
            source = extract_source(row)
            target = strip_decoder_label_prefix(row.get("target_text") or row.get("gold"))
            if not source or not target:
                counts["missing_source_or_target"] += 1
                continue
            rec = copy_metadata(row, dataset_name=dataset_name)
            rec.update(
                {
                    "direction": direction,
                    "source_text": source,
                    "target_text": target,
                    "input_text": build_translation_prompt(source, direction),
                }
            )
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["written"] += 1
    return counts


def build_classification_file(src_path: Path, dst_path: Path, *, dataset_name: str) -> dict[str, int]:
    counts = {"read": 0, "written": 0, "missing_source_or_target": 0, "unknown_label": 0}
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(src_path):
            counts["read"] += 1
            source = extract_source(row)
            raw_target = row.get("target_text") or row.get("label") or row.get("gold")
            target = normalize_label(raw_target)
            if target is None:
                counts["unknown_label"] += 1
                continue
            if not source:
                counts["missing_source_or_target"] += 1
                continue
            rec = copy_metadata(row, dataset_name=dataset_name)
            rec.update(
                {
                    "text": source,
                    "target_text": target,
                    "input_text": build_classification_prompt(source),
                }
            )
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["written"] += 1
    return counts


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {
        "input_root": args.input_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "source_view": args.source_view,
        "output_view": args.output_view,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        src_dir = args.input_root / dataset_name / args.source_view
        dst_dir = args.output_root / dataset_name / args.output_view
        translation_src = src_dir / "translation_test.jsonl"
        classification_src = src_dir / "classification_test.jsonl"
        if not translation_src.exists():
            raise FileNotFoundError(translation_src)
        if not classification_src.exists():
            raise FileNotFoundError(classification_src)

        translation_counts = build_translation_file(
            translation_src,
            dst_dir / "translation_test.jsonl",
            dataset_name=dataset_name,
        )
        classification_counts = build_classification_file(
            classification_src,
            dst_dir / "classification_test.jsonl",
            dataset_name=dataset_name,
        )
        report["datasets"][dataset_name] = {
            "translation": translation_counts,
            "classification": classification_counts,
            "output_dir": dst_dir.as_posix(),
        }
        print(
            f"{dataset_name}: translation={translation_counts['written']}/"
            f"{translation_counts['read']} classification={classification_counts['written']}/"
            f"{classification_counts['read']}"
        )

    report_path = args.output_root / "build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
