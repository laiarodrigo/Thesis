#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_TOKENS = ("<pt-br>", "<pt-pt>")
CONTROL_TOKENS = ("<cls>", "<pt-br>", "<pt-pt>")
LEGACY_CLASS_LABELS = {"BR", "PT", "br", "pt", "pt-br", "pt-pt", "equal", "igual"}
PREFIX_RE = re.compile(r"^\s*(<cls>|<pt-br>|<pt-pt>)\s*", flags=re.IGNORECASE)
DECODER_LABEL_RE = re.compile(r"^\s*(<pt-br>|<pt-pt>)\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit final-protocol rendered JSONL datasets for the translation-only, "
            "tokens-in-encoder, and labels-in-decoder views."
        )
    )
    parser.add_argument("--translation-only-dir", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--decoder-dir", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_control_prefix(text: str) -> str:
    raw = normalize_space(text)
    match = PREFIX_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def strip_decoder_label(text: str) -> str:
    raw = normalize_space(text)
    match = DECODER_LABEL_RE.match(raw)
    if match:
        raw = raw[match.end() :]
    return normalize_space(raw)


def first_token(text: str) -> str:
    normalized = normalize_space(text)
    return normalized.split(" ", 1)[0] if normalized else ""


def starts_with_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = normalize_space(text)
    return any(normalized.startswith(f"{token} ") or normalized == token for token in tokens)


def normalize_direction(raw: object) -> str:
    text = normalize_space(str(raw or "")).casefold()
    if text in {"translate_br2pt", "br2pt", "br-pt"}:
        return "br2pt"
    if text in {"translate_pt2br", "pt2br", "pt-br"}:
        return "pt2br"
    if text == "classification":
        return "classification"
    return text


def source_token_from_row(row: dict[str, Any], *, view: str) -> str:
    explicit = normalize_space(str(row.get("source_variant_label") or ""))
    if explicit in SOURCE_TOKENS:
        return explicit
    if view in {"T", "E"}:
        token = first_token(str(row.get("input_text") or ""))
        return token if token in SOURCE_TOKENS else ""
    if view == "D":
        token = first_token(str(row.get("target_text") or ""))
        return token if token in SOURCE_TOKENS else ""
    return ""


def translation_key(row: dict[str, Any], *, view: str) -> tuple[Any, ...]:
    source_text = (
        strip_control_prefix(str(row.get("input_text") or ""))
        if view in {"T", "E"}
        else normalize_space(str(row.get("input_text") or ""))
    )
    target_text = (
        strip_decoder_label(str(row.get("target_text") or ""))
        if view == "D"
        else normalize_space(str(row.get("target_text") or ""))
    )
    return (
        row.get("id"),
        normalize_space(str(row.get("dataset") or "")),
        normalize_space(str(row.get("bucket") or "")),
        normalize_direction(row.get("direction") or row.get("task")),
        source_token_from_row(row, view=view),
        source_text,
        target_text,
        bool(row.get("is_equal_pair", False)),
    )


def classification_key(row: dict[str, Any], *, view: str) -> tuple[Any, ...]:
    source_text = (
        strip_control_prefix(str(row.get("input_text") or ""))
        if view == "E"
        else normalize_space(str(row.get("input_text") or ""))
    )
    return (
        row.get("id"),
        normalize_space(str(row.get("dataset") or "")),
        normalize_space(str(row.get("bucket") or "")),
        source_text,
        normalize_space(str(row.get("target_text") or "")),
        bool(row.get("is_equal_pair", False)),
    )


def sample_counter_diff(left: Counter, right: Counter, *, limit: int = 5) -> dict[str, list[str]]:
    missing = list((left - right).elements())[:limit]
    extra = list((right - left).elements())[:limit]
    return {
        "missing_from_right": [repr(item) for item in missing],
        "extra_in_right": [repr(item) for item in extra],
    }


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    translations = [row for row in rows if normalize_space(str(row.get("task") or "")) == "translation"]
    classifications = [
        row for row in rows if normalize_space(str(row.get("task") or "")) == "classification"
    ]
    return translations, classifications


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)


def validate_view_format(
    *,
    view: str,
    split: str,
    translations: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    if view == "T" and classifications:
        add_error(errors, f"{view}/{split}: translation-only view contains classification rows")
    if view in {"E", "D"} and not classifications:
        add_error(errors, f"{view}/{split}: unified view has no classification rows")

    for idx, row in enumerate(translations):
        prefix = f"{view}/{split}/translation[{idx}]"
        source_label = source_token_from_row(row, view=view)
        if source_label not in SOURCE_TOKENS:
            add_error(errors, f"{prefix}: missing final source token")
        if row.get("loss_on_first_token_only") is not False:
            add_error(errors, f"{prefix}: translation row must have loss_on_first_token_only=false")
        if "loss_mask_prefix_tokens" in row:
            add_error(errors, f"{prefix}: final protocol must not use loss_mask_prefix_tokens")
        if view in {"T", "E"}:
            if not starts_with_any(str(row.get("input_text") or ""), SOURCE_TOKENS):
                add_error(errors, f"{prefix}: encoder input does not start with source token")
            if starts_with_any(str(row.get("target_text") or ""), SOURCE_TOKENS):
                add_error(errors, f"{prefix}: encoder-controlled target should not start with source token")
        if view == "D":
            if starts_with_any(str(row.get("input_text") or ""), CONTROL_TOKENS):
                add_error(errors, f"{prefix}: decoder-controlled input should not start with control token")
            if not starts_with_any(str(row.get("target_text") or ""), SOURCE_TOKENS):
                add_error(errors, f"{prefix}: decoder target does not start with source token")

        source_text = (
            strip_control_prefix(str(row.get("input_text") or ""))
            if view in {"T", "E"}
            else normalize_space(str(row.get("input_text") or ""))
        )
        target_text = (
            strip_decoder_label(str(row.get("target_text") or ""))
            if view == "D"
            else normalize_space(str(row.get("target_text") or ""))
        )
        if bool(row.get("is_equal_pair", False)) and source_text != target_text:
            add_warning(warnings, f"{prefix}: is_equal_pair=true but source and target differ")

    for idx, row in enumerate(classifications):
        prefix = f"{view}/{split}/classification[{idx}]"
        target = normalize_space(str(row.get("target_text") or ""))
        if target not in SOURCE_TOKENS:
            add_error(errors, f"{prefix}: classification target is not final token: {target!r}")
        if row.get("loss_on_first_token_only") is not True:
            add_error(errors, f"{prefix}: classification row must have loss_on_first_token_only=true")
        if "loss_mask_prefix_tokens" in row:
            add_error(errors, f"{prefix}: final protocol must not use loss_mask_prefix_tokens")
        if first_token(target) in LEGACY_CLASS_LABELS:
            add_error(errors, f"{prefix}: legacy classification label remains: {target!r}")
        if view == "E" and not starts_with_any(str(row.get("input_text") or ""), ("<cls>",)):
            add_error(errors, f"{prefix}: encoder classification input must start with <cls>")
        if view == "D" and starts_with_any(str(row.get("input_text") or ""), CONTROL_TOKENS):
            add_error(errors, f"{prefix}: decoder classification input should not contain control prefix")


def validate_equal_classification_pairs(
    *,
    view: str,
    split: str,
    classifications: list[dict[str, Any]],
    errors: list[str],
) -> None:
    groups: dict[tuple[Any, str], set[str]] = defaultdict(set)
    for row in classifications:
        if not bool(row.get("is_equal_pair", False)):
            continue
        source_text = (
            strip_control_prefix(str(row.get("input_text") or ""))
            if view == "E"
            else normalize_space(str(row.get("input_text") or ""))
        )
        groups[(row.get("id"), source_text)].add(normalize_space(str(row.get("target_text") or "")))

    for key, targets in groups.items():
        if targets != set(SOURCE_TOKENS):
            add_error(
                errors,
                f"{view}/{split}: equal classification group {key!r} has targets {sorted(targets)}",
            )


def audit_split(
    *,
    split: str,
    paths: dict[str, Path],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    rows_by_view = {
        view: read_jsonl(path)
        for view, path in paths.items()
    }
    hashes = {view: sha256_file(path) for view, path in paths.items()}
    task_counts: dict[str, dict[str, int]] = {}
    translations_by_view: dict[str, list[dict[str, Any]]] = {}
    classifications_by_view: dict[str, list[dict[str, Any]]] = {}

    for view, rows in rows_by_view.items():
        translations, classifications = split_rows(rows)
        translations_by_view[view] = translations
        classifications_by_view[view] = classifications
        task_counts[view] = dict(Counter(normalize_space(str(row.get("task") or "")) for row in rows))
        validate_view_format(
            view=view,
            split=split,
            translations=translations,
            classifications=classifications,
            errors=errors,
            warnings=warnings,
        )
        if view in {"E", "D"}:
            validate_equal_classification_pairs(
                view=view,
                split=split,
                classifications=classifications,
                errors=errors,
            )

    translation_counters = {
        view: Counter(translation_key(row, view=view) for row in translations)
        for view, translations in translations_by_view.items()
    }
    if translation_counters["T"] != translation_counters["E"]:
        add_error(
            errors,
            f"{split}: T and E translation rows differ: "
            f"{sample_counter_diff(translation_counters['T'], translation_counters['E'])}",
        )
    if translation_counters["T"] != translation_counters["D"]:
        add_error(
            errors,
            f"{split}: T and D translation rows differ: "
            f"{sample_counter_diff(translation_counters['T'], translation_counters['D'])}",
        )

    e_cls = Counter(classification_key(row, view="E") for row in classifications_by_view["E"])
    d_cls = Counter(classification_key(row, view="D") for row in classifications_by_view["D"])
    if e_cls != d_cls:
        add_error(
            errors,
            f"{split}: E and D classification rows differ: {sample_counter_diff(e_cls, d_cls)}",
        )

    return {
        "paths": {view: path.as_posix() for view, path in paths.items()},
        "sha256": hashes,
        "rows": {view: len(rows) for view, rows in rows_by_view.items()},
        "task_counts": task_counts,
        "translation_equal_rows": {
            view: sum(1 for row in rows if row.get("is_equal_pair"))
            for view, rows in translations_by_view.items()
        },
        "classification_equal_rows": {
            view: sum(1 for row in rows if row.get("is_equal_pair"))
            for view, rows in classifications_by_view.items()
        },
    }


def split_path(dataset_dir: Path, split: str) -> Path:
    path = dataset_dir / f"{split}.jsonl"
    if path.exists():
        return path
    if split == "train":
        alt = dataset_dir / "translation_train.jsonl"
    elif split == "valid":
        alt = dataset_dir / "translation_valid.jsonl"
    else:
        alt = dataset_dir / f"translation_{split}.jsonl"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find split {split!r} in {dataset_dir}")


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    split_reports: dict[str, Any] = {}

    dirs = {
        "T": args.translation_only_dir,
        "E": args.encoder_dir,
        "D": args.decoder_dir,
    }
    for view, directory in dirs.items():
        if not directory.is_dir():
            add_error(errors, f"{view}: missing directory {directory}")

    if not errors:
        for split in args.splits:
            paths = {
                view: split_path(directory, split)
                for view, directory in dirs.items()
            }
            split_reports[split] = audit_split(
                split=split,
                paths=paths,
                errors=errors,
                warnings=warnings,
            )

    report = {
        "status": "failed" if errors else ("warning" if warnings else "passed"),
        "errors": errors,
        "warnings": warnings,
        "splits": split_reports,
    }
    report_json = json.dumps(report, ensure_ascii=False, indent=2)
    print(report_json)
    if args.out_report is not None:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(report_json + "\n", encoding="utf-8")

    if errors or (warnings and args.fail_on_warning):
        sys.exit(1)


if __name__ == "__main__":
    main()
