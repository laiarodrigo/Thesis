#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable


PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a recovered mixed E dataset into paired E/D control-string "
            "datasets while preserving source row order and split membership."
        )
    )
    parser.add_argument("--mixed-train", type=Path, required=True)
    parser.add_argument("--mixed-valid", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--br-control", default="<pt-br>")
    parser.add_argument("--pt-control", default="<pt-pt>")
    parser.add_argument("--classification-control", default="<cls>")
    parser.add_argument(
        "--forbid-dataset-substring",
        default="",
        help="Abort if this case-insensitive substring appears in a row's dataset/source metadata.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args()


def normalize_space(value: object) -> str:
    return " ".join(str(value or "").split())


def coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_space(value).casefold() in {"1", "true", "yes", "y"}


def strip_prefix(value: object) -> tuple[str | None, str]:
    text = str(value or "")
    match = PREFIX_RE.match(text)
    if match is None:
        return None, normalize_space(text)
    return match.group(1).strip().casefold(), normalize_space(text[match.end() :])


def normalize_label(value: object, *, br_control: str, pt_control: str) -> str | None:
    text = normalize_space(value).casefold()
    if text in {"pt-br", "<pt-br>", "br", "brasil", "brasileiro", "brazilian"}:
        return br_control
    if text in {"pt-pt", "<pt-pt>", "pt", "portugal", "europeu", "european"}:
        return pt_control
    if text in {"equal", "igual", "same", "shared"}:
        return "equal"
    return None


def infer_translation_source_label(
    row: dict[str, Any],
    *,
    prefix: str | None,
    br_control: str,
    pt_control: str,
) -> str | None:
    explicit = normalize_label(
        row.get("source_variant_label"),
        br_control=br_control,
        pt_control=pt_control,
    )
    if explicit in {br_control, pt_control}:
        return explicit
    direction = normalize_space(row.get("direction") or row.get("task")).casefold()
    if direction in {"translate_br2pt", "br2pt", "br-pt"}:
        return br_control
    if direction in {"translate_pt2br", "pt2br", "pt-br"}:
        return pt_control
    if prefix == "br-pt":
        return br_control
    if prefix in {"pt-br", "pt-pt"}:
        return pt_control
    return None


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc


def render_translation(
    row: dict[str, Any],
    *,
    br_control: str,
    pt_control: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix, source = strip_prefix(row.get("input_text") or row.get("source_text"))
    target = normalize_space(row.get("target_text") or row.get("gold") or row.get("target"))
    label = infer_translation_source_label(
        row,
        prefix=prefix,
        br_control=br_control,
        pt_control=pt_control,
    )
    if not source or not target or label is None:
        raise ValueError(f"Invalid recovered translation row: {row}")
    is_equal = coerce_bool(row.get("is_equal_pair")) or source == target

    common = dict(row)
    common.update(
        {
            "task": "translation",
            "source_variant_label": label,
            "is_equal_pair": is_equal,
            "loss_on_first_token_only": False,
            "control_representation": "string",
        }
    )
    encoder = dict(common, input_text=f"{label} {source}", target_text=target)
    decoder = dict(common, input_text=source, target_text=f"{label} {target}")
    return encoder, decoder


def render_classification(
    row: dict[str, Any],
    *,
    br_control: str,
    pt_control: str,
    classification_control: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    _, source = strip_prefix(
        row.get("input_text") or row.get("source_text") or row.get("text")
    )
    label = normalize_label(
        row.get("target_text", row.get("label", row.get("gold"))),
        br_control=br_control,
        pt_control=pt_control,
    )
    if label == "equal":
        return None
    if not source or label not in {br_control, pt_control}:
        raise ValueError(f"Invalid recovered classification row: {row}")

    common = dict(row)
    common.update(
        {
            "task": "classification",
            "direction": "classification",
            "source_variant_label": label,
            "is_equal_pair": False,
            "target_text": label,
            # E follows recovered E: score/train the complete label string with
            # the normal full-vocabulary seq2seq loss. D follows recovered D:
            # train only the first label token so label-only classification rows
            # do not teach the decoder that the label is a complete translation.
            "loss_on_first_token_only": False,
            "control_representation": "string",
        }
    )
    encoder = dict(common, input_text=f"{classification_control} {source}")
    decoder = dict(common, input_text=source, loss_on_first_token_only=True)
    return encoder, decoder


def rewrite_split(
    source_path: Path,
    encoder_path: Path,
    decoder_path: Path,
    *,
    br_control: str,
    pt_control: str,
    classification_control: str,
    forbid_dataset_substring: str,
    progress_every: int,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    encoder_tmp = encoder_path.with_suffix(encoder_path.suffix + ".tmp")
    decoder_tmp = decoder_path.with_suffix(decoder_path.suffix + ".tmp")
    with ExitStack() as stack:
        encoder_fh = stack.enter_context(encoder_tmp.open("w", encoding="utf-8"))
        decoder_fh = stack.enter_context(decoder_tmp.open("w", encoding="utf-8"))
        for position, (_, row) in enumerate(iter_jsonl(source_path), start=1):
            if progress_every > 0 and position % progress_every == 0:
                print(f"[{source_path.name}] processed={position}", flush=True)
            dataset_name = normalize_space(row.get("dataset") or row.get("source") or "UNKNOWN")
            if (
                forbid_dataset_substring
                and forbid_dataset_substring.casefold() in dataset_name.casefold()
            ):
                raise RuntimeError(
                    f"Forbidden dataset metadata {dataset_name!r} in {source_path}:{position}"
                )
            counts[f"dataset:{dataset_name}"] += 1
            task = normalize_space(row.get("task")).casefold()
            if task in {"translation", "translate_br2pt", "translate_pt2br"}:
                encoder, decoder = render_translation(
                    row,
                    br_control=br_control,
                    pt_control=pt_control,
                )
                counts["translation"] += 1
                if encoder["is_equal_pair"]:
                    counts["translation_equal"] += 1
            elif task in {"classification", "classify"}:
                rendered = render_classification(
                    row,
                    br_control=br_control,
                    pt_control=pt_control,
                    classification_control=classification_control,
                )
                if rendered is None:
                    counts["classification_equal_dropped"] += 1
                    continue
                encoder, decoder = rendered
                counts["classification"] += 1
                counts[f"classification_label:{encoder['target_text']}"] += 1
            else:
                raise ValueError(f"Unsupported task {task!r} in {source_path}:{position}")
            encoder_fh.write(json.dumps(encoder, ensure_ascii=False) + "\n")
            decoder_fh.write(json.dumps(decoder, ensure_ascii=False) + "\n")

    if counts["classification_equal_dropped"]:
        raise RuntimeError(
            f"Recovered no-equal source unexpectedly contains equal classification rows: {dict(counts)}"
        )
    if not counts["translation"] or not counts["classification"]:
        raise RuntimeError(f"Missing task in recovered source: {dict(counts)}")
    encoder_tmp.replace(encoder_path)
    decoder_tmp.replace(decoder_path)
    return dict(counts)


def main() -> None:
    args = parse_args()
    for path in (args.mixed_train, args.mixed_valid):
        if not path.exists():
            raise FileNotFoundError(path)
    output_paths: dict[tuple[str, str], Path] = {}
    for mode in ("encoder_unified", "decoder_unified"):
        mode_root = args.out_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid"):
            output_paths[(mode, split)] = mode_root / f"{split}.jsonl"

    report: dict[str, Any] = {
        "source_protocol": "recovered encoder-unified mixed data",
        "mixed_train": str(args.mixed_train),
        "mixed_valid": str(args.mixed_valid),
        "br_control": args.br_control,
        "pt_control": args.pt_control,
        "encoder_classification_prefix": args.classification_control,
        "decoder_classification_prefix": None,
        "equal_translation_policy": "preserve recovered rows",
        "equal_classification_policy": "require absent",
        "classification_supervision": "full sequence, unrestricted vocabulary",
        "forbid_dataset_substring": args.forbid_dataset_substring or None,
        "splits": {},
    }
    for split, source_path in (("train", args.mixed_train), ("valid", args.mixed_valid)):
        report["splits"][split] = rewrite_split(
            source_path,
            output_paths[("encoder_unified", split)],
            output_paths[("decoder_unified", split)],
            br_control=args.br_control,
            pt_control=args.pt_control,
            classification_control=args.classification_control,
            forbid_dataset_substring=args.forbid_dataset_substring,
            progress_every=args.progress_every,
        )
    args.out_root.mkdir(parents=True, exist_ok=True)
    report_path = args.out_root / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
