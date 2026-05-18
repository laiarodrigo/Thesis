#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
TASK_PREFIX_RE = re.compile(r"^\s*<(br-pt|pt-br|id)>\s*", flags=re.IGNORECASE)

# This is intentionally conservative. Stage C should prefer rows where
# variant-specific edits are explicit and paraphrastic drift is small.
MARKER_VARIANTS = {
    "autocarro",
    "ônibus",
    "comboio",
    "trem",
    "telemóvel",
    "celular",
    "sumo",
    "suco",
    "bolachas",
    "biscoitos",
    "pequeno",
    "almoço",
    "manhã",
    "fixe",
    "legal",
    "miúdo",
    "miúda",
    "miúdos",
    "miúdas",
    "garoto",
    "garota",
    "garotos",
    "garotas",
    "rapaz",
    "rapazes",
    "moço",
    "moços",
    "rapariga",
    "raparigas",
    "moça",
    "moças",
    "fato",
    "terno",
    "carrinha",
    "caminhonete",
    "frigorífico",
    "geladeira",
    "gelado",
    "sorvete",
    "chávena",
    "xícara",
    "bicha",
    "fila",
    "propina",
    "mensalidade",
    "lembrar",
    "recordar",
    "ecrã",
    "tela",
    "ficheiro",
    "arquivo",
    "ordenador",
    "esferográfica",
    "caneta",
    "sítio",
    "site",
    "sítios",
    "sites",
    "tu",
    "você",
    "vocês",
    "consigo",
    "connosco",
    "conosco",
    "consigo",
    "contigo",
    "vocês",
}


GPT_SOURCE_HINTS = (
    "gpt_refresh",
    "gpt_frmt_mix",
    "gpt_refresh_2st",
    "r48_2st_refresh",
    # Stage B GPT-Wiki exports are written with empty dataset fields, so
    # subset construction must recover the source family from the path.
    "stageb_gpt_wiki",
    "gpt_wiki",
)


@dataclass
class CandidateRow:
    record_id: str
    source_text: str
    target_text: str
    dataset: str
    bucket: str
    direction: str
    source_path: str
    stage_bucket: str
    src_words: int
    tgt_words: int
    changed_spans: int
    changed_word_tokens: int
    marker_changed_tokens: int
    non_marker_changed_tokens: int
    edit_ratio: float
    structural_overlap: float
    paraphrase_score: float
    changed_spans_preview: str
    marker_preview: str
    keep_reason: str

    def to_json(self) -> dict:
        return {
            "record_id": self.record_id,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "dataset": self.dataset,
            "bucket": self.bucket,
            "direction": self.direction,
            "source_path": self.source_path,
            "stage_bucket": self.stage_bucket,
            "metrics": {
                "src_words": self.src_words,
                "tgt_words": self.tgt_words,
                "changed_spans": self.changed_spans,
                "changed_word_tokens": self.changed_word_tokens,
                "marker_changed_tokens": self.marker_changed_tokens,
                "non_marker_changed_tokens": self.non_marker_changed_tokens,
                "edit_ratio": self.edit_ratio,
                "structural_overlap": self.structural_overlap,
                "paraphrase_score": self.paraphrase_score,
            },
            "changed_spans_preview": self.changed_spans_preview,
            "marker_preview": self.marker_preview,
            "keep_reason": self.keep_reason,
        }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Build a high-confidence Stage C translation subset with bucketed filtering "
            "for exact-copy, FRMT, and filtered OpenSubs/other rows."
        )
    )
    parser.add_argument(
        "--translation-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Translation JSONL export(s) with input_text/target_text/task/dataset/bucket.",
    )
    parser.add_argument(
        "--csv-spec",
        action="append",
        default=[],
        help=(
            "CSV pair source in the form path[:dataset[:bucket]]. "
            "Expected columns include pt_PT and pt_BR."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "encoder_decoder" / "stage_c_subset",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-total", type=int, default=50000)
    parser.add_argument("--ratio-a", type=float, default=0.30)
    parser.add_argument("--ratio-b", type=float, default=0.40)
    parser.add_argument("--ratio-c", type=float, default=0.30)
    parser.add_argument("--max-changed-spans", type=int, default=4)
    parser.add_argument("--max-edit-ratio", type=float, default=0.30)
    parser.add_argument("--min-structural-overlap", type=float, default=0.72)
    parser.add_argument("--max-paraphrase-score", type=float, default=0.18)
    parser.add_argument("--max-non-marker-changed", type=int, default=6)
    parser.add_argument(
        "--max-non-marker-over-marker-gap",
        type=int,
        default=2,
        help=(
            "Reject rows where non-marker changed word tokens exceed marker-changed "
            "word tokens by more than this amount."
        ),
    )
    parser.add_argument(
        "--sample-per-bucket-preview",
        type=int,
        default=50,
        help="How many rows per Stage C bucket to save into preview CSV/JSONL.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000,
        help="Print progress every N raw rows processed.",
    )
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def extract_task_prefix(text: str) -> str | None:
    match = TASK_PREFIX_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip().lower()


def strip_task_prefix(text: str) -> str:
    return normalize_space(TASK_PREFIX_RE.sub("", text or ""))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_space(text))


def is_word(token: str) -> bool:
    return bool(token) and any(ch.isalnum() for ch in token)


def normalize_token(token: str) -> str:
    return token.casefold()


def is_marker_token(token: str) -> bool:
    norm = normalize_token(token)
    return norm in MARKER_VARIANTS


def contains_marker_tokens(text: str) -> bool:
    return any(is_marker_token(tok) for tok in tokenize(text) if is_word(tok))


def canonicalize_dataset(dataset: object, source_path: object) -> str:
    raw = normalize_space(str(dataset or ""))
    low = raw.casefold()
    if low == "frmt":
        return "FRMT"
    if low == "gpt":
        return "GPT"
    if low == "opensubs":
        return "OpenSubs"
    if low in {"unknown", ""}:
        source_low = str(source_path or "").casefold()
        if any(hint in source_low for hint in GPT_SOURCE_HINTS):
            return "GPT"
    return raw or "unknown"


def canonicalize_direction(task: object, direction: object, input_text: object) -> str:
    for value in (task, direction):
        text = normalize_space(str(value or "")).casefold()
        if text in {"translate_br2pt", "br2pt"}:
            return "translate_br2pt"
        if text in {"translate_pt2br", "pt2br"}:
            return "translate_pt2br"
    prefix = extract_task_prefix(str(input_text or ""))
    if prefix == "br-pt":
        return "translate_br2pt"
    if prefix == "pt-br":
        return "translate_pt2br"
    return "translation"


def parse_csv_spec(spec: str) -> tuple[Path, str, str]:
    parts = spec.split(":")
    path = Path(parts[0])
    dataset = parts[1] if len(parts) > 1 and parts[1].strip() else "GPT"
    bucket = parts[2] if len(parts) > 2 and parts[2].strip() else "csv"
    return path, dataset, bucket


def sequence_overlap(src_tokens: list[str], tgt_tokens: list[str]) -> float:
    if not src_tokens and not tgt_tokens:
        return 1.0
    ratio = SequenceMatcher(
        a=[normalize_token(t) for t in src_tokens],
        b=[normalize_token(t) for t in tgt_tokens],
        autojunk=False,
    ).ratio()
    return float(ratio)


def diff_metrics(source_text: str, target_text: str) -> dict:
    src_tokens = tokenize(source_text)
    tgt_tokens = tokenize(target_text)
    matcher = SequenceMatcher(
        a=[normalize_token(t) for t in src_tokens],
        b=[normalize_token(t) for t in tgt_tokens],
        autojunk=False,
    )

    changed_spans = 0
    changed_word_tokens = 0
    marker_changed_tokens = 0
    non_marker_changed_tokens = 0
    changed_previews: list[str] = []
    marker_hits: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_spans += 1
        src_chunk = src_tokens[i1:i2]
        tgt_chunk = tgt_tokens[j1:j2]
        changed_previews.append(
            f"{' '.join(src_chunk[:8]) or '<empty>'} => {' '.join(tgt_chunk[:8]) or '<empty>'}"
        )

        for tok in [*src_chunk, *tgt_chunk]:
            if not is_word(tok):
                continue
            changed_word_tokens += 1
            if is_marker_token(tok):
                marker_changed_tokens += 1
                marker_hits.append(tok)
            else:
                non_marker_changed_tokens += 1

    src_words = sum(1 for tok in src_tokens if is_word(tok))
    tgt_words = sum(1 for tok in tgt_tokens if is_word(tok))
    denom = max(src_words, tgt_words, 1)

    return {
        "src_words": src_words,
        "tgt_words": tgt_words,
        "changed_spans": changed_spans,
        "changed_word_tokens": changed_word_tokens,
        "marker_changed_tokens": marker_changed_tokens,
        "non_marker_changed_tokens": non_marker_changed_tokens,
        "edit_ratio": changed_word_tokens / denom,
        "structural_overlap": sequence_overlap(src_tokens, tgt_tokens),
        "paraphrase_score": non_marker_changed_tokens / denom,
        "changed_spans_preview": " | ".join(changed_previews[:4]),
        "marker_preview": ", ".join(sorted(dict.fromkeys(marker_hits))[:12]),
    }


def iter_translation_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_input_text = str(row.get("input_text", ""))
            input_text = strip_task_prefix(raw_input_text)
            target_text = normalize_space(str(row.get("target_text", "")))
            if not input_text or not target_text:
                continue
            yield {
                "record_id": str(row.get("id") or f"{path.name}:{line_no}"),
                "source_text": input_text,
                "target_text": target_text,
                "dataset": canonicalize_dataset(row.get("dataset"), path.as_posix()),
                "bucket": str(row.get("bucket") or "n/a"),
                "direction": canonicalize_direction(
                    row.get("task"),
                    row.get("direction"),
                    raw_input_text,
                ),
                "source_path": path.as_posix(),
            }


def iter_csv_pairs(path: Path, dataset: str, bucket: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [col for col in ("pt_PT", "pt_BR") if col not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} missing required columns: {', '.join(missing)}")
        for idx, row in enumerate(reader, start=1):
            pt_pt = normalize_space(str(row.get("pt_PT", "")))
            pt_br = normalize_space(str(row.get("pt_BR", "")))
            if not pt_pt or not pt_br:
                continue
            source_id = str(row.get("id") or idx)
            csv_bucket = str(row.get("length") or bucket or "csv")
            yield {
                "record_id": f"{source_id}:br2pt",
                "source_text": pt_br,
                "target_text": pt_pt,
                "dataset": dataset,
                "bucket": csv_bucket,
                "direction": "translate_br2pt",
                "source_path": path.as_posix(),
            }
            yield {
                "record_id": f"{source_id}:pt2br",
                "source_text": pt_pt,
                "target_text": pt_br,
                "dataset": dataset,
                "bucket": csv_bucket,
                "direction": "translate_pt2br",
                "source_path": path.as_posix(),
            }


def build_candidate(raw: dict, args: argparse.Namespace) -> CandidateRow | None:
    source_text = normalize_space(raw["source_text"])
    target_text = normalize_space(raw["target_text"])
    dataset = canonicalize_dataset(raw.get("dataset"), raw.get("source_path"))
    bucket = str(raw.get("bucket") or "n/a")
    dataset_key = dataset.strip().lower()

    if dataset_key not in {"frmt", "gpt"}:
        return None

    metrics = diff_metrics(source_text, target_text)

    if source_text == target_text:
        if contains_marker_tokens(source_text):
            return None
        return CandidateRow(
            record_id=str(raw["record_id"]),
            source_text=source_text,
            target_text=target_text,
            dataset=dataset,
            bucket=bucket,
            direction=str(raw.get("direction") or "translation"),
            source_path=str(raw.get("source_path") or ""),
            stage_bucket="A",
            keep_reason=f"{dataset_key}_equal_marker_safe",
            **metrics,
        )

    passes_core = (
        metrics["changed_spans"] <= args.max_changed_spans
        and metrics["edit_ratio"] <= args.max_edit_ratio
        and metrics["structural_overlap"] >= args.min_structural_overlap
        and metrics["paraphrase_score"] <= args.max_paraphrase_score
        and metrics["non_marker_changed_tokens"] <= args.max_non_marker_changed
        and (
            metrics["marker_changed_tokens"] == 0
            or (
                metrics["non_marker_changed_tokens"] - metrics["marker_changed_tokens"]
                <= args.max_non_marker_over_marker_gap
            )
        )
    )
    if not passes_core:
        return None

    # Apply the same core filter to both FRMT and GPT, then bucket by source.
    if dataset_key == "frmt":
        stage_bucket = "B"
        keep_reason = "frmt_filtered"
    elif dataset_key == "gpt":
        stage_bucket = "C"
        keep_reason = "gpt_filtered"
    else:
        return None

    return CandidateRow(
        record_id=str(raw["record_id"]),
        source_text=source_text,
        target_text=target_text,
        dataset=dataset,
        bucket=bucket,
        direction=str(raw.get("direction") or "translation"),
        source_path=str(raw.get("source_path") or ""),
        stage_bucket=stage_bucket,
        keep_reason=keep_reason,
        **metrics,
    )


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in ratios.items()}
    counts = {key: int(value) for key, value in raw.items()}
    assigned = sum(counts.values())
    remainder = total - assigned
    for key, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True):
        if remainder <= 0:
            break
        counts[key] += 1
        remainder -= 1
    return counts


def rebalance_bucket_counts(targets: dict[str, int], available: dict[str, int]) -> dict[str, int]:
    counts = {key: min(targets.get(key, 0), available.get(key, 0)) for key in targets}
    deficit = sum(targets.values()) - sum(counts.values())
    if deficit <= 0:
        return counts
    spare = {
        key: max(available.get(key, 0) - counts.get(key, 0), 0)
        for key in targets
    }
    for key in sorted(spare, key=lambda item: spare[item], reverse=True):
        if deficit <= 0:
            break
        take = min(spare[key], deficit)
        counts[key] += take
        deficit -= take
    return counts


def sample_bucket(rows: list[CandidateRow], count: int, rng: random.Random) -> list[CandidateRow]:
    if len(rows) <= count:
        return sorted(rows, key=lambda row: row.record_id)
    sample = rng.sample(rows, k=count)
    return sorted(sample, key=lambda row: row.record_id)


def reservoir_add(
    reservoir: list[CandidateRow],
    row: CandidateRow,
    *,
    seen_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if capacity <= 0:
        return
    if len(reservoir) < capacity:
        reservoir.append(row)
        return
    replace_at = rng.randrange(seen_count)
    if replace_at < capacity:
        reservoir[replace_at] = row


def write_jsonl(path: Path, rows: Iterable[CandidateRow]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Iterable[CandidateRow]) -> None:
    fieldnames = [
        "record_id",
        "stage_bucket",
        "keep_reason",
        "dataset",
        "bucket",
        "direction",
        "src_words",
        "tgt_words",
        "changed_spans",
        "changed_word_tokens",
        "marker_changed_tokens",
        "non_marker_changed_tokens",
        "edit_ratio",
        "structural_overlap",
        "paraphrase_score",
        "marker_preview",
        "changed_spans_preview",
        "source_text",
        "target_text",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row.record_id,
                    "stage_bucket": row.stage_bucket,
                    "keep_reason": row.keep_reason,
                    "dataset": row.dataset,
                    "bucket": row.bucket,
                    "direction": row.direction,
                    "src_words": row.src_words,
                    "tgt_words": row.tgt_words,
                    "changed_spans": row.changed_spans,
                    "changed_word_tokens": row.changed_word_tokens,
                    "marker_changed_tokens": row.marker_changed_tokens,
                    "non_marker_changed_tokens": row.non_marker_changed_tokens,
                    "edit_ratio": f"{row.edit_ratio:.4f}",
                    "structural_overlap": f"{row.structural_overlap:.4f}",
                    "paraphrase_score": f"{row.paraphrase_score:.4f}",
                    "marker_preview": row.marker_preview,
                    "changed_spans_preview": row.changed_spans_preview,
                    "source_text": row.source_text,
                    "target_text": row.target_text,
                    "source_path": row.source_path,
                }
            )


def main() -> None:
    args = parse_args()
    ratio_sum = args.ratio_a + args.ratio_b + args.ratio_c
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit("--ratio-a + --ratio-b + --ratio-c must equal 1.0")
    if not args.translation_jsonl and not args.csv_spec:
        raise SystemExit("Provide at least one --translation-jsonl or --csv-spec input.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    reservoirs: dict[str, list[CandidateRow]] = {"A": [], "B": [], "C": []}
    seen_per_bucket = Counter()
    reject_counts = Counter()
    dataset_counts = Counter()
    raw_rows = 0
    kept_candidates = 0

    def process_rows(rows: Iterable[dict], *, source_name: str) -> None:
        nonlocal raw_rows, kept_candidates
        start_raw = raw_rows
        start_kept = kept_candidates
        print(f"[stage-c] start source={source_name}")
        for raw in rows:
            raw_rows += 1
            dataset = str(raw.get("dataset") or "unknown")
            dataset_counts[dataset] += 1
            candidate = build_candidate(raw, args)
            if candidate is None:
                reject_counts[dataset] += 1
                continue
            kept_candidates += 1
            seen_per_bucket[candidate.stage_bucket] += 1
            reservoir_add(
                reservoirs[candidate.stage_bucket],
                candidate,
                seen_count=seen_per_bucket[candidate.stage_bucket],
                capacity=args.max_total,
                rng=rng,
            )
            if args.log_every > 0 and raw_rows % args.log_every == 0:
                print(
                    f"[stage-c] progress raw_rows={raw_rows} kept={kept_candidates} "
                    f"selected_pool_A={len(reservoirs['A'])}/{seen_per_bucket['A']} "
                    f"selected_pool_B={len(reservoirs['B'])}/{seen_per_bucket['B']} "
                    f"selected_pool_C={len(reservoirs['C'])}/{seen_per_bucket['C']} "
                    f"rejected_total={sum(reject_counts.values())}"
                )
        print(
            f"[stage-c] done source={source_name} "
            f"source_raw={raw_rows - start_raw} source_kept={kept_candidates - start_kept} "
            f"cumulative_raw={raw_rows} cumulative_kept={kept_candidates}"
        )

    for path in args.translation_jsonl:
        process_rows(iter_translation_jsonl(path), source_name=path.as_posix())
    for spec in args.csv_spec:
        csv_path, dataset, bucket = parse_csv_spec(spec)
        process_rows(
            iter_csv_pairs(csv_path, dataset, bucket),
            source_name=f"{csv_path.as_posix()}:{dataset}:{bucket}",
        )

    requested = allocate_counts(
        args.max_total,
        {"A": args.ratio_a, "B": args.ratio_b, "C": args.ratio_c},
    )
    available = {key: int(seen_per_bucket.get(key, 0)) for key in ("A", "B", "C")}
    selected_counts = rebalance_bucket_counts(requested, available)

    selected: list[CandidateRow] = []
    preview_rows: list[CandidateRow] = []
    for bucket_name in ("A", "B", "C"):
        sampled = sample_bucket(reservoirs[bucket_name], selected_counts[bucket_name], rng)
        selected.extend(sampled)
        preview_rows.extend(sample_bucket(sampled, min(args.sample_per_bucket_preview, len(sampled)), rng))

    selected.sort(key=lambda row: (row.stage_bucket, row.dataset.casefold(), row.record_id))
    preview_rows.sort(key=lambda row: (row.stage_bucket, row.dataset.casefold(), row.record_id))

    jsonl_path = args.out_dir / "stage_c_subset.jsonl"
    csv_path = args.out_dir / "stage_c_subset.csv"
    preview_jsonl_path = args.out_dir / "stage_c_subset_preview.jsonl"
    preview_csv_path = args.out_dir / "stage_c_subset_preview.csv"
    report_path = args.out_dir / "stage_c_subset_report.json"

    write_jsonl(jsonl_path, selected)
    write_csv(csv_path, selected)
    write_jsonl(preview_jsonl_path, preview_rows)
    write_csv(preview_csv_path, preview_rows)

    report = {
        "inputs": {
            "translation_jsonl": [path.as_posix() for path in args.translation_jsonl],
            "csv_spec": list(args.csv_spec),
        },
        "thresholds": {
            "max_total": args.max_total,
            "ratio_a": args.ratio_a,
            "ratio_b": args.ratio_b,
            "ratio_c": args.ratio_c,
            "max_changed_spans": args.max_changed_spans,
            "max_edit_ratio": args.max_edit_ratio,
            "min_structural_overlap": args.min_structural_overlap,
            "max_paraphrase_score": args.max_paraphrase_score,
            "max_non_marker_changed": args.max_non_marker_changed,
            "max_non_marker_over_marker_gap": args.max_non_marker_over_marker_gap,
        },
        "counts": {
            "raw_rows": raw_rows,
            "kept_candidates": kept_candidates,
            "selected_rows": len(selected),
            "requested_by_bucket": requested,
            "available_by_bucket": available,
            "selected_by_bucket": selected_counts,
            "raw_by_dataset": dict(sorted(dataset_counts.items())),
            "rejected_by_dataset": dict(sorted(reject_counts.items())),
        },
        "selected_by_dataset": dict(
            sorted(Counter(row.dataset for row in selected).items())
        ),
        "selected_by_stage_bucket": dict(
            sorted(Counter(row.stage_bucket for row in selected).items())
        ),
        "selected_by_source_bucket": dict(
            sorted(Counter(f"{row.dataset}:{row.bucket}" for row in selected).items())
        ),
        "output_files": {
            "jsonl": jsonl_path.as_posix(),
            "csv": csv_path.as_posix(),
            "preview_jsonl": preview_jsonl_path.as_posix(),
            "preview_csv": preview_csv_path.as_posix(),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Raw rows: {raw_rows}")
    print(f"Candidate rows after filtering: {kept_candidates}")
    print(f"Selected Stage C rows: {len(selected)}")
    for bucket_name in ("A", "B", "C"):
        print(
            f"  bucket {bucket_name}: available={available[bucket_name]} "
            f"requested={requested[bucket_name]} selected={selected_counts[bucket_name]}"
        )
    print(f"Wrote JSONL: {jsonl_path}")
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote preview JSONL: {preview_jsonl_path}")
    print(f"Wrote preview CSV: {preview_csv_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
