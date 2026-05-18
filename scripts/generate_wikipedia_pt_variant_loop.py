#!/usr/bin/env python3
"""
Run pt-PT/pt-BR generation across multiple Wikipedia inspiration JSON files.

This wrapper calls scripts/generate_pt_variant_prompts_csv.py once per input
JSON batch and writes one output CSV per inspiration file.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Loop over Wikipedia inspiration JSON files and generate pt-PT/pt-BR "
            "CSV batches with one output CSV per input file."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root / "data" / "wikipedia_pt_inspiration",
        help="Directory containing ptwiki_inspiration_batch_*.json files.",
    )
    parser.add_argument(
        "--input-glob",
        default="ptwiki_inspiration_batch_*.json",
        help="Glob used inside --input-dir to select input JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "wikipedia_pt_variant_csv",
        help="Directory where generated CSV files will be written.",
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=repo_root / "scripts" / "generate_pt_variant_prompts_csv.py",
        help="Path to the underlying generator script.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=repo_root / "bla.env",
        help="Environment file forwarded to the generator script.",
    )
    parser.add_argument(
        "--per-file-total",
        type=int,
        default=50,
        help="How many total examples to request per Wikipedia JSON file.",
    )
    parser.add_argument(
        "--reference-pairs",
        type=int,
        default=4,
        help="How many inspiration passages to expose per request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Long-example request size per API call.",
    )
    parser.add_argument(
        "--short-batch-size",
        type=int,
        default=4,
        help="Short-example request size per API call.",
    )
    parser.add_argument(
        "--story-batch-size",
        type=int,
        default=2,
        help="Story-example request size per API call.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=1.0,
        help="Generation over-request multiplier forwarded to the generator.",
    )
    parser.add_argument(
        "--max-candidate-batch",
        type=int,
        default=6,
        help="Maximum request size after multiplier expansion.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=2.0,
        help="Pause between API calls in the generator.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=140,
        help="Maximum API attempts per generation mode, forwarded to the generator.",
    )
    parser.add_argument(
        "--rotate-thread-every-requested-items",
        type=int,
        default=0,
        help=(
            "Rotate the generator thread id after this many requested items have been tried "
            "within a generation mode. 0 disables it."
        ),
    )
    parser.add_argument(
        "--min-variant-differences",
        type=int,
        default=1,
        help="Minimum clear variant differences required per pair.",
    )
    parser.add_argument(
        "--min-enforced-variant-token-edits",
        type=int,
        default=0,
        help="Hard post-filter minimum token edit count between pt_PT and pt_BR.",
    )
    parser.add_argument(
        "--min-enforced-variant-edit-ratio",
        type=float,
        default=0.0,
        help="Hard post-filter minimum edit ratio between pt_PT and pt_BR.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to existing per-batch CSVs instead of overwriting them.",
    )
    parser.add_argument(
        "--rotate-thread-on-backend-error",
        action="store_true",
        default=True,
        help="Rotate thread ids inside the generator after repeated backend errors.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="Stop the loop immediately if one batch generation fails.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="Optional max number of inspiration JSON files to process. 0 means all.",
    )
    parser.add_argument(
        "--start-batch-number",
        type=int,
        default=1,
        help="Start processing from this Wikipedia batch number.",
    )
    parser.add_argument(
        "--carry-last-csv-reference",
        action="store_true",
        default=False,
        help="Pass the most recent successful CSV as an extra reference file to the next batch.",
    )
    parser.add_argument(
        "--rotate-thread-every-batch",
        action="store_true",
        default=False,
        help="Force a fresh IAEDU thread id for every batch generation call.",
    )
    parser.add_argument(
        "--rotate-thread-every-n-batches",
        type=int,
        default=0,
        help="Rotate to a fresh IAEDU thread id every N processed batches. 0 disables it.",
    )
    parser.add_argument(
        "--plain-only",
        action="store_true",
        default=False,
        help="Use the generator's plain sentence mode without long/short/story framing.",
    )
    parser.add_argument(
        "--strict-variant-filter",
        action="store_true",
        default=False,
        help="Enable Stage-C-style strict filtering in the generator.",
    )
    parser.add_argument(
        "--reject-entity-drift",
        action="store_true",
        default=False,
        help="Reject candidate pairs that change names, places, institutions or exact-number entities.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[wiki-loop] {message}", flush=True)


def build_output_path(output_dir: Path, input_path: Path) -> Path:
    stem = input_path.stem.replace("ptwiki_inspiration_", "pt_variant_prompts_wikipedia_")
    return output_dir / f"{stem}.csv"


def extract_batch_number(path: Path) -> int:
    match = re.search(r"_batch_(\d+)$", path.stem)
    if match:
        return int(match.group(1))
    return 0


def load_env_value(path: Path | None, key: str) -> str | None:
    if path is None or not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        if lhs.strip() != key:
            continue
        return rhs.strip().strip('"').strip("'")
    return None


def rotated_thread_id(base_thread_id: str, batch_number: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "", base_thread_id).strip("-_")
    if not base:
        base = "thread"
    suffix = f"b{batch_number:04d}-{uuid.uuid4().hex[:6]}"
    max_base_len = max(8, 64 - len(suffix) - 1)
    base = base[:max_base_len].rstrip("-_")
    return f"{base}-{suffix}"


def grouped_thread_id(base_thread_id: str, group_index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_-]", "", base_thread_id).strip("-_")
    if not base:
        base = "thread"
    suffix = f"g{group_index:04d}-{uuid.uuid4().hex[:6]}"
    max_base_len = max(8, 64 - len(suffix) - 1)
    base = base[:max_base_len].rstrip("-_")
    return f"{base}-{suffix}"


def find_latest_existing_output(output_dir: Path, before_batch_number: int) -> Path | None:
    candidates = []
    for path in output_dir.glob("pt_variant_prompts_wikipedia_batch_*.csv"):
        batch_number = extract_batch_number(path)
        if batch_number < before_batch_number:
            candidates.append((batch_number, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def main() -> None:
    args = parse_args()

    if args.per_file_total <= 0:
        raise SystemExit("--per-file-total must be > 0")
    if args.reference_pairs <= 0:
        raise SystemExit("--reference-pairs must be > 0")
    if args.batch_size <= 0 or args.short_batch_size <= 0 or args.story_batch_size <= 0:
        raise SystemExit("Batch sizes must be > 0")
    if args.candidate_multiplier < 1.0:
        raise SystemExit("--candidate-multiplier must be >= 1.0")
    if args.max_candidate_batch <= 0:
        raise SystemExit("--max-candidate-batch must be > 0")
    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be > 0")
    if args.rotate_thread_every_requested_items < 0:
        raise SystemExit("--rotate-thread-every-requested-items must be >= 0")
    if args.min_variant_differences <= 0:
        raise SystemExit("--min-variant-differences must be > 0")
    if args.min_enforced_variant_token_edits < 0:
        raise SystemExit("--min-enforced-variant-token-edits must be >= 0")
    if not (0.0 <= args.min_enforced_variant_edit_ratio <= 1.0):
        raise SystemExit("--min-enforced-variant-edit-ratio must be between 0 and 1")
    if args.start_batch_number <= 0:
        raise SystemExit("--start-batch-number must be > 0")
    if args.rotate_thread_every_n_batches < 0:
        raise SystemExit("--rotate-thread-every-n-batches must be >= 0")
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.generator_script.exists():
        raise SystemExit(f"Generator script not found: {args.generator_script}")
    if args.env_file is not None and not args.env_file.exists():
        raise SystemExit(f"Env file not found: {args.env_file}")

    input_paths = sorted(args.input_dir.glob(args.input_glob), key=extract_batch_number)
    input_paths = [
        path for path in input_paths if extract_batch_number(path) >= args.start_batch_number
    ]
    if args.limit_files > 0:
        input_paths = input_paths[: args.limit_files]
    if not input_paths:
        raise SystemExit(
            f"No input files matched {args.input_glob!r} under {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, int]] = []
    base_thread_id = load_env_value(args.env_file, "IAEDU_THREAD_ID")
    base_short_thread_id = load_env_value(args.env_file, "IAEDU_SHORT_THREAD_ID")
    previous_output_csv = None
    if args.carry_last_csv_reference:
        previous_output_csv = find_latest_existing_output(
            args.output_dir,
            before_batch_number=extract_batch_number(input_paths[0]),
        )
        if previous_output_csv is not None:
            log(f"using prior CSV as initial extra reference: {previous_output_csv.name}")

    for index, input_path in enumerate(input_paths, start=1):
        batch_number = extract_batch_number(input_path)
        output_csv = build_output_path(args.output_dir, input_path)
        cmd = [
            sys.executable,
            str(args.generator_script),
            "--examples-file",
            str(input_path),
            "--dedupe-against-dir",
            str(args.output_dir),
            "--dedupe-exclude-file",
            str(output_csv),
            "--prompt-style",
            "minimal_lexical",
            "--disable-topic-tags",
            "--exclude-equal",
            "--min-variant-differences",
            str(args.min_variant_differences),
            "--min-enforced-variant-token-edits",
            str(args.min_enforced_variant_token_edits),
            "--min-enforced-variant-edit-ratio",
            str(args.min_enforced_variant_edit_ratio),
            "--reference-pairs",
            str(args.reference_pairs),
            "--total",
            str(args.per_file_total),
            "--batch-size",
            str(args.batch_size),
            "--short-batch-size",
            str(args.short_batch_size),
            "--story-batch-size",
            str(args.story_batch_size),
            "--candidate-multiplier",
            str(args.candidate_multiplier),
            "--max-candidate-batch",
            str(args.max_candidate_batch),
            "--pause-seconds",
            str(args.pause_seconds),
            "--max-attempts",
            str(args.max_attempts),
            "--rotate-thread-every-requested-items",
            str(args.rotate_thread_every_requested_items),
            "--output-csv",
            str(output_csv),
        ]
        if args.plain_only:
            cmd.append("--plain-only")
        if args.env_file is not None:
            cmd.extend(["--env-file", str(args.env_file)])
        if args.rotate_thread_every_batch and base_thread_id:
            cmd.extend(["--thread-id", rotated_thread_id(base_thread_id, batch_number)])
        elif args.rotate_thread_every_n_batches > 0 and base_thread_id:
            group_index = ((index - 1) // args.rotate_thread_every_n_batches) + 1
            cmd.extend(["--thread-id", grouped_thread_id(base_thread_id, group_index)])
        if args.rotate_thread_every_batch and base_short_thread_id:
            cmd.extend(["--short-thread-id", rotated_thread_id(base_short_thread_id, batch_number)])
        elif args.rotate_thread_every_n_batches > 0 and base_short_thread_id:
            group_index = ((index - 1) // args.rotate_thread_every_n_batches) + 1
            cmd.extend(["--short-thread-id", grouped_thread_id(base_short_thread_id, group_index)])
        if args.carry_last_csv_reference and previous_output_csv is not None:
            cmd.extend(["--extra-examples-file", str(previous_output_csv)])
        if args.strict_variant_filter:
            cmd.append("--strict-variant-filter")
        if args.reject_entity_drift:
            cmd.append("--reject-entity-drift")
        if args.append:
            cmd.append("--append")
        if args.rotate_thread_on_backend_error:
            cmd.append("--rotate-thread-on-backend-error")

        log(
            f"[{index}/{len(input_paths)}] generating from {input_path.name} "
            f"-> {output_csv.name} (total={args.per_file_total}, "
            f"extra_ref={previous_output_csv.name if previous_output_csv else 'none'}, "
            f"fresh_thread={'yes' if args.rotate_thread_every_batch else 'no'})"
        )
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append((input_path, result.returncode))
            log(
                f"failed for {input_path.name} with exit code {result.returncode}"
            )
            if args.stop_on_error:
                break
        else:
            previous_output_csv = output_csv

    succeeded = len(input_paths) - len(failures)
    log(
        f"finished: succeeded={succeeded}, failed={len(failures)}, "
        f"output_dir={args.output_dir}"
    )
    if failures:
        for input_path, code in failures:
            log(f"failure: {input_path.name} (exit_code={code})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
