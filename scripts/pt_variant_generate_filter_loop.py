#!/usr/bin/env python3
"""
Iterative pipeline for pt-PT/pt-BR data refresh:
1) Generate new rows with generate_pt_variant_prompts_csv.py
2) Auto-detect weak interchangeable PT/BR substitutions
3) Rewrite BR side to keep stronger variant contrast
4) Repeat for multiple cycles
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
DEFAULT_SEED_WEAK_PAIRS: set[tuple[str, str]] = {
    ("decidida", "determinada"),
    ("determinada", "decidida"),
    ("decidido", "determinado"),
    ("determinado", "decidido"),
    ("calmo", "tranquilo"),
    ("tranquilo", "calmo"),
    ("calma", "tranquila"),
    ("tranquila", "calma"),
    ("calmos", "tranquilos"),
    ("tranquilos", "calmos"),
    ("calmas", "tranquilas"),
    ("tranquilas", "calmas"),
    ("velho", "antigo"),
    ("antigo", "velho"),
    ("velha", "antiga"),
    ("antiga", "velha"),
    ("velhos", "antigos"),
    ("antigos", "velhos"),
    ("velhas", "antigas"),
    ("antigas", "velhas"),
    ("viu", "avistou"),
    ("avistou", "viu"),
    ("viram", "avistaram"),
    ("avistaram", "viram"),
    ("viam", "avistavam"),
    ("avistavam", "viam"),
    ("lentamente", "devagar"),
    ("devagar", "lentamente"),
    ("rapariga", "moca"),
    ("moca", "rapariga"),
    ("observava", "olhava"),
    ("olhava", "observava"),
    ("fotografias", "fotos"),
    ("fotos", "fotografias"),
}
DEFAULT_DIALECT_MARKER_PAIRS: set[tuple[str, str]] = {
    ("comboios", "trens"),
    ("comboio", "trem"),
    ("autocarros", "onibus"),
    ("autocarro", "onibus"),
    ("telemoveis", "celulares"),
    ("telemovel", "celular"),
    ("sumo", "suco"),
    ("esplanada", "varanda"),
    ("montra", "vitrine"),
    ("chavena", "xicara"),
    ("colectiva", "coletiva"),
    ("colectivo", "coletivo"),
    ("registo", "registro"),
    ("humidos", "umidos"),
    ("humidas", "umidas"),
    ("humido", "umido"),
    ("humida", "umida"),
    ("trilhos", "trilhas"),
    ("trilho", "trilha"),
    ("raparigas", "garotas"),
    ("rapariga", "garota"),
    ("miudos", "garotos"),
    ("miudo", "garoto"),
}


def log(message: str) -> None:
    print(f"[loop] {message}", flush=True)


def fold_text(text: str) -> str:
    raw = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(fold_text(text))


def tokens_raw(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def variant_edit_stats(pt_text: str, br_text: str) -> tuple[int, float]:
    ta = tokens(pt_text)
    tb = tokens(br_text)
    if not ta and not tb:
        return 0, 0.0
    matcher = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    edits = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        edits += max(i2 - i1, j2 - j1)
    denom = max(len(ta), len(tb), 1)
    return edits, edits / denom


def count_dialect_marker_pairs(
    pt_text: str,
    br_text: str,
    marker_pairs: set[tuple[str, str]],
) -> int:
    pt_fold = fold_text(pt_text or "")
    br_fold = fold_text(br_text or "")
    matched = 0
    for pt_tok, br_tok in marker_pairs:
        if not pt_tok or not br_tok:
            continue
        pt_pat = re.compile(rf"\b{re.escape(pt_tok)}\b")
        br_pat = re.compile(rf"\b{re.escape(br_tok)}\b")
        if pt_pat.search(pt_fold) and br_pat.search(br_fold):
            matched += 1
    return matched


def replace_first_token_case_aware(text: str, old_token: str, new_token: str) -> tuple[str, bool]:
    pattern = re.compile(rf"\b{re.escape(old_token)}\b", flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        current = match.group(0)
        if current.isupper():
            return new_token.upper()
        if current and current[0].isupper():
            return new_token.capitalize()
        return new_token.lower()

    updated, n = pattern.subn(repl, text, count=1)
    return updated, n > 0


def parse_weak_pair_spec(spec: str) -> tuple[str, str]:
    if ":" in spec:
        left, right = spec.split(":", 1)
    elif "," in spec:
        left, right = spec.split(",", 1)
    elif "=" in spec:
        left, right = spec.split("=", 1)
    else:
        raise ValueError(f"Invalid weak pair '{spec}'. Use pt:br format.")
    pt_tok = fold_text(left).strip()
    br_tok = fold_text(right).strip()
    if not pt_tok or not br_tok:
        raise ValueError(f"Invalid weak pair '{spec}'. Empty token.")
    return pt_tok, br_tok


def parse_marker_pair_spec(spec: str) -> tuple[str, str]:
    left, right = parse_weak_pair_spec(spec)
    return left, right


def load_seed_weak_pairs(
    *,
    include_default: bool,
    pair_specs: list[str],
    pairs_file: Path | None,
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set(DEFAULT_SEED_WEAK_PAIRS if include_default else set())

    for spec in pair_specs:
        out.add(parse_weak_pair_spec(spec))

    if pairs_file is not None:
        if not pairs_file.exists():
            raise SystemExit(f"Weak-pairs file not found: {pairs_file}")
        for raw in pairs_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.add(parse_weak_pair_spec(line))

    return out


def load_dialect_marker_pairs(
    *,
    include_default: bool,
    pair_specs: list[str],
    pairs_file: Path | None,
) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set(DEFAULT_DIALECT_MARKER_PAIRS if include_default else set())

    for spec in pair_specs:
        out.add(parse_marker_pair_spec(spec))

    if pairs_file is not None:
        if not pairs_file.exists():
            raise SystemExit(f"Dialect-marker pairs file not found: {pairs_file}")
        for raw in pairs_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            out.add(parse_marker_pair_spec(line))

    return out


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def save_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_reversible_weak_pairs(
    rows: list[dict[str, str]],
    *,
    min_pair_count: int,
    min_reverse_count: int,
    min_token_total: int,
    balanced_ratio_low: float,
    balanced_ratio_high: float,
    ignore_tokens: set[str],
    seed_pairs: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], dict[str, Any], dict[str, float]]:
    pt_counts: Counter[str] = Counter()
    br_counts: Counter[str] = Counter()
    replace_pairs: Counter[tuple[str, str]] = Counter()

    for row in rows:
        ta = tokens(row.get("pt_PT", ""))
        tb = tokens(row.get("pt_BR", ""))
        pt_counts.update(ta)
        br_counts.update(tb)

        matcher = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            if i2 - i1 == 1 and j2 - j1 == 1:
                pt_tok = ta[i1]
                br_tok = tb[j1]
                if pt_tok != br_tok:
                    replace_pairs[(pt_tok, br_tok)] += 1

    token_bias: dict[str, float] = {}
    for tok in set(pt_counts) | set(br_counts):
        token_bias[tok] = (br_counts[tok] + 1) / (pt_counts[tok] + 1)

    def is_balanced(tok: str) -> bool:
        if tok in ignore_tokens:
            return False
        total = pt_counts[tok] + br_counts[tok]
        if total < min_token_total:
            return False
        ratio = token_bias.get(tok, 1.0)
        return balanced_ratio_low <= ratio <= balanced_ratio_high

    weak_pairs: set[tuple[str, str]] = set()
    reversible_examples: list[dict[str, Any]] = []
    for (pt_tok, br_tok), count in replace_pairs.items():
        reverse = replace_pairs.get((br_tok, pt_tok), 0)
        if count < min_pair_count or reverse < min_reverse_count:
            continue
        if not (is_balanced(pt_tok) and is_balanced(br_tok)):
            continue
        weak_pairs.add((pt_tok, br_tok))
        reversible_examples.append(
            {
                "pt_to_br": [pt_tok, br_tok],
                "count": count,
                "reverse_count": reverse,
                "pt_bias": round(token_bias.get(pt_tok, 1.0), 4),
                "br_bias": round(token_bias.get(br_tok, 1.0), 4),
            }
        )

    injected = [
        pair
        for pair in sorted(seed_pairs)
        if pair[0] not in ignore_tokens and pair[1] not in ignore_tokens
    ]
    weak_pairs.update(injected)

    summary = {
        "weak_pairs_count": len(weak_pairs),
        "seed_pairs_injected": [list(pair) for pair in injected[:100]],
        "top_reversible_pairs": sorted(
            reversible_examples, key=lambda item: item["count"], reverse=True
        )[:30],
    }
    return weak_pairs, summary, token_bias


def rewrite_rows_from_weak_pairs(
    rows: list[dict[str, str]],
    weak_pairs: set[tuple[str, str]],
    token_bias: dict[str, float],
    *,
    min_strong_diffs: int,
    strong_bias_low: float,
    strong_bias_high: float,
    pt_rewrite_share: float,
) -> tuple[int, list[dict[str, Any]]]:
    changed = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        pt = row.get("pt_PT", "")
        br = row.get("pt_BR", "")
        try:
            row_id = int(str(row.get("id", "")).strip())
        except Exception:
            row_id = 0
        ta_fold = tokens(pt)
        tb_fold = tokens(br)
        ta_raw = tokens_raw(pt)
        tb_raw = tokens_raw(br)
        matcher = difflib.SequenceMatcher(a=ta_fold, b=tb_fold, autojunk=False)

        weak_ops: list[tuple[str, str, str, str]] = []
        strong_diffs = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
                pt_tok = ta_fold[i1]
                br_tok = tb_fold[j1]
                pt_tok_raw = ta_raw[i1] if i1 < len(ta_raw) else pt_tok
                br_tok_raw = tb_raw[j1] if j1 < len(tb_raw) else br_tok
                if (pt_tok, br_tok) in weak_pairs:
                    weak_ops.append((pt_tok, br_tok, pt_tok_raw, br_tok_raw))
                else:
                    pt_ratio = token_bias.get(pt_tok, 1.0)
                    br_ratio = token_bias.get(br_tok, 1.0)
                    if (
                        pt_ratio <= strong_bias_low
                        or pt_ratio >= strong_bias_high
                        or br_ratio <= strong_bias_low
                        or br_ratio >= strong_bias_high
                    ):
                        strong_diffs += 1
            elif tag == "replace":
                # Multi-token replacement already indicates stronger lexical divergence.
                strong_diffs += 1

        if not weak_ops or strong_diffs < min_strong_diffs:
            continue

        new_pt = pt
        new_br = br
        applied: list[dict[str, Any]] = []
        for pt_tok, br_tok, pt_tok_raw, br_tok_raw in weak_ops:
            # Deterministic split so we do not always normalize BR only.
            checksum = row_id + sum(ord(ch) for ch in (pt_tok + br_tok))
            rewrite_pt_side = pt_rewrite_share > 0 and (checksum % 100) < int(pt_rewrite_share * 100)
            if rewrite_pt_side:
                updated, ok = replace_first_token_case_aware(new_pt, pt_tok_raw, br_tok_raw)
                if ok:
                    new_pt = updated
                    applied.append(
                        {
                            "pair": [pt_tok, br_tok],
                            "side": "pt_PT",
                            "from": pt_tok_raw,
                            "to": br_tok_raw,
                        }
                    )
            else:
                updated, ok = replace_first_token_case_aware(new_br, br_tok_raw, pt_tok_raw)
                if ok:
                    new_br = updated
                    applied.append(
                        {
                            "pair": [pt_tok, br_tok],
                            "side": "pt_BR",
                            "from": br_tok_raw,
                            "to": pt_tok_raw,
                        }
                    )

        if new_br == br and new_pt == pt:
            continue

        row["pt_PT"] = new_pt
        row["pt_BR"] = new_br
        if "pt_BR_words" in row:
            row["pt_BR_words"] = str(token_count(new_br))
        if "pt_PT_words" in row:
            row["pt_PT_words"] = str(token_count(new_pt))
        changed += 1
        if len(examples) < 25:
            examples.append(
                {
                    "id": row.get("id", ""),
                    "applied_pairs": applied,
                    "pt_pt_before": pt,
                    "pt_pt_after": new_pt,
                    "pt_br_before": br,
                    "pt_br_after": new_br,
                }
            )

    return changed, examples


def run_generator(
    generator_script: Path,
    output_csv: Path,
    total: int,
    forwarded_args: list[str],
) -> None:
    cmd = [
        sys.executable,
        str(generator_script),
        "--append",
        "--output-csv",
        str(output_csv),
        "--total",
        str(total),
    ] + forwarded_args
    log("Running generator: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def max_row_id(rows: list[dict[str, str]]) -> int:
    best = 0
    for row in rows:
        try:
            best = max(best, int(str(row.get("id", "")).strip()))
        except Exception:
            continue
    return best


def drop_weak_new_rows(
    rows: list[dict[str, str]],
    *,
    previous_max_id: int,
    min_token_edits: int,
    min_edit_ratio: float,
) -> tuple[list[dict[str, str]], int, list[dict[str, Any]]]:
    kept: list[dict[str, str]] = []
    dropped = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        try:
            rid = int(str(row.get("id", "")).strip())
        except Exception:
            kept.append(row)
            continue

        if rid <= previous_max_id:
            kept.append(row)
            continue

        pt = row.get("pt_PT", "")
        br = row.get("pt_BR", "")
        edits, ratio = variant_edit_stats(pt, br)
        if edits < min_token_edits or ratio < min_edit_ratio:
            dropped += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "id": rid,
                        "token_edits": edits,
                        "edit_ratio": round(ratio, 4),
                        "pt_pt": pt,
                        "pt_br": br,
                    }
                )
            continue
        kept.append(row)

    return kept, dropped, examples


def drop_rows_missing_dialect_markers(
    rows: list[dict[str, str]],
    *,
    min_pairs: int,
    marker_pairs: set[tuple[str, str]],
    only_new_after: int | None = None,
    id_min: int | None = None,
    id_max: int | None = None,
) -> tuple[list[dict[str, str]], int, list[dict[str, Any]]]:
    if min_pairs <= 0:
        return rows, 0, []
    if not marker_pairs:
        raise SystemExit(
            "Dialect-marker gate enabled but no marker pairs were configured."
        )

    kept: list[dict[str, str]] = []
    dropped = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        rid_raw = str(row.get("id", "")).strip()
        if not rid_raw.isdigit():
            kept.append(row)
            continue
        rid = int(rid_raw)

        if only_new_after is not None and rid <= only_new_after:
            kept.append(row)
            continue
        if id_min is not None and rid < id_min:
            kept.append(row)
            continue
        if id_max is not None and rid > id_max:
            kept.append(row)
            continue

        pt = row.get("pt_PT", "")
        br = row.get("pt_BR", "")
        marker_count = count_dialect_marker_pairs(pt, br, marker_pairs)
        if marker_count < min_pairs:
            dropped += 1
            if len(examples) < 25:
                examples.append(
                    {
                        "id": rid,
                        "marker_pairs": marker_count,
                        "required_min_pairs": min_pairs,
                        "pt_pt": pt,
                        "pt_br": br,
                    }
                )
            continue
        kept.append(row)

    return kept, dropped, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Loop: generate new pt-variant rows, auto-rewrite weak pairs, repeat."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="CSV to append and clean in-place.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="How many generate+clean cycles to execute.",
    )
    parser.add_argument(
        "--per-cycle-total",
        type=int,
        default=40,
        help="How many rows to request in each generation cycle.",
    )
    parser.add_argument(
        "--generator-script",
        type=Path,
        default=Path("scripts/generate_pt_variant_prompts_csv.py"),
        help="Path to generation script.",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        default=False,
        help="Run only cleaning phase in each cycle.",
    )
    parser.add_argument(
        "--continue-on-generate-error",
        action="store_true",
        default=False,
        help="Continue to cleaning even if generation fails in a cycle.",
    )
    parser.add_argument(
        "--no-default-seed-pairs",
        action="store_true",
        default=False,
        help="Disable built-in weak-pair seeds (e.g., decidida:determinada).",
    )
    parser.add_argument(
        "--weak-pair",
        action="append",
        default=[],
        help="Add explicit weak pair in pt:br format. Repeatable.",
    )
    parser.add_argument(
        "--weak-pairs-file",
        type=Path,
        default=None,
        help="Optional text file with one weak pair per line (pt:br).",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=2,
        help="Min count for a PT->BR replacement pair to be considered weak.",
    )
    parser.add_argument(
        "--min-reverse-count",
        type=int,
        default=1,
        help="Min reverse count for BR->PT replacement to mark a weak reversible pair.",
    )
    parser.add_argument(
        "--min-token-total",
        type=int,
        default=10,
        help="Min total token frequency (PT+BR) to include in weak-pair detection.",
    )
    parser.add_argument(
        "--balanced-ratio-low",
        type=float,
        default=0.75,
        help="Lower bound of BR/PT frequency ratio to treat token as balanced.",
    )
    parser.add_argument(
        "--balanced-ratio-high",
        type=float,
        default=1.33,
        help="Upper bound of BR/PT frequency ratio to treat token as balanced.",
    )
    parser.add_argument(
        "--min-strong-diffs",
        type=int,
        default=1,
        help="Only rewrite weak pairs when row already has at least this many stronger diffs.",
    )
    parser.add_argument(
        "--strong-bias-low",
        type=float,
        default=0.56,
        help="Lower bound for strong-difference bias threshold.",
    )
    parser.add_argument(
        "--strong-bias-high",
        type=float,
        default=1.80,
        help="Upper bound for strong-difference bias threshold.",
    )
    parser.add_argument(
        "--pt-rewrite-share",
        type=float,
        default=0.35,
        help=(
            "Share in [0,1] of weak-pair rewrites applied on pt_PT side "
            "(the rest are applied on pt_BR side)."
        ),
    )
    parser.add_argument(
        "--ignore-token",
        action="append",
        default=["de", "do", "da", "dos", "das"],
        help="Token to ignore in weak-pair discovery. Repeatable.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("data/pt_variant_generate_filter_loop_report.json"),
        help="Where to save cycle report.",
    )
    parser.add_argument(
        "--max-example-logs",
        type=int,
        default=8,
        help="How many rewritten examples to print per cycle.",
    )
    parser.add_argument(
        "--disable-new-heavy-rewrite",
        action="store_true",
        default=False,
        help="Disable post-generation heavy-rewrite gate for newly added rows.",
    )
    parser.add_argument(
        "--new-min-token-edits",
        type=int,
        default=3,
        help="Heavy-rewrite gate for new rows: minimum token edits.",
    )
    parser.add_argument(
        "--new-min-edit-ratio",
        type=float,
        default=0.12,
        help="Heavy-rewrite gate for new rows: minimum edit ratio in [0,1].",
    )
    parser.add_argument(
        "--min-dialect-marker-pairs",
        type=int,
        default=0,
        help=(
            "Reject rows that do not contain at least this number of explicit "
            "PT->BR dialect marker pairs."
        ),
    )
    parser.add_argument(
        "--no-default-dialect-marker-pairs",
        action="store_true",
        default=False,
        help="Disable built-in dialect marker pairs.",
    )
    parser.add_argument(
        "--dialect-marker-pair",
        action="append",
        default=[],
        help="Add explicit dialect marker pair in pt:br format. Repeatable.",
    )
    parser.add_argument(
        "--dialect-marker-pairs-file",
        type=Path,
        default=None,
        help="Optional text file with one dialect marker pair per line (pt:br).",
    )
    parser.add_argument(
        "--enforce-dialect-on-new",
        action="store_true",
        default=False,
        help="Apply dialect-marker rejection gate on newly generated rows.",
    )
    parser.add_argument(
        "--enforce-dialect-id-min",
        type=int,
        default=None,
        help="Optional lower ID bound to enforce dialect-marker gate on existing rows.",
    )
    parser.add_argument(
        "--enforce-dialect-id-max",
        type=int,
        default=None,
        help="Optional upper ID bound to enforce dialect-marker gate on existing rows.",
    )
    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to generator script (use after --).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cycles <= 0:
        raise SystemExit("--cycles must be > 0")
    if args.per_cycle_total <= 0:
        raise SystemExit("--per-cycle-total must be > 0")
    if args.min_pair_count <= 0 or args.min_reverse_count <= 0:
        raise SystemExit("--min-pair-count and --min-reverse-count must be > 0")
    if not (0.0 < args.balanced_ratio_low <= args.balanced_ratio_high):
        raise SystemExit("invalid balanced ratio bounds")
    if args.min_strong_diffs < 0:
        raise SystemExit("--min-strong-diffs must be >= 0")
    if not (0.0 <= args.pt_rewrite_share <= 1.0):
        raise SystemExit("--pt-rewrite-share must be between 0 and 1")
    if args.new_min_token_edits < 0:
        raise SystemExit("--new-min-token-edits must be >= 0")
    if not (0.0 <= args.new_min_edit_ratio <= 1.0):
        raise SystemExit("--new-min-edit-ratio must be between 0 and 1")
    if args.min_dialect_marker_pairs < 0:
        raise SystemExit("--min-dialect-marker-pairs must be >= 0")
    if (
        args.enforce_dialect_id_min is not None
        and args.enforce_dialect_id_max is not None
        and args.enforce_dialect_id_min > args.enforce_dialect_id_max
    ):
        raise SystemExit("--enforce-dialect-id-min cannot be greater than --enforce-dialect-id-max")

    output_csv = args.output_csv
    if not output_csv.exists():
        raise SystemExit(f"CSV not found: {output_csv}")

    forwarded = list(args.generator_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    report: dict[str, Any] = {"cycles": []}
    ignore_tokens = {fold_text(tok) for tok in args.ignore_token if tok.strip()}
    seed_pairs = load_seed_weak_pairs(
        include_default=not args.no_default_seed_pairs,
        pair_specs=args.weak_pair,
        pairs_file=args.weak_pairs_file,
    )
    log(f"Seed weak pairs loaded: {len(seed_pairs)}")
    dialect_marker_pairs = load_dialect_marker_pairs(
        include_default=not args.no_default_dialect_marker_pairs,
        pair_specs=args.dialect_marker_pair,
        pairs_file=args.dialect_marker_pairs_file,
    )
    if args.min_dialect_marker_pairs > 0:
        log(f"Dialect marker pairs loaded: {len(dialect_marker_pairs)}")

    for cycle in range(1, args.cycles + 1):
        log(f"=== Cycle {cycle}/{args.cycles} ===")
        cycle_info: dict[str, Any] = {"cycle": cycle}
        before_rows, _ = load_rows(output_csv)
        previous_max_id = max_row_id(before_rows)
        cycle_info["previous_max_id"] = previous_max_id

        if not args.skip_generate:
            try:
                run_generator(
                    generator_script=args.generator_script,
                    output_csv=output_csv,
                    total=args.per_cycle_total,
                    forwarded_args=forwarded,
                )
                cycle_info["generation_status"] = "ok"
            except subprocess.CalledProcessError as exc:
                cycle_info["generation_status"] = "error"
                cycle_info["generation_returncode"] = exc.returncode
                log(f"Generation failed in cycle {cycle}: returncode={exc.returncode}")
                if not args.continue_on_generate_error:
                    report["cycles"].append(cycle_info)
                    break
        else:
            cycle_info["generation_status"] = "skipped"

        rows, fieldnames = load_rows(output_csv)
        if (not args.disable_new_heavy_rewrite) and cycle_info.get("generation_status") == "ok":
            rows, dropped, drop_examples = drop_weak_new_rows(
                rows,
                previous_max_id=previous_max_id,
                min_token_edits=args.new_min_token_edits,
                min_edit_ratio=args.new_min_edit_ratio,
            )
            cycle_info["new_rows_dropped_for_weak_rewrite"] = dropped
            cycle_info["new_rows_drop_examples"] = drop_examples
            log(f"Dropped weak new rows in cycle {cycle}: {dropped}")

        weak_pairs, weak_summary, token_bias = discover_reversible_weak_pairs(
            rows,
            min_pair_count=args.min_pair_count,
            min_reverse_count=args.min_reverse_count,
            min_token_total=args.min_token_total,
            balanced_ratio_low=args.balanced_ratio_low,
            balanced_ratio_high=args.balanced_ratio_high,
            ignore_tokens=ignore_tokens,
            seed_pairs=seed_pairs,
        )
        cycle_info["weak_pair_summary"] = weak_summary
        log(f"Detected weak reversible pairs: {len(weak_pairs)}")

        changed, examples = rewrite_rows_from_weak_pairs(
            rows,
            weak_pairs,
            token_bias,
            min_strong_diffs=args.min_strong_diffs,
            strong_bias_low=args.strong_bias_low,
            strong_bias_high=args.strong_bias_high,
            pt_rewrite_share=args.pt_rewrite_share,
        )
        cycle_info["rows_rewritten"] = changed
        cycle_info["example_rewrites"] = examples

        if (
            args.min_dialect_marker_pairs > 0
            and args.enforce_dialect_on_new
            and cycle_info.get("generation_status") == "ok"
        ):
            rows, dropped, drop_examples = drop_rows_missing_dialect_markers(
                rows,
                min_pairs=args.min_dialect_marker_pairs,
                marker_pairs=dialect_marker_pairs,
                only_new_after=previous_max_id,
            )
            cycle_info["new_rows_dropped_for_missing_dialect_markers"] = dropped
            cycle_info["new_rows_missing_dialect_marker_examples"] = drop_examples
            log(f"Dropped new rows without enough dialect markers in cycle {cycle}: {dropped}")

        if (
            args.min_dialect_marker_pairs > 0
            and (
                args.enforce_dialect_id_min is not None
                or args.enforce_dialect_id_max is not None
            )
        ):
            rows, dropped, drop_examples = drop_rows_missing_dialect_markers(
                rows,
                min_pairs=args.min_dialect_marker_pairs,
                marker_pairs=dialect_marker_pairs,
                id_min=args.enforce_dialect_id_min,
                id_max=args.enforce_dialect_id_max,
            )
            cycle_info["rows_dropped_for_missing_dialect_markers_in_range"] = dropped
            cycle_info["rows_missing_dialect_marker_examples_in_range"] = drop_examples
            log(f"Dropped rows without enough dialect markers in configured ID range: {dropped}")

        save_rows(output_csv, rows, fieldnames)
        log(f"Rewritten rows in cycle {cycle}: {changed}")
        for sample in examples[: args.max_example_logs]:
            parts = [f"ID {sample['id']}", f"pairs={sample['applied_pairs']}"]
            if sample.get("pt_pt_before") != sample.get("pt_pt_after"):
                parts.append(
                    "PT: "
                    + repr(sample.get("pt_pt_before", ""))
                    + " -> "
                    + repr(sample.get("pt_pt_after", ""))
                )
            if sample.get("pt_br_before") != sample.get("pt_br_after"):
                parts.append(
                    "BR: "
                    + repr(sample.get("pt_br_before", ""))
                    + " -> "
                    + repr(sample.get("pt_br_after", ""))
                )
            log(" | ".join(parts))

        report["cycles"].append(cycle_info)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote report to {args.report_json.resolve()}")


if __name__ == "__main__":
    main()
