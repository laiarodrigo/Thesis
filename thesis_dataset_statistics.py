#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def require_module(name: str, install_hint: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Missing dependency '{name}'. {install_hint}"
        ) from exc


duckdb = require_module(
    "duckdb",
    "Use a Python environment with duckdb installed, for example the repo-local ./thesis/bin/python.",
)
yaml = require_module(
    "yaml",
    "Use a Python environment with PyYAML installed, for example the repo-local ./thesis/bin/python.",
)
transformers = require_module(
    "transformers",
    "Use a Python environment with transformers installed, for example the repo-local ./thesis/bin/python.",
)
AutoTokenizer = transformers.AutoTokenizer


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent
DEFAULT_REFERENCE_CONFIG_CANDIDATES = [
    DEFAULT_REPO_ROOT
    / "configs"
    / "encoder_decoder"
    / "t5gemma2"
    / "comparison_staged"
    / "translation_fullft_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml",
    DEFAULT_REPO_ROOT
    / "configs"
    / "encoder_decoder"
    / "t5gemma2_4b"
    / "comparison_staged"
    / "translation_r48_stageB_gpt_wikipedia_plus_frmt_label_first_with_cls.yaml",
    DEFAULT_REPO_ROOT
    / "configs"
    / "encoder_decoder"
    / "t5gemma2_4b"
    / "comparison_staged"
    / "translation_r48_stageB_gpt_wikipedia_plus_frmt_with_cls.yaml",
]
DEFAULT_PROJECT_DB_CANDIDATES = [
    DEFAULT_REPO_ROOT / "data" / "duckdb" / "subs_project.duckdb",
]
DEFAULT_SOURCE_DB_CANDIDATES = [
    DEFAULT_REPO_ROOT / "data" / "duckdb" / "subs.duckdb",
]
DEFAULT_PTBR_DB_CANDIDATES = [
    DEFAULT_REPO_ROOT / "data" / "duckdb" / "subs_ptbr_filtered.duckdb",
    DEFAULT_REPO_ROOT / "data" / "duckdb" / "subs_filtered_final.duckdb",
    DEFAULT_REPO_ROOT / "data" / "duckdb" / "ptbrvarid_only.duckdb",
]
DEFAULT_PTBR_SAMPLED_CSV_CANDIDATES = [
    DEFAULT_REPO_ROOT / "data" / "ptbrvarid" / "translated_stageb_pairs" / "sampled_rows.csv",
    DEFAULT_REPO_ROOT
    / "data"
    / "ptbrvarid"
    / "translated_stageb_pairs_r48_500_t50"
    / "sampled_rows_resampled.csv",
]
VALID_TOPUP_SEED = 42
TRANSLATION_VALID_MIN_ROWS = 200
STAGE_B_MIX_VALID_MIN_ROWS = 200
STAGE_A_PTBRVARID_TARGET_SHARE = 0.33
DEFAULT_STAGE_A_PTBRVARID_EXCLUDED_DOMAINS = "social_media,web"


def log(message: str) -> None:
    print(message, flush=True)


def normalize_space(text: Any) -> str:
    return " ".join(str(text or "").split())


def parse_domain_list(raw: str | None) -> set[str]:
    return {
        normalize_space(item).casefold()
        for item in str(raw or "").split(",")
        if normalize_space(item)
    }


def as_relative(repo_root: Path, path: Path | str) -> str:
    path_obj = Path(path).resolve() if isinstance(path, Path) else (repo_root / path).resolve()
    try:
        return path_obj.relative_to(repo_root).as_posix()
    except ValueError:
        return path_obj.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def maybe_repo_local_path(repo_root: Path, raw: str) -> Path | None:
    raw = str(raw or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    if raw.startswith(("outputs/", "data/", "configs/", "./", "../")):
        return (repo_root / candidate).resolve()
    return None


def find_first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def discover_reference_config(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        resolved = requested if requested.is_absolute() else (repo_root / requested).resolve()
        if not resolved.exists():
            raise SystemExit(f"Reference config not found: {resolved}")
        return resolved
    for candidate in DEFAULT_REFERENCE_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit("Could not auto-discover a reference thesis config. Pass --reference-config.")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover_project_db(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        resolved = requested if requested.is_absolute() else (repo_root / requested).resolve()
        if not resolved.exists():
            raise SystemExit(f"Project DB not found: {resolved}")
        return resolved
    candidate = find_first_existing(DEFAULT_PROJECT_DB_CANDIDATES)
    if candidate is None:
        raise SystemExit("Could not find subs_project.duckdb.")
    return candidate


def discover_source_db(repo_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        resolved = requested if requested.is_absolute() else (repo_root / requested).resolve()
        if not resolved.exists():
            raise SystemExit(f"Source DB not found: {resolved}")
        return resolved
    candidate = find_first_existing(DEFAULT_SOURCE_DB_CANDIDATES)
    if candidate is None:
        raise SystemExit("Could not find subs.duckdb.")
    return candidate


def try_open_duckdb(path: Path, *, required_tables: set[str] | None = None) -> tuple[bool, str]:
    try:
        con = duckdb.connect(path.as_posix(), read_only=True)
        try:
            if required_tables:
                tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
                missing = sorted(required_tables.difference(tables))
                if missing:
                    return False, f"missing tables: {', '.join(missing)}"
        finally:
            con.close()
    except Exception as exc:
        return False, repr(exc)
    return True, "ok"


@dataclass
class DiscoveryNote:
    requested: str
    used: str
    status: str
    detail: str


def discover_ptbr_db(
    repo_root: Path,
    requested: Path | None,
    notes: list[str],
) -> Path:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested if requested.is_absolute() else (repo_root / requested).resolve())
    candidates.extend(DEFAULT_PTBR_DB_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            notes.append(f"PtBrVId DB candidate missing: {candidate}")
            continue
        ok, detail = try_open_duckdb(candidate, required_tables={"ptbrvarid"})
        if ok:
            notes.append(f"Using PtBrVId DB: {candidate}")
            return candidate
        notes.append(f"Rejected PtBrVId DB candidate {candidate}: {detail}")
    raise SystemExit("Could not find a readable PtBrVId DuckDB with table 'ptbrvarid'.")


def discover_ptbr_sampled_csv(repo_root: Path, requested: Path | None) -> Path | None:
    candidates: list[Path] = []
    if requested is not None:
        candidates.append(requested if requested.is_absolute() else (repo_root / requested).resolve())
    candidates.extend(DEFAULT_PTBR_SAMPLED_CSV_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_gpt_pairs_dir(repo_root: Path, notes: list[str]) -> Path:
    preferred = repo_root / "data" / "encoder_decoder" / "t5gemma2" / "compare_staged_v2" / "stageB_gpt_wiki"
    fallback = repo_root / "data" / "encoder_decoder" / "t5gemma2"
    if (preferred / "pairs_train.jsonl").exists():
        notes.append(f"Using Stage B GPT/Wikipedia pair exports from {preferred}")
        return preferred
    if (fallback / "pairs_train.jsonl").exists():
        notes.append(
            "Stage B GPT/Wikipedia compare_staged_v2 directory is missing; "
            f"falling back to root T5Gemma2 split exports in {fallback}"
        )
        return fallback
    raise SystemExit("Could not find GPT/Wikipedia pair exports (pairs_train.jsonl).")


def discover_ptbr_translated_dir(repo_root: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        resolved = requested if requested.is_absolute() else (repo_root / requested).resolve()
        if not resolved.exists():
            raise SystemExit(f"PtBrVId translated-stage directory not found: {resolved}")
        return resolved
    base = repo_root / "data" / "encoder_decoder" / "t5gemma2"
    candidates = sorted(base.glob("ptbrvarid_translated_stageB*"))
    for candidate in candidates:
        if (candidate / "build_report.json").exists():
            return candidate
    return None


def find_line_numbers(path: Path, pattern: str) -> list[int]:
    regex = re.compile(pattern)
    matches: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            if regex.search(line):
                matches.append(idx)
    return matches


def output_dir_to_config_map(repo_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for cfg_path in sorted((repo_root / "configs").rglob("*.yaml")):
        try:
            cfg = load_yaml(cfg_path)
        except Exception:
            continue
        training_cfg = cfg.get("training") or {}
        output_dir = training_cfg.get("output_dir")
        if not output_dir:
            continue
        local = maybe_repo_local_path(repo_root, str(output_dir))
        if local is not None:
            mapping[local.resolve().as_posix()] = cfg_path.resolve()
    return mapping


def resolve_tokenizer_source(
    repo_root: Path,
    reference_config: Path,
    output_dir_map: dict[str, Path],
) -> tuple[str, list[str]]:
    """Follow the real staged training chain to the tokenizer actually used.

    The staged configs often point model.base_model at an earlier output_dir
    rather than a Hugging Face id directly. For thesis token counts we need the
    tokenizer that the trainer resolves at runtime, so this function walks that
    config/output-dir chain until it reaches a concrete tokenizer source.
    """
    chain: list[str] = []
    seen: set[Path] = set()
    current = reference_config.resolve()

    while True:
        if current in seen:
            raise SystemExit(
                "Tokenizer resolution loop detected while following base_model chain: "
                + " -> ".join(chain)
            )
        seen.add(current)
        cfg = load_yaml(current)
        model_cfg = cfg.get("model") or {}
        raw_base_model = str(model_cfg.get("base_model") or "").strip()
        if not raw_base_model:
            raise SystemExit(f"Config has no model.base_model: {current}")
        chain.append(f"{current.relative_to(repo_root).as_posix()} :: {raw_base_model}")

        local_path = maybe_repo_local_path(repo_root, raw_base_model)
        if local_path is None:
            return raw_base_model, chain

        resolved_local = local_path.resolve()
        if (resolved_local / "tokenizer_config.json").exists():
            return resolved_local.as_posix(), chain

        upstream_cfg = output_dir_map.get(resolved_local.as_posix())
        if upstream_cfg is None:
            # Stage outputs may not exist yet. If the path is local but we cannot trace it
            # back to a config, use it directly and let tokenizer loading fail clearly later.
            return resolved_local.as_posix(), chain
        current = upstream_cfg.resolve()


@dataclass(frozen=True)
class PairSplitStats:
    pairs: int
    equal_pairs: int

    @property
    def translation_rows(self) -> int:
        return 2 * self.pairs

    @property
    def translation_non_equal_rows(self) -> int:
        return 2 * max(0, self.pairs - self.equal_pairs)

    @property
    def classification_rows(self) -> int:
        return (2 * self.pairs) - self.equal_pairs

    @property
    def classification_non_equal_rows(self) -> int:
        return 2 * max(0, self.pairs - self.equal_pairs)


@dataclass
class FilteredFrmtSplitStats:
    translation_rows: int = 0
    translation_non_equal_rows: int = 0
    translation_equal_rows: int = 0
    classification_rows: int = 0
    classification_non_equal_rows: int = 0
    classification_equal_rows: int = 0
    decision_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class VariantTextSource:
    count: int
    iterator_factory: Callable[[], Iterator[str]]
    source_path: str
    unit: str
    notes: list[str] = field(default_factory=list)


@dataclass
class DatasetSplitSummary:
    dataset: str
    split: str
    unit: str
    rows: int
    translation_examples: int | None
    classification_examples: int | None
    pt_br_rows: int | None
    pt_pt_rows: int | None
    source_path: str
    notes: list[str] = field(default_factory=list)


@dataclass
class StageProfileResult:
    name: str
    data_path: str
    split_counts: dict[str, Counter[str]]
    referenced_by_configs: list[str]
    status: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Context:
    repo_root: Path
    output_dir: Path
    reference_config: Path
    tokenizer_source: str
    tokenizer_chain: list[str]
    project_db: Path
    source_db: Path
    ptbr_db: Path
    ptbr_db_notes: list[str]
    ptbr_sampled_csv: Path | None
    ptbr_excluded_domains: set[str]
    gpt_pairs_dir: Path
    gpt_pairs_notes: list[str]
    ptbr_translated_dir: Path | None
    token_batch_size: int
    duckdb_batch_size: int
    token_log_every: int


def connect_project_db(project_db: Path, source_db: Path):
    con = duckdb.connect(project_db.as_posix(), read_only=True)
    con.execute(f"ATTACH '{source_db.as_posix()}' AS src (READ_ONLY)")
    return con


def scalar(con: Any, query: str, params: list[Any] | None = None) -> Any:
    row = con.execute(query, params or []).fetchone()
    return row[0] if row else None


def list_rows(con: Any, query: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
    return con.execute(query, params or []).fetchall()


def compute_project_pair_stats(project_db: Path, source_db: Path) -> tuple[dict[str, PairSplitStats], dict[str, PairSplitStats]]:
    """Recover split counts from the same DuckDB views used by the pipeline.

    OpenSubs/FRMT/Gold split logic is defined in build_project_db.py. This
    function reads the built project DB instead of guessing counts, which lets
    the script verify repository-specific behavior such as OpenSubs being
    train-only in train_data and FRMT dev being the only source of valid rows.
    """
    con = connect_project_db(project_db, source_db)
    try:
        opensubs_pairs = int(
            scalar(
                con,
                """
                SELECT COUNT(*)
                FROM train_data
                WHERE dataset='OpenSubs' AND split='train'
                """,
            )
        )
        opensubs_equal = int(
            scalar(
                con,
                r"""
                SELECT COALESCE(SUM(
                    CASE
                      WHEN regexp_replace(trim(text_pt_br), '\s+', ' ', 'g')
                         = regexp_replace(trim(text_pt_pt), '\s+', ' ', 'g')
                      THEN 1 ELSE 0
                    END
                ), 0)
                FROM train_data
                WHERE dataset='OpenSubs' AND split='train'
                """,
            )
        )

        frmt_stats: dict[str, PairSplitStats] = {}
        for split in ("train", "valid"):
            pairs = int(
                scalar(
                    con,
                    "SELECT COUNT(*) FROM frmt_dev_split_v WHERE split=?",
                    [split],
                )
            )
            equal_pairs = int(
                scalar(
                    con,
                    r"""
                    SELECT COALESCE(SUM(
                        CASE
                          WHEN regexp_replace(trim(text_pt_br), '\s+', ' ', 'g')
                             = regexp_replace(trim(text_pt_pt), '\s+', ' ', 'g')
                          THEN 1 ELSE 0
                        END
                    ), 0)
                    FROM frmt_dev_split_v
                    WHERE split=?
                    """,
                    [split],
                )
            )
            frmt_stats[split] = PairSplitStats(pairs=pairs, equal_pairs=equal_pairs)

        frmt_test_pairs = int(scalar(con, "SELECT COUNT(*) FROM frmt_test_clean_v"))
        frmt_test_equal = int(
            scalar(
                con,
                r"""
                SELECT COALESCE(SUM(
                    CASE
                      WHEN regexp_replace(trim(text_pt_br), '\s+', ' ', 'g')
                         = regexp_replace(trim(text_pt_pt), '\s+', ' ', 'g')
                      THEN 1 ELSE 0
                    END
                ), 0)
                FROM frmt_test_clean_v
                """,
            )
        )
        frmt_stats["test"] = PairSplitStats(pairs=frmt_test_pairs, equal_pairs=frmt_test_equal)
    finally:
        con.close()

    opensubs = {
        "train": PairSplitStats(pairs=opensubs_pairs, equal_pairs=opensubs_equal),
        "valid": PairSplitStats(pairs=0, equal_pairs=0),
        "test": PairSplitStats(pairs=0, equal_pairs=0),
    }
    return opensubs, frmt_stats


def load_frmt_pairs(project_db: Path, source_db: Path) -> dict[str, list[dict[str, str]]]:
    con = connect_project_db(project_db, source_db)
    try:
        out: dict[str, list[dict[str, str]]] = {}
        for split in ("train", "valid"):
            rows = list_rows(
                con,
                "SELECT bucket, text_pt_br, text_pt_pt FROM frmt_dev_split_v WHERE split=? ORDER BY bucket, text_pt_br, text_pt_pt",
                [split],
            )
            out[split] = [
                {
                    "bucket": str(bucket or "n/a"),
                    "pt_br": str(text_pt_br or ""),
                    "pt_pt": str(text_pt_pt or ""),
                }
                for bucket, text_pt_br, text_pt_pt in rows
            ]
        rows = list_rows(
            con,
            "SELECT bucket, text_pt_br, text_pt_pt FROM frmt_test_clean_v ORDER BY bucket, text_pt_br, text_pt_pt",
        )
        out["test"] = [
            {
                "bucket": str(bucket or "n/a"),
                "pt_br": str(text_pt_br or ""),
                "pt_pt": str(text_pt_pt or ""),
            }
            for bucket, text_pt_br, text_pt_pt in rows
        ]
        return out
    finally:
        con.close()


def load_gold_pairs(project_db: Path, source_db: Path) -> list[dict[str, str]]:
    con = connect_project_db(project_db, source_db)
    try:
        rows = list_rows(
            con,
            "SELECT text_pt_br, ref_pt_pt_manual FROM gold_test ORDER BY text_pt_br, ref_pt_pt_manual",
        )
        return [
            {"pt_br": str(text_pt_br or ""), "pt_pt": str(ref_pt_pt_manual or "")}
            for text_pt_br, ref_pt_pt_manual in rows
        ]
    finally:
        con.close()


def load_gpt_pair_stats(gpt_pairs_dir: Path) -> tuple[dict[str, PairSplitStats], dict[str, list[dict[str, Any]]]]:
    stats: dict[str, PairSplitStats] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = gpt_pairs_dir / f"pairs_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing GPT/Wikipedia pair file: {path}")
        rows = read_jsonl(path)
        equal_pairs = sum(1 for row in rows if bool(row.get("is_equal")))
        stats[split] = PairSplitStats(pairs=len(rows), equal_pairs=equal_pairs)
        rows_by_split[split] = rows
    return stats, rows_by_split


def load_ptbr_translated_pair_stats(
    ptbr_translated_dir: Path | None,
) -> tuple[dict[str, PairSplitStats] | None, dict[str, list[dict[str, Any]]] | None]:
    if ptbr_translated_dir is None:
        return None, None
    stats: dict[str, PairSplitStats] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        path = ptbr_translated_dir / f"pairs_{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing PtBrVId translated pair file: {path}")
        rows = read_jsonl(path)
        equal_pairs = sum(1 for row in rows if bool(row.get("is_equal")))
        stats[split] = PairSplitStats(pairs=len(rows), equal_pairs=equal_pairs)
        rows_by_split[split] = rows
    return stats, rows_by_split


def stageb_filter_module(repo_root: Path):
    module_dir = (
        repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
    )
    if module_dir.as_posix() not in sys.path:
        sys.path.insert(0, module_dir.as_posix())
    try:
        return importlib.import_module("frmt_stageb_filter")
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"Could not import frmt_stageb_filter.py from {module_dir}"
        ) from exc


def compute_filtered_frmt_stats(repo_root: Path, frmt_pairs: dict[str, list[dict[str, str]]]) -> dict[str, FilteredFrmtSplitStats]:
    module = stageb_filter_module(repo_root)
    config = module.FrmtFilterConfig()
    out: dict[str, FilteredFrmtSplitStats] = {}
    for split in ("train", "valid"):
        stats = FilteredFrmtSplitStats()
        kept_texts: set[str] = set()
        for line_no, row in enumerate(frmt_pairs[split], start=1):
            bucket = row["bucket"]
            pt_br = row["pt_br"]
            pt_pt = row["pt_pt"]
            translation_rows = [
                {
                    "bucket": bucket,
                    "task": "translate_br2pt",
                    "direction": "translate_br2pt",
                    "input_text": f"<br-pt> {pt_br}",
                    "target_text": pt_pt,
                },
                {
                    "bucket": bucket,
                    "task": "translate_pt2br",
                    "direction": "translate_pt2br",
                    "input_text": f"<pt-br> {pt_pt}",
                    "target_text": pt_br,
                },
            ]
            for translation_row in translation_rows:
                result = module.evaluate_frmt_translation_row(
                    translation_row,
                    source_path=f"frmt_{split}",
                    line_no=line_no,
                    config=config,
                )
                stats.decision_reasons[result["decision_reason"]] += 1
                if result["decision"] != "keep":
                    continue
                stats.translation_rows += 1
                if result["decision_reason"].startswith("equal"):
                    stats.translation_equal_rows += 1
                else:
                    stats.translation_non_equal_rows += 1
                kept_texts.add(result["source_text"])
                kept_texts.add(result["target_text"])

        for row in frmt_pairs[split]:
            pt_br = normalize_space(row["pt_br"])
            pt_pt = normalize_space(row["pt_pt"])
            if pt_br == pt_pt:
                if pt_br in kept_texts:
                    stats.classification_rows += 1
                    stats.classification_equal_rows += 1
                continue
            if pt_br in kept_texts:
                stats.classification_rows += 1
                stats.classification_non_equal_rows += 1
            if pt_pt in kept_texts:
                stats.classification_rows += 1
                stats.classification_non_equal_rows += 1

        out[split] = stats
    return out


def count_ptbr_raw_stats(ptbr_db: Path) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    con = duckdb.connect(ptbr_db.as_posix(), read_only=True)
    try:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        if "ptbrvarid" not in tables:
            raise SystemExit(f"Table 'ptbrvarid' not found in {ptbr_db}")

        columns = {row[1] for row in con.execute("PRAGMA table_info('ptbrvarid')").fetchall()}
        if "dataset" in columns:
            query = """
                SELECT COALESCE(split, 'train') AS split, label, COUNT(*)
                FROM ptbrvarid
                WHERE dataset='PtBrVId'
                GROUP BY 1,2
                ORDER BY 1,2
            """
        else:
            query = """
                SELECT COALESCE(split, 'train') AS split, label, COUNT(*)
                FROM ptbrvarid
                GROUP BY 1,2
                ORDER BY 1,2
            """
        split_variant_counts: dict[str, dict[str, int]] = defaultdict(dict)
        total_by_split: dict[str, int] = defaultdict(int)
        for split, label, count in con.execute(query).fetchall():
            split_norm = normalize_space(split).lower() or "train"
            label_norm = normalize_space(label)
            split_variant_counts[split_norm][label_norm] = int(count)
            total_by_split[split_norm] += int(count)
        return dict(split_variant_counts), dict(total_by_split)
    finally:
        con.close()


def load_ptbr_leftover_count(
    ptbr_db: Path,
    sampled_rows_csv: Path | None,
    excluded_domains: set[str] | None = None,
) -> int | None:
    if sampled_rows_csv is None or not sampled_rows_csv.exists():
        return None
    excluded_domains = excluded_domains or set()
    exclusions: set[tuple[str, str]] = set()
    with sampled_rows_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"label", "source_text"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            return None
        for row in reader:
            label_raw = normalize_space(row.get("label") or "")
            source_text = normalize_space(row.get("source_text") or "")
            if not source_text:
                continue
            if label_raw.lower() in {"pt-br", "pt_br", "ptbr"}:
                label = "pt-br"
            elif label_raw.lower() in {"pt-pt", "pt_pt", "ptpt"}:
                label = "pt-pt"
            else:
                continue
            exclusions.add((label, source_text))

    con = duckdb.connect(ptbr_db.as_posix(), read_only=True)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info('ptbrvarid')").fetchall()}
        has_dataset = "dataset" in columns
        if has_dataset:
            rows = con.execute(
                """
                SELECT COALESCE(split, 'train') AS split, COALESCE(domain, '') AS domain, label, text_pt_br, text_pt_pt
                FROM ptbrvarid
                WHERE dataset='PtBrVId'
                """
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT COALESCE(split, 'train') AS split, COALESCE(domain, '') AS domain, label, text_pt_br, text_pt_pt
                FROM ptbrvarid
                """
            ).fetchall()
        kept = 0
        seen: set[tuple[str, str]] = set()
        for split, domain, label, text_pt_br, text_pt_pt in rows:
            if normalize_space(split).lower() != "train":
                continue
            if normalize_space(domain).casefold() in excluded_domains:
                continue
            label_norm = normalize_space(label).lower().replace("_", "-")
            if label_norm in {"pt-br", "ptbr"}:
                canonical_label = "pt-br"
                source_text = normalize_space(text_pt_br)
            elif label_norm in {"pt-pt", "ptpt"}:
                canonical_label = "pt-pt"
                source_text = normalize_space(text_pt_pt)
            else:
                continue
            if not source_text:
                continue
            key = (canonical_label, source_text)
            if key in exclusions or key in seen:
                continue
            seen.add(key)
            kept += 1
        return kept
    finally:
        con.close()


def build_project_text_iterator(
    project_db: Path,
    source_db: Path,
    query: str,
    params: list[Any] | None = None,
    batch_size: int = 50_000,
) -> Callable[[], Iterator[str]]:
    def iterator() -> Iterator[str]:
        con = connect_project_db(project_db, source_db)
        try:
            cur = con.execute(query, params or [])
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for (text,) in rows:
                    norm = normalize_space(text)
                    if norm:
                        yield norm
        finally:
            con.close()

    return iterator


def build_ptbr_text_iterator(
    ptbr_db: Path,
    *,
    split: str,
    label: str,
    batch_size: int = 50_000,
) -> Callable[[], Iterator[str]]:
    def iterator() -> Iterator[str]:
        con = duckdb.connect(ptbr_db.as_posix(), read_only=True)
        try:
            columns = {row[1] for row in con.execute("PRAGMA table_info('ptbrvarid')").fetchall()}
            has_dataset = "dataset" in columns
            if has_dataset:
                query = """
                    SELECT
                      CASE
                        WHEN lower(label) IN ('pt-br', 'pt_br', 'ptbr') THEN text_pt_br
                        ELSE text_pt_pt
                      END AS text_value
                    FROM ptbrvarid
                    WHERE dataset='PtBrVId' AND lower(split)=? AND lower(label)=?
                """
            else:
                query = """
                    SELECT
                      CASE
                        WHEN lower(label) IN ('pt-br', 'pt_br', 'ptbr') THEN text_pt_br
                        ELSE text_pt_pt
                      END AS text_value
                    FROM ptbrvarid
                    WHERE lower(split)=? AND lower(label)=?
                """
            cur = con.execute(query, [split.lower(), label.lower()])
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for (text,) in rows:
                    norm = normalize_space(text)
                    if norm:
                        yield norm
        finally:
            con.close()

    return iterator


def build_jsonl_text_iterator(
    path: Path,
    field: str,
) -> Callable[[], Iterator[str]]:
    def iterator() -> Iterator[str]:
        for row in iter_jsonl(path):
            norm = normalize_space(row.get(field))
            if norm:
                yield norm

    return iterator


def build_pairs_text_iterator(
    pairs: list[dict[str, Any]],
    field: str,
) -> Callable[[], Iterator[str]]:
    def iterator() -> Iterator[str]:
        for row in pairs:
            norm = normalize_space(row.get(field))
            if norm:
                yield norm

    return iterator


def count_jsonl_rows_by_dataset(path: Path, fallback_dataset: str | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        dataset = normalize_space(row.get("dataset") or fallback_dataset or "UNKNOWN")
        counts[dataset] += 1
    return counts


def simulate_topup_counts(
    train_sequence: list[str],
    existing_counts: Counter[str],
    *,
    min_rows: int,
    seed: int,
) -> tuple[Counter[str], Counter[str]]:
    total_existing = sum(existing_counts.values())
    if total_existing >= min_rows:
        return Counter(existing_counts), Counter()
    sample_size = min_rows - total_existing
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for dataset in train_sequence:
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(dataset)
            continue
        idx = rng.randrange(seen)
        if idx < sample_size:
            reservoir[idx] = dataset
    topup = Counter(reservoir)
    final = Counter(existing_counts)
    final.update(topup)
    return final, topup


def build_dataset_summaries(
    ctx: Context,
    opensubs_stats: dict[str, PairSplitStats],
    frmt_stats: dict[str, PairSplitStats],
    gpt_stats: dict[str, PairSplitStats],
    ptbr_raw_split_counts: dict[str, dict[str, int]],
    ptbr_raw_totals: dict[str, int],
    ptbr_translated_stats: dict[str, PairSplitStats] | None,
    gold_pairs: list[dict[str, str]],
) -> tuple[list[DatasetSplitSummary], dict[tuple[str, str, str], VariantTextSource]]:
    summaries: list[DatasetSplitSummary] = []
    token_sources: dict[tuple[str, str, str], VariantTextSource] = {}

    for split in ("train", "valid", "test"):
        stats = opensubs_stats[split]
        if stats.pairs <= 0:
            continue
        summaries.append(
            DatasetSplitSummary(
                dataset="OpenSubs",
                split=split,
                unit="pair",
                rows=stats.pairs,
                translation_examples=stats.translation_rows,
                classification_examples=stats.classification_rows,
                pt_br_rows=stats.pairs,
                pt_pt_rows=stats.pairs,
                source_path="data/duckdb/subs_project.duckdb::train_data",
                notes=["Counts come from the unified project train_data view after test-pair guarding."],
            )
        )
        token_sources[("OpenSubs", split, "pt-BR")] = VariantTextSource(
            count=stats.pairs,
            iterator_factory=build_project_text_iterator(
                ctx.project_db,
                ctx.source_db,
                "SELECT text_pt_br FROM train_data WHERE dataset='OpenSubs' AND split='train'",
                batch_size=ctx.duckdb_batch_size,
            ),
            source_path="data/duckdb/subs_project.duckdb::train_data.text_pt_br",
            unit="pair_text",
        )
        token_sources[("OpenSubs", split, "pt-PT")] = VariantTextSource(
            count=stats.pairs,
            iterator_factory=build_project_text_iterator(
                ctx.project_db,
                ctx.source_db,
                "SELECT text_pt_pt FROM train_data WHERE dataset='OpenSubs' AND split='train'",
                batch_size=ctx.duckdb_batch_size,
            ),
            source_path="data/duckdb/subs_project.duckdb::train_data.text_pt_pt",
            unit="pair_text",
        )

    for split in ("train", "valid", "test"):
        stats = frmt_stats[split]
        if stats.pairs <= 0:
            continue
        source_path = (
            "data/duckdb/subs_project.duckdb::frmt_dev_split_v"
            if split in {"train", "valid"}
            else "data/duckdb/subs_project.duckdb::frmt_test_clean_v"
        )
        summaries.append(
            DatasetSplitSummary(
                dataset="FRMT",
                split=split,
                unit="pair",
                rows=stats.pairs,
                translation_examples=stats.translation_rows,
                classification_examples=stats.classification_rows,
                pt_br_rows=stats.pairs,
                pt_pt_rows=stats.pairs,
                source_path=source_path,
                notes=["Counts come from the FRMT dev hash split for train/valid and the clean FRMT test split."],
            )
        )
        if split in {"train", "valid"}:
            token_sources[("FRMT", split, "pt-BR")] = VariantTextSource(
                count=stats.pairs,
                iterator_factory=build_project_text_iterator(
                    ctx.project_db,
                    ctx.source_db,
                    "SELECT text_pt_br FROM frmt_dev_split_v WHERE split=?",
                    [split],
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{source_path}.text_pt_br[{split}]",
                unit="pair_text",
            )
            token_sources[("FRMT", split, "pt-PT")] = VariantTextSource(
                count=stats.pairs,
                iterator_factory=build_project_text_iterator(
                    ctx.project_db,
                    ctx.source_db,
                    "SELECT text_pt_pt FROM frmt_dev_split_v WHERE split=?",
                    [split],
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{source_path}.text_pt_pt[{split}]",
                unit="pair_text",
            )
        else:
            token_sources[("FRMT", split, "pt-BR")] = VariantTextSource(
                count=stats.pairs,
                iterator_factory=build_project_text_iterator(
                    ctx.project_db,
                    ctx.source_db,
                    "SELECT text_pt_br FROM frmt_test_clean_v",
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{source_path}.text_pt_br",
                unit="pair_text",
            )
            token_sources[("FRMT", split, "pt-PT")] = VariantTextSource(
                count=stats.pairs,
                iterator_factory=build_project_text_iterator(
                    ctx.project_db,
                    ctx.source_db,
                    "SELECT text_pt_pt FROM frmt_test_clean_v",
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{source_path}.text_pt_pt",
                unit="pair_text",
            )

    gpt_pairs_dir_rel = as_relative(ctx.repo_root, ctx.gpt_pairs_dir)
    for split in ("train", "valid", "test"):
        stats = gpt_stats[split]
        summaries.append(
            DatasetSplitSummary(
                dataset="GPT-Wikipedia",
                split=split,
                unit="pair",
                rows=stats.pairs,
                translation_examples=stats.translation_rows,
                classification_examples=stats.classification_rows,
                pt_br_rows=stats.pairs,
                pt_pt_rows=stats.pairs,
                source_path=f"{gpt_pairs_dir_rel}/pairs_{split}.jsonl",
                notes=["Counts come from the real pair split exported from the merged Wikipedia variant CSV."],
            )
        )
        pairs_path = ctx.gpt_pairs_dir / f"pairs_{split}.jsonl"
        token_sources[("GPT-Wikipedia", split, "pt-BR")] = VariantTextSource(
            count=stats.pairs,
            iterator_factory=build_jsonl_text_iterator(pairs_path, "pt_br"),
            source_path=as_relative(ctx.repo_root, pairs_path),
            unit="pair_text",
        )
        token_sources[("GPT-Wikipedia", split, "pt-PT")] = VariantTextSource(
            count=stats.pairs,
            iterator_factory=build_jsonl_text_iterator(pairs_path, "pt_pt"),
            source_path=as_relative(ctx.repo_root, pairs_path),
            unit="pair_text",
        )

    gold_translation_test = ctx.repo_root / "data" / "encoder_decoder" / "t5gemma2" / "golden_collection" / "translation_test.jsonl"
    gold_classification_test = ctx.repo_root / "data" / "encoder_decoder" / "t5gemma2" / "golden_collection" / "classification_test.jsonl"
    gold_translation_examples = count_jsonl_lines(gold_translation_test)
    gold_classification_examples = count_jsonl_lines(gold_classification_test)
    summaries.append(
        DatasetSplitSummary(
            dataset="Golden Collection",
            split="test",
            unit="pair",
            rows=len(gold_pairs),
            translation_examples=gold_translation_examples,
            classification_examples=gold_classification_examples,
            pt_br_rows=len(gold_pairs),
            pt_pt_rows=len(gold_pairs),
            source_path="data/duckdb/subs_project.duckdb::gold_test",
            notes=["Golden Collection is treated as a test-only evaluation dataset in this repository."],
        )
    )
    token_sources[("Golden Collection", "test", "pt-BR")] = VariantTextSource(
        count=len(gold_pairs),
        iterator_factory=build_pairs_text_iterator(gold_pairs, "pt_br"),
        source_path="data/duckdb/subs_project.duckdb::gold_test.text_pt_br",
        unit="pair_text",
    )
    token_sources[("Golden Collection", "test", "pt-PT")] = VariantTextSource(
        count=len(gold_pairs),
        iterator_factory=build_pairs_text_iterator(gold_pairs, "pt_pt"),
        source_path="data/duckdb/subs_project.duckdb::gold_test.ref_pt_pt_manual",
        unit="pair_text",
    )

    for split in sorted(ptbr_raw_totals):
        counts = ptbr_raw_split_counts.get(split, {})
        pt_br_rows = int(counts.get("pt-BR", 0))
        pt_pt_rows = int(counts.get("pt-PT", 0))
        summaries.append(
            DatasetSplitSummary(
                dataset="PtBrVId-Raw",
                split=split,
                unit="variant_text",
                rows=int(ptbr_raw_totals[split]),
                translation_examples=None,
                classification_examples=None,
                pt_br_rows=pt_br_rows,
                pt_pt_rows=pt_pt_rows,
                source_path=as_relative(ctx.repo_root, ctx.ptbr_db),
                notes=["Counts come from the filtered PtBrVId table used by the augmentation scripts, not from guessed split ratios."],
            )
        )
        if pt_br_rows:
            token_sources[("PtBrVId-Raw", split, "pt-BR")] = VariantTextSource(
                count=pt_br_rows,
                iterator_factory=build_ptbr_text_iterator(
                    ctx.ptbr_db,
                    split=split,
                    label="pt-br",
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{as_relative(ctx.repo_root, ctx.ptbr_db)}::ptbrvarid[label=pt-BR,split={split}]",
                unit="variant_text",
            )
        if pt_pt_rows:
            token_sources[("PtBrVId-Raw", split, "pt-PT")] = VariantTextSource(
                count=pt_pt_rows,
                iterator_factory=build_ptbr_text_iterator(
                    ctx.ptbr_db,
                    split=split,
                    label="pt-pt",
                    batch_size=ctx.duckdb_batch_size,
                ),
                source_path=f"{as_relative(ctx.repo_root, ctx.ptbr_db)}::ptbrvarid[label=pt-PT,split={split}]",
                unit="variant_text",
            )

    if ptbr_translated_stats is not None and ctx.ptbr_translated_dir is not None:
        dir_rel = as_relative(ctx.repo_root, ctx.ptbr_translated_dir)
        for split in ("train", "valid", "test"):
            stats = ptbr_translated_stats[split]
            summaries.append(
                DatasetSplitSummary(
                    dataset="PtBrVId-TranslatedStageB",
                    split=split,
                    unit="pair",
                    rows=stats.pairs,
                    translation_examples=stats.translation_rows,
                    classification_examples=stats.classification_rows,
                    pt_br_rows=stats.pairs,
                    pt_pt_rows=stats.pairs,
                    source_path=f"{dir_rel}/pairs_{split}.jsonl",
                    notes=["Counts come from the exported translated canonical PtBrVId pairs used by the optional Stage B translation mix."],
                )
            )
            pairs_path = ctx.ptbr_translated_dir / f"pairs_{split}.jsonl"
            if stats.pairs > 0:
                token_sources[("PtBrVId-TranslatedStageB", split, "pt-BR")] = VariantTextSource(
                    count=stats.pairs,
                    iterator_factory=build_jsonl_text_iterator(pairs_path, "pt_br"),
                    source_path=as_relative(ctx.repo_root, pairs_path),
                    unit="pair_text",
                )
                token_sources[("PtBrVId-TranslatedStageB", split, "pt-PT")] = VariantTextSource(
                    count=stats.pairs,
                    iterator_factory=build_jsonl_text_iterator(pairs_path, "pt_pt"),
                    source_path=as_relative(ctx.repo_root, pairs_path),
                    unit="pair_text",
                )

    return summaries, token_sources


def topup_for_stage_a(existing_rows: int) -> int:
    return max(0, TRANSLATION_VALID_MIN_ROWS - existing_rows)


def downsampled_opensubs_count_for_ptbrvarid(
    *,
    opensubs_count: int,
    ptbrvarid_count: int,
    target_share: float,
) -> int:
    if target_share <= 0.0 or ptbrvarid_count <= 0:
        return opensubs_count
    allowed = int((ptbrvarid_count * (1.0 - target_share)) / target_share)
    return min(opensubs_count, allowed)


def stage_profile_registry(
    ctx: Context,
    opensubs_stats: dict[str, PairSplitStats],
    frmt_stats: dict[str, PairSplitStats],
    gpt_stats: dict[str, PairSplitStats],
    frmt_filtered: dict[str, FilteredFrmtSplitStats],
    ptbr_translated_stats: dict[str, PairSplitStats] | None,
    ptbr_leftover_count: int | None,
) -> dict[str, StageProfileResult]:
    """Reconstruct Stage A / Stage B composition from builder scripts and data.

    The repo contains multiple similarly named configs whose paths do not always
    match their names, so stage counts are recovered from the actual builder
    semantics plus on-disk JSONL exports when available. This is where the
    thesis-facing stage tables document top-up behavior, FRMT filtering, and the
    optional PtBrVId Stage B translation rows.
    """
    profiles: dict[str, StageProfileResult] = {}

    def add_profile(name: str, data_path: str, split_counts: dict[str, Counter[str]], notes: list[str], status: str = "computed") -> None:
        profiles[name] = StageProfileResult(
            name=name,
            data_path=data_path,
            split_counts=split_counts,
            referenced_by_configs=[],
            status=status,
            notes=notes,
        )

    add_profile(
        "stageA_opensubs_only_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/translation_{split}.jsonl",
        {
            "train": Counter({"OpenSubs": opensubs_stats["train"].translation_rows}),
            "valid": Counter({"OpenSubs": 0}),
        },
        [
            "Counts follow export_encdec_data.py over train_data filtered to dataset=OpenSubs.",
            "OpenSubs contributes no source valid rows in train_data, so the exported valid file is empty.",
            "This clean-source profile intentionally ignores any legacy stageA_opensubs_only directory that may already have been augmented with PtBrVId rows.",
        ],
        status="expected_clean",
    )

    add_profile(
        "stageA_opensubs_only_classification",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/classification_{split}.jsonl",
        {
            "train": Counter({"OpenSubs": opensubs_stats["train"].classification_rows}),
            "valid": Counter({"OpenSubs": 0}),
        },
        [
            "Counts follow export_encdec_data.py over train_data filtered to dataset=OpenSubs.",
            "OpenSubs contributes no source valid rows in train_data, so the exported valid file is empty.",
            "This clean-source profile intentionally ignores any legacy stageA_opensubs_only directory that may already have been augmented with PtBrVId rows.",
        ],
        status="expected_clean",
    )

    stagea_frmt_valid_translation = frmt_stats["valid"].translation_rows
    add_profile(
        "stageA_opensubs_frmt_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/translation_{split}.jsonl",
        {
            "train": Counter(
                {
                    "OpenSubs": opensubs_stats["train"].translation_rows,
                    "FRMT": frmt_stats["train"].translation_rows,
                }
            ),
            "valid": Counter(
                {
                    "FRMT": stagea_frmt_valid_translation,
                    "OpenSubs": topup_for_stage_a(stagea_frmt_valid_translation),
                }
            ),
        },
        [
            "build_translation_stageA_opensubs_frmt.py tops translation valid to 200 rows from the OpenSubs train file.",
            f"Current deterministic top-up: {topup_for_stage_a(stagea_frmt_valid_translation)} OpenSubs rows added to valid.",
            "This profile reports the literal source mix. For thesis interpretation, see the effective-train assumption profile below.",
        ],
    )

    add_profile(
        "stageA_opensubs_frmt_effective_train_assume_opensubs_only",
        "effective-train assumption over data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/{translation_train,translation_valid}.jsonl",
        {
            "train": Counter({"OpenSubs": opensubs_stats["train"].translation_rows}),
            "valid": Counter(
                {
                    "FRMT": stagea_frmt_valid_translation,
                    "OpenSubs": topup_for_stage_a(stagea_frmt_valid_translation),
                }
            ),
        },
        [
            "Thesis assumption: although Stage A was exported with OpenSubs+FRMT, OpenSubs is large enough that FRMT train exposure is treated as negligible.",
            "Train counts intentionally collapse the effective Stage A exposure to OpenSubs-only; valid remains the real built valid split.",
        ],
        status="thesis_assumption",
    )

    add_profile(
        "stageA_opensubs_frmt_classification",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/classification_{split}.jsonl",
        {
            "train": Counter(
                {
                    "OpenSubs": opensubs_stats["train"].classification_rows,
                    "FRMT": frmt_stats["train"].classification_rows,
                }
            ),
            "valid": Counter({"FRMT": frmt_stats["valid"].classification_rows}),
        },
        [
            "build_translation_stageA_opensubs_frmt.py sets CLASSIFICATION_VALID_MIN_ROWS=0, so no train-to-valid top-up is applied here.",
        ],
    )

    add_profile(
        "stageA_opensubs_only_with_cls_mixed",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_with_cls/{split}.jsonl",
        {
            "train": Counter(
                {
                    "OpenSubs": opensubs_stats["train"].translation_rows
                    + opensubs_stats["train"].classification_rows
                }
            ),
            "valid": Counter({"OpenSubs": STAGE_B_MIX_VALID_MIN_ROWS}),
        },
        [
            "run_export_stageA_opensubs_only_with_cls.sh reuses build_stageB_gpt_wiki_frmt_translation_plus_cls.py.",
            "Because the source valid files are empty, the mixed valid split is topped up to 200 rows by sampling from train.",
            "Counts are kept as clean OpenSubs-only expected counts so a legacy contaminated on-disk export cannot inject duplicated PtBrVId rows into this thesis table.",
        ],
        status="expected_clean",
    )

    add_profile(
        "stageA_opensubs_only_label_first_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_noequal/{split}.jsonl",
        {
            "train": Counter({"OpenSubs": opensubs_stats["train"].translation_non_equal_rows}),
            "valid": Counter({"OpenSubs": TRANSLATION_VALID_MIN_ROWS}),
        },
        [
            "build_translation_stageA_opensubs_frmt_label_first.py drops equal translation rows.",
            "Because the OpenSubs valid source file is empty, the label-first valid split is topped up to 200 rows from train.",
            "Counts are kept as clean OpenSubs-only expected counts so a legacy contaminated on-disk export cannot inject duplicated PtBrVId rows into this thesis table.",
        ],
        status="expected_clean",
    )

    add_profile(
        "stageA_opensubs_only_label_first_with_cls_mixed",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_noequal/{split}.jsonl",
        {
            "train": Counter(
                {
                    "OpenSubs": opensubs_stats["train"].translation_non_equal_rows
                    + opensubs_stats["train"].classification_non_equal_rows
                }
            ),
            "valid": Counter({"OpenSubs": STAGE_B_MIX_VALID_MIN_ROWS}),
        },
        [
            "build_stageB_gpt_wiki_frmt_label_first_with_cls.py is also used for the Stage A label-first+cls export.",
            "Equal translation rows are dropped by default and equal classification rows are always dropped.",
            "Because the source valid files are empty, the mixed valid split is topped up to 200 rows from train.",
            "Counts are kept as clean OpenSubs-only expected counts so a legacy contaminated on-disk export cannot inject duplicated PtBrVId rows into this thesis table.",
        ],
        status="expected_clean",
    )

    stagea_frmt_label_valid = frmt_stats["valid"].translation_non_equal_rows
    add_profile(
        "stageA_opensubs_frmt_label_first_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt_label_first_noequal/{split}.jsonl",
        {
            "train": Counter(
                {
                    "OpenSubs": opensubs_stats["train"].translation_non_equal_rows,
                    "FRMT": frmt_stats["train"].translation_non_equal_rows,
                }
            ),
            "valid": Counter(
                {
                    "FRMT": stagea_frmt_label_valid,
                    "OpenSubs": topup_for_stage_a(stagea_frmt_label_valid),
                }
            ),
        },
        [
            "This profile is built from stageA_opensubs_frmt translation files, then rewritten into label-first form with equal rows removed.",
            f"Current deterministic top-up: {topup_for_stage_a(stagea_frmt_label_valid)} OpenSubs rows added to valid.",
            "This profile reports the literal source mix. For thesis interpretation, see the effective-train assumption profile below.",
        ],
    )

    add_profile(
        "stageA_opensubs_frmt_label_first_effective_train_assume_opensubs_only",
        "effective-train assumption over data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt_label_first_noequal/{train,valid}.jsonl",
        {
            "train": Counter({"OpenSubs": opensubs_stats["train"].translation_non_equal_rows}),
            "valid": Counter(
                {
                    "FRMT": stagea_frmt_label_valid,
                    "OpenSubs": topup_for_stage_a(stagea_frmt_label_valid),
                }
            ),
        },
        [
            "Thesis assumption: for Stage A label-first runs built from OpenSubs+FRMT, FRMT train exposure is treated as negligible relative to OpenSubs.",
            "Train counts intentionally collapse the effective Stage A exposure to OpenSubs-only; valid remains the real built valid split.",
        ],
        status="thesis_assumption",
    )

    add_profile(
        "legacy_gpt_frmt_unfiltered_translation",
        "data/encoder_decoder/t5gemma2/gpt_frmt_mix/translation_{split}.jsonl",
        {
            "train": Counter(
                {
                    "FRMT": frmt_stats["train"].translation_rows,
                    "GPT-Wikipedia": gpt_stats["train"].translation_rows,
                }
            ),
            "valid": Counter(
                {
                    "FRMT": frmt_stats["valid"].translation_rows,
                    "GPT-Wikipedia": gpt_stats["valid"].translation_rows,
                }
            ),
        },
        [
            "Legacy non-staged GPT+FRMT translation family: no PtBrVId augmentation and no Stage B FRMT filtering.",
            "When on-disk gpt_frmt_mix files are absent, counts are reconstructed from the strict GPT pair split plus the unfiltered FRMT train/valid split.",
            "run_export_translation_gpt_frmt.sh exports this family from a caller-specified source table, so the script records the ambiguity instead of pretending the source table is fixed.",
        ],
    )

    add_profile(
        "stageB_gpt_wiki_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/translation_{split}.jsonl",
        {
            "train": Counter({"GPT-Wikipedia": gpt_stats["train"].translation_rows}),
            "valid": Counter({"GPT-Wikipedia": gpt_stats["valid"].translation_rows}),
        },
        [
            "Counts come from the strict pair split built by step2_build_all_tasks_from_csv.py and step3_split_tasks.py.",
            "The local workspace falls back to data/encoder_decoder/t5gemma2/{translation,classification,pairs}_*.jsonl when the compare_staged_v2 directory is absent.",
        ],
    )

    add_profile(
        "stageB_gpt_wiki_classification",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/classification_{split}.jsonl",
        {
            "train": Counter({"GPT-Wikipedia": gpt_stats["train"].classification_rows}),
            "valid": Counter({"GPT-Wikipedia": gpt_stats["valid"].classification_rows}),
        },
        [
            "Counts come from the strict pair split built by step2_build_all_tasks_from_csv.py and step3_split_tasks.py.",
        ],
    )

    add_profile(
        "stageB_gpt_wiki_frmt_mix_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_{split}.jsonl",
        {
            "train": Counter(
                {
                    "FRMT": frmt_filtered["train"].translation_rows,
                    "GPT-Wikipedia": gpt_stats["train"].translation_rows,
                }
            ),
            "valid": Counter(
                {
                    "FRMT": frmt_filtered["valid"].translation_rows,
                    "GPT-Wikipedia": gpt_stats["valid"].translation_rows,
                }
            ),
        },
        [
            "build_translation_gpt_wiki_frmt_mix.py applies frmt_stageb_filter.py to FRMT translation rows before mixing with GPT/Wikipedia.",
            "No train-to-valid top-up happens at this mix stage.",
        ],
    )

    add_profile(
        "stageB_gpt_wiki_frmt_mix_classification",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/classification_{split}.jsonl",
        {
            "train": Counter(
                {
                    "FRMT": frmt_filtered["train"].classification_rows,
                    "GPT-Wikipedia": gpt_stats["train"].classification_rows,
                }
            ),
            "valid": Counter(
                {
                    "FRMT": frmt_filtered["valid"].classification_rows,
                    "GPT-Wikipedia": gpt_stats["valid"].classification_rows,
                }
            ),
        },
        [
            "FRMT classification rows are kept only when their text appears in the filtered FRMT translation keep-set.",
            "No train-to-valid top-up happens at this mix stage.",
        ],
    )

    translation_plus_cls_train = Counter(
        {
            "FRMT": frmt_filtered["train"].translation_rows + frmt_filtered["train"].classification_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["train"].translation_rows + gpt_stats["train"].classification_non_equal_rows,
        }
    )
    translation_plus_cls_valid_existing = Counter(
        {
            "FRMT": frmt_filtered["valid"].translation_rows + frmt_filtered["valid"].classification_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["valid"].translation_rows + gpt_stats["valid"].classification_non_equal_rows,
        }
    )
    translation_plus_cls_train_sequence = (
        ["FRMT"] * frmt_filtered["train"].translation_rows
        + ["GPT-Wikipedia"] * gpt_stats["train"].translation_rows
        + ["FRMT"] * frmt_filtered["train"].classification_non_equal_rows
        + ["GPT-Wikipedia"] * gpt_stats["train"].classification_non_equal_rows
    )
    translation_plus_cls_valid_final, translation_plus_cls_topup = simulate_topup_counts(
        translation_plus_cls_train_sequence,
        translation_plus_cls_valid_existing,
        min_rows=STAGE_B_MIX_VALID_MIN_ROWS,
        seed=VALID_TOPUP_SEED,
    )
    add_profile(
        "stageB_gpt_wiki_frmt_translation_plus_cls_mixed",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_translation_plus_cls_noequal/{split}.jsonl",
        {
            "train": translation_plus_cls_train,
            "valid": translation_plus_cls_valid_final,
        },
        [
            "build_stageB_gpt_wiki_frmt_translation_plus_cls.py keeps translation rows as-is, drops equal classification rows, and tops valid up to 200 rows if needed.",
            f"Current deterministic valid top-up from train: {sum(translation_plus_cls_topup.values())} rows.",
        ],
    )

    label_first_translation_train = Counter(
        {
            "FRMT": frmt_filtered["train"].translation_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["train"].translation_non_equal_rows,
        }
    )
    label_first_translation_valid = Counter(
        {
            "FRMT": frmt_filtered["valid"].translation_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["valid"].translation_non_equal_rows,
        }
    )
    add_profile(
        "stageB_gpt_wiki_frmt_label_first_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/{split}.jsonl",
        {
            "train": label_first_translation_train,
            "valid": label_first_translation_valid,
        },
        [
            "build_stageB_gpt_wiki_frmt_label_first.py rewrites only the translation file into label-first form and drops equal rows.",
            "No valid top-up is applied in this translation-only label-first builder.",
        ],
    )

    label_first_with_cls_train = Counter(
        {
            "FRMT": frmt_filtered["train"].translation_non_equal_rows
            + frmt_filtered["train"].classification_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["train"].translation_non_equal_rows
            + gpt_stats["train"].classification_non_equal_rows,
        }
    )
    label_first_with_cls_valid_existing = Counter(
        {
            "FRMT": frmt_filtered["valid"].translation_non_equal_rows
            + frmt_filtered["valid"].classification_non_equal_rows,
            "GPT-Wikipedia": gpt_stats["valid"].translation_non_equal_rows
            + gpt_stats["valid"].classification_non_equal_rows,
        }
    )
    label_first_with_cls_train_sequence = (
        ["FRMT"] * frmt_filtered["train"].translation_non_equal_rows
        + ["GPT-Wikipedia"] * gpt_stats["train"].translation_non_equal_rows
        + ["FRMT"] * frmt_filtered["train"].classification_non_equal_rows
        + ["GPT-Wikipedia"] * gpt_stats["train"].classification_non_equal_rows
    )
    label_first_with_cls_valid_final, label_first_with_cls_topup = simulate_topup_counts(
        label_first_with_cls_train_sequence,
        label_first_with_cls_valid_existing,
        min_rows=STAGE_B_MIX_VALID_MIN_ROWS,
        seed=VALID_TOPUP_SEED,
    )
    add_profile(
        "stageB_gpt_wiki_frmt_label_first_with_cls_mixed",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_with_cls_noequal/{split}.jsonl",
        {
            "train": label_first_with_cls_train,
            "valid": label_first_with_cls_valid_final,
        },
        [
            "This profile describes what build_stageB_gpt_wiki_frmt_label_first_with_cls.py would build: label-first translation rows plus BR/PT-only classification rows, with valid topped up to 200 if needed.",
            f"Current deterministic valid top-up from train: {sum(label_first_with_cls_topup.values())} rows.",
            "Some configs named '*label_first_with_cls*' currently point to the translation-only label-first path instead; see schema_and_dataflow_notes.md.",
        ],
    )

    if ptbr_translated_stats is not None:
        add_profile(
            "stageB_gpt_wiki_frmt_mix_optional_ptbrvarid_translation",
            "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_{split}.jsonl + optional PTBRVID_DIR",
            {
                "train": Counter(
                    {
                        "FRMT": frmt_filtered["train"].translation_rows,
                        "GPT-Wikipedia": gpt_stats["train"].translation_rows,
                        "PtBrVId-TranslatedStageB": ptbr_translated_stats["train"].translation_rows,
                    }
                ),
                "valid": Counter(
                    {
                        "FRMT": frmt_filtered["valid"].translation_rows,
                        "GPT-Wikipedia": gpt_stats["valid"].translation_rows,
                        "PtBrVId-TranslatedStageB": ptbr_translated_stats["valid"].translation_rows,
                    }
                ),
            },
            [
                "run_export_stageB_gpt_wikipedia_plus_frmt.sh supports optional --ptbrvarid-train/--ptbrvarid-valid inputs via PTBRVID_DIR.",
                "Clean PtBrVId Stage B configs should point to dedicated plus_ptbrvarid data directories so these rows are not conflated with the non-PtBrVId Stage B mix.",
            ],
        )

    if ptbr_leftover_count is not None:
        stagea_ptbr_opensubs_rows = downsampled_opensubs_count_for_ptbrvarid(
            opensubs_count=opensubs_stats["train"].classification_rows,
            ptbrvarid_count=ptbr_leftover_count,
            target_share=STAGE_A_PTBRVARID_TARGET_SHARE,
        )
        add_profile(
            "stageA_opensubs_plus_ptbrvarid_classification_source_mix_clean",
            "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid/classification_{split}.jsonl",
            {
                "train": Counter(
                    {
                        "OpenSubs": stagea_ptbr_opensubs_rows,
                        "PtBrVId-Raw": ptbr_leftover_count,
                    }
                ),
                "valid": Counter({"OpenSubs": 0}),
            },
            [
                "build_stageA_opensubs_ptbrvarid_cls.py augments Stage A classification with leftover PtBrVId rows excluded from the translated Stage B sample.",
                "The PtBrVId count is the unique leftover count computed from the filtered PtBrVId DuckDB plus the sampled-rows exclusion CSV; it does not use any previously augmented JSONL file.",
                f"The OpenSubs classification side is downsampled to target a PtBrVId share of {STAGE_A_PTBRVARID_TARGET_SHARE:.2f}.",
                "This expected-clean profile is intentionally not overridden by on-disk JSONL counts, because legacy stageA_opensubs_only files may already contain duplicated/previously added PtBrVId rows.",
            ],
            status="expected_clean",
        )

        add_profile(
            "stageA_opensubs_plus_ptbrvarid_with_cls_mixed",
            "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_with_cls/{split}.jsonl",
            {"train": Counter(), "valid": Counter()},
            [
                "Actual file profile for the clean Setup 1 Stage A export derived from stageA_opensubs_plus_ptbrvarid.",
                "This profile should be populated from the new clean on-disk JSONL after rebuilding; it is separated from legacy stageA_opensubs_only_with_cls to avoid duplicated PtBrVId counts.",
            ],
            status="actual_file_expected_after_build",
        )

        add_profile(
            "stageA_opensubs_plus_ptbrvarid_label_first_with_cls_mixed",
            "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_noequal/{split}.jsonl",
            {"train": Counter(), "valid": Counter()},
            [
                "Actual file profile for the clean Setup 2 Stage A export without equal translation rows.",
                "This profile should be populated from the new clean on-disk JSONL after rebuilding; it is separated from legacy stageA_opensubs_only_label_first_with_cls_noequal to avoid duplicated PtBrVId counts.",
            ],
            status="actual_file_expected_after_build",
        )

        add_profile(
            "stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_mixed",
            "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal/{split}.jsonl",
            {"train": Counter(), "valid": Counter()},
            [
                "Actual file profile for the clean Setup 2 Stage A equal-translation-row export.",
                "This profile should be populated from the new clean on-disk JSONL after rebuilding; it is separated from legacy equal-row attempts to avoid duplicated PtBrVId counts.",
            ],
            status="actual_file_expected_after_build",
        )

    add_profile(
        "stageB_gpt_refresh2_frmt_mix_translation",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/translation_{split}.jsonl",
        {"train": Counter(), "valid": Counter()},
        [
            "No deterministic fallback was computed here because the GPT refresh 2 exported JSONL files are not present in this workspace.",
        ],
        status="missing_source",
    )

    add_profile(
        "stageB_gpt_refresh2_frmt_mix_classification",
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/classification_{split}.jsonl",
        {"train": Counter(), "valid": Counter()},
        [
            "No deterministic fallback was computed here because the GPT refresh 2 exported JSONL files are not present in this workspace.",
        ],
        status="missing_source",
    )

    return profiles


def scan_stage_config_references(repo_root: Path, profiles: dict[str, StageProfileResult]) -> list[str]:
    path_to_profiles = {
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/translation_train.jsonl": ["stageA_opensubs_only_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/translation_valid.jsonl": ["stageA_opensubs_only_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/classification_train.jsonl": ["stageA_opensubs_only_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_clean/classification_valid.jsonl": ["stageA_opensubs_only_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_train.jsonl": ["stageA_opensubs_only_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/translation_valid.jsonl": ["stageA_opensubs_only_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_train.jsonl": ["stageA_opensubs_only_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only/classification_valid.jsonl": ["stageA_opensubs_only_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid/classification_train.jsonl": ["stageA_opensubs_plus_ptbrvarid_classification_source_mix_clean"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid/classification_valid.jsonl": ["stageA_opensubs_plus_ptbrvarid_classification_source_mix_clean"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/translation_train.jsonl": [
            "stageA_opensubs_frmt_translation",
            "stageA_opensubs_frmt_effective_train_assume_opensubs_only",
        ],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/translation_valid.jsonl": [
            "stageA_opensubs_frmt_translation",
            "stageA_opensubs_frmt_effective_train_assume_opensubs_only",
        ],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/classification_train.jsonl": ["stageA_opensubs_frmt_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt/classification_valid.jsonl": ["stageA_opensubs_frmt_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_with_cls/train.jsonl": ["stageA_opensubs_only_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_with_cls/valid.jsonl": ["stageA_opensubs_only_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_noequal/train.jsonl": ["stageA_opensubs_only_label_first_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_noequal/valid.jsonl": ["stageA_opensubs_only_label_first_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_noequal/train.jsonl": ["stageA_opensubs_only_label_first_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_only_label_first_with_cls_noequal/valid.jsonl": ["stageA_opensubs_only_label_first_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_with_cls/train.jsonl": ["stageA_opensubs_plus_ptbrvarid_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_with_cls/valid.jsonl": ["stageA_opensubs_plus_ptbrvarid_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_noequal/train.jsonl": ["stageA_opensubs_plus_ptbrvarid_label_first_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_noequal/valid.jsonl": ["stageA_opensubs_plus_ptbrvarid_label_first_with_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal/train.jsonl": ["stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal/valid.jsonl": ["stageA_opensubs_plus_ptbrvarid_label_first_with_cls_equal_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt_label_first_noequal/train.jsonl": [
            "stageA_opensubs_frmt_label_first_translation",
            "stageA_opensubs_frmt_label_first_effective_train_assume_opensubs_only",
        ],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageA_opensubs_frmt_label_first_noequal/valid.jsonl": [
            "stageA_opensubs_frmt_label_first_translation",
            "stageA_opensubs_frmt_label_first_effective_train_assume_opensubs_only",
        ],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/translation_train.jsonl": ["stageB_gpt_wiki_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/translation_valid.jsonl": ["stageB_gpt_wiki_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/classification_train.jsonl": ["stageB_gpt_wiki_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki/classification_valid.jsonl": ["stageB_gpt_wiki_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_train.jsonl": ["stageB_gpt_wiki_frmt_mix_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/translation_valid.jsonl": ["stageB_gpt_wiki_frmt_mix_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/classification_train.jsonl": ["stageB_gpt_wiki_frmt_mix_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_mix/classification_valid.jsonl": ["stageB_gpt_wiki_frmt_mix_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_translation_plus_cls_noequal/train.jsonl": ["stageB_gpt_wiki_frmt_translation_plus_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_translation_plus_cls_noequal/valid.jsonl": ["stageB_gpt_wiki_frmt_translation_plus_cls_mixed"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/train.jsonl": ["stageB_gpt_wiki_frmt_label_first_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_wiki_frmt_label_first_noequal/valid.jsonl": ["stageB_gpt_wiki_frmt_label_first_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/translation_train.jsonl": ["stageB_gpt_refresh2_frmt_mix_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/translation_valid.jsonl": ["stageB_gpt_refresh2_frmt_mix_translation"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/classification_train.jsonl": ["stageB_gpt_refresh2_frmt_mix_classification"],
        "data/encoder_decoder/t5gemma2/compare_staged_v2/stageB_gpt_refresh2_frmt_mix/classification_valid.jsonl": ["stageB_gpt_refresh2_frmt_mix_classification"],
        "data/encoder_decoder/t5gemma2/gpt_frmt_mix/translation_train.jsonl": ["legacy_gpt_frmt_unfiltered_translation"],
        "data/encoder_decoder/t5gemma2/gpt_frmt_mix/translation_valid.jsonl": ["legacy_gpt_frmt_unfiltered_translation"],
    }

    anomalies: list[str] = []
    config_dirs = [
        repo_root / "configs" / "encoder_decoder" / "t5gemma2" / "comparison_staged",
        repo_root / "configs" / "encoder_decoder" / "t5gemma2_4b" / "comparison_staged",
        repo_root / "configs" / "encoder_decoder" / "t5gemma2",
        repo_root / "configs" / "encoder_decoder" / "t5gemma2_4b",
    ]
    for cfg_dir in config_dirs:
        if not cfg_dir.exists():
            continue
        for cfg_path in sorted(cfg_dir.glob("*.yaml")):
            name_lower = cfg_path.name.lower()
            if "smoke" in name_lower or "stagec" in name_lower:
                continue
            cfg = load_yaml(cfg_path)
            dataset_cfg = cfg.get("dataset") or {}
            for key in ("train_path", "valid_path"):
                rel = str(dataset_cfg.get(key) or "").strip()
                if not rel:
                    continue
                profile_names = path_to_profiles.get(rel, [])
                for profile_name in profile_names:
                    profiles[profile_name].referenced_by_configs.append(as_relative(repo_root, cfg_path))

            train_path = str(dataset_cfg.get("train_path") or "")
            if "with_cls" in name_lower and "label_first_noequal" in train_path and "translation_plus_cls" not in train_path:
                anomalies.append(
                    f"{as_relative(repo_root, cfg_path)} is named '*with_cls*' but points to {train_path}, "
                    "which is the translation-only label-first path."
                )
            if "ptbrvarid" in name_lower and "ptbrvarid" not in train_path.lower():
                anomalies.append(
                    f"{as_relative(repo_root, cfg_path)} is named '*ptbrvarid*' but reuses data path {train_path}; "
                    "the PtBrVId augmentation must be inferred from the build history of that shared directory."
                )
            if "opensubs_frmt" in name_lower and "stagea_opensubs_only/" in train_path.lower():
                anomalies.append(
                    f"{as_relative(repo_root, cfg_path)} is named '*opensubs_frmt*' but currently points to {train_path}."
                )
    for profile in profiles.values():
        profile.referenced_by_configs = sorted(dict.fromkeys(profile.referenced_by_configs))
    return anomalies


def compute_stage_profile_actual_file_override(
    repo_root: Path,
    profile: StageProfileResult,
) -> StageProfileResult:
    if profile.status == "expected_clean":
        return profile
    if "{split}" not in profile.data_path:
        return profile
    if (
        profile.name.startswith("stageA_opensubs_plus_ptbrvarid_")
        and profile.status == "actual_file_expected_after_build"
    ):
        profile_dir = repo_root / profile.data_path.split("/{split}.jsonl", 1)[0]
        report_path = profile_dir / "build_report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {}
            source_values = [
                normalize_space(report.get(key))
                for key in (
                    "translation_train",
                    "translation_valid",
                    "classification_train",
                    "classification_valid",
                )
                if report.get(key)
            ]
            contaminated_sources = [
                value
                for value in source_values
                if "/stageA_opensubs_only/" in value
                or value.endswith("/stageA_opensubs_only/translation_train.jsonl")
                or value.endswith("/stageA_opensubs_only/classification_train.jsonl")
            ]
            if contaminated_sources:
                return StageProfileResult(
                    name=profile.name,
                    data_path=profile.data_path,
                    split_counts=profile.split_counts,
                    referenced_by_configs=profile.referenced_by_configs,
                    status="contaminated_actual_ignored",
                    notes=list(profile.notes)
                    + [
                        "Existing on-disk build_report.json points back to legacy stageA_opensubs_only inputs, so this actual file is ignored to avoid counting duplicated PtBrVId rows.",
                        "Rebuild from stageA_opensubs_plus_ptbrvarid before using this profile as an actual-file count.",
                    ],
                )
    file_counts: dict[str, Counter[str]] = {}
    all_found = True
    any_nonempty = False
    notes = list(profile.notes)
    for split in ("train", "valid"):
        rel = profile.data_path.replace("{split}", split)
        path = repo_root / rel
        if not path.exists():
            all_found = False
            continue
        if path.stat().st_size == 0:
            all_found = False
            notes.append(f"On-disk file is empty, so theoretical counts were kept: {rel}")
            continue
        any_nonempty = True
        fallback_dataset = None
        if "stageB_gpt_wiki" in rel and "compare_staged_v2/stageB_gpt_wiki" in rel:
            fallback_dataset = "GPT-Wikipedia"
        counts = count_jsonl_rows_by_dataset(path, fallback_dataset=fallback_dataset)
        if fallback_dataset is None and "MISSING" in counts:
            all_found = False
            notes.append(
                "On-disk file lacks per-row dataset labels, so theoretical counts were kept for "
                f"{rel} instead of replacing them with ambiguous 'MISSING' counts."
            )
            continue
        file_counts[split] = counts
    if not any_nonempty:
        return profile
    merged = dict(profile.split_counts)
    merged.update(file_counts)
    status = "actual_file" if all_found else "mixed_actual_and_theoretical"
    return StageProfileResult(
        name=profile.name,
        data_path=profile.data_path,
        split_counts=merged,
        referenced_by_configs=profile.referenced_by_configs,
        status=status,
        notes=notes,
    )


def build_stage_rows(
    repo_root: Path,
    profiles: dict[str, StageProfileResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile_name in sorted(profiles):
        profile = compute_stage_profile_actual_file_override(repo_root, profiles[profile_name])
        if profile.status == "missing_source":
            rows.append(
                {
                    "stage_entry": profile.name,
                    "split": "",
                    "dataset": "",
                    "rows": "",
                    "data_path": profile.data_path,
                    "referenced_by_configs": "; ".join(profile.referenced_by_configs),
                    "status": profile.status,
                    "notes": " ".join(profile.notes),
                }
            )
            continue
        for split in ("train", "valid"):
            counts = profile.split_counts.get(split, Counter())
            if not counts:
                rows.append(
                    {
                        "stage_entry": profile.name,
                        "split": split,
                        "dataset": "",
                        "rows": 0,
                        "data_path": profile.data_path,
                        "referenced_by_configs": "; ".join(profile.referenced_by_configs),
                        "status": profile.status,
                        "notes": " ".join(profile.notes),
                    }
                )
                continue
            for dataset, count in sorted(counts.items()):
                rows.append(
                    {
                        "stage_entry": profile.name,
                        "split": split,
                        "dataset": dataset,
                        "rows": int(count),
                        "data_path": profile.data_path,
                        "referenced_by_configs": "; ".join(profile.referenced_by_configs),
                        "status": profile.status,
                        "notes": " ".join(profile.notes),
                    }
                )
    return rows


def compute_token_stats(
    tokenizer: Any,
    label: str,
    source: VariantTextSource,
    batch_size: int,
    log_every: int,
) -> tuple[int, float, int, float]:
    total_tokens = 0
    total_rows = 0
    unique_token_ids: set[int] = set()
    batch: list[str] = []

    def flush_batch() -> None:
        nonlocal total_tokens, total_rows, unique_token_ids, batch
        if not batch:
            return
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = enc["input_ids"]
        total_tokens += sum(len(ids) for ids in input_ids)
        total_rows += len(input_ids)
        for ids in input_ids:
            unique_token_ids.update(ids)
        batch = []
        if log_every > 0 and total_rows % log_every == 0:
            log(
                f"[tokenize] {label}: rows={total_rows} "
                f"tokens={total_tokens} unique_tokenizer_ids={len(unique_token_ids)}"
            )

    for text in source.iterator_factory():
        batch.append(text)
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()

    mean_tokens = (total_tokens / total_rows) if total_rows else 0.0
    unique_tokenizer_ids = len(unique_token_ids)
    tokenizer_type_token_ratio = (unique_tokenizer_ids / total_tokens) if total_tokens else 0.0
    return total_tokens, mean_tokens, unique_tokenizer_ids, tokenizer_type_token_ratio


def build_preprocessing_rows(ptbr_db: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    """Read preprocessing counters from the PtBrVId metrics tables.

    The thesis figures around preprocessing should come from the DuckDB metrics
    emitted by the real PtBrVId filtering pipeline. We therefore read
    ptbrvarid_metrics for per-step removals and ptbrvarid_jt_dropped_examples
    only for schema/provenance notes, not to invent counts independently.
    """
    notes: list[str] = []
    schema_notes: list[str] = []
    con = duckdb.connect(ptbr_db.as_posix(), read_only=True)
    try:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        if "ptbrvarid_metrics" not in tables:
            raise SystemExit(f"Table 'ptbrvarid_metrics' not found in {ptbr_db}")
        metric_columns = [row[1] for row in con.execute("PRAGMA table_info('ptbrvarid_metrics')").fetchall()]
        schema_notes.append(
            f"ptbrvarid_metrics columns in {as_relative(DEFAULT_REPO_ROOT, ptbr_db)}: {', '.join(metric_columns)}"
        )
        rows = con.execute(
            """
            SELECT
              dataset,
              domain,
              split,
              raw,
              after_nonempty,
              after_jusText,
              after_author,
              after_clean,
              after_dedup,
              after_filters,
              after_IQR,
              drop_empty,
              drop_jusText,
              drop_author,
              drop_clean,
              drop_dedup,
              drop_filters,
              drop_IQR
            FROM ptbrvarid_metrics
            ORDER BY dataset, domain, split
            """
        ).fetchall()
        step_rows: list[dict[str, Any]] = []
        justext_rows: list[dict[str, Any]] = []
        for (
            dataset,
            domain,
            split,
            raw,
            after_nonempty,
            after_jusText,
            after_author,
            after_clean,
            after_dedup,
            after_filters,
            after_iqr,
            drop_empty,
            drop_jusText,
            drop_author,
            drop_clean,
            drop_dedup,
            drop_filters,
            drop_iqr,
        ) in rows:
            step_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "empty",
                        "removed_rows": int(drop_empty),
                        "rows_after_step": int(after_nonempty),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "jusText",
                        "removed_rows": int(drop_jusText),
                        "rows_after_step": int(after_jusText),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "author",
                        "removed_rows": int(drop_author),
                        "rows_after_step": int(after_author),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "clean",
                        "removed_rows": int(drop_clean),
                        "rows_after_step": int(after_clean),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "dedup",
                        "removed_rows": int(drop_dedup),
                        "rows_after_step": int(after_dedup),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "filters",
                        "removed_rows": int(drop_filters),
                        "rows_after_step": int(after_filters),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                    {
                        "dataset": dataset,
                        "domain": domain,
                        "split": split,
                        "step_name": "IQR",
                        "removed_rows": int(drop_iqr),
                        "rows_after_step": int(after_iqr),
                        "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                    },
                ]
            )
            justext_rows.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "split": split,
                    "raw_rows": int(raw),
                    "after_jusText_rows": int(after_jusText),
                    "jusText_removed_rows": int(drop_jusText),
                    "final_retained_rows": int(after_iqr),
                    "source_db": as_relative(DEFAULT_REPO_ROOT, ptbr_db),
                }
            )
        if "ptbrvarid_jt_dropped_examples" in tables:
            jt_columns = [row[1] for row in con.execute("PRAGMA table_info('ptbrvarid_jt_dropped_examples')").fetchall()]
            schema_notes.append(
                "ptbrvarid_jt_dropped_examples columns in "
                f"{as_relative(DEFAULT_REPO_ROOT, ptbr_db)}: {', '.join(jt_columns)}"
            )
        return step_rows, justext_rows, notes, schema_notes
    finally:
        con.close()


def build_summary_markdown(
    ctx: Context,
    dataset_rows: list[DatasetSplitSummary],
    token_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    preprocessing_rows: list[dict[str, Any]],
    justext_rows: list[dict[str, Any]],
    config_anomalies: list[str],
) -> str:
    def lookup(dataset: str, split: str) -> DatasetSplitSummary | None:
        for row in dataset_rows:
            if row.dataset == dataset and row.split == split:
                return row
        return None

    def stage_lookup(stage_entry: str, split: str, dataset: str) -> int | None:
        for row in stage_rows:
            if row["stage_entry"] == stage_entry and row["split"] == split and row["dataset"] == dataset:
                return int(row["rows"])
        return None

    def fmt_count(value: int | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:,}"

    total_justext_removed = sum(int(row["jusText_removed_rows"]) for row in justext_rows)
    opensubs = lookup("OpenSubs", "train")
    frmt_train = lookup("FRMT", "train")
    frmt_valid = lookup("FRMT", "valid")
    frmt_test = lookup("FRMT", "test")
    gpt_train = lookup("GPT-Wikipedia", "train")
    gold_test = lookup("Golden Collection", "test")
    ptbr_raw = lookup("PtBrVId-Raw", "train")
    ptbr_translated_train = lookup("PtBrVId-TranslatedStageB", "train")

    lines = [
        "# Thesis Dataset Statistics Summary",
        "",
        "## Resolved Inputs",
        f"- Repository root: `{ctx.repo_root}`",
        f"- Reference config: `{as_relative(ctx.repo_root, ctx.reference_config)}`",
        f"- Tokenizer source: `{ctx.tokenizer_source}`",
        f"- Project DB: `{as_relative(ctx.repo_root, ctx.project_db)}`",
        f"- Source DB: `{as_relative(ctx.repo_root, ctx.source_db)}`",
        f"- PtBrVId DB used for preprocessing/raw stats: `{as_relative(ctx.repo_root, ctx.ptbr_db)}`",
        "",
        "## Key Findings",
        "- OpenSubs contributes train rows only in the project training view. The source valid count for dataset=OpenSubs is zero.",
        "- Validation top-up to 200 rows is implemented in data-building scripts, not in the trainer. The training scripts consume already-built JSONL files.",
        "- Golden Collection is wired as a test-only evaluation dataset in the current repository layout.",
        "- PtBrVId is loaded directly from a DuckDB table in the Stage A augmentation code, not from the unified subs_project training view.",
        "- The script distinguishes the older unfiltered GPT+FRMT family from the staged GPT-Wikipedia+FRMT family that applies FRMT filtering.",
        "- For thesis-facing Stage A summaries, OpenSubs+FRMT training can also be interpreted with an effective-exposure assumption where train FRMT is ignored relative to OpenSubs.",
        "",
        "## Dataset Counts",
    ]
    if opensubs is not None:
        lines.append(
            f"- OpenSubs train source pairs: {opensubs.rows:,} "
            f"(translation examples: {opensubs.translation_examples:,}; classification examples: {opensubs.classification_examples:,})."
        )
    if frmt_train is not None and frmt_valid is not None and frmt_test is not None:
        lines.append(
            f"- FRMT clean pairs: train={frmt_train.rows:,}, valid={frmt_valid.rows:,}, test={frmt_test.rows:,}."
        )
    if gpt_train is not None:
        gpt_valid = lookup("GPT-Wikipedia", "valid")
        gpt_test = lookup("GPT-Wikipedia", "test")
        lines.append(
            f"- GPT-Wikipedia pairs: train={gpt_train.rows:,}, valid={gpt_valid.rows if gpt_valid else 0:,}, "
            f"test={gpt_test.rows if gpt_test else 0:,}."
        )
    if gold_test is not None:
        lines.append(
            f"- Golden Collection test pairs: {gold_test.rows:,} "
            f"(translation examples: {gold_test.translation_examples:,}; classification examples: {gold_test.classification_examples:,})."
        )
    if ptbr_raw is not None:
        lines.append(
            f"- PtBrVId filtered raw texts: {ptbr_raw.rows:,} train rows "
            f"(pt-BR={ptbr_raw.pt_br_rows:,}, pt-PT={ptbr_raw.pt_pt_rows:,})."
        )
    if ptbr_translated_train is not None:
        ptbr_translated_test = lookup("PtBrVId-TranslatedStageB", "test")
        lines.append(
            f"- PtBrVId translated canonical pairs: train={ptbr_translated_train.rows:,}, "
            f"valid={lookup('PtBrVId-TranslatedStageB', 'valid').rows if lookup('PtBrVId-TranslatedStageB', 'valid') else 0:,}, "
            f"test={ptbr_translated_test.rows if ptbr_translated_test else 0:,}."
        )

    lines.extend(
        [
            "",
            "## Stage Highlights",
            f"- Stage A OpenSubs-only translation: train OpenSubs={fmt_count(stage_lookup('stageA_opensubs_only_translation', 'train', 'OpenSubs'))}, valid OpenSubs={fmt_count(stage_lookup('stageA_opensubs_only_translation', 'valid', 'OpenSubs'))}.",
            f"- Stage A clean OpenSubs+PtBrVId classification source mix: train OpenSubs={fmt_count(stage_lookup('stageA_opensubs_plus_ptbrvarid_classification_source_mix_clean', 'train', 'OpenSubs'))}, "
            f"train PtBrVId={fmt_count(stage_lookup('stageA_opensubs_plus_ptbrvarid_classification_source_mix_clean', 'train', 'PtBrVId-Raw'))}.",
            f"- Stage A OpenSubs+FRMT thesis assumption: effective train OpenSubs={fmt_count(stage_lookup('stageA_opensubs_frmt_effective_train_assume_opensubs_only', 'train', 'OpenSubs'))}, "
            f"effective train FRMT={fmt_count(stage_lookup('stageA_opensubs_frmt_effective_train_assume_opensubs_only', 'train', 'FRMT') or 0)}, "
            f"valid FRMT={fmt_count(stage_lookup('stageA_opensubs_frmt_effective_train_assume_opensubs_only', 'valid', 'FRMT'))}, "
            f"valid OpenSubs top-up={fmt_count(stage_lookup('stageA_opensubs_frmt_effective_train_assume_opensubs_only', 'valid', 'OpenSubs'))}.",
            f"- Legacy GPT+FRMT unfiltered translation: train GPT={fmt_count(stage_lookup('legacy_gpt_frmt_unfiltered_translation', 'train', 'GPT-Wikipedia'))}, "
            f"train FRMT={fmt_count(stage_lookup('legacy_gpt_frmt_unfiltered_translation', 'train', 'FRMT'))}, "
            f"valid GPT={fmt_count(stage_lookup('legacy_gpt_frmt_unfiltered_translation', 'valid', 'GPT-Wikipedia'))}, "
            f"valid FRMT={fmt_count(stage_lookup('legacy_gpt_frmt_unfiltered_translation', 'valid', 'FRMT'))}.",
            f"- Stage B GPT+FRMT filtered translation mix: train GPT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_mix_translation', 'train', 'GPT-Wikipedia'))}, "
            f"train FRMT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_mix_translation', 'train', 'FRMT'))}, "
            f"valid GPT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_mix_translation', 'valid', 'GPT-Wikipedia'))}, "
            f"valid FRMT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_mix_translation', 'valid', 'FRMT'))}.",
            f"- Stage B label-first translation-only: train GPT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_label_first_translation', 'train', 'GPT-Wikipedia'))}, "
            f"train FRMT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_label_first_translation', 'train', 'FRMT'))}, "
            f"valid GPT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_label_first_translation', 'valid', 'GPT-Wikipedia'))}, "
            f"valid FRMT={fmt_count(stage_lookup('stageB_gpt_wiki_frmt_label_first_translation', 'valid', 'FRMT'))}.",
            "",
            "## Preprocessing",
            f"- Total jusText removals recorded in ptbrvarid_metrics: {total_justext_removed:,}.",
            f"- Preprocessing summary rows written: {len(preprocessing_rows):,}; jusText domain rows written: {len(justext_rows):,}.",
        ]
    )

    if token_rows:
        lines.extend(["", "## Tokenization", f"- Token statistic rows written: {len(token_rows):,}."])

    if config_anomalies:
        lines.extend(["", "## Config Anomalies"])
        for anomaly in config_anomalies:
            lines.append(f"- {anomaly}")

    lines.append("")
    return "\n".join(lines)


def build_schema_notes_markdown(
    ctx: Context,
    config_anomalies: list[str],
    ptbr_schema_notes: list[str],
) -> str:
    stagea_mix_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_translation_stageA_opensubs_frmt.py"
    )
    stagea_label_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_translation_stageA_opensubs_frmt_label_first.py"
    )
    stageb_mix_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_translation_gpt_wiki_frmt_mix.py"
    )
    stageb_with_cls_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_stageB_gpt_wiki_frmt_translation_plus_cls.py"
    )
    stageb_label_with_cls_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_stageB_gpt_wiki_frmt_label_first_with_cls.py"
    )
    ptbr_stagea_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "build_stageA_opensubs_ptbrvarid_cls.py"
    )
    project_db_script = ctx.repo_root / "scripts" / "project" / "build_project_db.py"
    run_stageb_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "run_export_stageB_gpt_wikipedia_plus_frmt.sh"
    )
    legacy_gpt_frmt_script = (
        ctx.repo_root
        / "scripts"
        / "encoder_decoder"
        / "single_task_models"
        / "t5gemma2_4b"
        / "run_export_translation_gpt_frmt.sh"
    )
    slurm_train_script = ctx.repo_root / "scripts" / "slurm" / "train_t5gemma2_4b_translation_compare_staged.sbatch"
    stagea_mix_translation_lines = find_line_numbers(
        stagea_mix_script,
        r"TRANSLATION_VALID_MIN_ROWS\s*=\s*200",
    )
    stagea_mix_classification_lines = find_line_numbers(
        stagea_mix_script,
        r"CLASSIFICATION_VALID_MIN_ROWS\s*=\s*0",
    )
    stagea_label_valid_lines = find_line_numbers(
        stagea_label_script,
        r"TRANSLATION_VALID_MIN_ROWS\s*=\s*200",
    )
    stageb_with_cls_valid_lines = find_line_numbers(
        stageb_with_cls_script,
        r"VALID_MIN_ROWS\s*=\s*200",
    )
    stageb_label_with_cls_valid_lines = find_line_numbers(
        stageb_label_with_cls_script,
        r"VALID_MIN_ROWS\s*=\s*200",
    )

    lines = [
        "# Schema And Dataflow Notes",
        "",
        "## Resolved Files",
        f"- Reference config: `{as_relative(ctx.repo_root, ctx.reference_config)}`",
        f"- Tokenizer source: `{ctx.tokenizer_source}`",
        f"- Project DB: `{as_relative(ctx.repo_root, ctx.project_db)}`",
        f"- Source DB: `{as_relative(ctx.repo_root, ctx.source_db)}`",
        f"- PtBrVId DB: `{as_relative(ctx.repo_root, ctx.ptbr_db)}`",
    ]
    if ctx.ptbr_sampled_csv is not None:
        lines.append(f"- PtBrVId sampled-rows CSV: `{as_relative(ctx.repo_root, ctx.ptbr_sampled_csv)}`")
    if ctx.ptbr_excluded_domains:
        lines.append(f"- PtBrVId Stage A excluded domains: `{', '.join(sorted(ctx.ptbr_excluded_domains))}`")
    if ctx.ptbr_translated_dir is not None:
        lines.append(f"- PtBrVId translated-stage dir: `{as_relative(ctx.repo_root, ctx.ptbr_translated_dir)}`")
    lines.extend(["", "## Tokenizer Resolution", "The tokenizer is resolved from the reference config by following model.base_model until a non-local model id or a directory with tokenizer files is found."])
    for item in ctx.tokenizer_chain:
        lines.append(f"- {item}")
    lines.extend(["", "## Split Logic"])
    lines.append(
        f"- OpenSubs/FRMT/Gold are assembled in `{as_relative(ctx.repo_root, project_db_script)}`. "
        "OpenSubs enters the unified train_data view as train-only, FRMT dev is split into train/valid, and Gold is test-only."
    )
    lines.append(
        f"- Stage A OpenSubs-only exports come from `run_export_stageA_opensubs_frmt.sh` with `--dataset-include OpenSubs`; "
        "that means the exported valid file is empty because train_data has no OpenSubs valid rows. "
        "The thesis statistics use `stageA_opensubs_only_clean` as the clean source name so old `stageA_opensubs_only` rebuilds cannot leak PtBrVId rows into OpenSubs-only counts."
    )
    lines.append(
        f"- Stage A OpenSubs+FRMT translation valid top-up is implemented in `{as_relative(ctx.repo_root, stagea_mix_script)}` "
        f"(VALID_MIN_ROWS lines: {stagea_mix_translation_lines}; "
        f"classification top-up lines: {stagea_mix_classification_lines})."
    )
    lines.append(
        f"- Stage A label-first valid top-up is implemented in `{as_relative(ctx.repo_root, stagea_label_script)}` "
        f"(VALID_MIN_ROWS lines: {stagea_label_valid_lines})."
    )
    lines.append(
        "- For thesis-facing stage summaries, the script also emits an effective-train assumption profile for Stage A OpenSubs+FRMT runs: "
        "train exposure is treated as OpenSubs-only because the OpenSubs volume dwarfs FRMT, while valid remains the real built split."
    )
    lines.append(
        f"- Stage B GPT+FRMT uses `{as_relative(ctx.repo_root, stageb_mix_script)}` for the filtered mix, "
        f"`{as_relative(ctx.repo_root, stageb_with_cls_script)}` for translation+classification mixing "
        f"(VALID_MIN_ROWS lines: {stageb_with_cls_valid_lines}), and "
        f"`{as_relative(ctx.repo_root, stageb_label_with_cls_script)}` for the label-first+cls builder "
        f"(VALID_MIN_ROWS lines: {stageb_label_with_cls_valid_lines})."
    )
    lines.append(
        f"- The older non-staged GPT+FRMT translation family is represented separately via `{as_relative(ctx.repo_root, legacy_gpt_frmt_script)}`. "
        "That exporter takes a caller-specified source table and does not apply the Stage B FRMT filter or any PtBrVId augmentation."
    )
    lines.append(
        f"- The trainer consumes already-built JSONL files via `{as_relative(ctx.repo_root, slurm_train_script)}` and "
        "does not move rows between train and valid during training itself."
    )
    lines.extend(["", "## PtBrVId Loading"])
    lines.append(
        f"- Stage A PtBrVId augmentation is loaded directly from DuckDB in `{as_relative(ctx.repo_root, ptbr_stagea_script)}`. "
        f"The helper resolves `subs_ptbr_filtered.duckdb` first and falls back to `subs_filtered_final.duckdb` when needed."
    )
    lines.append(
        "- Stage A PtBrVId thesis counts use the unique leftover count from DuckDB plus the sampled-rows exclusion CSV. "
        "They also apply the Stage A domain exclusion set and do not use the legacy `stageA_opensubs_only/classification_train.jsonl` file, because that file may already contain previously added PtBrVId rows."
    )
    for note in ctx.ptbr_db_notes:
        lines.append(f"- {note}")
    lines.extend(["", "## Golden Collection"])
    lines.append(
        f"- Golden Collection is treated as test-only. Training Slurm wrappers explicitly reject configs that reference Golden Collection for training."
    )
    lines.extend(["", "## Stage B Optional PtBrVId"])
    lines.append(
        f"- `{as_relative(ctx.repo_root, run_stageb_script)}` supports optional PtBrVId translation rows via `PTBRVID_DIR`, "
        "and the current clean PtBrVId configs point to dedicated `plus_ptbrvarid` Stage B data directories."
    )
    lines.extend(["", "## PtBrVId Schema"])
    for note in ptbr_schema_notes:
        lines.append(f"- {note}")
    if config_anomalies:
        lines.extend(["", "## Config/Path Anomalies"])
        for anomaly in config_anomalies:
            lines.append(f"- {anomaly}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute thesis-ready dataset statistics from the real repository training pipeline, "
            "including split counts, token counts, Stage A/Stage B composition, and PtBrVId preprocessing metrics."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--reference-config", type=Path, default=None)
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--project-db", type=Path, default=None)
    parser.add_argument("--source-db", type=Path, default=None)
    parser.add_argument("--ptbrvarid-db", type=Path, default=None)
    parser.add_argument("--ptbrvarid-sampled-csv", type=Path, default=None)
    parser.add_argument(
        "--ptbrvarid-exclude-domains",
        default=DEFAULT_STAGE_A_PTBRVARID_EXCLUDED_DOMAINS,
        help=(
            "Comma-separated PtBrVId domains excluded by the Stage A augmentation. "
            "Default matches run_export_stageA_opensubs_plus_ptbrvarid_cls.sh usage."
        ),
    )
    parser.add_argument("--ptbrvarid-translated-dir", type=Path, default=None)
    parser.add_argument("--token-batch-size", type=int, default=2048)
    parser.add_argument("--duckdb-batch-size", type=int, default=50_000)
    parser.add_argument("--token-log-every", type=int, default=100_000)
    parser.add_argument("--skip-token-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = discover_reference_config(repo_root, args.reference_config)
    project_db = discover_project_db(repo_root, args.project_db)
    source_db = discover_source_db(repo_root, args.source_db)
    ptbr_db_notes: list[str] = []
    ptbr_db = discover_ptbr_db(repo_root, args.ptbrvarid_db, ptbr_db_notes)
    ptbr_sampled_csv = discover_ptbr_sampled_csv(repo_root, args.ptbrvarid_sampled_csv)
    ptbr_excluded_domains = parse_domain_list(args.ptbrvarid_exclude_domains)
    gpt_pairs_notes: list[str] = []
    gpt_pairs_dir = discover_gpt_pairs_dir(repo_root, gpt_pairs_notes)
    ptbr_translated_dir = discover_ptbr_translated_dir(repo_root, args.ptbrvarid_translated_dir)

    output_dir_map = output_dir_to_config_map(repo_root)
    if args.tokenizer_source:
        tokenizer_source = args.tokenizer_source
        tokenizer_chain = [f"manual override :: {tokenizer_source}"]
    else:
        tokenizer_source, tokenizer_chain = resolve_tokenizer_source(repo_root, reference_config, output_dir_map)

    ctx = Context(
        repo_root=repo_root,
        output_dir=output_dir,
        reference_config=reference_config,
        tokenizer_source=tokenizer_source,
        tokenizer_chain=tokenizer_chain,
        project_db=project_db,
        source_db=source_db,
        ptbr_db=ptbr_db,
        ptbr_db_notes=ptbr_db_notes,
        ptbr_sampled_csv=ptbr_sampled_csv,
        ptbr_excluded_domains=ptbr_excluded_domains,
        gpt_pairs_dir=gpt_pairs_dir,
        gpt_pairs_notes=gpt_pairs_notes,
        ptbr_translated_dir=ptbr_translated_dir,
        token_batch_size=int(args.token_batch_size),
        duckdb_batch_size=int(args.duckdb_batch_size),
        token_log_every=int(args.token_log_every),
    )

    log("[stats] loading project pair statistics...")
    opensubs_stats, frmt_stats = compute_project_pair_stats(project_db, source_db)
    log("[stats] loading FRMT and Gold canonical pairs...")
    frmt_pairs = load_frmt_pairs(project_db, source_db)
    gold_pairs = load_gold_pairs(project_db, source_db)
    log("[stats] loading GPT/Wikipedia pair splits...")
    gpt_stats, _gpt_pairs = load_gpt_pair_stats(gpt_pairs_dir)
    log("[stats] loading PtBrVId translated pair splits...")
    ptbr_translated_stats, _ptbr_translated_pairs = load_ptbr_translated_pair_stats(ptbr_translated_dir)
    log("[stats] computing filtered FRMT Stage B counts...")
    frmt_filtered = compute_filtered_frmt_stats(repo_root, frmt_pairs)
    log("[stats] loading PtBrVId raw counts...")
    ptbr_raw_split_counts, ptbr_raw_totals = count_ptbr_raw_stats(ptbr_db)
    ptbr_leftover_count = load_ptbr_leftover_count(
        ptbr_db,
        ptbr_sampled_csv,
        excluded_domains=ptbr_excluded_domains,
    )

    dataset_summaries, token_sources = build_dataset_summaries(
        ctx,
        opensubs_stats,
        frmt_stats,
        gpt_stats,
        ptbr_raw_split_counts,
        ptbr_raw_totals,
        ptbr_translated_stats,
        gold_pairs,
    )

    log("[stats] building stage profiles...")
    stage_profiles = stage_profile_registry(
        ctx,
        opensubs_stats,
        frmt_stats,
        gpt_stats,
        frmt_filtered,
        ptbr_translated_stats,
        ptbr_leftover_count,
    )
    config_anomalies = scan_stage_config_references(repo_root, stage_profiles)
    stage_rows = build_stage_rows(repo_root, stage_profiles)

    log("[stats] reading preprocessing metrics...")
    preprocessing_rows, justext_rows, _pre_notes, ptbr_schema_notes = build_preprocessing_rows(ptbr_db)

    log("[stats] writing dataset_counts_per_split.csv ...")
    dataset_count_rows = [
        {
            "dataset": row.dataset,
            "split": row.split,
            "unit": row.unit,
            "rows": row.rows,
            "translation_examples": row.translation_examples if row.translation_examples is not None else "",
            "classification_examples": row.classification_examples if row.classification_examples is not None else "",
            "pt_br_rows": row.pt_br_rows if row.pt_br_rows is not None else "",
            "pt_pt_rows": row.pt_pt_rows if row.pt_pt_rows is not None else "",
            "source_path": row.source_path,
            "notes": " ".join(row.notes),
        }
        for row in dataset_summaries
    ]
    write_csv(
        output_dir / "dataset_counts_per_split.csv",
        dataset_count_rows,
        [
            "dataset",
            "split",
            "unit",
            "rows",
            "translation_examples",
            "classification_examples",
            "pt_br_rows",
            "pt_pt_rows",
            "source_path",
            "notes",
        ],
    )

    token_rows: list[dict[str, Any]] = []
    if args.skip_token_counts:
        log("[stats] skipping token counts (--skip-token-counts).")
    else:
        log("[stats] loading tokenizer...")
        tokenizer_kwargs: dict[str, Any] = {"use_fast": True}
        if args.tokenizer_local_files_only:
            tokenizer_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        for key in sorted(token_sources):
            dataset, split, variant = key
            source = token_sources[key]
            label = f"{dataset}/{split}/{variant}"
            log(f"[stats] tokenizing {label} from {source.source_path} ...")
            total_tokens, mean_tokens, unique_tokenizer_ids, tokenizer_type_token_ratio = compute_token_stats(
                tokenizer,
                label,
                source,
                batch_size=ctx.token_batch_size,
                log_every=ctx.token_log_every,
            )
            token_rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "variant": variant,
                    "rows": source.count,
                    "total_tokens": total_tokens,
                    "mean_tokens": f"{mean_tokens:.6f}",
                    "unique_tokenizer_ids": unique_tokenizer_ids,
                    "tokenizer_type_token_ratio": f"{tokenizer_type_token_ratio:.8f}",
                    "tokenizer_source": tokenizer_source,
                    "source_path": source.source_path,
                    "unit": source.unit,
                    "notes": " ".join(source.notes),
                }
            )
        write_csv(
            output_dir / "token_counts_per_split_variant.csv",
            token_rows,
            [
                "dataset",
                "split",
                "variant",
                "rows",
                "total_tokens",
                "mean_tokens",
                "unique_tokenizer_ids",
                "tokenizer_type_token_ratio",
                "tokenizer_source",
                "source_path",
                "unit",
                "notes",
            ],
        )

    log("[stats] writing stage_dataset_counts.csv ...")
    write_csv(
        output_dir / "stage_dataset_counts.csv",
        stage_rows,
        [
            "stage_entry",
            "split",
            "dataset",
            "rows",
            "data_path",
            "referenced_by_configs",
            "status",
            "notes",
        ],
    )

    log("[stats] writing preprocessing_step_removals.csv ...")
    write_csv(
        output_dir / "preprocessing_step_removals.csv",
        preprocessing_rows,
        [
            "dataset",
            "domain",
            "split",
            "step_name",
            "removed_rows",
            "rows_after_step",
            "source_db",
        ],
    )

    log("[stats] writing justext_removals_by_domain.csv ...")
    write_csv(
        output_dir / "justext_removals_by_domain.csv",
        justext_rows,
        [
            "dataset",
            "domain",
            "split",
            "raw_rows",
            "after_jusText_rows",
            "jusText_removed_rows",
            "final_retained_rows",
            "source_db",
        ],
    )

    summary_md = build_summary_markdown(
        ctx,
        dataset_summaries,
        token_rows,
        stage_rows,
        preprocessing_rows,
        justext_rows,
        config_anomalies,
    )
    (output_dir / "thesis_dataset_statistics_summary.md").write_text(summary_md, encoding="utf-8")

    schema_md = build_schema_notes_markdown(
        ctx,
        config_anomalies,
        ptbr_schema_notes,
    )
    (output_dir / "schema_and_dataflow_notes.md").write_text(schema_md, encoding="utf-8")

    log("[stats] done")
    log(f"[stats] summary -> {output_dir / 'thesis_dataset_statistics_summary.md'}")
    log(f"[stats] dataset counts -> {output_dir / 'dataset_counts_per_split.csv'}")
    if not args.skip_token_counts:
        log(f"[stats] token counts -> {output_dir / 'token_counts_per_split_variant.csv'}")
    log(f"[stats] stage counts -> {output_dir / 'stage_dataset_counts.csv'}")
    log(f"[stats] preprocessing -> {output_dir / 'preprocessing_step_removals.csv'}")
    log(f"[stats] jusText -> {output_dir / 'justext_removals_by_domain.csv'}")
    log(f"[stats] dataflow notes -> {output_dir / 'schema_and_dataflow_notes.md'}")


if __name__ == "__main__":
    main()
