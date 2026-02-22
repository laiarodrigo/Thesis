#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, Optional

import duckdb


SYSTEM_TRANSLATION = (
    "És um assistente especialista em português europeu e português do Brasil. "
    "A tua tarefa é converter frases entre as duas variantes, mantendo o significado, "
    "o registo e um estilo natural."
)

SYSTEM_CLASSIFICATION = (
    "És um linguista especializado em português europeu e português do Brasil. "
    "A tua tarefa é identificar a variante de português de uma frase."
)

USER_TRANSL_BR2PT = (
    "Converte o seguinte texto de português do Brasil para português europeu, "
    "mantendo o sentido e soando natural em português europeu.\n\n"
    "Texto: {source}"
)

USER_TRANSL_PT2BR = (
    "Converte o seguinte texto de português europeu para português do Brasil, "
    "mantendo o sentido e soando natural em português do Brasil.\n\n"
    "Texto: {source}"
)

USER_CLASSIFICATION = (
    "Qual é a variante de português desta frase? "
    "Responde apenas com \"Português do Brasil\" ou \"Português Europeu\".\n\n"
    "Frase: {source}"
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_project_db = repo_root / "data" / "duckdb" / "subs_project.duckdb"
    default_source_db = repo_root / "data" / "duckdb" / "subs.duckdb"
    default_out_dir = repo_root / "data" / "decoder_only" / "qwen3"
    default_gpt_dir = repo_root / "data" / "encoder_decoder" / "t5gemma2"

    parser = argparse.ArgumentParser(
        description=(
            "Export train/valid examples from DuckDB views into Axolotl chat JSONL "
            "using the same task construction used in scripts/train_qwen_lora.py."
        )
    )
    parser.add_argument("--project-db", type=Path, default=default_project_db)
    parser.add_argument("--source-db", type=Path, default=default_source_db)
    parser.add_argument("--train-view", default="train_data")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-view", default="train_data")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--pt2br-prob",
        type=float,
        default=0.7,
        help="Probability of choosing PT->BR when both sides are available.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["multitask", "translation", "classification"],
        default="multitask",
        help="Choose which tasks to export into chat JSONL.",
    )
    parser.add_argument(
        "--gpt-only",
        action="store_true",
        default=False,
        help=(
            "Use GPT task JSONL files from Step 2/3 instead of DuckDB views. "
            "This bypasses train_data/test_data schema."
        ),
    )
    parser.add_argument("--gpt-translation-train-file", type=Path, default=default_gpt_dir / "translation_train.jsonl")
    parser.add_argument("--gpt-translation-valid-file", type=Path, default=default_gpt_dir / "translation_valid.jsonl")
    parser.add_argument("--gpt-classification-train-file", type=Path, default=default_gpt_dir / "classification_train.jsonl")
    parser.add_argument("--gpt-classification-valid-file", type=Path, default=default_gpt_dir / "classification_valid.jsonl")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--train-file", default="train.jsonl")
    parser.add_argument("--valid-file", default="valid.jsonl")
    return parser.parse_args()


def as_clean_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def connect_project(project_db: Path, source_db: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(project_db.as_posix(), read_only=read_only)
    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        con.execute(f"ATTACH '{source_db.as_posix()}' AS src")
    return con


def stream_from_view(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    split: Optional[str] = None,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    offset = 0

    base_query = f"""
        SELECT
          dataset,
          split,
          source,
          bucket,
          theme,
          domain,
          label,
          text_pt_br,
          text_pt_pt,
          ref_pt_pt_manual,
          ref_pt_pt_deepl
        FROM {view_name}
    """

    params = []
    if split is not None:
        base_query += " WHERE lower(split) = ?"
        params.append(split.lower())

    while True:
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                break
            cur_batch = min(batch_size, remaining)
        else:
            cur_batch = batch_size

        batch = con.execute(base_query + " LIMIT ? OFFSET ?", [*params, cur_batch, offset]).df()
        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield row.to_dict()

        offset += len(batch)


def count_rows_for_split(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    split: Optional[str],
) -> int:
    query = f"SELECT COUNT(*) FROM {view_name}"
    params = []
    if split is not None:
        query += " WHERE lower(split) = ?"
        params.append(split.lower())
    return int(con.execute(query, params).fetchone()[0])


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def strip_task_prefix(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"^<[^>]+>\s*", "", str(text).strip())


def normalize_class_label(label: Optional[str]) -> Optional[str]:
    t = (label or "").strip().lower()
    if t == "pt-br":
        return "pt-BR"
    if t == "pt-pt":
        return "pt-PT"
    return None


def export_split_from_gpt_files(
    *,
    translation_file: Path,
    classification_file: Path,
    out_path: Path,
    task_mode: str,
) -> Dict:
    include_translation = task_mode in {"multitask", "translation"}
    include_classification = task_mode in {"multitask", "classification"}

    task_counts = Counter()
    dataset_counts = Counter()
    row_count = 0
    written_count = 0
    skipped_equal = 0

    with out_path.open("w", encoding="utf-8") as fh:
        if include_translation:
            if not translation_file.exists():
                raise FileNotFoundError(f"Missing GPT translation file: {translation_file}")
            for row in iter_jsonl(translation_file):
                row_count += 1
                source = as_clean_text(row.get("source_text")) or strip_task_prefix(row.get("input_text"))
                target = as_clean_text(row.get("target_text"))
                direction = (row.get("direction") or "").strip().lower()
                if not direction:
                    pref = str(row.get("input_text") or "").strip()
                    if pref.startswith("<br-pt>"):
                        direction = "br2pt"
                    elif pref.startswith("<pt-br>"):
                        direction = "pt2br"

                if not source or not target or direction not in {"br2pt", "pt2br"}:
                    continue

                ex = {
                    "task": "translate_br2pt" if direction == "br2pt" else "translate_pt2br",
                    "source": source,
                    "target": target,
                    "dataset": "GPT",
                }
                fh.write(json.dumps(to_axolotl_chat_record(ex), ensure_ascii=False) + "\n")
                written_count += 1
                task_counts[ex["task"]] += 1
                dataset_counts["GPT"] += 1

        if include_classification:
            if not classification_file.exists():
                raise FileNotFoundError(f"Missing GPT classification file: {classification_file}")
            for row in iter_jsonl(classification_file):
                row_count += 1
                source = as_clean_text(row.get("text")) or strip_task_prefix(row.get("input_text"))
                target = normalize_class_label(as_clean_text(row.get("label")))
                if not source:
                    continue
                if target is None:
                    skipped_equal += 1
                    continue

                ex = {
                    "task": "classify",
                    "source": source,
                    "target": target,
                    "dataset": "GPT",
                }
                fh.write(json.dumps(to_axolotl_chat_record(ex), ensure_ascii=False) + "\n")
                written_count += 1
                task_counts[ex["task"]] += 1
                dataset_counts["GPT"] += 1

    out = {
        "rows": row_count,
        "rows_selected": row_count,
        "examples": written_count,
        "tasks": dict(task_counts),
        "datasets": dict(dataset_counts),
        "path": out_path.as_posix(),
    }
    if skipped_equal:
        out["skipped_equal_labels"] = skipped_equal
    return out


def build_examples_from_row(
    row: Dict,
    rng: random.Random,
    pt2br_prob: float,
    task_mode: str,
) -> list[Dict]:
    examples = []
    include_translation = task_mode in {"multitask", "translation"}
    include_classification = task_mode in {"multitask", "classification"}

    dataset = row.get("dataset")
    src_br = as_clean_text(row.get("text_pt_br"))
    src_pt = as_clean_text(row.get("text_pt_pt"))
    label = as_clean_text(row.get("label"))

    has_br = bool(src_br)
    has_pt = bool(src_pt)

    if dataset == "OpenSubs":
        if include_translation and has_br and has_pt:
            if rng.random() < pt2br_prob:
                examples.append(
                    {"task": "translate_pt2br", "source": src_pt, "target": src_br, "dataset": dataset}
                )
            else:
                examples.append(
                    {"task": "translate_br2pt", "source": src_br, "target": src_pt, "dataset": dataset}
                )
        if include_classification and has_br and has_pt:
            examples.append({"task": "classify", "source": src_br, "target": "pt-BR", "dataset": dataset})
            examples.append({"task": "classify", "source": src_pt, "target": "pt-PT", "dataset": dataset})
        return examples

    if dataset == "PtBrVarId":
        if include_classification and label in ("pt-BR", "pt-PT"):
            text = src_br if has_br else src_pt
            if text:
                examples.append({"task": "classify", "source": text, "target": label, "dataset": dataset})
        return examples

    if include_translation and has_br and has_pt:
        if rng.random() < pt2br_prob:
            examples.append({"task": "translate_pt2br", "source": src_pt, "target": src_br, "dataset": dataset})
        else:
            examples.append({"task": "translate_br2pt", "source": src_br, "target": src_pt, "dataset": dataset})

    if include_classification and has_br:
        examples.append({"task": "classify", "source": src_br, "target": "pt-BR", "dataset": dataset})
    if include_classification and has_pt:
        examples.append({"task": "classify", "source": src_pt, "target": "pt-PT", "dataset": dataset})

    return examples


def to_axolotl_chat_record(example: Dict) -> Dict:
    task = example["task"]
    source = example["source"]
    target = example["target"]

    if task == "translate_br2pt":
        system_prompt = SYSTEM_TRANSLATION
        user_content = USER_TRANSL_BR2PT.format(source=source)
    elif task == "translate_pt2br":
        system_prompt = SYSTEM_TRANSLATION
        user_content = USER_TRANSL_PT2BR.format(source=source)
    elif task == "classify":
        system_prompt = SYSTEM_CLASSIFICATION
        user_content = USER_CLASSIFICATION.format(source=source)
    else:
        raise ValueError(f"Unknown task: {task}")

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ]
    }


def export_split(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str,
    split: str,
    limit: Optional[int],
    batch_size: int,
    out_path: Path,
    rng: random.Random,
    pt2br_prob: float,
    task_mode: str,
) -> Dict:
    row_count = 0
    written_count = 0
    task_counts = Counter()
    dataset_counts = Counter()

    with out_path.open("w", encoding="utf-8") as fh:
        for row in stream_from_view(
            con=con,
            view_name=view_name,
            split=split,
            batch_size=batch_size,
            limit=limit,
        ):
            row_count += 1
            examples = build_examples_from_row(
                row,
                rng=rng,
                pt2br_prob=pt2br_prob,
                task_mode=task_mode,
            )
            for ex in examples:
                record = to_axolotl_chat_record(ex)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1
                task_counts[ex["task"]] += 1
                dataset_counts[ex.get("dataset") or "UNKNOWN"] += 1

            if row_count % 100_000 == 0:
                print(f"[{split}] processed rows={row_count} written_examples={written_count}")

    return {
        "rows": row_count,
        "rows_selected": row_count,
        "examples": written_count,
        "tasks": dict(task_counts),
        "datasets": dict(dataset_counts),
        "path": out_path.as_posix(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_path = args.out_dir / args.train_file
    valid_path = args.out_dir / args.valid_file

    if args.gpt_only:
        print("GPT-only mode: exporting from GPT task JSONL files (Step 2/3 outputs).")
        print(f"  translation train: {args.gpt_translation_train_file}")
        print(f"  translation valid: {args.gpt_translation_valid_file}")
        if args.task_mode in {"multitask", "classification"}:
            print(f"  classification train: {args.gpt_classification_train_file}")
            print(f"  classification valid: {args.gpt_classification_valid_file}")

        train_stats = export_split_from_gpt_files(
            translation_file=args.gpt_translation_train_file,
            classification_file=args.gpt_classification_train_file,
            out_path=train_path,
            task_mode=args.task_mode,
        )
        valid_stats = export_split_from_gpt_files(
            translation_file=args.gpt_translation_valid_file,
            classification_file=args.gpt_classification_valid_file,
            out_path=valid_path,
            task_mode=args.task_mode,
        )
    else:
        con = connect_project(args.project_db, args.source_db, read_only=True)
        try:
            train_source_rows = count_rows_for_split(
                con,
                view_name=args.train_view,
                split=args.train_split,
            )
            valid_source_rows = count_rows_for_split(
                con,
                view_name=args.valid_view,
                split=args.valid_split,
            )

            print("Sanity check (source rows from DuckDB views):")
            print(
                f"  train source: view={args.train_view} split={args.train_split} rows={train_source_rows}"
            )
            print(
                f"  valid source: view={args.valid_view} split={args.valid_split} rows={valid_source_rows}"
            )

            train_stats = export_split(
                con,
                view_name=args.train_view,
                split=args.train_split,
                limit=args.train_limit,
                batch_size=args.batch_size,
                out_path=train_path,
                rng=rng,
                pt2br_prob=args.pt2br_prob,
                task_mode=args.task_mode,
            )

            valid_stats = export_split(
                con,
                view_name=args.valid_view,
                split=args.valid_split,
                limit=args.valid_limit,
                batch_size=args.batch_size,
                out_path=valid_path,
                rng=rng,
                pt2br_prob=args.pt2br_prob,
                task_mode=args.task_mode,
            )
        finally:
            con.close()

    print("\nExport completed.")
    print(f"Task mode: {args.task_mode}")
    print(f"GPT-only filter: {args.gpt_only}")
    print(f"Train file: {train_stats['path']}")
    print(
        f"  rows_total={train_stats['rows']} rows_selected={train_stats['rows_selected']} "
        f"examples={train_stats['examples']}"
    )
    print(f"  tasks={train_stats['tasks']}")
    if "skipped_equal_labels" in train_stats:
        print(f"  skipped_equal_labels={train_stats['skipped_equal_labels']}")
    print(f"Valid file: {valid_stats['path']}")
    print(
        f"  rows_total={valid_stats['rows']} rows_selected={valid_stats['rows_selected']} "
        f"examples={valid_stats['examples']}"
    )
    print(f"  tasks={valid_stats['tasks']}")
    if "skipped_equal_labels" in valid_stats:
        print(f"  skipped_equal_labels={valid_stats['skipped_equal_labels']}")

    train_jsonl_lines = sum(1 for _ in train_path.open("r", encoding="utf-8"))
    valid_jsonl_lines = sum(1 for _ in valid_path.open("r", encoding="utf-8"))
    print("\nSanity check (Axolotl JSONL inputs):")
    print(f"  train jsonl lines={train_jsonl_lines} file={train_path}")
    print(f"  valid jsonl lines={valid_jsonl_lines} file={valid_path}")


if __name__ == "__main__":
    main()
