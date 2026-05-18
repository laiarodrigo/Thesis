#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from generate_pt_variant_prompts_csv import (  # noqa: E402
    collect_text_fragments,
    extract_json_object,
    is_retryable_backend_error,
    mask_secret,
    normalize_space,
    resolve_api_config,
    rotated_thread_id,
    send_agent_message,
)


DOMAIN_HINTS = {
    "journalistic": "jornalístico",
    "legal": "jurídico",
    "literature": "literário",
    "politics": "político",
    "social_media": "redes sociais",
    "web": "web",
}


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Sample PtBrVId rows by (domain, label) from DuckDB and translate them "
            "to the missing Portuguese variant via the IAEDU API."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=repo_root / "data" / "duckdb" / "subs_ptbr_filtered.duckdb",
        help=(
            "DuckDB file containing table ptbrvarid. If the default path does not "
            "exist, the script falls back to data/duckdb/subs_filtered_final.duckdb."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="PtBrVId",
        help="Dataset tag inside ptbrvarid (default: PtBrVId).",
    )
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated split filter. Use '*' or empty to keep every split.",
    )
    parser.add_argument(
        "--domains",
        default="",
        help="Optional comma-separated domain allowlist.",
    )
    parser.add_argument(
        "--labels",
        default="pt-BR,pt-PT",
        help="Comma-separated label allowlist (default: pt-BR,pt-PT).",
    )
    parser.add_argument(
        "--rows-per-group",
        type=int,
        default=1000,
        help="How many sampled rows to translate for each (domain, label) group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=48,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Rows per translation request.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum concurrent API requests.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum attempts per batch.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=3.0,
        help="Base retry backoff in seconds.",
    )
    parser.add_argument(
        "--allow-short-groups",
        action="store_true",
        help=(
            "Allow groups with fewer than --rows-per-group rows and sample the full "
            "available group instead of failing."
        ),
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only sample rows and write manifests. Do not call the API.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse any existing translated_pairs.csv in out-dir and skip already "
            "translated sample_ids."
        ),
    )
    parser.add_argument(
        "--sampled-rows-csv",
        type=Path,
        default=None,
        help=(
            "Optional pre-sampled CSV manifest. When provided, skip DuckDB sampling "
            "and translate exactly the rows listed there."
        ),
    )
    parser.add_argument(
        "--batch-json-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory where each API batch is also written as a JSON "
            "file for inspection or resumable offline handling."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "ptbrvarid" / "translated_stageb_pairs",
        help="Output directory.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=repo_root / ".env",
        help="Optional .env-style file with IAEDU credentials.",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="IAEDU endpoint. If omitted, uses IAEDU_ENDPOINT from env.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="IAEDU API key. If omitted, uses IAEDU_API_KEY from env.",
    )
    parser.add_argument(
        "--channel-id",
        default=None,
        help="IAEDU channel id. If omitted, uses IAEDU_CHANNEL_ID from env.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="IAEDU base thread id. Every batch request gets a fresh derived thread id.",
    )
    parser.add_argument(
        "--short-thread-id",
        default=None,
        help=(
            "Optional IAEDU short thread id. If omitted, uses IAEDU_SHORT_THREAD_ID "
            "from env when available."
        ),
    )
    parser.add_argument(
        "--user-info",
        default="{}",
        help="Mandatory IAEDU user_info field as JSON string.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="Optional IAEDU user_id.",
    )
    parser.add_argument(
        "--user-context",
        default=None,
        help="Optional IAEDU user_context JSON string.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="HTTP timeout per request in seconds.",
    )
    return parser.parse_args()


def split_csv_arg(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def canonical_label(label: str) -> str:
    value = normalize_space(label).casefold()
    if value in {"pt-br", "pt_br", "ptbr"}:
        return "pt-BR"
    if value in {"pt-pt", "pt_pt", "ptpt"}:
        return "pt-PT"
    return normalize_space(label)


def infer_target_variant(source_label: str) -> str:
    source = canonical_label(source_label)
    if source == "pt-BR":
        return "pt-PT"
    if source == "pt-PT":
        return "pt-BR"
    raise ValueError(f"Unsupported source label: {source_label}")


def infer_direction(source_label: str) -> str:
    source = canonical_label(source_label)
    if source == "pt-BR":
        return "br2pt"
    if source == "pt-PT":
        return "pt2br"
    raise ValueError(f"Unsupported source label: {source_label}")


def source_text_from_row(row: dict[str, Any]) -> str:
    label = canonical_label(str(row["label"]))
    if label == "pt-BR":
        return normalize_space(str(row.get("text_pt_br") or ""))
    if label == "pt-PT":
        return normalize_space(str(row.get("text_pt_pt") or ""))
    raise ValueError(f"Unsupported label: {row['label']}")


def output_pair_columns(source_label: str, source_text: str, translated_text: str) -> tuple[str, str]:
    label = canonical_label(source_label)
    if label == "pt-BR":
        return source_text, translated_text
    if label == "pt-PT":
        return translated_text, source_text
    raise ValueError(f"Unsupported label: {source_label}")


def maybe_resolve_db_path(path: Path) -> Path:
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[2] / "data" / "duckdb" / "subs_filtered_final.duckdb"
    if path.name == "subs_ptbr_filtered.duckdb" and fallback.exists():
        log(f"Default db path not found. Falling back to {fallback.resolve()}")
        return fallback
    return path


def load_group_rows(
    *,
    db_path: Path,
    dataset: str,
    splits: list[str],
    domains: list[str],
    labels: list[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Missing dependency 'duckdb'. Use an environment with duckdb installed.") from exc

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {name for (name,) in con.execute("SHOW TABLES").fetchall()}
        if "ptbrvarid" not in tables:
            raise SystemExit(f"Table 'ptbrvarid' not found in {db_path}")

        column_names = {
            str(name).casefold()
            for _, name, *_ in con.execute("PRAGMA table_info('ptbrvarid')").fetchall()
        }
        has_dataset_column = "dataset" in column_names

        query = """
            SELECT
              COALESCE(split, '') AS split,
              lower(trim(COALESCE(domain, ''))) AS domain,
              label,
              text_pt_br,
              text_pt_pt
            FROM ptbrvarid
            WHERE lower(trim(COALESCE(domain, ''))) <> ''
              AND label IN (?, ?)
        """
        params: list[Any] = ["pt-BR", "pt-PT"]
        if has_dataset_column:
            query = query.replace(
                "WHERE lower(trim(COALESCE(domain, ''))) <> ''",
                "WHERE dataset = ?\n              AND lower(trim(COALESCE(domain, ''))) <> ''",
            )
            params.insert(0, dataset)
        rows = con.execute(query, params).fetchall()
    finally:
        con.close()

    split_filter = {normalize_space(item).casefold() for item in splits}
    domain_filter = {normalize_space(item).casefold() for item in domains}
    label_filter = {canonical_label(item) for item in labels}

    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for split, domain, label, text_pt_br, text_pt_pt in rows:
        split_norm = normalize_space(str(split or "")).casefold()
        domain_norm = normalize_space(str(domain or "")).casefold()
        label_norm = canonical_label(str(label or ""))
        if split_filter and split_norm not in split_filter:
            continue
        if domain_filter and domain_norm not in domain_filter:
            continue
        if label_norm not in label_filter:
            continue

        row = {
            "split": normalize_space(str(split or "")),
            "domain": domain_norm,
            "label": label_norm,
            "text_pt_br": normalize_space(str(text_pt_br or "")),
            "text_pt_pt": normalize_space(str(text_pt_pt or "")),
        }
        source_text = source_text_from_row(row)
        if not source_text:
            continue
        dedupe_key = source_text.casefold()
        grouped[(domain_norm, label_norm)][dedupe_key] = row

    return {key: list(inner.values()) for key, inner in grouped.items()}


def sample_group_rows(
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    rows_per_group: int,
    seed: int,
    allow_short_groups: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if rows_per_group <= 0:
        raise SystemExit("--rows-per-group must be > 0")

    sampled: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    sample_counter = 1
    group_keys = sorted(grouped_rows)
    shortages: list[str] = []

    for domain, label in group_keys:
        pool = list(grouped_rows[(domain, label)])
        rng = random.Random(f"{seed}:{domain}:{label}")
        rng.shuffle(pool)
        available = len(pool)
        requested = rows_per_group
        selected_n = min(available, requested) if allow_short_groups else requested
        if available < requested:
            shortages.append(f"{domain}/{label}={available}")
            if not allow_short_groups:
                continue
        chosen = pool[:selected_n]
        report_rows.append(
            {
                "domain": domain,
                "label": label,
                "available_rows": available,
                "selected_rows": len(chosen),
            }
        )
        for row in chosen:
            source_text = source_text_from_row(row)
            item = dict(row)
            item["sample_id"] = f"ptbrvid_{sample_counter:06d}"
            item["source_text"] = source_text
            item["target_variant"] = infer_target_variant(label)
            item["direction"] = infer_direction(label)
            sampled.append(item)
            sample_counter += 1

    if shortages and not allow_short_groups:
        joined = ", ".join(shortages)
        raise SystemExit(
            "Some groups do not have enough rows for the requested sample size. "
            f"Short groups: {joined}. Re-run with --allow-short-groups to cap them."
        )

    return sampled, report_rows


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def build_translation_prompt(*, domain: str, source_label: str, target_label: str, rows: list[dict[str, Any]]) -> str:
    schema = '{"translations":[{"id":"...", "translated_text":"..."}]}'
    items = [{"id": row["sample_id"], "text": row["source_text"]} for row in rows]
    domain_hint = DOMAIN_HINTS.get(domain, domain)
    return f"""
Traduz exatamente {len(rows)} textos de {source_label} para {target_label}.
Domínio/registo principal: {domain_hint}.

Devolve apenas JSON válido com este formato:
{schema}

Regras obrigatórias:
1) Mantém o significado exatamente.
2) Não sobretraduzas.
3) Não faças paráfrases livres.
4) Não introduzas sinónimos, reformulações ou floreados quando não forem necessários para adaptar à variante alvo.
5) Faz apenas as mudanças mínimas necessárias entre variantes. Se o texto já funcionar bem na variante alvo, podes mantê-lo igual.
6) Preserva nomes próprios, números, datas, siglas, URLs, hashtags, menções, emoji, segmentação frásica e pontuação sempre que possível.
7) Mantém o mesmo registo, tom e grau de formalidade do original.
8) Não acrescentes nem retires informação.
9) Para cada item, devolve uma única tradução no mesmo "id".
10) Não devolvas markdown, comentários, notas nem chaves extra.

Itens:
{json.dumps(items, ensure_ascii=False, indent=2)}
""".strip()


def parse_translation_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw_items = (
        payload.get("translations")
        or payload.get("traducoes")
        or payload.get("items")
        or payload.get("rows")
    )
    if not isinstance(raw_items, list):
        return []

    out: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        sample_id = normalize_space(
            str(
                item.get("id")
                or item.get("sample_id")
                or item.get("source_id")
                or ""
            )
        )
        translated_text = normalize_space(
            str(
                item.get("translated_text")
                or item.get("translation")
                or item.get("target_text")
                or item.get("texto_traduzido")
                or item.get("tradução")
                or item.get("traducao")
                or ""
            )
        )
        if sample_id and translated_text:
            out.append({"sample_id": sample_id, "translated_text": translated_text})
    return out


def iter_json_objects_from_text(raw: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    length = len(raw)
    while idx < length:
        start = raw.find("{", idx)
        if start == -1:
            break
        try:
            parsed, end = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
        idx = start + end
    return objects


def extract_translations_from_raw(raw: str) -> list[dict[str, str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        direct = parse_translation_payload(extract_json_object(text))
        if direct:
            return direct
    except Exception:
        pass

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        if isinstance(parsed, dict):
            direct = parse_translation_payload(parsed)
            if direct:
                return direct
        for fragment in collect_text_fragments(parsed):
            try:
                nested = parse_translation_payload(extract_json_object(fragment))
                if nested:
                    return nested
            except Exception:
                continue

    for obj in iter_json_objects_from_text(text):
        direct = parse_translation_payload(obj)
        if direct:
            return direct

        for fragment in collect_text_fragments(obj):
            try:
                nested = parse_translation_payload(extract_json_object(fragment))
                if nested:
                    return nested
            except Exception:
                continue

    raise ValueError("Could not parse translations from agent response.")


def validate_batch_response(
    *,
    batch_rows: list[dict[str, Any]],
    parsed_rows: list[dict[str, str]],
) -> dict[str, str]:
    expected_ids = {row["sample_id"] for row in batch_rows}
    seen: dict[str, str] = {}
    for item in parsed_rows:
        sample_id = item["sample_id"]
        if sample_id not in expected_ids:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate translation for sample_id={sample_id}")
        translated_text = normalize_space(item["translated_text"])
        if not translated_text:
            raise ValueError(f"Empty translation for sample_id={sample_id}")
        seen[sample_id] = translated_text

    missing = sorted(expected_ids - set(seen))
    if missing:
        raise ValueError(f"Missing translations for {len(missing)} ids: {', '.join(missing[:5])}")
    return seen


def translate_batch(
    *,
    batch_rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    max_attempts: int,
    retry_backoff_seconds: float,
    raw_failure_dir: Path,
    batch_name: str,
) -> dict[str, Any]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")

    domain = str(batch_rows[0]["domain"])
    source_label = str(batch_rows[0]["label"])
    target_label = str(batch_rows[0]["target_variant"])
    prompt = build_translation_prompt(
        domain=domain,
        source_label=source_label,
        target_label=target_label,
        rows=batch_rows,
    )

    last_exc: Exception | None = None
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        thread_id = rotated_thread_id(str(base_config["thread_id"]))
        config = dict(base_config)
        config["thread_id"] = thread_id
        try:
            raw = send_agent_message(config, prompt)
            last_raw = raw
            parsed = extract_translations_from_raw(raw)
            translations = validate_batch_response(batch_rows=batch_rows, parsed_rows=parsed)
            return {
                "ok": True,
                "rows": batch_rows,
                "translations": translations,
                "attempts": attempt,
                "thread_id": thread_id,
                "batch_name": batch_name,
            }
        except Exception as exc:
            last_exc = exc
            wait_s = retry_backoff_seconds * attempt
            if attempt < max_attempts:
                if is_retryable_backend_error(exc):
                    log(f"[{batch_name}] attempt {attempt}/{max_attempts} failed with retryable error: {exc}")
                else:
                    log(f"[{batch_name}] attempt {attempt}/{max_attempts} failed: {exc}")
                time.sleep(wait_s)

    raw_failure_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_failure_dir / f"{batch_name}.txt"
    if last_raw:
        raw_path.write_text(last_raw, encoding="utf-8")
    return {
        "ok": False,
        "rows": batch_rows,
        "error": str(last_exc or "unknown error"),
        "attempts": max_attempts,
        "raw_path": raw_path.as_posix() if last_raw else "",
        "batch_name": batch_name,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_sampled_rows_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        for row in reader:
            label = canonical_label(str(row.get("label") or ""))
            source_text = normalize_space(str(row.get("source_text") or ""))
            text_pt_br = normalize_space(str(row.get("text_pt_br") or ""))
            text_pt_pt = normalize_space(str(row.get("text_pt_pt") or ""))
            if not label or not source_text:
                continue
            rows.append(
                {
                    "sample_id": normalize_space(str(row.get("sample_id") or "")),
                    "split": normalize_space(str(row.get("split") or "")),
                    "domain": normalize_space(str(row.get("domain") or "")).casefold(),
                    "label": label,
                    "target_variant": normalize_space(str(row.get("target_variant") or infer_target_variant(label))),
                    "direction": normalize_space(str(row.get("direction") or infer_direction(label))),
                    "source_text": source_text,
                    "text_pt_br": text_pt_br,
                    "text_pt_pt": text_pt_pt,
                }
            )
    return rows


def build_group_report_from_sampled_rows(sampled_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter((row["domain"], row["label"]) for row in sampled_rows)
    report_rows: list[dict[str, Any]] = []
    for (domain, label), count in sorted(counts.items()):
        report_rows.append(
            {
                "domain": domain,
                "label": label,
                "available_rows": count,
                "selected_rows": count,
            }
        )
    return report_rows


def build_domain_batches(
    sampled_rows: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[str], dict[str, list[tuple[str, list[dict[str, Any]]]]], int]:
    domain_order = sorted({row["domain"] for row in sampled_rows})
    batches_by_domain: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    total_batches = 0
    for domain in domain_order:
        domain_items = [row for row in sampled_rows if row["domain"] == domain]
        labels_in_domain = sorted({row["label"] for row in domain_items})
        domain_batches: list[tuple[str, list[dict[str, Any]]]] = []
        for label in labels_in_domain:
            group_items = [row for row in domain_items if row["label"] == label]
            for batch_idx, batch_rows in enumerate(chunk_rows(group_items, batch_size), start=1):
                batch_name = f"{domain}_{label}_{batch_idx:04d}".replace("/", "_")
                domain_batches.append((batch_name, batch_rows))
        batches_by_domain[domain] = domain_batches
        total_batches += len(domain_batches)
    return domain_order, batches_by_domain, total_batches


def write_batch_json_files(
    batch_json_dir: Path,
    batches_by_domain: dict[str, list[tuple[str, list[dict[str, Any]]]]],
) -> None:
    batch_json_dir.mkdir(parents=True, exist_ok=True)
    for domain_batches in batches_by_domain.values():
        for batch_name, batch_rows in domain_batches:
            payload = {
                "batch_name": batch_name,
                "rows": batch_rows,
            }
            (batch_json_dir / f"{batch_name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


TRANSLATED_FIELDNAMES = [
    "sample_id",
    "dataset",
    "split",
    "domain",
    "source_label",
    "target_variant",
    "direction",
    "source_text",
    "translated_text",
    "pt_BR",
    "pt_PT",
    "batch_name",
    "thread_id",
    "attempts",
]

FAILURE_FIELDNAMES = [
    "sample_id",
    "dataset",
    "split",
    "domain",
    "source_label",
    "target_variant",
    "direction",
    "source_text",
    "batch_name",
    "attempts",
    "error",
    "raw_path",
]


def checkpoint_progress(
    *,
    translated_pairs_path: Path,
    failures_path: Path,
    merged_translated_by_id: dict[str, dict[str, Any]],
    merged_failure_by_id: dict[str, dict[str, Any]],
) -> None:
    merged_translated_rows = sorted(merged_translated_by_id.values(), key=lambda row: str(row["sample_id"]))
    merged_failure_rows = sorted(merged_failure_by_id.values(), key=lambda row: str(row["sample_id"]))
    write_csv(translated_pairs_path, merged_translated_rows, TRANSLATED_FIELDNAMES)
    write_csv(failures_path, merged_failure_rows, FAILURE_FIELDNAMES)


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be > 0")
    if args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0")

    sampled_rows_path = args.out_dir / "sampled_rows.csv"
    translated_pairs_path = args.out_dir / "translated_pairs.csv"
    failures_path = args.out_dir / "translation_failures.csv"
    group_report_path = args.out_dir / "group_report.csv"
    build_report_path = args.out_dir / "build_report.json"
    raw_failure_dir = args.out_dir / "raw_failures"
    batch_json_dir = args.batch_json_dir or (args.out_dir / "batch_json")

    splits = split_csv_arg(args.splits)
    if any(item == "*" for item in splits):
        splits = []
    domains = split_csv_arg(args.domains)
    labels = [canonical_label(item) for item in split_csv_arg(args.labels)]
    if not labels:
        labels = ["pt-BR", "pt-PT"]

    sampled_rows: list[dict[str, Any]]
    group_report: list[dict[str, Any]]
    db_path_text: str | None
    if args.sampled_rows_csv is not None:
        if not args.sampled_rows_csv.exists():
            raise SystemExit(f"Sampled rows CSV not found: {args.sampled_rows_csv}")
        sampled_rows = read_sampled_rows_csv(args.sampled_rows_csv)
        if not sampled_rows:
            raise SystemExit(f"Sampled rows CSV is empty or invalid: {args.sampled_rows_csv}")
        group_report = build_group_report_from_sampled_rows(sampled_rows)
        db_path_text = None
    else:
        args.db = maybe_resolve_db_path(args.db)
        if not args.db.exists():
            raise SystemExit(f"DuckDB file not found: {args.db}")
        grouped_rows = load_group_rows(
            db_path=args.db,
            dataset=args.dataset,
            splits=splits,
            domains=domains,
            labels=labels,
        )
        if not grouped_rows:
            raise SystemExit("No PtBrVId rows matched the provided filters.")

        sampled_rows, group_report = sample_group_rows(
            grouped_rows,
            rows_per_group=args.rows_per_group,
            seed=args.seed,
            allow_short_groups=bool(args.allow_short_groups),
        )
        if not sampled_rows:
            raise SystemExit("Sampling produced zero rows.")
        db_path_text = args.db.as_posix()

    sampled_fieldnames = [
        "sample_id",
        "dataset",
        "split",
        "domain",
        "label",
        "target_variant",
        "direction",
        "source_text",
        "text_pt_br",
        "text_pt_pt",
    ]
    sampled_export_rows = []
    for row in sampled_rows:
        sampled_export_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": args.dataset,
                "split": row["split"],
                "domain": row["domain"],
                "label": row["label"],
                "target_variant": row["target_variant"],
                "direction": row["direction"],
                "source_text": row["source_text"],
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
            }
        )
    write_csv(sampled_rows_path, sampled_export_rows, sampled_fieldnames)
    write_csv(group_report_path, group_report, ["domain", "label", "available_rows", "selected_rows"])

    build_report: dict[str, Any] = {
        "db": db_path_text,
        "dataset": args.dataset,
        "splits": splits,
        "domains": domains,
        "labels": labels,
        "rows_per_group": args.rows_per_group,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "allow_short_groups": bool(args.allow_short_groups),
        "sample_only": bool(args.sample_only),
        "sampled_rows_csv_input": args.sampled_rows_csv.as_posix() if args.sampled_rows_csv is not None else None,
        "sampled_rows_csv": sampled_rows_path.as_posix(),
        "group_report_csv": group_report_path.as_posix(),
        "sampled_total": len(sampled_rows),
        "selected_by_group": group_report,
    }

    if args.sample_only:
        build_report_path.parent.mkdir(parents=True, exist_ok=True)
        build_report_path.write_text(json.dumps(build_report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Sample-only mode. Wrote sampled rows to {sampled_rows_path.resolve()}")
        log(f"Wrote group report to {group_report_path.resolve()}")
        log(f"Wrote build report to {build_report_path.resolve()}")
        return

    config = resolve_api_config(args)
    log(
        "Resolved IAEDU config: "
        f"endpoint={config['endpoint']}, channel_id={config['channel_id']}, "
        f"thread_id={config['thread_id']}, api_key={mask_secret(config['api_key'])}"
    )

    existing_translated_rows = read_csv_rows(translated_pairs_path) if args.resume else []
    existing_failure_rows = read_csv_rows(failures_path) if args.resume else []
    existing_translated_by_id = {
        normalize_space(str(row.get("sample_id") or "")): row
        for row in existing_translated_rows
        if normalize_space(str(row.get("sample_id") or ""))
    }
    if existing_translated_by_id:
        before_count = len(sampled_rows)
        sampled_rows = [row for row in sampled_rows if row["sample_id"] not in existing_translated_by_id]
        skipped_count = before_count - len(sampled_rows)
        log(
            f"Resume mode: found {len(existing_translated_by_id)} existing translated rows; "
            f"skipping {skipped_count} sampled rows already completed."
        )
        build_report["resume_existing_translated"] = len(existing_translated_by_id)
        build_report["resume_skipped_already_translated"] = skipped_count

    translated_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    translated_by_group: Counter[str] = Counter()
    failed_by_group: Counter[str] = Counter()
    merged_translated_by_id = dict(existing_translated_by_id)
    translated_ids = set(merged_translated_by_id)
    merged_failure_by_id: dict[str, dict[str, Any]] = {}
    for row in existing_failure_rows:
        sample_id = normalize_space(str(row.get("sample_id") or ""))
        if sample_id and sample_id not in translated_ids:
            merged_failure_by_id[sample_id] = row
    domain_order, batches_by_domain, total_batches = build_domain_batches(
        sampled_rows,
        batch_size=args.batch_size,
    )
    write_batch_json_files(batch_json_dir, batches_by_domain)
    completed_batches = 0

    log(
        f"Submitting {total_batches} batches for {len(sampled_rows)} sampled rows "
        f"(batch_size={args.batch_size}, max_workers={args.max_workers}, domain_order={domain_order})."
    )
    log(f"Wrote batch JSON files to {batch_json_dir.resolve()}")

    for domain_idx, domain in enumerate(domain_order, start=1):
        domain_items = [row for row in sampled_rows if row["domain"] == domain]
        labels_in_domain = sorted({row["label"] for row in domain_items})
        domain_batches = batches_by_domain[domain]

        log(
            f"Starting domain {domain_idx}/{len(domain_order)}: {domain} "
            f"(labels={labels_in_domain}, rows={len(domain_items)}, batches={len(domain_batches)})"
        )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    translate_batch,
                    batch_rows=batch_rows,
                    base_config=config,
                    max_attempts=args.max_attempts,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    raw_failure_dir=raw_failure_dir,
                    batch_name=batch_name,
                )
                for batch_name, batch_rows in domain_batches
            ]
            for future in as_completed(futures):
                result = future.result()
                rows = result["rows"]
                group_key = f"{rows[0]['domain']}/{rows[0]['label']}"
                if result["ok"]:
                    translations = result["translations"]
                    for row in rows:
                        translated_text = translations[row["sample_id"]]
                        pt_br, pt_pt = output_pair_columns(row["label"], row["source_text"], translated_text)
                        translated_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "dataset": args.dataset,
                                "split": row["split"],
                                "domain": row["domain"],
                                "source_label": row["label"],
                                "target_variant": row["target_variant"],
                                "direction": row["direction"],
                                "source_text": row["source_text"],
                                "translated_text": translated_text,
                                "pt_BR": pt_br,
                                "pt_PT": pt_pt,
                                "batch_name": result["batch_name"],
                                "thread_id": result["thread_id"],
                                "attempts": result["attempts"],
                            }
                        )
                        merged_translated_by_id[row["sample_id"]] = translated_rows[-1]
                        translated_ids.add(row["sample_id"])
                        merged_failure_by_id.pop(row["sample_id"], None)
                        translated_by_group[group_key] += 1
                    log(f"[{result['batch_name']}] translated {len(rows)} rows in {result['attempts']} attempt(s).")
                else:
                    for row in rows:
                        failure_rows.append(
                            {
                                "sample_id": row["sample_id"],
                                "dataset": args.dataset,
                                "split": row["split"],
                                "domain": row["domain"],
                                "source_label": row["label"],
                                "target_variant": row["target_variant"],
                                "direction": row["direction"],
                                "source_text": row["source_text"],
                                "batch_name": result["batch_name"],
                                "attempts": result["attempts"],
                                "error": result["error"],
                                "raw_path": result.get("raw_path", ""),
                            }
                        )
                        if row["sample_id"] not in translated_ids:
                            merged_failure_by_id[row["sample_id"]] = failure_rows[-1]
                        failed_by_group[group_key] += 1
                    log(f"[{result['batch_name']}] failed after {result['attempts']} attempts: {result['error']}")

                completed_batches += 1
                if completed_batches % 10 == 0:
                    checkpoint_progress(
                        translated_pairs_path=translated_pairs_path,
                        failures_path=failures_path,
                        merged_translated_by_id=merged_translated_by_id,
                        merged_failure_by_id=merged_failure_by_id,
                    )
                    log(
                        f"Checkpointed progress after {completed_batches} completed batches "
                        f"(translated={len(merged_translated_by_id)}, failed={len(merged_failure_by_id)})."
                    )

        log(f"Finished domain {domain}: translated={sum(v for k, v in translated_by_group.items() if k.startswith(domain + '/'))}, failed={sum(v for k, v in failed_by_group.items() if k.startswith(domain + '/'))}")

    translated_rows.sort(key=lambda row: row["sample_id"])
    failure_rows.sort(key=lambda row: row["sample_id"])
    merged_failure_rows = sorted(merged_failure_by_id.values(), key=lambda row: str(row["sample_id"]))
    merged_translated_rows = sorted(merged_translated_by_id.values(), key=lambda row: str(row["sample_id"]))
    checkpoint_progress(
        translated_pairs_path=translated_pairs_path,
        failures_path=failures_path,
        merged_translated_by_id=merged_translated_by_id,
        merged_failure_by_id=merged_failure_by_id,
    )

    build_report.update(
        {
            "translated_pairs_csv": translated_pairs_path.as_posix(),
            "failures_csv": failures_path.as_posix(),
            "batch_json_dir": batch_json_dir.as_posix(),
            "translated_total": len(merged_translated_rows),
            "failed_total": len(merged_failure_rows),
            "translated_new_this_run": len(translated_rows),
            "failed_new_this_run": len(failure_rows),
            "translated_by_group": dict(translated_by_group),
            "failed_by_group": dict(failed_by_group),
        }
    )
    build_report_path.parent.mkdir(parents=True, exist_ok=True)
    build_report_path.write_text(json.dumps(build_report, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Wrote sampled rows to {sampled_rows_path.resolve()}")
    log(f"Wrote translated pairs to {translated_pairs_path.resolve()}")
    log(f"Wrote failure log to {failures_path.resolve()}")
    log(f"Wrote build report to {build_report_path.resolve()}")


if __name__ == "__main__":
    main()
