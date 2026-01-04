import os
import random
import numpy as np
import torch
import duckdb
import pathlib
import pandas as pd

from typing import Iterator, Dict, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sacrebleu import corpus_bleu  # pip install sacrebleu

from collections import defaultdict, Counter
from dataclasses import dataclass

import json
from datetime import datetime
import re
import matplotlib.pyplot as plt


# -------------------------
# CONFIG – EDIT THESE
# -------------------------
USE_ADAPTER = True
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 123

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DB_PATH = BASE_DIR / "data" / "duckdb" / "subs_project.duckdb"
SOURCE_DB_PATH  = BASE_DIR / "data" / "duckdb" / "subs.duckdb"

PROJECT_DB_STR = PROJECT_DB_PATH.as_posix()
SOURCE_DB_STR  = SOURCE_DB_PATH.as_posix()

ADAPTER_DIR = Path("/cfs/home/u036584/models/qwen3-0_6b-ptbr-ptpt-lora-adapter-cluster_full")
TOKENIZER_PATH = Path("/cfs/home/u036584/models/qwen3-0_6b-ptbr-ptpt-lora-adapter-cluster_full-tokenizer")

MAX_LENGTH = 1024

EVAL_VIEW  = "test_data"
EVAL_SPLIT = "test"        # set to None if your view doesn't have split=test
EVAL_ROW_LIMIT = None        # None for all rows
GEN_BATCH_SIZE = 128

# streaming buffer (examples) to keep memory bounded
MAX_EXAMPLE_BUFFER = 512

# only bucket-level breakdowns (plus "overall")
GROUP_BY_BUCKET_ONLY = True
MIN_GROUP_SUPPORT = 2  # only print bucket groups with >= this many examples

N_EXAMPLES_PRINT = 40
PRINT_RAW_MODEL_OUTPUT = True

pd.set_option("display.max_colwidth", 180)

import re

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

def clean_hyp(text: str) -> str:
    t = (text or "").strip()
    t = _THINK_RE.sub("", t).strip()
    return t



# -------------------------
# DUCKDB: connect + streaming from VIEWS
# -------------------------

def connect_project(
    read_only: bool = True,
    memory_limit: str = "3.5GB",
    threads: int = 4,
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(
        PROJECT_DB_STR,
        read_only=read_only,
        config={
            "memory_limit": memory_limit,
            "threads": threads,
        },
    )

    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        con.execute(f"ATTACH '{SOURCE_DB_STR}' AS src")

    return con


def stream_from_view(
    view_name: str,
    split: Optional[str] = None,
    batch_size: int = 2_000,   # smaller is safer locally
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    con = connect_project(read_only=True)
    offset = 0

    base_query = f"""
        SELECT
          dataset,
          split,
          source,
          bucket,
          theme,
          label,
          text_pt_br,
          text_pt_pt,
          ref_pt_pt_manual,
          ref_pt_pt_deepl
        FROM {view_name}
    """

    if split is not None:
        base_query += f" WHERE lower(split) = '{split.lower()}'"

    while True:
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                break
            cur_batch = min(batch_size, remaining)
        else:
            cur_batch = batch_size

        batch = con.execute(base_query + f" LIMIT {cur_batch} OFFSET {offset}").df()
        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield row.to_dict()

        offset += len(batch)

    con.close()


# -------------------------
# PROMPTS (copied from train_script)
# -------------------------

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


# -------------------------
# Example builder for EVAL (deterministic)
# -------------------------

def build_eval_examples_from_row(row: Dict) -> List[Dict]:
    examples = []

    dataset = row.get("dataset")
    src_br  = row.get("text_pt_br")
    src_pt  = row.get("text_pt_pt")
    label   = row.get("label")

    # bucket is the only field we break down on (plus overall)
    meta_base = {
        "dataset": dataset,
        "bucket": row.get("bucket"),
    }

    has_br = bool(src_br)
    has_pt = bool(src_pt)

    # OpenSubs not expected in test_data, but keep logic safe
    if dataset == "OpenSubs":
        if has_br and has_pt:
            examples.append({
                "task": "translate_br2pt",
                "source": src_br,
                "target": src_pt,
                "meta": {**meta_base, "direction": "br2pt"},
            })
            examples.append({
                "task": "translate_pt2br",
                "source": src_pt,
                "target": src_br,
                "meta": {**meta_base, "direction": "pt2br"},
            })

        if has_br:
            examples.append({
                "task": "classify",
                "source": src_br,
                "target": "pt-BR",
                "meta": {**meta_base, "var": "pt-BR"},
            })
        if has_pt:
            examples.append({
                "task": "classify",
                "source": src_pt,
                "target": "pt-PT",
                "meta": {**meta_base, "var": "pt-PT"},
            })
        return examples

    # PtBrVarId: classification only, trust label
    if dataset == "PtBrVarId":
        if label in ("pt-BR", "pt-PT"):
            text = src_br if src_br else src_pt
            if text:
                examples.append({
                    "task": "classify",
                    "source": text,
                    "target": label,
                    "meta": {**meta_base, "var": label},
                })
        return examples

    # FRMT / Gold / others: translation if both sides exist + classification for each side
    if has_br and has_pt:
        examples.append({
            "task": "translate_br2pt",
            "source": src_br,
            "target": src_pt,
            "meta": {**meta_base, "direction": "br2pt"},
        })
        examples.append({
            "task": "translate_pt2br",
            "source": src_pt,
            "target": src_br,
            "meta": {**meta_base, "direction": "pt2br"},
        })

    if has_br:
        examples.append({
            "task": "classify",
            "source": src_br,
            "target": "pt-BR",
            "meta": {**meta_base, "var": "pt-BR"},
        })
    if has_pt:
        examples.append({
            "task": "classify",
            "source": src_pt,
            "target": "pt-PT",
            "meta": {**meta_base, "var": "pt-PT"},
        })

    return examples


def build_prompt_from_example(ex: Dict, tok) -> str:
    task   = ex["task"]
    source = ex["source"]

    if task == "translate_br2pt":
        system_prompt = SYSTEM_TRANSLATION
        user_content  = USER_TRANSL_BR2PT.format(source=source)
    elif task == "translate_pt2br":
        system_prompt = SYSTEM_TRANSLATION
        user_content  = USER_TRANSL_PT2BR.format(source=source)
    elif task == "classify":
        system_prompt = SYSTEM_CLASSIFICATION
        user_content  = USER_CLASSIFICATION.format(source=source)
    else:
        raise ValueError(f"Unknown task {task}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -------------------------
# Model loading
# -------------------------

def load_model_and_tokenizer():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    if USE_ADAPTER:
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True)
    else:
        print("WARNING: Not using adapter; loading base model and tokenizer only.")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if USE_ADAPTER:
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    else:
        model = base_model

    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")

    print("Model loaded.")
    return model, tok


# -------------------------
# Generation helper
# -------------------------

def generate_batch(model, tok, prompts: List[str]) -> List[str]:
    device = next(model.parameters()).device
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=128,  # helps stop runaway generations
        )

    preds = []
    for i in range(out.size(0)):
        seq_len = enc["input_ids"].shape[1]
        gen_ids = out[i, seq_len:]
        preds.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
    return preds


# -------------------------
# Classification label parsing
# -------------------------

def normalize_class_prediction(text: str) -> Optional[str]:
    t = (text or "").lower().strip()

    if "português do brasil" in t or "portugues do brasil" in t or "pt-br" in t or "brasil" in t:
        return "pt-BR"
    if "português europeu" in t or "portugues europeu" in t or "pt-pt" in t or "portugal" in t:
        return "pt-PT"
    if t.startswith("pt-br"):
        return "pt-BR"
    if t.startswith("pt-pt"):
        return "pt-PT"
    return None


# -------------------------
# Metrics
# -------------------------

def safe_bucket(val) -> str:
    if val is None:
        return "UNKNOWN"
    if isinstance(val, str):
        v = val.strip()
        return v if v else "UNKNOWN"
    return str(val)

def group_keys_from_meta(meta: Dict) -> List[str]:
    # Only overall + bucket breakdown
    keys = ["overall"]
    keys.append(f"bucket={safe_bucket(meta.get('bucket'))}")
    return keys

@dataclass
class ClsStats:
    tp: Counter
    fp: Counter
    fn: Counter
    cm: Counter
    total: int = 0
    correct: int = 0
    unknown: int = 0   

def update_cls(stats: ClsStats, gold: str, pred: str):
    stats.total += 1
    stats.cm[(gold, pred)] += 1  # NEW

    if pred == "UNKNOWN":
        stats.unknown += 1

    if pred == gold:
        stats.correct += 1
        stats.tp[gold] += 1
    else:
        stats.fn[gold] += 1
        stats.fp[pred] += 1


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0

def cls_report(stats: ClsStats, labels=("pt-BR", "pt-PT")) -> Dict:
    acc = stats.correct / stats.total if stats.total else 0.0
    unk_rate = stats.unknown / stats.total if stats.total else 0.0

    per = {}
    f1s = []
    recalls = []

    for lab in labels:
        tp = stats.tp[lab]
        fp = stats.fp[lab]
        fn = stats.fn[lab]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = _f1(precision, recall)

        per[lab] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
        f1s.append(f1)
        recalls.append(recall)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "unknown_rate": unk_rate,
        "n": stats.total,
        "per_class": per,
    }


@dataclass
class TransStats:
    refs: List[str]
    hyps: List[str]

def bleu_score(stats: TransStats) -> float:
    if not stats.refs:
        return float("nan")
    return corpus_bleu(stats.hyps, [stats.refs]).score


# -------------------------
# MAIN EVAL (streaming + bucket breakdown)
# -------------------------

def _process_batch(
    batch: List[Dict],
    model,
    tok,
    cls_by_group,
    trans_all_by_group,
    trans_dir_by_group,
    *,
    gen_batch_size: int = GEN_BATCH_SIZE,
    # optional: pass lists from main() if you want example printing
    cls_examples: Optional[List[Dict]] = None,
    trans_examples: Optional[List[Dict]] = None,
    max_examples_print: int = 8,
    print_raw_model_output: bool = True,
    cls_out_fh=None,
    trans_out_fh=None,
):
    """
    Process a list of eval examples.
    - Builds prompts
    - Runs generation in chunks of size `gen_batch_size` (important for VRAM control)
    - Updates classification + translation metric accumulators
    - Optionally stores a few qualitative examples in cls_examples/trans_examples
    """

    # Small helper: append up to max_examples_print
    def _append_example(buf: Optional[List[Dict]], item: Dict):
        if buf is None:
            return
        if len(buf) < max_examples_print:
            buf.append(item)

    # Run in generation chunks to avoid big VRAM spikes
    for start in range(0, len(batch), gen_batch_size):
        chunk = batch[start : start + gen_batch_size]

        prompts = [build_prompt_from_example(ex, tok) for ex in chunk]
        preds = generate_batch(model, tok, prompts)
        preds = [clean_hyp(p) for p in preds]

        for ex, pred in zip(chunk, preds):
            meta = ex.get("meta", {}) or {}
            gkeys = group_keys_from_meta(meta)

            task = ex.get("task", "")

            if task.startswith("translate"):
                ref = ex.get("target", "")
                direction = meta.get("direction", "UNKNOWN")

                # NEW: write translation record (only here, after ref/direction exist)
                if trans_out_fh is not None:
                    rec = {
                        "task": task,
                        "dataset": meta.get("dataset"),
                        "bucket": meta.get("bucket"),
                        "direction": direction,
                        "source": ex.get("source", ""),
                        "ref": ref,
                        "hyp": pred,
                    }
                    trans_out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # store a few translation examples for qualitative inspection
                _append_example(trans_examples, {
                    "dataset": meta.get("dataset"),
                    "bucket": meta.get("bucket"),
                    "direction": direction,
                    "source": ex.get("source", ""),
                    "ref": ref,
                    "hyp": pred,
                })

                for gk in gkeys:
                    trans_all_by_group[gk].refs.append(ref)
                    trans_all_by_group[gk].hyps.append(pred)

                    trans_dir_by_group[gk][direction].refs.append(ref)
                    trans_dir_by_group[gk][direction].hyps.append(pred)

            elif task == "classify":
                gold = ex.get("target")
                norm = normalize_class_prediction(pred) or "UNKNOWN"

                # write classification record (only here)
                if cls_out_fh is not None:
                    rec = {
                        "task": "classify",
                        "dataset": meta.get("dataset"),
                        "bucket": meta.get("bucket"),
                        "source": ex.get("source", ""),
                        "gold": gold,
                        "pred_norm": norm,
                        "pred_raw": pred if print_raw_model_output else None,
                    }
                    cls_out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                _append_example(cls_examples, {
                    "dataset": meta.get("dataset"),
                    "bucket": meta.get("bucket"),
                    "source": ex.get("source", ""),
                    "gold": gold,
                    "pred_norm": norm,
                    "pred_raw": pred if print_raw_model_output else None,
                })

                for gk in gkeys:
                    update_cls(cls_by_group[gk], gold, norm)

            else:
                continue

def main():
    model, tok = load_model_and_tokenizer()

    cls_by_group = defaultdict(lambda: ClsStats(tp=Counter(), fp=Counter(), fn=Counter(), cm=Counter()))

    trans_all_by_group = defaultdict(lambda: TransStats(refs=[], hyps=[]))
    trans_dir_by_group = defaultdict(lambda: defaultdict(lambda: TransStats(refs=[], hyps=[])))

    example_buffer: List[Dict] = []
    cls_examples: List[Dict] = []
    trans_examples: List[Dict] = []

    # -------------------------
    # NEW: output files (JSONL)
    # -------------------------
    out_dir = BASE_DIR / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    trans_path = out_dir / f"{run_id}_translations.jsonl"
    cls_path   = out_dir / f"{run_id}_classification.jsonl"

    with open(trans_path, "w", encoding="utf-8") as trans_fh, open(cls_path, "w", encoding="utf-8") as cls_fh:
        print("Streaming rows and evaluating...", flush=True)

        row_count = 0
        for row in stream_from_view(
            EVAL_VIEW,
            split=EVAL_SPLIT,
            batch_size=2_000,
            limit=EVAL_ROW_LIMIT,
        ):
            row_count += 1
            example_buffer.extend(build_eval_examples_from_row(row))

            while len(example_buffer) >= MAX_EXAMPLE_BUFFER:
                batch = example_buffer[:MAX_EXAMPLE_BUFFER]
                example_buffer = example_buffer[MAX_EXAMPLE_BUFFER:]
                _process_batch(
                    batch,
                    model,
                    tok,
                    cls_by_group,
                    trans_all_by_group,
                    trans_dir_by_group,
                    gen_batch_size=GEN_BATCH_SIZE,
                    cls_examples=cls_examples,
                    trans_examples=trans_examples,
                    max_examples_print=N_EXAMPLES_PRINT,
                    print_raw_model_output=PRINT_RAW_MODEL_OUTPUT,
                    # NEW: stream predictions to disk
                    cls_out_fh=cls_fh,
                    trans_out_fh=trans_fh,
                )

            if row_count % 1000 == 0:
                print(f"... streamed {row_count} rows (buffer={len(example_buffer)})", flush=True)

        if example_buffer:
            _process_batch(
                example_buffer,
                model,
                tok,
                cls_by_group,
                trans_all_by_group,
                trans_dir_by_group,
                gen_batch_size=GEN_BATCH_SIZE,
                cls_examples=cls_examples,
                trans_examples=trans_examples,
                max_examples_print=N_EXAMPLES_PRINT,
                print_raw_model_output=PRINT_RAW_MODEL_OUTPUT,
                # NEW: stream predictions to disk
                cls_out_fh=cls_fh,
                trans_out_fh=trans_fh,
            )

    print("\nDone generating. Metrics:\n")
    print(f"Saved translation predictions to: {trans_path}")
    print(f"Saved classification predictions to: {cls_path}")

    def _safe_name(s: str) -> str:
        s = s.replace("overall", "overall")
        s = re.sub(r"[^A-Za-z0-9._=-]+", "_", s)
        return s[:120]

    def save_confusion_matrix_png(
        stats: ClsStats,
        out_path: Path,
        *,
        title: str,
        labels=("pt-BR", "pt-PT"),
        cmap: str = "YlGnBu",   # brighter than default
    ):
        rows = list(labels)
        cols = list(labels)

        # Build 2x2 matrix (ignore UNKNOWN and any other labels)
        M = np.zeros((len(rows), len(cols)), dtype=int)
        for i, g in enumerate(rows):
            for j, p in enumerate(cols):
                M[i, j] = stats.cm.get((g, p), 0)

        n_included = int(M.sum())
        n_excluded = stats.total - n_included  # includes UNKNOWN or any other label pairs

        fig = plt.figure(figsize=(7.6, 6.2), dpi=240)
        ax = fig.add_subplot(111)

        # Brighter look: explicit cmap + scaling
        vmax = max(1, M.max())
        im = ax.imshow(M, cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)

        ax.set_title(f"{title} (n={n_included}, excluded={n_excluded})", fontsize=16, pad=12)
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("Gold", fontsize=14)

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(rows)))
        ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=12)
        ax.set_yticklabels(rows, fontsize=12)

        # Colorbar for readability
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)

        # Annotate counts with contrast-aware text color
        thresh = 0.55 * vmax
        for i in range(len(rows)):
            for j in range(len(cols)):
                val = M[i, j]
                txt_color = "white" if val > thresh else "black"
                ax.text(
                    j, i, str(val),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=txt_color,
                )

        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)



    # ---- Translation ----
    print("\n=== TRANSLATION (BLEU) ===")
    for gk in sorted(trans_all_by_group.keys(), key=lambda k: (0, k) if k == "overall" else (1, k)):
        n = len(trans_all_by_group[gk].refs)
        if gk != "overall" and n < MIN_GROUP_SUPPORT:
            continue

        bleu_all = bleu_score(trans_all_by_group[gk])
        print(f"\n[{gk}] n={n} BLEU={bleu_all:.2f}")

        for direction in ("br2pt", "pt2br"):
            st = trans_dir_by_group[gk].get(direction)
            if not st:
                continue
            if gk != "overall" and len(st.refs) < MIN_GROUP_SUPPORT:
                continue
            print(f"  - {direction}: n={len(st.refs)} BLEU={bleu_score(st):.2f}")

    # ---- Classification ----
    print("\n=== CLASSIFICATION (accuracy + balanced_acc + macro-F1 + per-class) ===")
    for gk in sorted(cls_by_group.keys(), key=lambda k: (0, k) if k == "overall" else (1, k)):
        stats = cls_by_group[gk]
        if gk != "overall" and stats.total < MIN_GROUP_SUPPORT:
            continue

        rep = cls_report(stats, labels=("pt-BR", "pt-PT"))
        print(
            f"\n[{gk}] n={rep['n']} "
            f"accuracy={rep['accuracy']*100:.2f}% "
            f"balanced_acc={rep['balanced_accuracy']*100:.2f}% "
            f"macro_f1={rep['macro_f1']*100:.2f}% "
            f"unknown={rep['unknown_rate']*100:.2f}%"
        )

        for lab, m in rep["per_class"].items():
            print(
                f"  - {lab}: "
                f"precision={m['precision']*100:.2f} "
                f"recall={m['recall']*100:.2f} "
                f"f1={m['f1']*100:.2f} "
                f"support={m['support']}"
            )

    print("\nSaving confusion-matrix figures...")
    for gk, stats in cls_by_group.items():
        if gk != "overall" and stats.total < MIN_GROUP_SUPPORT:
            continue
        name = _safe_name(gk)
        cm_path = out_dir / f"{run_id}_confmat_{name}.png"
        save_confusion_matrix_png(
            stats,
            cm_path,
            title=gk,
            labels=("pt-BR", "pt-PT"),
            cmap="YlGnBu",
        )
        print(f"  - {gk}: {cm_path}")


if __name__ == "__main__":
    main()
