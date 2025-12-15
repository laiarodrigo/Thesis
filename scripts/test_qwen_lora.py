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


# -------------------------
# CONFIG – EDIT THESE
# -------------------------
USE_ADAPTER = False
MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 123

from pathlib import Path

# Assuming same repo layout locally:
#   repo_root/
#     data/duckdb/subs_project.duckdb
#     data/duckdb/subs.duckdb
#     scripts/test_script.py  (this file)
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DB_PATH = BASE_DIR / "data" / "duckdb" / "subs_project.duckdb"
SOURCE_DB_PATH  = BASE_DIR / "data" / "duckdb" / "subs.duckdb"

PROJECT_DB_STR = PROJECT_DB_PATH.as_posix()
SOURCE_DB_STR  = SOURCE_DB_PATH.as_posix()

# >>> EDIT THIS: where you copied your trained adapter
ADAPTER_DIR = Path("/path/to/qwen3-0_6b-ptbr-ptpt-lora-adapter-cluster_full")

# >>> EDIT THIS: where you copied the tokenizer saved in train_script
TOKENIZER_PATH = Path("/path/to/qwen3-0_6b-ptbr-ptpt-lora-adapter-cluster_full-tokenizer")

MAX_LENGTH = 1024

# Use the test_data view here
EVAL_VIEW  = "test_data"
# If your test_data has a split column with 'test', set EVAL_SPLIT = "test".
# If not, leave as None to use all rows in the view.
EVAL_SPLIT = 'test'

EVAL_ROW_LIMIT = 10  # set to an integer for quicker testing, or None for all rows
GEN_BATCH_SIZE   = 8

pd.set_option("display.max_colwidth", 180)


# -------------------------
# DUCKDB: connect + streaming from VIEWS
# -------------------------

def connect_project(
    read_only: bool = True,
    memory_limit: str = "3.5GB",   # cap DuckDB memory
    threads: int = 4,            # number of DuckDB threads
) -> duckdb.DuckDBPyConnection:
    """
    Connect to subs_project.duckdb and ensure the 'src' catalog is attached.
    Also set DuckDB memory + threading limits for this connection.
    """
    con = duckdb.connect(
        PROJECT_DB_STR,
        read_only=read_only,
        config={
            "memory_limit": memory_limit,  # e.g. "512MB", "1GB", "2GB"
            "threads": threads,            # e.g. 2, 4, 8
        },
    )

    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        con.execute(f"ATTACH '{SOURCE_DB_STR}' AS src")

    return con

def stream_from_view(
    view_name: str,
    split: Optional[str] = None,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream rows from train_data or test_data.

    Unified row structure:
      dataset, split, source, bucket, theme, label,
      text_pt_br, text_pt_pt, ref_pt_pt_manual, ref_pt_pt_deepl.
    """
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
    """
    Deterministic examples for evaluation.

    Each example:
      {"task": "translate_br2pt"|"translate_pt2br"|"classify",
       "source": <str>, "target": <str> or <label>, "meta": {...}}
    """
    examples = []

    dataset = row["dataset"]
    src_br  = row["text_pt_br"]
    src_pt  = row["text_pt_pt"]
    label   = row["label"]

    has_br = bool(src_br)
    has_pt = bool(src_pt)

    # --- OpenSubs: two translation directions + classification on both sides ---
    if dataset == "OpenSubs":
        if has_br and has_pt:
            examples.append({
                "task": "translate_br2pt",
                "source": src_br,
                "target": src_pt,
                "meta": {"dataset": dataset, "direction": "br2pt"},
            })
            examples.append({
                "task": "translate_pt2br",
                "source": src_pt,
                "target": src_br,
                "meta": {"dataset": dataset, "direction": "pt2br"},
            })

        if has_br:
            examples.append({
                "task": "classify",
                "source": src_br,
                "target": "pt-BR",
                "meta": {"dataset": dataset, "var": "pt-BR"},
            })
        if has_pt:
            examples.append({
                "task": "classify",
                "source": src_pt,
                "target": "pt-PT",
                "meta": {"dataset": dataset, "var": "pt-PT"},
            })

        return examples

    # --- PtBrVarId: classification only, using explicit label ---
    if dataset == "PtBrVarId":
        if label in ("pt-BR", "pt-PT"):
            text = src_br if src_br else src_pt
            if text:
                examples.append({
                    "task": "classify",
                    "source": text,
                    "target": label,
                    "meta": {"dataset": dataset, "var": label},
                })
        return examples

    # --- FRMT / Gold / others: generic rule ---
    if has_br and has_pt:
        examples.append({
            "task": "translate_br2pt",
            "source": src_br,
            "target": src_pt,
            "meta": {"dataset": dataset, "direction": "br2pt"},
        })
        examples.append({
            "task": "translate_pt2br",
            "source": src_pt,
            "target": src_br,
            "meta": {"dataset": dataset, "direction": "pt2br"},
        })

    if has_br:
        examples.append({
            "task": "classify",
            "source": src_br,
            "target": "pt-BR",
            "meta": {"dataset": dataset, "var": "pt-BR"},
        })
    if has_pt:
        examples.append({
            "task": "classify",
            "source": src_pt,
            "target": "pt-PT",
            "meta": {"dataset": dataset, "var": "pt-PT"},
        })

    return examples


def build_prompt_from_example(ex: Dict, tok) -> str:
    """Create the chat-style prompt used for generation (no assistant turn)."""
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
    prompt_str = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_str


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
        # ---- use the fine-tuned tokenizer you saved ----
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), use_fast=True)
    else:
        print("WARNING: Not using adapter; loading base model and tokenizer only.")
        # ---- just use the original tokenizer from MODEL_ID ----
        tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # base model (always needed)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if USE_ADAPTER:
        # ---- load LoRA adapter on top ----
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    else:
        # ---- no adapter: just use the base model ----
        model = base_model

    print("Model loaded.")
    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")

    return model, tok

# -------------------------
# Generation helper
# -------------------------

def generate_batch(
    model,
    tok,
    prompts: List[str],
) -> List[str]:
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
        )

    preds = []
    for i in range(out.size(0)):
        # left padding: number of non-pad tokens is the prompt length
        prompt_len = int(enc["attention_mask"][i].sum().item())
        gen_ids = out[i, prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        preds.append(text)
    return preds


# -------------------------
# Classification label parsing
# -------------------------

def normalize_class_prediction(text: str) -> Optional[str]:
    """
    Map raw model text to 'pt-BR' or 'pt-PT' when possible.
    Very simple heuristics aligned with your instruction.
    """
    t = text.lower().strip()

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
# MAIN EVAL
# -------------------------

def main():
    model, tok = load_model_and_tokenizer()

    # For BLEU: keep global, and also per-direction BR→PT / PT→BR
    translation_refs_all = []
    translation_hyps_all = []

    br2pt_refs = []
    br2pt_hyps = []
    pt2br_refs = []
    pt2br_hyps = []

    # For classification
    class_gold = []
    class_pred = []

    all_examples = []

    print("Building eval examples...")
    # 1) build eval examples from test_data
    for row in stream_from_view(
        EVAL_VIEW,
        split=EVAL_SPLIT,
        batch_size=10_000,
        limit=EVAL_ROW_LIMIT,
    ):
        print(f"Processing row from dataset={row['dataset']}, split={row['split']}")
        all_examples.extend(build_eval_examples_from_row(row))

    print(f"Total eval examples: {len(all_examples)}")

    # 2) loop over examples in generation batches
    for i in range(0, len(all_examples), GEN_BATCH_SIZE):
        batch = all_examples[i : i + GEN_BATCH_SIZE]
        prompts = [build_prompt_from_example(ex, tok) for ex in batch]
        preds = generate_batch(model, tok, prompts)

        for ex, pred in zip(batch, preds):
            if ex["task"].startswith("translate"):
                ref = ex["target"]
                translation_refs_all.append(ref)
                translation_hyps_all.append(pred)

                direction = ex["meta"].get("direction")
                if direction == "br2pt":
                    br2pt_refs.append(ref)
                    br2pt_hyps.append(pred)
                elif direction == "pt2br":
                    pt2br_refs.append(ref)
                    pt2br_hyps.append(pred)

            elif ex["task"] == "classify":
                gold = ex["target"]
                norm = normalize_class_prediction(pred)
                if norm is None:
                    norm = "UNKNOWN"
                class_gold.append(gold)
                class_pred.append(norm)

    # 3) BLEU for translation
    if translation_refs_all:
        bleu_all = corpus_bleu(translation_hyps_all, [translation_refs_all]).score
        print(f"\nTranslation BLEU (all directions combined): {bleu_all:.2f}")

        if br2pt_refs:
            bleu_br2pt = corpus_bleu(br2pt_hyps, [br2pt_refs]).score
            print(f"  BR → PT BLEU: {bleu_br2pt:.2f}")

        if pt2br_refs:
            bleu_pt2br = corpus_bleu(pt2br_hyps, [pt2br_refs]).score
            print(f"  PT → BR BLEU: {bleu_pt2br:.2f}")
    else:
        print("\nNo translation examples found in test_data.")

    # 4) Accuracy for classification
    if class_gold:
        total = len(class_gold)
        correct = sum(1 for g, p in zip(class_gold, class_pred) if g == p)
        acc = correct / total
        print(f"\nClassification accuracy: {acc*100:.2f}% ({correct}/{total})")
    else:
        print("\nNo classification examples found in test_data.")


if __name__ == "__main__":
    main()
