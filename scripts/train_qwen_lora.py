#!/usr/bin/env python
import random
import numpy as np
import torch
import duckdb
import pathlib
import pandas as pd

from typing import Iterator, Dict, Optional
from datasets import Dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# -------------------------
# CONFIG
# -------------------------

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 123

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

PROJECT_DB_PATH = BASE_DIR / "data" / "duckdb" / "subs_project.duckdb"
SOURCE_DB_PATH  = BASE_DIR / "data" / "duckdb" / "subs.duckdb"

# change to "cluster_full" on HLT
RUN_MODE = "local_debug"   # "local_debug" or "cluster_full"

if RUN_MODE == "local_debug":
    # short + tiny run just to make sure everything works
    MAX_LENGTH = 512            # shorter context to save memory
    OPUS_LIMIT = 50_000         # or even 10_000 if needed
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 1
    MAX_STEPS = 200             # tiny training run
    USE_CPU = False             # set True if GPU still OOM and you just want to test
else:  # "cluster_full"
    MAX_LENGTH = 2048           # your real context length
    OPUS_LIMIT = None           # full OPUS
    PER_DEVICE_BATCH = 4        # per GPU
    GRAD_ACCUM = 8              # effective batch = 32 sequences
    MAX_STEPS = 8000            # or whatever you and your advisor want
    USE_CPU = False             # we want GPUs on HLT

pd.set_option("display.max_colwidth", 180)


# -------------------------
# DUCKDB STREAMING HELPERS
# -------------------------

def find_ptbrvarid_table(con: duckdb.DuckDBPyConnection) -> str:
    """
    Try to find the PtBrVarId base table in subs.duckdb.
    Adjust this if your table name is different.
    """
    cand_df = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema='main' AND lower(table_name) LIKE 'ptbrvarid%'
    """).df()
    if cand_df.empty:
        raise RuntimeError("Could not find a PtBrVarId table in subs.duckdb (name like 'ptbrvarid%').")
    base = cand_df[cand_df.table_type == "BASE TABLE"]
    if not base.empty:
        return base.table_name.iloc[0]
    return cand_df.table_name.iloc[0]


def stream_opus(
    db_path: str,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """
    OPUS / OpenSubs: parallel pt-BR / pt-PT pairs.
    """
    con = duckdb.connect(db_path, read_only=True)

    base_query = """
        SELECT
            sent_pt_br AS text_pt_br,
            sent_pt_pt AS text_pt_pt
        FROM opus_moses_filtered
    """

    offset = 0
    while True:
        if limit is not None:
            remaining = limit - offset
            if remaining <= 0:
                break
            cur_batch = min(batch_size, remaining)
        else:
            cur_batch = batch_size

        batch = con.execute(
            base_query + f" LIMIT {cur_batch} OFFSET {offset}"
        ).df()

        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield {
                "dataset": "OpenSubs",
                "source": "opus_moses_filtered",
                "bucket": "n/a",
                "theme": "n/a",
                "label": None,
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
                "ref_pt_pt_manual": None,
                "ref_pt_pt_deepl": None,
            }

        offset += len(batch)

    con.close()


def stream_ptbrvarid_split(
    db_path: str,
    split_kinds=("train", "validation"),
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    """
    Stream PtBrVarId rows for given splits from subs.duckdb.
    - train instance: split_kinds = ('train', 'validation')
    - test instance : split_kinds = ('test',)
    """
    con = duckdb.connect(db_path, read_only=True)

    ptbr_table = find_ptbrvarid_table(con)
    print(f"Using PtBrVarId table: {ptbr_table} | splits: {split_kinds}")

    split_list = ",".join(f"'{s}'" for s in split_kinds)
    base_query = f"""
        SELECT
            dataset,
            domain,
            split,
            label,
            text_pt_br,
            text_pt_pt
        FROM {ptbr_table}
        WHERE dataset = 'PtBrVId'
          AND lower(split) IN ({split_list})
    """

    offset = 0
    while True:
        batch = con.execute(
            base_query + f" LIMIT {batch_size} OFFSET {offset}"
        ).df()

        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield {
                "dataset": "PtBrVarId",
                "source": "liaad/PtBrVId",
                "bucket": row.get("domain", "n/a") or "n/a",
                "theme": "n/a",
                "label": row["label"],       # 'pt-BR' or 'pt-PT'
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
                "ref_pt_pt_manual": None,
                "ref_pt_pt_deepl": None,
            }

        offset += len(batch)

    con.close()


def stream_ptbrvarid_train(db_path: str, batch_size: int = 10_000) -> Iterator[Dict]:
    return stream_ptbrvarid_split(db_path, split_kinds=("train", "validation"), batch_size=batch_size)


def stream_ptbrvarid_test(db_path: str, batch_size: int = 10_000) -> Iterator[Dict]:
    return stream_ptbrvarid_split(db_path, split_kinds=("test",), batch_size=batch_size)


def stream_frmt_dev(
    db_path: str,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    con = duckdb.connect(db_path, read_only=True)

    base_query = """
        SELECT bucket, text_pt_br, text_pt_pt
        FROM frmt_dev
    """

    offset = 0
    while True:
        batch = con.execute(
            base_query + f" LIMIT {batch_size} OFFSET {offset}"
        ).df()

        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield {
                "dataset": "FRMT",
                "source": "google-research/frmt",
                "bucket": row.get("bucket", "n/a") or "n/a",
                "theme": "n/a",
                "label": None,
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
                "ref_pt_pt_manual": None,
                "ref_pt_pt_deepl": None,
            }

        offset += len(batch)

    con.close()


def stream_frmt_test(
    db_path: str,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    con = duckdb.connect(db_path, read_only=True)

    base_query = """
        SELECT bucket, text_pt_br, text_pt_pt
        FROM frmt_test
    """

    offset = 0
    while True:
        batch = con.execute(
            base_query + f" LIMIT {batch_size} OFFSET {offset}"
        ).df()

        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield {
                "dataset": "FRMT",
                "source": "google-research/frmt",
                "bucket": row.get("bucket", "n/a") or "n/a",
                "theme": "n/a",
                "label": None,
                "text_pt_br": row["text_pt_br"],
                "text_pt_pt": row["text_pt_pt"],
                "ref_pt_pt_manual": None,
                "ref_pt_pt_deepl": None,
            }

        offset += len(batch)

    con.close()


def stream_gold_test(
    db_path: str,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    con = duckdb.connect(db_path, read_only=True)

    base_query = """
        SELECT bucket, theme, text_pt_br, ref_pt_pt_manual, ref_pt_pt_deepl
        FROM gold_test
    """

    offset = 0
    while True:
        batch = con.execute(
            base_query + f" LIMIT {batch_size} OFFSET {offset}"
        ).df()

        if batch.empty:
            break

        for _, row in batch.iterrows():
            yield {
                "dataset": "Gold",
                "source": "joaosanches/golden_collection",
                "bucket": row.get("bucket", "n/a") or "n/a",
                "theme": row.get("theme", "n/a") or "n/a",
                "label": None,
                "text_pt_br": row["text_pt_br"],              # pt-BR
                "text_pt_pt": row["ref_pt_pt_manual"],        # treat manual EP as pt-PT text
                "ref_pt_pt_manual": row["ref_pt_pt_manual"],
                "ref_pt_pt_deepl": row["ref_pt_pt_deepl"],
            }

        offset += len(batch)

    con.close()


def stream_training_rows(
    opus_limit: Optional[int] = None,
    opus_batch: int = 10_000,
    ptbr_batch: int = 10_000,
    frmt_batch: int = 10_000,
) -> Iterator[Dict]:
    """
    Train instance:
      - OPUS (translation + classification)
      - PtBrVarId train+validation (classification only, since no parallel)
      - FRMT dev (translation + classification when both sides exist)
    """
    src_path = SOURCE_DB_PATH.as_posix()
    proj_path = PROJECT_DB_PATH.as_posix()

    print("Streaming OPUS/OpenSubs (train)...")
    for row in stream_opus(src_path, batch_size=opus_batch, limit=opus_limit):
        yield row

    print("Streaming PtBrVarId train+val (train)...")
    for row in stream_ptbrvarid_train(src_path, batch_size=ptbr_batch):
        yield row

    print("Streaming FRMT dev (train)...")
    for row in stream_frmt_dev(proj_path, batch_size=frmt_batch):
        yield row


def stream_test_rows(
    ptbr_batch: int = 10_000,
    frmt_batch: int = 10_000,
    gold_batch: int = 10_000,
) -> Iterator[Dict]:
    """
    Test instance:
      - PtBrVarId test
      - FRMT test
      - Gold Collection
    """
    src_path = SOURCE_DB_PATH.as_posix()
    proj_path = PROJECT_DB_PATH.as_posix()

    print("Streaming PtBrVarId test (eval)...")
    for row in stream_ptbrvarid_test(src_path, batch_size=ptbr_batch):
        yield row

    print("Streaming FRMT test (eval)...")
    for row in stream_frmt_test(proj_path, batch_size=frmt_batch):
        yield row

    print("Streaming Gold Collection (eval)...")
    for row in stream_gold_test(proj_path, batch_size=gold_batch):
        yield row


# -------------------------
# MAIN
# -------------------------

def main():
    # --- seeds ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- build training IterableDataset (streaming) ---
    def duckdb_row_generator():
        yield from stream_training_rows(
            opus_limit=OPUS_LIMIT,
            opus_batch=10_000,
            ptbr_batch=10_000,
            frmt_batch=10_000,
        )

    raw_train = IterableDataset.from_generator(duckdb_row_generator)

    # --- build eval dataset in memory ---
    eval_rows = []
    for i, row in enumerate(stream_test_rows()):
        eval_rows.append(row)
        if i >= 10_000:   # cap for practicality
            break

    eval_df = pd.DataFrame(eval_rows)
    print(f"Eval rows collected: {len(eval_df)}")
    eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)

    # -------------------------
    # Tokenizer & Base Model
    # -------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and not USE_CPU else torch.float32,
        device_map="auto" if not USE_CPU else None,
        trust_remote_code=True,
    )

    if USE_CPU:
        model.to("cpu")

    # -------------------------
    # Prompts
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
    # Example builder
    # -------------------------
    def build_examples_from_row(row: Dict):
        """
        Return a *list* of examples for this row.
        Each example is a dict: {"task": ..., "source": ..., "target": ...}

        - OPUS (dataset == 'OpenSubs'):
            * Always has both texts -> translation + classification for each side.
        - PtBrVarId:
            * Only classification, using the explicit 'label' column.
            * No translation (no rows with two texts).
        - FRMT:
            * If both text_pt_br and text_pt_pt exist -> translation + classification (both sides).
            * Otherwise -> classification only for whichever side exists.
        - Gold:
            * We treat text_pt_br as pt-BR, ref_pt_pt_manual as pt-PT.
            * If both exist -> translation (BR->PT) + classification for both sides.
            * Otherwise -> classification only for available side(s).
        """
        examples = []

        dataset = row["dataset"]
        src_br = row["text_pt_br"]
        src_pt = row["text_pt_pt"]   # for Gold, this is ref_pt_pt_manual
        label = row["label"]

        # --- 1) OPUS: translation + classification from both columns ---
        if dataset == "OpenSubs":
            if src_br and src_pt:
                # translation
                examples.append({
                    "task": "translate_br2pt",
                    "source": src_br,
                    "target": src_pt,
                })
                # classification BR
                examples.append({
                    "task": "classify",
                    "source": src_br,
                    "target": "pt-BR",
                })
                # classification PT
                examples.append({
                    "task": "classify",
                    "source": src_pt,
                    "target": "pt-PT",
                })
            return examples

        # --- 2) PtBrVarId: classification only ---
        if dataset == "PtBrVarId":
            if label in ("pt-BR", "pt-PT"):
                # choose whichever text exists (you said only one per row)
                text = src_br if src_br else src_pt
                if text:
                    examples.append({
                        "task": "classify",
                        "source": text,
                        "target": label,
                    })
            return examples

        # --- 3) FRMT / Gold / others: generic rule ---
        has_br = bool(src_br)
        has_pt = bool(src_pt)

        # a) translation if both sides exist
        if has_br and has_pt:
            examples.append({
                "task": "translate_br2pt",
                "source": src_br,
                "target": src_pt,
            })

        # b) classification for all available texts
        if has_br:
            examples.append({
                "task": "classify",
                "source": src_br,
                "target": "pt-BR",
            })
        if has_pt:
            examples.append({
                "task": "classify",
                "source": src_pt,
                "target": "pt-PT",
            })

        return examples

    # -------------------------
    # Preprocess / tokenization
    # -------------------------
    def preprocess_batch(batch):
        examples = []
        for i in range(len(batch["dataset"])):
            row = {k: batch[k][i] for k in batch}
            row_examples = build_examples_from_row(row)
            examples.extend(row_examples)

        if not examples:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        texts = []
        for ex in examples:
            task = ex["task"]
            source = ex["source"]
            target = ex["target"]

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
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target},
            ]
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        tokenized = tok(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
        )

        labels = tokenized["input_ids"].copy()
        pad_id = tok.pad_token_id
        labels = [
            [(tid if tid != pad_id else -100) for tid in seq]
            for seq in labels
        ]

        tokenized["labels"] = labels
        return tokenized

    # tokenized train/eval
    train_tokenized = raw_train.map(
        preprocess_batch,
        batched=True,
        remove_columns=[
            "dataset", "source", "bucket", "theme",
            "label", "text_pt_br", "text_pt_pt",
            "ref_pt_pt_manual", "ref_pt_pt_deepl",
        ],
    )

    eval_tokenized = eval_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=[
            "dataset", "source", "bucket", "theme",
            "label", "text_pt_br", "text_pt_pt",
            "ref_pt_pt_manual", "ref_pt_pt_deepl",
        ],
    )

    # -------------------------
    # LoRA
    # -------------------------
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model_lora = get_peft_model(model, lora_config)
    model_lora.print_trainable_parameters()

    # -------------------------
    # TrainingArguments & Trainer
    # -------------------------
    training_args = TrainingArguments(
        output_dir=f"./qwen3-0_6b-ptbr-ptpt-lora-{RUN_MODE}",

        # core training
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=200,
        max_steps=MAX_STEPS,

        # logging
        logging_strategy="steps",
        logging_steps=50,

        # evaluation
        do_eval=True,
        eval_strategy="steps",
        eval_steps=1000,

        # saving
        save_strategy="steps",
        save_steps=1000,

        # precision & device
        bf16=torch.cuda.is_available() and not USE_CPU,
        fp16=False,
        use_cpu=USE_CPU,

        # important for our custom preprocess_batch
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
    )

    trainer.train()
    trainer.evaluate()

    adapter_dir = f"./qwen3-0_6b-ptbr-ptpt-lora-adapter-{RUN_MODE}"
    trainer.save_model(adapter_dir)
    tok.save_pretrained(f"{adapter_dir}-tokenizer")
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
