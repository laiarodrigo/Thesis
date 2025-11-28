import os
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
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from types import MethodType


# -------------------------
# CONFIG
# -------------------------

MODEL_ID = "Qwen/Qwen3-0.6B"
SEED = 123

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # repo root (one level above scripts/)
PROJECT_DB_PATH = BASE_DIR / "data" / "duckdb" / "subs_project.duckdb"
SOURCE_DB_PATH  = BASE_DIR / "data" / "duckdb" / "subs.duckdb"

PROJECT_DB_STR = PROJECT_DB_PATH.as_posix()
SOURCE_DB_STR  = SOURCE_DB_PATH.as_posix()

RUN_MODE = "cluster_full"   # "local_debug" or "cluster_full"

if RUN_MODE == "local_debug":
    MAX_LENGTH = 512
    TRAIN_LIMIT = 50_000
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 1
    MAX_STEPS = 2000
    USE_CPU = False

    VAL_MAX_ROWS = 2_000
else:  # "cluster_full"
    MAX_LENGTH = 2048
    TRAIN_LIMIT = None
    PER_DEVICE_BATCH = 2
    GRAD_ACCUM = 4
    MAX_STEPS = 900_000
    USE_CPU = False

    VAL_MAX_ROWS = 100_000    # <= fixed cap for validation size


pd.set_option("display.max_colwidth", 180)


# -------------------------
# DUCKDB: connect + streaming from VIEWS
# -------------------------

def connect_project(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Connect to subs_project.duckdb and ensure the 'src' catalog is attached.
    The views train_data/test_data may reference src.main.*.
    """
    con = duckdb.connect(PROJECT_DB_STR, read_only=read_only)
    dbl = con.execute("PRAGMA database_list").df()
    if not (dbl["name"] == "src").any():
        con.execute(f"ATTACH '{SOURCE_DB_STR}' AS src")
    return con


def stream_from_view(
    view_name: str,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream rows from train_data or test_data (views in subs_project.duckdb).
    We keep the unified row structure: dataset, source, bucket, theme, label,
    text_pt_br, text_pt_pt, ref_pt_pt_manual, ref_pt_pt_deepl.
    """
    con = connect_project(read_only=True)
    offset = 0

    base_query = f"""
        SELECT
          dataset, source, bucket, theme, label,
          text_pt_br, text_pt_pt,
          ref_pt_pt_manual, ref_pt_pt_deepl
        FROM {view_name}
    """

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


def stream_training_rows_from_project(
    train_limit: Optional[int] = None,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    """
    Training instance:
      - All rows from main.train_data view (which already merges OPUS, PtBrVarId, FRMT dev, Gold as you defined).
    """
    print("Streaming rows from train_data...")
    yield from stream_from_view(
        view_name="train_data",
        batch_size=batch_size,
        limit=train_limit,
    )


def stream_test_rows_from_project(
    test_limit: Optional[int] = None,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    """
    Test instance:
      - All rows from main.test_data view (PtBrVarId test, FRMT test, Gold, etc.).
    NOTE: we no longer use this during training-time evaluation; it's reserved for final test.
    """
    print("Streaming rows from test_data...")
    yield from stream_from_view(
        view_name="test_data",
        batch_size=batch_size,
        limit=test_limit,
    )


# -------------------------
# Helper: build validation dataset by random sampling from train_data
# -------------------------

def build_val_dataset_from_train(
    val_max_rows: int,
    seed: int = SEED,
) -> Dataset:
    """
    Build an in-memory validation Dataset by randomly sampling rows from train_data,
    with different probabilities per dataset (stratified sampling).

    - val_max_rows: upper cap on number of validation rows.
    - seed: for reproducibility.

    NOTE: this does NOT guarantee disjointness from the training stream, because
    we don't have stable IDs to filter them out, but for a large corpus the
    overlap is usually negligible.
    """
    rng = random.Random(seed)
    val_rows = []

    # Adjust these names/probabilities based on your actual `dataset` values
    p_by_dataset = {
        "OpenSubs": 0.02,   # OPUS / subtitles: big, use low prob
        "OPUS": 0.02,       # in case it's named 'OPUS'
        "PtBrVarId": 0.30,  # more weight
        "FRMT": 0.30,       # more weight
        # any other dataset will use default below
    }

    for row in stream_training_rows_from_project(train_limit=None, batch_size=10_000):
        ds = row.get("dataset")
        p = p_by_dataset.get(ds, 0.05)  # default prob for unknown datasets

        if rng.random() < p:
            val_rows.append(row)
            if len(val_rows) >= val_max_rows:
                break

    val_df = pd.DataFrame(val_rows)
    print("Validation rows collected from train_data:", len(val_df))
    if "dataset" in val_df.columns:
        print("Validation rows by dataset:\n", val_df["dataset"].value_counts())

    return Dataset.from_pandas(val_df, preserve_index=False)



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

    # --- build training IterableDataset (streaming from view) ---
    def duckdb_row_generator():
        """
        Stream rows from train_data and occasionally print which dataset
        a row is coming from, just for monitoring.
        """
        for i, row in enumerate(
            stream_training_rows_from_project(
                train_limit=TRAIN_LIMIT,
                batch_size=10_000,
            )
        ):
            # print with a small probability, e.g. 0.001 = 0.1% of rows
            if random.random() < 0.001:
                print(f"[stream] row {i} from dataset={row.get('dataset')}")
            yield row


    raw_train = IterableDataset.from_generator(duckdb_row_generator)

    # --- build validation dataset in memory (sampled from train_data) ---
    eval_ds = build_val_dataset_from_train(
        val_max_rows=VAL_MAX_ROWS,
        seed=SEED,
    )

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

        - OpenSubs (dataset == 'OpenSubs'):
            * Both texts -> translation + classification for each side.
        - PtBrVarId:
            * Only classification, using the explicit 'label' column.
            * No translation (no rows with two texts).
        - FRMT:
            * If both text_pt_br and text_pt_pt exist -> translation + classification (both sides).
            * Otherwise -> classification only for whichever side exists.
        - Gold:
            * We treat text_pt_br as pt-BR, text_pt_pt (manual) as pt-PT.
            * If both exist -> translation (BR->PT) + classification for both sides.
            * Otherwise -> classification only for available side(s).
        """
        examples = []

        dataset = row["dataset"]
        src_br  = row["text_pt_br"]
        src_pt  = row["text_pt_pt"]   # for Gold, this is manual EP
        label   = row["label"]

        # --- 1) OpenSubs: translation + classification from both columns ---
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
            task   = ex["task"]
            source = ex["source"]
            target = ex["target"]

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
                continue

            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": user_content},
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

    # ---- Disable PEFT model-card writing to avoid PermissionError ----
    def _no_model_card(self, output_dir: str):
        return
    model_lora.create_or_update_model_card = MethodType(_no_model_card, model_lora)

    # -------------------------
    # TrainingArguments & Trainer
    # -------------------------

    # use a writable base dir for checkpoints
    BASE_OUT = pathlib.Path.home() / "models"
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BASE_OUT / f"qwen3-0_6b-ptbr-ptpt-lora-{RUN_MODE}"),

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
        evaluation_strategy="steps",
        evaluation_steps=1000,              # how often to run on the fixed val set

        # saving
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        overwrite_output_dir=True,

        # precision & device
        bf16=torch.cuda.is_available() and not USE_CPU,
        fp16=False,
        use_cpu=USE_CPU,

        # important for our custom preprocess_batch
        remove_unused_columns=False,

        # early stopping / best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.evaluate()

    # final adapter save
    adapter_dir = BASE_OUT / f"qwen3-0_6b-ptbr-ptpt-lora-adapter-{RUN_MODE}"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir) + "-tokenizer")
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
