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
    DataCollatorForLanguageModeling,
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
    MAX_STEPS = 2_000
    USE_CPU = False

    VAL_MAX_ROWS = 2_000
else:  # "cluster_full"
    MAX_LENGTH = 512
    TRAIN_LIMIT = None
    PER_DEVICE_BATCH = 10
    GRAD_ACCUM = 4
    MAX_STEPS = 900_000
    USE_CPU = False

    VAL_MAX_ROWS = 100_000    # upper cap for validation size

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
    split: Optional[str] = None,
    batch_size: int = 10_000,
    limit: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream rows from train_data or test_data (views in subs_project.duckdb).

    If `split` is not None, only rows with that split (case-insensitive)
    are returned (e.g. 'train', 'valid', 'test').

    We keep the unified row structure:
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


def stream_training_rows_from_project(
    train_limit: Optional[int] = None,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    """
    Training instance:
      - All rows from train_data with split='train'.
      - Includes OpenSubs, PtBrVarId train, FRMT train, etc.
    """
    print("Streaming rows from train_data (split=train)...")
    yield from stream_from_view(
        view_name="train_data",
        split="train",
        batch_size=batch_size,
        limit=train_limit,
    )


def stream_valid_rows_from_project(
    valid_limit: Optional[int] = None,
    batch_size: int = 10_000,
) -> Iterator[Dict]:
    """
    Validation instance:
      - All rows from train_data with split='valid'.
      - Includes PtBrVarId valid, FRMT valid (1% of dev), etc..
    """
    print("Streaming rows from train_data (split=valid)...")
    yield from stream_from_view(
        view_name="train_data",
        split="valid",
        batch_size=batch_size,
        limit=valid_limit,
    )


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

    # --- build training IterableDataset (split='train') ---
    def duckdb_row_generator():
        """
        Stream rows from train_data (split=train) and occasionally print which
        dataset a row is coming from, just for monitoring.
        """
        for i, row in enumerate(
            stream_training_rows_from_project(
                train_limit=TRAIN_LIMIT,
                batch_size=10_000,
            )
        ):
            # print with a small probability, e.g. 0.001 = 0.1% of rows
            if random.random() < 0.001:
                print(f"[stream train] row {i} from dataset={row.get('dataset')}, split={row.get('split')}")
            yield row

    raw_train = IterableDataset.from_generator(duckdb_row_generator)

    # --- build validation dataset in memory (split='valid' from train_data) ---
    val_rows = []
    for i, row in enumerate(
        stream_valid_rows_from_project(
            valid_limit=None,
            batch_size=10_000,
        )
    ):
        val_rows.append(row)
        if len(val_rows) >= VAL_MAX_ROWS:
            break

    val_df = pd.DataFrame(val_rows)
    print(f"Validation rows collected (split='valid' from train_data): {len(val_df)}")
    if not val_df.empty and "dataset" in val_df.columns:
        print("Validation rows by dataset:\n", val_df["dataset"].value_counts())

    eval_ds = Dataset.from_pandas(val_df, preserve_index=False)

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
            * One translation direction per row (randomly chosen, BR→PT or PT→BR).
            * Classification for both sides if present.
        - PtBrVarId:
            * Classification only, using the explicit 'label' column.
        - FRMT / Gold / others:
            * One translation direction per row if both sides exist (randomly chosen).
            * Classification for all available sides.
        """
        examples = []

        dataset = row["dataset"]
        src_br  = row["text_pt_br"]
        src_pt  = row["text_pt_pt"]   # for Gold, this is manual EP
        label   = row["label"]

        # --- 1) OpenSubs: one translation direction + classification from both columns ---
        if dataset == "OpenSubs":
            if src_br and src_pt:
                r = random.random()
                if r > 0.3:
                    # PT -> BR translation
                    examples.append({
                        "task": "translate_pt2br",
                        "source": src_pt,
                        "target": src_br,
                    })
                else:
                    # BR -> PT translation
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

        # a) one translation direction if both sides exist
        if has_br and has_pt:
            r = random.random()
            if r > 0.3:
                # PT -> BR translation
                examples.append({
                    "task": "translate_pt2br",
                    "source": src_pt,
                    "target": src_br,
                })
            else:
                # BR -> PT translation
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
            # No examples generated for this batch
            return {"input_ids": [], "attention_mask": []}

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

        # Dynamic padding: no padding here, just truncation
        tokenized = tok(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        # No labels here – the collator will create them (copy of input_ids with pads masked as -100)
        return tokenized

    # tokenized train/eval
    train_tokenized = raw_train.map(
        preprocess_batch,
        batched=True,
        remove_columns=[
            "dataset", "split", "source", "bucket", "theme",
            "label", "text_pt_br", "text_pt_pt",
            "ref_pt_pt_manual", "ref_pt_pt_deepl",
        ],
    )

    eval_tokenized = eval_ds.map(
        preprocess_batch,
        batched=True,
        remove_columns=[
            "dataset", "split", "source", "bucket", "theme",
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

    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tok,
    mlm=False,
)

    # ---- Disable PEFT model-card writing to avoid PermissionError ----
    def _no_model_card(self, output_dir: str):
        return
    model_lora.create_or_update_model_card = MethodType(_no_model_card, model_lora)

    # -------------------------
    # TrainingArguments & Trainer
    # -------------------------

    # -------------------------
    # TrainingArguments & Trainer
    # -------------------------

    # use a writable base dir for checkpoints
    BASE_OUT = pathlib.Path.home() / "models"
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    # where all checkpoints for this run mode live
    checkpoint_root = BASE_OUT / f"qwen3-0_6b-ptbr-ptpt-lora-{RUN_MODE}"

    training_args = TrainingArguments(
        output_dir=str(checkpoint_root),

        # core training
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        dataloader_num_workers=32,                      
        dataloader_pin_memory=True,
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
        eval_steps=10_000,  # how often to run on the fixed val set

        # saving
        save_strategy="steps",
        save_steps=100_000,
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
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=50)],
    )

    # --- resume from latest checkpoint if it exists ---
    latest_ckpt = None
    if checkpoint_root.exists():
        ckpts = sorted(
            [p for p in checkpoint_root.glob("checkpoint-*") if p.is_dir()],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if ckpts:
            latest_ckpt = ckpts[-1]
            print(f"Resuming training from checkpoint: {latest_ckpt}")
        else:
            print("No checkpoints found in checkpoint_root, starting from scratch.")
    else:
        print("checkpoint_root does not exist yet, starting from scratch.")

    if latest_ckpt is not None:
        trainer.train(resume_from_checkpoint=str(latest_ckpt))
    else:
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
