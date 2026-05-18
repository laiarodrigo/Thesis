# scripts/encoder_decoder/multitask/task_protocol.py
from __future__ import annotations

import re

TRANSLATE_BR2PT_PROMPT = "BR"
TRANSLATE_PT2BR_PROMPT = "PT"
CLASSIFY_PTBR_TOKEN = "BR"
CLASSIFY_PTPT_TOKEN = "PT"
CLASSIFY_EQUAL_TOKEN = "igual"

# Legacy aliases kept so older imports keep working while the protocol moves
# from decoder-side special tokens to existing-vocabulary prompt tokens.
TR_BR2PT = TRANSLATE_BR2PT_PROMPT
TR_PT2BR = TRANSLATE_PT2BR_PROMPT
CLS = ""
LBL_PTBR = CLASSIFY_PTBR_TOKEN
LBL_PTPT = CLASSIFY_PTPT_TOKEN
LBL_EQUAL = CLASSIFY_EQUAL_TOKEN

CLASS_LABELS = {"pt-br", "pt-pt", "equal"}
CLASS_LABEL_TO_TOKEN = {
    "pt-br": CLASSIFY_PTBR_TOKEN,
    "pt-pt": CLASSIFY_PTPT_TOKEN,
    "equal": CLASSIFY_EQUAL_TOKEN,
}
VALID_TRANSLATION_PROMPTS = frozenset(
    {TRANSLATE_BR2PT_PROMPT, TRANSLATE_PT2BR_PROMPT}
)
VALID_CLASS_PAYLOADS = frozenset(
    CLASS_LABELS
    | set(CLASS_LABEL_TO_TOKEN.values())
    | {token.lower() for token in CLASS_LABEL_TO_TOKEN.values()}
)
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def strip_encoder_prefix(text: str) -> tuple[str | None, str]:
    m = ENCODER_TASK_PREFIX_RE.match(text or "")
    if not m:
        return None, normalize_space(text or "")
    prefix = m.group(1).strip().lower()
    clean = normalize_space((text or "")[m.end() :])
    return prefix, clean


def map_encoder_prefix_to_translation_prompt(prefix: str) -> str | None:
    # Legacy source-side format: <br-pt> / <pt-br> / <id>
    if prefix == "br-pt":
        return TRANSLATE_BR2PT_PROMPT
    if prefix == "pt-br":
        return TRANSLATE_PT2BR_PROMPT
    if prefix == "id":
        return None
    raise ValueError(f"Unknown encoder prefix: {prefix}")


def map_task_to_translation_prompt(task: str) -> str | None:
    raw = normalize_space(task).lower()
    if raw in {"translate_br2pt", "br-pt", TRANSLATE_BR2PT_PROMPT.lower()}:
        return TRANSLATE_BR2PT_PROMPT
    if raw in {"translate_pt2br", "pt-br", TRANSLATE_PT2BR_PROMPT.lower()}:
        return TRANSLATE_PT2BR_PROMPT
    return None


def build_prompted_input(prompt: str | None, text: str) -> str:
    clean = normalize_space(text)
    if not prompt:
        return clean
    if not clean:
        return prompt
    return f"{prompt} {clean}"


def build_translation_input_from_encoder_prefix(prefix: str, text: str) -> str:
    return build_prompted_input(map_encoder_prefix_to_translation_prompt(prefix), text)


def split_translation_prompt(text: str) -> tuple[str | None, str]:
    clean = normalize_space(text)
    if not clean:
        return None, ""
    first, sep, rest = clean.partition(" ")
    if first in VALID_TRANSLATION_PROMPTS:
        return first, rest.strip()
    return None, clean


def normalize_class_label(label: str) -> str:
    v = normalize_space(label).lower()
    if v not in CLASS_LABELS:
        raise ValueError(f"Invalid class label: {label}")
    return v


def class_label_to_decoder_payload(label: str) -> str:
    return CLASS_LABEL_TO_TOKEN[normalize_class_label(label)]


def decoder_payload_to_class_label(payload: str) -> str | None:
    raw = normalize_space(payload).lstrip("▁")
    if not raw:
        return None
    low = raw.lower()
    if low in CLASS_LABELS:
        return low
    if raw.upper() == CLASSIFY_PTBR_TOKEN:
        return "pt-br"
    if raw.upper() == CLASSIFY_PTPT_TOKEN:
        return "pt-pt"
    if low == CLASSIFY_EQUAL_TOKEN.lower():
        return "equal"
    return None
