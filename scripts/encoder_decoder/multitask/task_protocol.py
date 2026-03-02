# scripts/encoder_decoder/multitask/task_protocol.py
from __future__ import annotations
import re

TR_BR2PT = "<TR_BR2PT>"
TR_PT2BR = "<TR_PT2BR>"
CLS = "<CLS>"

CLASS_LABELS = {"pt-br", "pt-pt", "equal"}
ENCODER_TASK_PREFIX_RE = re.compile(r"^\s*<([^>]+)>\s*")

def strip_encoder_prefix(text: str) -> tuple[str | None, str]:
    m = ENCODER_TASK_PREFIX_RE.match(text or "")
    if not m:
        return None, (text or "").strip()
    prefix = m.group(1).strip().lower()
    clean = (text or "")[m.end():].strip()
    return prefix, clean

def map_encoder_prefix_to_decoder_task(prefix: str) -> str:
    # existing format: <br-pt> / <pt-br> / <id>
    if prefix == "br-pt":
        return TR_BR2PT
    if prefix == "pt-br":
        return TR_PT2BR
    if prefix == "id":
        return CLS
    raise ValueError(f"Unknown encoder prefix: {prefix}")

def build_decoder_target(task_token: str, payload: str) -> str:
    payload = (payload or "").strip()
    if not payload:
        raise ValueError("Empty payload for decoder target")
    return f"{task_token} {payload}"

def normalize_class_label(label: str) -> str:
    v = (label or "").strip().lower()
    if v not in CLASS_LABELS:
        raise ValueError(f"Invalid class label: {label}")
    return v
