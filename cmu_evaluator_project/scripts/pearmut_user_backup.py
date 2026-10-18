#!/usr/bin/env python3
"""Build a completion-only, per-user Pearmut backup without private mappings."""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
COMPLETED_VALUES = {"completed", "completed_foreign"}


class UserBackupError(Exception):
    """Base class for safe user-backup failures."""


class UserBackupAuthorizationError(UserBackupError):
    """The supplied capability is not valid for this campaign and user."""


class UserBackupNotFoundError(UserBackupError):
    """The requested public campaign assignment does not exist."""


class UserBackupIncompleteError(UserBackupError):
    """The user has not completed the assignment."""


class UserBackupStateError(UserBackupError):
    """The authoritative persistent state is missing or malformed."""


@dataclass(frozen=True)
class UserBackupArchive:
    filename: str
    content: bytes
    metadata: dict[str, Any]
    raw_rows: tuple[dict[str, Any], ...]


def _validated_id(label: str, value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise UserBackupAuthorizationError(f"Invalid {label}")
    return value


def _signing_key_bytes(signing_key: str) -> bytes:
    value = signing_key.encode("utf-8")
    if len(value) < 32:
        raise UserBackupStateError(
            "PEARMUT_BACKUP_SIGNING_KEY must contain at least 32 characters"
        )
    return value


def make_backup_token(signing_key: str, campaign_id: str, user_id: str) -> str:
    campaign = _validated_id("campaign ID", campaign_id)
    user = _validated_id("user ID", user_id)
    message = f"pearmut-user-backup-v1\0{campaign}\0{user}".encode("utf-8")
    digest = hmac.new(_signing_key_bytes(signing_key), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def backup_token_is_valid(
    signing_key: str, campaign_id: str, user_id: str, supplied_token: str
) -> bool:
    try:
        expected = make_backup_token(signing_key, campaign_id, user_id)
    except UserBackupError:
        return False
    return hmac.compare_digest(expected, supplied_token)


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise UserBackupNotFoundError(f"Unknown {description}") from error
    except json.JSONDecodeError as error:
        raise UserBackupStateError(f"Malformed persistent {description}") from error


def _progress_is_complete(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value) and all(_progress_is_complete(child) for child in value)
    if isinstance(value, dict):
        return bool(value) and all(_progress_is_complete(child) for child in value.values())
    return value in COMPLETED_VALUES


def _passage_ids_for_user(campaign: dict[str, Any], user_id: str) -> list[str]:
    users = campaign.get("info", {}).get("users", [])
    data = campaign.get("data", [])
    if isinstance(data, dict):
        task = data.get(user_id)
    else:
        try:
            task = data[users.index(user_id)]
        except (ValueError, IndexError, TypeError):
            task = None
    if not isinstance(task, list):
        raise UserBackupNotFoundError("Unknown user assignment")

    passage_ids: list[str] = []
    for document in task:
        if not isinstance(document, list):
            continue
        for segment in document:
            if not isinstance(segment, dict):
                continue
            raw_item_id = segment.get("item_id")
            try:
                item_id = json.loads(raw_item_id) if isinstance(raw_item_id, str) else raw_item_id
            except json.JSONDecodeError:
                continue
            if isinstance(item_id, list) and len(item_id) >= 2:
                passage_id = item_id[1]
                if (
                    isinstance(passage_id, str)
                    and SAFE_ID.fullmatch(passage_id)
                    and passage_id not in passage_ids
                ):
                    passage_ids.append(passage_id)

    if not passage_ids:
        public_ids = campaign.get("metadata", {}).get("passage_ids", [])
        if isinstance(public_ids, list):
            passage_ids = [
                value
                for value in public_ids
                if isinstance(value, str) and SAFE_ID.fullmatch(value)
            ]
    return passage_ids


def _assignment_id(campaign: dict[str, Any], user_id: str) -> str:
    metadata = campaign.get("metadata", {})
    role = metadata.get("collection_role")
    pool_id = metadata.get("pool_id")
    if role == "pilot_screening" and isinstance(pool_id, str) and SAFE_ID.fullmatch(pool_id):
        return f"pilot_{pool_id}__{user_id}"
    if role == "external_web_smoke_test":
        return f"external_smoke_01__{user_id}"
    return f"{campaign['campaign_id']}__{user_id}"


def _backup_filename(
    campaign: dict[str, Any], user_id: str, passage_ids: list[str]
) -> str:
    role = str(campaign.get("metadata", {}).get("collection_role", "assignment"))
    if role == "pilot_screening":
        middle = "pilot_screening"
    elif role.startswith("test"):
        middle = f"test_{passage_ids[0]}" if passage_ids else "test"
    elif role == "external_web_smoke_test":
        middle = "external_smoke_test"
    else:
        middle = re.sub(r"[^A-Za-z0-9._-]+", "_", role).strip("_") or "assignment"
    return f"{user_id}_{middle}_backup.zip"


def _user_raw_rows(output_path: Path, user_id: str) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    try:
        raw_lines = output_path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError as error:
        raise UserBackupStateError("Persistent annotation log is missing") from error

    selected = bytearray()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise UserBackupStateError(
                f"Malformed persistent annotation row {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise UserBackupStateError(
                f"Malformed persistent annotation row {line_number}"
            )
        if row.get("user_id") == user_id:
            selected.extend(raw_line)
            if not raw_line.endswith((b"\n", b"\r")):
                selected.extend(b"\n")
            rows.append(row)
    return bytes(selected), tuple(rows)


def build_user_backup(
    pearmut_root: str | Path,
    campaign_id: str,
    user_id: str,
    supplied_token: str,
    signing_key: str,
    *,
    exported_at: datetime | None = None,
) -> UserBackupArchive:
    """Return a ZIP scoped to one completed public campaign assignment."""
    campaign = _validated_id("campaign ID", campaign_id)
    user = _validated_id("user ID", user_id)
    if not backup_token_is_valid(signing_key, campaign, user, supplied_token):
        raise UserBackupAuthorizationError("Backup token is not valid for this user")

    root = Path(pearmut_root)
    campaign_payload = _load_json(
        root / "data/campaigns" / f"{campaign}.json", "campaign"
    )
    if not isinstance(campaign_payload, dict) or campaign_payload.get("campaign_id") != campaign:
        raise UserBackupStateError("Persistent campaign ID mismatch")

    users = campaign_payload.get("info", {}).get("users", [])
    if user not in users:
        raise UserBackupNotFoundError("Unknown user assignment")

    all_progress = _load_json(root / "data/progress.json", "progress")
    try:
        user_progress = all_progress[campaign][user]
    except (KeyError, TypeError):
        raise UserBackupNotFoundError("Unknown user progress") from None
    if not isinstance(user_progress, dict) or not _progress_is_complete(
        user_progress.get("progress")
    ):
        raise UserBackupIncompleteError(
            "Backup becomes available after this assignment is complete"
        )

    raw_content, raw_rows = _user_raw_rows(
        root / "data/outputs" / f"{campaign}.jsonl", user
    )
    passage_ids = _passage_ids_for_user(campaign_payload, user)
    timestamp = (exported_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    campaign_metadata = campaign_payload.get("metadata", {})
    metadata = {
        "schema_version": "1.0",
        "campaign_id": campaign,
        "user_id": user,
        "assignment_id": _assignment_id(campaign_payload, user),
        "passage_id": passage_ids[0] if len(passage_ids) == 1 else None,
        "passage_ids": passage_ids,
        "exported_at": timestamp.isoformat().replace("+00:00", "Z"),
        "purpose": campaign_metadata.get("purpose")
        or campaign_metadata.get("collection_role"),
        "eligible_for_gold": bool(campaign_metadata.get("eligible_for_gold", False)),
        "raw_row_count": len(raw_rows),
        "authority": "annotator_copy_only_server_logs_remain_authoritative",
    }
    progress_subset = {campaign: {user: user_progress}}

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("annotations.jsonl", raw_content)
        archive.writestr(
            "progress.json",
            json.dumps(progress_subset, ensure_ascii=False, indent=2) + "\n",
        )
        archive.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )

    return UserBackupArchive(
        filename=_backup_filename(campaign_payload, user, passage_ids),
        content=archive_buffer.getvalue(),
        metadata=metadata,
        raw_rows=raw_rows,
    )
