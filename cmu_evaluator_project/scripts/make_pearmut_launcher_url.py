#!/usr/bin/env python3
"""Create a validated public wake-and-redirect link for a Pearmut annotator."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

from pearmut_user_backup import UserBackupError, make_backup_token


DEFAULT_LAUNCHER_BASE = (
    "https://laiarodrigo.github.io/Thesis/cmu-evaluator-launcher/"
)
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def normalize_railway_service_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not hostname.endswith(".up.railway.app")
        or hostname == ".up.railway.app"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "service URL must be exactly https://<host>.up.railway.app"
        )
    return urlunsplit(("https", hostname, "", "", ""))


def validate_id(label: str, value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{label} must contain 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


def make_launcher_url(
    service_url: str,
    campaign_id: str,
    user_id: str,
    signing_key: str,
    launcher_base: str = DEFAULT_LAUNCHER_BASE,
) -> str:
    service = normalize_railway_service_url(service_url)
    campaign = validate_id("campaign ID", campaign_id)
    user = validate_id("user ID", user_id)
    launcher = urlsplit(launcher_base)
    if launcher.scheme != "https" or not launcher.netloc or launcher.query or launcher.fragment:
        raise ValueError("launcher base must be an HTTPS URL without query or fragment")
    backup_token = make_backup_token(signing_key, campaign, user)
    fragment = urlencode(
        {
            "service": service,
            "campaign_id": campaign,
            "user_id": user,
            "backup_token": backup_token,
        }
    )
    return urlunsplit(
        (launcher.scheme, launcher.netloc, launcher.path, "", fragment)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service_url", help="https://<host>.up.railway.app")
    parser.add_argument("campaign_id")
    parser.add_argument("user_id")
    parser.add_argument(
        "--signing-key-file",
        type=Path,
        required=True,
        help="private local file containing PEARMUT_BACKUP_SIGNING_KEY",
    )
    parser.add_argument("--launcher-base", default=DEFAULT_LAUNCHER_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        signing_key = args.signing_key_file.read_text(encoding="utf-8").strip()
        print(
            make_launcher_url(
                args.service_url,
                args.campaign_id,
                args.user_id,
                signing_key,
                args.launcher_base,
            )
        )
    except (OSError, ValueError, UserBackupError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
