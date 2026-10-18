#!/usr/bin/env python3
"""Verify that one Pearmut campaign has persistent submissions for one user."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(root: Path, campaign_id: str, user_id: str) -> dict[str, object]:
    campaign_path = root / "data/campaigns" / f"{campaign_id}.json"
    output_path = root / "data/outputs" / f"{campaign_id}.jsonl"
    progress_path = root / "data/progress.json"
    for path in (campaign_path, output_path, progress_path):
        if not path.is_file():
            raise ValueError(f"missing expected Pearmut file: {path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if campaign.get("campaign_id") != campaign_id:
        raise ValueError("runtime campaign ID mismatch")
    if campaign_id not in progress or user_id not in progress[campaign_id]:
        raise ValueError("expected campaign/user missing from progress.json")
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("annotation JSONL exists but has no submissions")
    wrong_users = sorted({row.get("user_id") for row in rows if row.get("user_id") != user_id})
    if wrong_users:
        raise ValueError(f"unexpected user IDs in annotation JSONL: {wrong_users}")
    return {
        "campaign_id": campaign_id,
        "user_id": user_id,
        "submission_count": len(rows),
        "progress": progress[campaign_id][user_id].get("progress"),
        "campaign_path": str(campaign_path),
        "annotation_path": str(output_path),
        "progress_path": str(progress_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pearmut_root", type=Path)
    parser.add_argument("campaign_id")
    parser.add_argument("user_id")
    args = parser.parse_args()
    try:
        result = verify(args.pearmut_root.resolve(), args.campaign_id, args.user_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
