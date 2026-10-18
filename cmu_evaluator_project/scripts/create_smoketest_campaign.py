#!/usr/bin/env python3
"""Create a disposable Pearmut campaign by changing only campaign identity."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from campaign_common import CampaignError, atomic_write_json, read_json, require_nonempty_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        source = read_json(args.source)
        original_id = require_nonempty_string(source.get("campaign_id"), "source campaign_id")
        smoke_id = require_nonempty_string(args.campaign_id, "campaign_id")
        if smoke_id == original_id:
            raise CampaignError("smoke-test campaign_id must differ from the real campaign_id")

        smoke = copy.deepcopy(source)
        smoke["campaign_id"] = smoke_id

        restored = copy.deepcopy(smoke)
        restored["campaign_id"] = original_id
        if restored != source:
            raise CampaignError("smoke-test derivation changed more than campaign_id")

        atomic_write_json(args.output, smoke, args.overwrite)
        print(
            f"Built disposable campaign {args.output}: "
            f"{original_id} -> {smoke_id}; only campaign_id changed"
        )
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
