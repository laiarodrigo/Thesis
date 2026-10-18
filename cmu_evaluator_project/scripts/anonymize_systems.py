#!/usr/bin/env python3
"""Create a deterministic private real-to-anonymous system mapping."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from campaign_common import (
    CampaignError,
    atomic_write_json,
    load_config,
    read_json,
    require_nonempty_string,
    resolve_path,
    spreadsheet_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing mapping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config, path_root = load_config(args.config)
        campaign_id = require_nonempty_string(config.get("campaign_id"), "campaign_id")
        paths = config.get("paths")
        if not isinstance(paths, dict):
            raise CampaignError("config.paths must be an object")
        dataset_path = resolve_path(path_root, paths.get("candidate_dataset"), "paths.candidate_dataset")
        mapping_path = resolve_path(path_root, paths.get("private_mapping"), "paths.private_mapping")
        dataset = read_json(dataset_path)
        if dataset.get("campaign_id") != campaign_id:
            raise CampaignError("candidate dataset campaign_id does not match configuration")
        systems = dataset.get("systems")
        if not isinstance(systems, list) or not systems:
            raise CampaignError("candidate dataset has no systems")
        real_ids = []
        for index, system in enumerate(systems):
            if not isinstance(system, dict):
                raise CampaignError(f"dataset.systems[{index}] must be an object")
            real_ids.append(require_nonempty_string(system.get("system_id"), f"dataset.systems[{index}].system_id"))
        if len(real_ids) != len(set(real_ids)):
            raise CampaignError("candidate dataset contains duplicate system IDs")

        settings = config.get("anonymization")
        if not isinstance(settings, dict):
            raise CampaignError("config.anonymization must be an object")
        seed = settings.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise CampaignError("anonymization.seed must be an integer")
        prefix = require_nonempty_string(settings.get("label_prefix"), "anonymization.label_prefix")

        shuffled = list(real_ids)
        random.Random(seed).shuffle(shuffled)
        mappings = [
            {"real_system_id": real_id, "anonymous_system_id": f"{prefix}-{spreadsheet_label(index)}"}
            for index, real_id in enumerate(shuffled)
        ]
        anonymous_ids = [entry["anonymous_system_id"] for entry in mappings]
        if len(anonymous_ids) != len(set(anonymous_ids)):
            raise CampaignError("anonymous labels are not unique")
        if set(real_ids) & set(anonymous_ids):
            raise CampaignError("an anonymous label collides with a real system ID")

        payload = {
            "schema_version": "1.0",
            "researcher_only": True,
            "campaign_id": campaign_id,
            "assignment_id": config.get("assignment_id"),
            "annotator_id": config.get("annotator_id"),
            "seed": seed,
            "anonymous_label_prefix": prefix,
            "system_count": len(real_ids),
            "mappings": mappings,
        }
        atomic_write_json(mapping_path, payload, args.overwrite)
        print(f"Built private mapping {mapping_path} for {len(real_ids)} systems")
    except CampaignError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
