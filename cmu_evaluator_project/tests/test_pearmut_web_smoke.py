from __future__ import annotations

import asyncio
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
SCRIPTS = PROJECT / "scripts"
PEARMUT = REPOSITORY / "thesis/bin/pearmut"
sys.path.insert(0, str(SCRIPTS))

from campaign_common import candidate_id, item_key  # noqa: E402
from normalize_pearmut_annotations import normalize  # noqa: E402


def response_json(response) -> object:
    return json.loads(response.body.decode("utf-8"))


class PearmutWebSmokeTest(unittest.TestCase):
    maxDiff = None

    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")

    def test_two_links_export_normalize_restore_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active"
            restore = root / "restore"
            fixtures = root / "fixtures"
            fixtures.mkdir(parents=True)
            campaigns = []
            manifest_assignments = []
            documents = []
            for ordinal, (campaign_id, passage_id, role) in enumerate(
                [
                    ("smoke-pilot-shared", "pilot-p", "pilot_screening"),
                    ("smoke-test-shared", "test-p", "test_provisional"),
                ],
                1,
            ):
                document_id = f"d{ordinal}"
                sentence_id = "s1"
                key = item_key(document_id, passage_id, sentence_id)
                source = f"Fonte {ordinal}."
                candidate = f"Candidato {ordinal}."
                campaigns.append(
                    {
                        "campaign_id": campaign_id,
                        "info": {
                            "assignment": "task-based",
                            "users": ["SMOKE-A01"],
                            "protocol": "cESA",
                            "show_model_names": False,
                            "show_progress": True,
                            "shuffle": False,
                            "textfield": "visible",
                        },
                        "data": [
                            [
                                [
                                    {
                                        "item_id": key,
                                        "src": source,
                                        "tgt": {"System-A": candidate},
                                    }
                                ]
                            ]
                        ],
                    }
                )
                mapping_path = fixtures / f"mapping-{ordinal}.json"
                self.write_json(
                    mapping_path,
                    {
                        "campaign_id": campaign_id,
                        "assignment_id": f"smoke-{ordinal}",
                        "annotator_id": "SMOKE-A01",
                        "mappings": [
                            {
                                "real_system_id": "model-1",
                                "anonymous_system_id": "System-A",
                            }
                        ],
                    },
                )
                manifest_assignments.append(
                    {
                        "pool_id": "smoke_pool",
                        "collection_role": role,
                        "campaign_id": campaign_id,
                        "user_id": "SMOKE-A01",
                        "annotator_id": "SMOKE-A01",
                        "assignment_id": f"smoke-{ordinal}",
                        "passage_ids": [passage_id],
                        "mapping_path": str(mapping_path),
                        "raw_exports": [str(active / "data/outputs" / f"{campaign_id}.jsonl")],
                    }
                )
                documents.append(
                    {
                        "document_id": document_id,
                        "passages": [
                            {
                                "passage_id": passage_id,
                                "items": [
                                    {
                                        "item_key": key,
                                        "document_id": document_id,
                                        "passage_id": passage_id,
                                        "sentence_id": sentence_id,
                                        "position": 1,
                                        "source_bp": source,
                                        "candidates": {"model-1": candidate},
                                        "candidate_ids": {
                                            "model-1": candidate_id(passage_id, sentence_id, "model-1")
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                )
            campaigns_path = fixtures / "campaign-input.json"
            self.write_json(campaigns_path, campaigns)
            env = os.environ | {
                "PEARMUT_ROOT": str(active),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            add = subprocess.run(
                [str(PEARMUT), "add", str(campaigns_path), "--url", "http://127.0.0.1"],
                cwd=REPOSITORY,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(add.returncode, 0, add.stdout + add.stderr)

            original_root = os.environ.get("PEARMUT_ROOT")
            os.environ["PEARMUT_ROOT"] = str(active)
            try:
                # Other tests may already have imported Pearmut with the repository
                # root. Reload its root-owning module so this test remains strictly
                # isolated even when the full suite runs in one interpreter.
                pearmut_utils = importlib.import_module("pearmut.utils")
                importlib.reload(pearmut_utils)
                pearmut_app = importlib.reload(importlib.import_module("pearmut.app"))

                downloaded_campaigns = response_json(
                    asyncio.run(
                        pearmut_app._download_campaigns(
                            ["smoke-pilot-shared", "smoke-test-shared"],
                            "campaigns.json",
                        )
                    )
                )
                self.assertEqual(len(downloaded_campaigns), 2)
                for campaign_id, expected_passage in [
                    ("smoke-pilot-shared", "pilot-p"),
                    ("smoke-test-shared", "test-p"),
                ]:
                    response = asyncio.run(pearmut_app.serve_annotate())
                    self.assertEqual(response.status_code, 200)
                    next_item = response_json(
                        asyncio.run(
                            pearmut_app._get_next_item(
                                pearmut_app.NextItemRequest(
                                    campaign_id=campaign_id, user_id="SMOKE-A01"
                                )
                            )
                        )
                    )
                    item = next_item["payload"]
                    self.assertIn(expected_passage, item[0]["item_id"])
                    submitted = {
                        "item": item,
                        "annotation": [
                            {
                                "System-A": {
                                    "score": 100,
                                    "error_spans": [],
                                    "textfield": None,
                                }
                            }
                        ],
                        "actions": [
                            {"time": 1000.0, "action": "load"},
                            {"time": 1010.0, "action": "submit"},
                        ],
                        "comment": "",
                    }
                    self.assertEqual(
                        response_json(
                            asyncio.run(
                                pearmut_app._log_response(
                                    pearmut_app.LogResponseRequest(
                                        campaign_id=campaign_id,
                                        user_id="SMOKE-A01",
                                        item_i=0,
                                        payload=submitted,
                                    )
                                )
                            )
                        ),
                        "ok",
                    )
                    resumed = response_json(
                        asyncio.run(
                            pearmut_app._get_next_item(
                                pearmut_app.NextItemRequest(
                                    campaign_id=campaign_id, user_id="SMOKE-A01"
                                )
                            )
                        )
                    )
                    self.assertEqual(resumed["status"], "goodbye")

                annotations = response_json(
                    asyncio.run(
                        pearmut_app._download_annotations(
                            ["smoke-pilot-shared", "smoke-test-shared"],
                            "annotations.json",
                        )
                    )
                )
                tokens = {row["campaign_id"]: row["token"] for row in downloaded_campaigns}
                progress = response_json(
                    asyncio.run(
                        pearmut_app._download_progress(
                            ["smoke-pilot-shared", "smoke-test-shared"],
                            [
                                tokens["smoke-pilot-shared"],
                                tokens["smoke-test-shared"],
                            ],
                            "progress.json",
                        )
                    )
                )
                self.assertEqual(set(annotations), {"smoke-pilot-shared", "smoke-test-shared"})
                self.assertTrue(
                    all(
                        value["SMOKE-A01"]["progress"] == ["completed"]
                        for value in [progress["smoke-pilot-shared"], progress["smoke-test-shared"]]
                    )
                )
            finally:
                if original_root is None:
                    os.environ.pop("PEARMUT_ROOT", None)
                else:
                    os.environ["PEARMUT_ROOT"] = original_root

            dataset = {
                "schema_version": "2.1",
                "campaign_id": "smoke-source",
                "systems": [{"system_id": "model-1"}],
                "documents": documents,
            }
            manifest = {
                "schema_version": "2.0",
                "pool_id": "smoke_pool",
                "annotators": [
                    {"user_id": "SMOKE-A01", "screening_status": "pending"}
                ],
                "assignments": manifest_assignments,
            }
            normalized = normalize(
                dataset, manifest, fixtures / "manifest.json", {}, "preserve"
            )
            self.assertEqual(normalized["record_count"], 2)
            self.assertEqual(
                {record["campaign_id"] for record in normalized["records"]},
                {"smoke-pilot-shared", "smoke-test-shared"},
            )

            downloaded_campaigns_path = fixtures / "campaigns.json"
            annotations_path = fixtures / "annotations.json"
            progress_path = fixtures / "progress.json"
            self.write_json(downloaded_campaigns_path, downloaded_campaigns)
            self.write_json(annotations_path, annotations)
            self.write_json(progress_path, progress)
            restore_env = os.environ | {
                "PEARMUT_ROOT": str(restore),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            imported = subprocess.run(
                [
                    str(PEARMUT),
                    "add-existing",
                    str(downloaded_campaigns_path),
                    "--progress",
                    str(progress_path),
                    "--annotations",
                    str(annotations_path),
                    "--url",
                    "http://127.0.0.1",
                ],
                cwd=REPOSITORY,
                env=restore_env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)
            restored_progress = json.loads((restore / "data/progress.json").read_text())
            self.assertEqual(
                restored_progress["smoke-test-shared"]["SMOKE-A01"]["progress"],
                ["completed"],
            )
            self.assertEqual(
                len((restore / "data/outputs/smoke-pilot-shared.jsonl").read_text().splitlines()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
