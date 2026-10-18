from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_annotation_campaigns import build_outputs  # noqa: E402
from assignment_allocation import allocate_two_annotators  # noqa: E402
from analyze_annotation_disagreement import analyze  # noqa: E402
from campaign_common import (  # noqa: E402
    CampaignError,
    candidate_id,
    canonical_candidate_index,
    item_key,
    load_config,
    resolve_path,
)
from normalize_pearmut_annotations import normalize  # noqa: E402
from summarize_screening_coverage import summarize  # noqa: E402


class AnnotationInfrastructureTest(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")

    def dataset(self, passage_count: int = 1) -> dict:
        passages = []
        for number in range(1, passage_count + 1):
            passage_id = f"p{number}"
            sentence_id = f"s{number}"
            key = item_key(f"d{number}", passage_id, sentence_id)
            passages.append(
                {
                    "document_id": f"d{number}",
                    "passages": [
                        {
                            "passage_id": passage_id,
                            "items": [
                                {
                                    "item_key": key,
                                    "document_id": f"d{number}",
                                    "passage_id": passage_id,
                                    "sentence_id": sentence_id,
                                    "position": 1,
                                    "source_bp": f"Fonte {number}.",
                                    "candidates": {
                                        "model-1": f"Candidato 1-{number}.",
                                        "model-2": f"Candidato 2-{number}.",
                                    },
                                    "candidate_ids": {
                                        "model-1": candidate_id(passage_id, sentence_id, "model-1"),
                                        "model-2": candidate_id(passage_id, sentence_id, "model-2"),
                                    },
                                }
                            ],
                        }
                    ],
                }
            )
        return {
            "schema_version": "2.1",
            "campaign_id": "synthetic-source",
            "systems": [
                {"system_id": "model-1"},
                {"system_id": "model-2"},
            ],
            "documents": passages,
        }

    def assignment_config(self, root: Path, dataset: dict, granularity: str, assignments: list[dict]) -> Path:
        self.write_json(root / "candidates.json", dataset)
        (root / "instructions.md").write_text("# Synthetic instructions\n", encoding="utf-8")
        config = {
            "campaign_project_id": "synthetic-project",
            "path_root": ".",
            "paths": {
                "candidate_dataset": "candidates.json",
                "pearmut_instructions": "instructions.md",
                "assignment_manifest": "manifest.json",
            },
            "candidate_systems": ["model-1", "model-2"],
            "assignment_strategy": {"link_granularity": granularity},
            "anonymization": {"label_prefix": "System"},
            "assignments": assignments,
            "pearmut": {
                "assignment": "task-based",
                "protocol": "cESA",
                "page_layout": "one_sentence_per_page_with_full_source_context",
                "instructions_format": "markdown",
                "show_model_names": False,
                "show_progress": True,
                "shuffle": False,
                "textfield": "visible",
                "special_tokens": ["[MISSING]"],
            },
            "expected_counts": {},
        }
        self.write_json(root / "config.json", config)
        return root / "config.json"

    def mapping(self, assignment_id: str, annotator_id: str, campaign_id: str, model_1_label: str) -> dict:
        other = "System-B" if model_1_label == "System-A" else "System-A"
        return {
            "campaign_id": campaign_id,
            "assignment_id": assignment_id,
            "annotator_id": annotator_id,
            "mappings": [
                {"real_system_id": "model-1", "anonymous_system_id": model_1_label},
                {"real_system_id": "model-2", "anonymous_system_id": other},
            ],
        }

    def raw_row(self, user: str, timestamp: float, label: str, candidate_text: str, source_key: str) -> dict:
        return {
            "user_id": user,
            "item_i": 0,
            "item": [{"item_id": source_key, "src": "Fonte 1.", "tgt": {label: candidate_text}}],
            "annotation": [
                {
                    label: {
                        "score": 90,
                        "error_spans": [],
                        "textfield": None,
                    }
                }
            ],
            "actions": [
                {"time": timestamp, "action": "load"},
                {"time": timestamp + 10, "action": "submit"},
            ],
        }

    def make_normalization_case(self, root: Path, annotators: list[tuple[str, str]]) -> tuple[dict, dict, dict[str, list[Path]]]:
        dataset = self.dataset()
        key = item_key("d1", "p1", "s1")
        assignments = []
        overrides: dict[str, list[Path]] = {}
        for index, (annotator_id, label) in enumerate(annotators, 1):
            assignment_id = f"assignment-{index}"
            campaign_id = f"campaign-{index}"
            mapping_path = root / f"mapping-{index}.json"
            raw_path = root / f"raw-{index}.jsonl"
            self.write_json(mapping_path, self.mapping(assignment_id, annotator_id, campaign_id, label))
            self.write_json(
                raw_path,
                self.raw_row(f"pearmut-{index}", float(index * 100), label, "Candidato 1-1.", key),
            )
            assignments.append(
                {
                    "annotator_id": annotator_id,
                    "assignment_id": assignment_id,
                    "campaign_id": campaign_id,
                    "mapping_path": str(mapping_path),
                    "raw_exports": [],
                }
            )
            overrides[assignment_id] = [raw_path]
        manifest = {"assignments": assignments}
        return dataset, manifest, overrides

    def test_different_labels_resolve_to_same_candidate_and_three_records_survive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest, overrides = self.make_normalization_case(
                root,
                [("annotator-a", "System-A"), ("annotator-b", "System-B"), ("annotator-c", "System-A")],
            )
            payload = normalize(dataset, manifest, root / "manifest.json", overrides, "error")
            self.assertEqual(payload["record_count"], 3)
            self.assertEqual(
                {record["candidate_id"] for record in payload["records"]},
                {candidate_id("p1", "s1", "model-1")},
            )
            self.assertEqual(
                {record["provenance"]["presentation_label"] for record in payload["records"]},
                {"System-A", "System-B"},
            )

    def test_candidate_ids_never_silently_collide(self) -> None:
        dataset = self.dataset()
        duplicate = json.loads(json.dumps(dataset["documents"][0]))
        duplicate["document_id"] = "different-document"
        duplicate["passages"][0]["items"][0]["document_id"] = "different-document"
        duplicate["passages"][0]["items"][0]["item_key"] = item_key(
            "different-document", "p1", "s1"
        )
        dataset["documents"].append(duplicate)
        with self.assertRaisesRegex(CampaignError, "candidate_id collision"):
            canonical_candidate_index(dataset)

    def test_missing_mapping_and_unknown_candidate_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest, overrides = self.make_normalization_case(
                root, [("annotator-a", "System-A")]
            )
            raw_path = overrides["assignment-1"][0]
            bad = self.raw_row(
                "pearmut-1", 100.0, "System-Z", "Candidato 1-1.", item_key("d1", "p1", "s1")
            )
            self.write_json(raw_path, bad)
            with self.assertRaisesRegex(CampaignError, "missing anonymization mapping"):
                normalize(dataset, manifest, root / "manifest.json", overrides, "error")

            self.write_json(
                raw_path,
                self.raw_row("pearmut-1", 100.0, "System-A", "Candidato 1-1.", item_key("d1", "unknown", "s1")),
            )
            with self.assertRaisesRegex(CampaignError, "unknown candidate item"):
                normalize(dataset, manifest, root / "manifest.json", overrides, "error")

            bad_mapping = self.mapping(
                "assignment-1", "annotator-a", "campaign-1", "System-A"
            )
            bad_mapping["mappings"][0]["real_system_id"] = "unknown-system"
            self.write_json(root / "mapping-1.json", bad_mapping)
            self.write_json(
                raw_path,
                self.raw_row("pearmut-1", 100.0, "System-A", "Candidato 1-1.", item_key("d1", "p1", "s1")),
            )
            with self.assertRaisesRegex(CampaignError, "mapping systems do not match"):
                normalize(dataset, manifest, root / "manifest.json", overrides, "error")

    def test_overlap_and_both_link_granularities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments = [
                {
                    "annotator_id": "a",
                    "assignment_id": "a-p1",
                    "campaign_id": "c-a-p1",
                    "passage_ids": ["p1"],
                    "anonymization_seed": 1,
                    "campaign_output": "campaign-a.json",
                    "mapping_output": "mapping-a.json",
                },
                {
                    "annotator_id": "b",
                    "assignment_id": "b-p1",
                    "campaign_id": "c-b-p1",
                    "passage_ids": ["p1"],
                    "anonymization_seed": 0,
                    "campaign_output": "campaign-b.json",
                    "mapping_output": "mapping-b.json",
                },
            ]
            config = self.assignment_config(
                root, self.dataset(2), "one_passage_per_link", assignments
            )
            outputs, _, manifest = build_outputs(config)
            self.assertEqual(manifest["passage_allocations"], {"a-p1": ["p1"], "b-p1": ["p1"]})
            campaigns = [payload for _, payload in outputs if payload.get("data")]
            self.assertEqual([payload["metadata"]["passage_ids"] for payload in campaigns], [["p1"], ["p1"]])
            mappings = [payload for _, payload in outputs if payload.get("researcher_only")]
            labels = [
                next(
                    row["anonymous_system_id"]
                    for row in mapping["mappings"]
                    if row["real_system_id"] == "model-1"
                )
                for mapping in mappings
            ]
            self.assertEqual(set(labels), {"System-A", "System-B"})

            assignments = [
                {
                    "annotator_id": "a",
                    "assignment_id": "a-multi",
                    "campaign_id": "c-a-multi",
                    "passage_ids": ["p1", "p2"],
                    "anonymization_seed": 3,
                    "campaign_output": "campaign-multi.json",
                    "mapping_output": "mapping-multi.json",
                }
            ]
            config = self.assignment_config(
                root, self.dataset(2), "multiple_passages_per_link", assignments
            )
            outputs, _, manifest = build_outputs(config)
            campaign = next(payload for _, payload in outputs if payload.get("data"))
            self.assertEqual(manifest["passage_allocations"]["a-multi"], ["p1", "p2"])
            self.assertEqual(campaign["metadata"]["passage_ids"], ["p1", "p2"])
            self.assertEqual(campaign["metadata"]["judged_source_item_count"], 2)

    def test_balanced_two_annotator_allocator_representative_cases(self) -> None:
        cases = [(10, 2), (10, 3), (10, 4), (5, 5), (11, 4)]
        for passage_count, annotator_count in cases:
            with self.subTest(X=passage_count, Y=annotator_count):
                passages = [f"p{index}" for index in range(passage_count)]
                annotators = [f"a{index}" for index in range(annotator_count)]
                first = allocate_two_annotators(passages, annotators, seed=41)
                second = allocate_two_annotators(passages, annotators, seed=41)
                self.assertEqual(first, second)
                self.assertEqual(len(first), passage_count)
                self.assertEqual(
                    sum(len(row["annotator_ids"]) for row in first),
                    2 * passage_count,
                )
                self.assertEqual(
                    [row["passage_id"] for row in first], passages
                )
                for row in first:
                    self.assertEqual(len(row["annotator_ids"]), 2)
                    self.assertEqual(len(set(row["annotator_ids"])), 2)
                loads = {
                    annotator: sum(
                        annotator in row["annotator_ids"] for row in first
                    )
                    for annotator in annotators
                }
                self.assertLessEqual(max(loads.values()) - min(loads.values()), 1)
                pair_counts = {}
                for row in first:
                    pair = tuple(sorted(row["annotator_ids"]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                possible_pairs = annotator_count * (annotator_count - 1) // 2
                self.assertEqual(
                    len(pair_counts), min(passage_count, possible_pairs)
                )
                self.assertLessEqual(
                    max(pair_counts.values()) - min(pair_counts.values()), 1
                )

    def test_exact_pool_01_allocation(self) -> None:
        passages = [
            "ted2020-1188_s9.2-s18",
            "ted2020-2279_s83-s91",
            "ted2020-1565_s7.2-s14",
            "ted2020-2568_s127-s135",
        ]
        users = [f"A{index:02d}" for index in range(1, 9)]
        first = allocate_two_annotators(passages, users, seed=20261009)
        second = allocate_two_annotators(passages, users, seed=20261009)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(len(set(row["annotator_ids"])) == 2 for row in first))
        self.assertEqual(
            {user: sum(user in row["annotator_ids"] for row in first) for user in users},
            {user: 1 for user in users},
        )
        self.assertEqual(len({tuple(sorted(row["annotator_ids"])) for row in first}), 4)

    def test_shared_pilot_and_test_campaigns_preserve_user_task_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "candidates.json", self.dataset(4))
            (root / "instructions.md").write_text("# Instructions\n", encoding="utf-8")
            users = [f"A{index:02d}" for index in range(1, 9)]
            pilot_assignments = [
                {
                    "user_id": user,
                    "assignment_id": f"pilot__{user}",
                    "passage_ids": ["p1"],
                }
                for user in users
            ]
            config = {
                "schema_version": "2.0",
                "pool_id": "pool_01",
                "campaign_project_id": "synthetic-pool",
                "path_root": ".",
                "paths": {
                    "pearmut_instructions": "instructions.md",
                    "assignment_manifest": "manifest.json",
                    "private_mapping_dir": "mappings",
                },
                "annotators": [
                    {"user_id": user, "screening_status": "pending"}
                    for user in users
                ],
                "anonymization": {"label_prefix": "System"},
                "campaigns": [
                    {
                        "campaign_id": "pilot-shared",
                        "campaign_project_id": "pilot",
                        "role": "pilot_screening",
                        "candidate_dataset": "candidates.json",
                        "campaign_output": "pilot.json",
                        "users": users,
                        "assignment_strategy": {
                            "link_granularity": "one_passage_per_link",
                            "seed": 11,
                        },
                        "assignments": pilot_assignments,
                    },
                    {
                        "campaign_id": "test-shared",
                        "campaign_project_id": "test",
                        "role": "test_provisional",
                        "candidate_dataset": "candidates.json",
                        "campaign_output": "test.json",
                        "users": users,
                        "selected_passage_ids": ["p1", "p2", "p3", "p4"],
                        "assignment_strategy": {
                            "link_granularity": "one_passage_per_link",
                            "allocation": "balanced_exact_coverage",
                            "required_annotators_per_passage": 2,
                            "seed": 20261009,
                        },
                        "assignments": [],
                    },
                ],
                "pearmut": {
                    "assignment": "task-based",
                    "protocol": "cESA",
                    "page_layout": "one_sentence_per_page_with_full_source_context",
                    "instructions_format": "markdown",
                    "show_model_names": False,
                    "show_progress": True,
                    "shuffle": False,
                    "textfield": "visible",
                    "special_tokens": ["[MISSING]"],
                },
            }
            self.write_json(root / "config.json", config)
            outputs, _, manifest = build_outputs(root / "config.json")
            campaigns = {
                payload["campaign_id"]: payload
                for _, payload in outputs
                if payload.get("campaign_id") in {"pilot-shared", "test-shared"}
                and "data" in payload
            }
            self.assertEqual(set(campaigns), {"pilot-shared", "test-shared"})
            for campaign in campaigns.values():
                self.assertEqual(campaign["info"]["users"], users)
                self.assertEqual(len(campaign["data"]), len(users))
            pilot_task_shapes = [
                [
                    [(segment["item_id"], segment["src"]) for segment in document]
                    for document in task
                ]
                for task in campaigns["pilot-shared"]["data"]
            ]
            self.assertTrue(
                all(shape == pilot_task_shapes[0] for shape in pilot_task_shapes)
            )
            test_rows = [
                row for row in manifest["assignments"] if row["campaign_id"] == "test-shared"
            ]
            self.assertEqual([row["user_id"] for row in test_rows], users)
            self.assertTrue(all(len(row["passage_ids"]) == 1 for row in test_rows))
            coverage = {}
            for row in test_rows:
                coverage.setdefault(row["passage_ids"][0], set()).add(row["user_id"])
            self.assertEqual(set(coverage), {"p1", "p2", "p3", "p4"})
            self.assertTrue(all(len(value) == 2 for value in coverage.values()))

    def test_multi_user_normalizer_filters_rows_and_preserves_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(2)
            campaign_id = "shared-test"
            raw_path = root / "shared.jsonl"
            assignments = []
            for index, (user_id, passage_id, label) in enumerate(
                [("A01", "p1", "System-A"), ("A02", "p2", "System-B")], 1
            ):
                assignment_id = f"assignment-{index}"
                mapping_path = root / f"mapping-{index}.json"
                self.write_json(
                    mapping_path,
                    self.mapping(assignment_id, user_id, campaign_id, label),
                )
                assignments.append(
                    {
                        "pool_id": "pool_01",
                        "campaign_id": campaign_id,
                        "collection_role": "test_provisional",
                        "user_id": user_id,
                        "annotator_id": user_id,
                        "assignment_id": assignment_id,
                        "passage_ids": [passage_id],
                        "mapping_path": str(mapping_path),
                        "raw_exports": [str(raw_path)],
                    }
                )
            manifest = {
                "schema_version": "2.0",
                "pool_id": "pool_01",
                "annotators": [
                    {"user_id": "A01", "screening_status": "pending"},
                    {"user_id": "A02", "screening_status": "accepted"},
                ],
                "assignments": assignments,
            }
            rows = [
                self.raw_row("A01", 100.0, "System-A", "Candidato 1-1.", item_key("d1", "p1", "s1")),
                self.raw_row("A02", 200.0, "System-B", "Candidato 1-2.", item_key("d2", "p2", "s2")),
                self.raw_row("A01", 300.0, "System-A", "Candidato 1-1.", item_key("d1", "p1", "s1")),
            ]
            raw_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            payload = normalize(dataset, manifest, root / "manifest.json", {}, "preserve")
            self.assertEqual(payload["record_count"], 3)
            self.assertEqual(
                [record["user_id"] for record in payload["records"]],
                ["A01", "A02", "A01"],
            )
            self.assertEqual(
                [record["assignment_id"] for record in payload["records"]],
                ["assignment-1", "assignment-2", "assignment-1"],
            )
            self.assertFalse(payload["records"][0]["eligible_for_gold"])
            self.assertTrue(payload["records"][1]["eligible_for_gold"])
            self.assertTrue(payload["records"][0]["provenance"]["repeat_resolution_required"])
            self.assertTrue(payload["records"][2]["provenance"]["repeat_resolution_required"])

            wrong = dict(rows[0], user_id="A99")
            raw_path.write_text(json.dumps(wrong) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CampaignError, "unexpected user_id"):
                normalize(dataset, manifest, root / "manifest.json", {}, "preserve")
            missing = dict(rows[0])
            missing.pop("user_id")
            raw_path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(CampaignError, "user_id"):
                normalize(dataset, manifest, root / "manifest.json", {}, "preserve")

    def test_test_pool_01_source_is_exact_xml(self) -> None:
        source = json.loads(
            (PROJECT / "data/prepared/test_pool_01/source/source.json").read_text(encoding="utf-8")
        )
        expected = [
            ("ted2020-1188_s9.2-s18", 10),
            ("ted2020-2279_s83-s91", 10),
            ("ted2020-1565_s7.2-s14", 10),
            ("ted2020-2568_s127-s135", 10),
        ]
        self.assertEqual(source["source_item_count"], 40)
        observed = []
        for document in source["documents"]:
            xml_path = REPOSITORY / document["source_xml"]
            xml_rows = {
                str(node.attrib["id"]): "".join(node.itertext())
                for node in ET.parse(xml_path).getroot().findall(".//s")
            }
            passage = document["passages"][0]
            observed.append((passage["passage_id"], passage["sentence_count"]))
            for sentence in passage["sentences"]:
                self.assertEqual(sentence["source_bp"], xml_rows[sentence["sentence_id"]])
                self.assertNotIn("raw_source_bp", sentence)
        self.assertEqual(observed, expected)

    def test_accepted_coverage_reports_replacement_assignments(self) -> None:
        manifest = {
            "pool_id": "pool_01",
            "annotators": [
                {"user_id": "A01", "screening_status": "accepted"},
                {"user_id": "A02", "screening_status": "rejected"},
                {"user_id": "A03", "screening_status": "accepted"},
                {"user_id": "A04", "screening_status": "rejected"},
            ],
            "assignments": [
                {"collection_role": "test_provisional", "user_id": "A01", "passage_ids": ["p1"]},
                {"collection_role": "test_provisional", "user_id": "A02", "passage_ids": ["p1"]},
                {"collection_role": "test_provisional", "user_id": "A03", "passage_ids": ["p2"]},
                {"collection_role": "test_provisional", "user_id": "A04", "passage_ids": ["p2"]},
            ],
        }
        payload = summarize(manifest)
        self.assertEqual(
            [(row["passage_id"], row["accepted_annotation_count"], row["replacement_assignments_needed"]) for row in payload["coverage"]],
            [("p1", 1, 1), ("p2", 1, 1)],
        )
        self.assertEqual(payload["replacement_assignment_count"], 2)

    def test_balanced_allocator_rejects_fewer_than_two_annotators(self) -> None:
        with self.assertRaisesRegex(CampaignError, "at least two annotators"):
            allocate_two_annotators(["p1"], ["a1"], seed=1)

    def test_automatic_two_annotator_campaigns_use_one_passage_per_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(5)
            self.write_json(root / "candidates.json", dataset)
            (root / "instructions.md").write_text(
                "# Synthetic instructions\n", encoding="utf-8"
            )
            config = {
                "campaign_project_id": "synthetic-test",
                "path_root": ".",
                "paths": {
                    "candidate_dataset": "candidates.json",
                    "pearmut_instructions": "instructions.md",
                    "assignment_manifest": "private/manifest.json",
                    "campaign_output_dir": "campaigns",
                    "private_mapping_dir": "private/mappings",
                },
                "selected_passage_ids": [f"p{index}" for index in range(1, 6)],
                "candidate_systems": ["model-1", "model-2"],
                "annotators": ["a", "b", "c"],
                "assignment_strategy": {
                    "link_granularity": "one_passage_per_link",
                    "allocation": "balanced_exact_coverage",
                    "required_annotators_per_passage": 2,
                    "seed": 17,
                },
                "anonymization": {"label_prefix": "System"},
                "assignments": [],
                "pearmut": {
                    "assignment": "task-based",
                    "protocol": "cESA",
                    "page_layout": "one_sentence_per_page_with_full_source_context",
                    "instructions_format": "markdown",
                    "show_model_names": False,
                    "show_progress": True,
                    "shuffle": False,
                    "textfield": "visible",
                    "special_tokens": ["[MISSING]"],
                },
                "expected_counts": {
                    "passages": 5,
                    "assignments": 10,
                    "candidate_judgments": 20,
                    "individual_annotations": 20,
                },
            }
            self.write_json(root / "config.json", config)
            outputs, _, manifest = build_outputs(root / "config.json")
            self.assertEqual(manifest["assignment_count"], 10)
            self.assertTrue(
                all(len(row["passage_ids"]) == 1 for row in manifest["assignments"])
            )
            coverage = {}
            for row in manifest["assignments"]:
                coverage.setdefault(row["passage_ids"][0], set()).add(
                    row["annotator_id"]
                )
            self.assertEqual(set(coverage), set(config["selected_passage_ids"]))
            self.assertTrue(all(len(annotators) == 2 for annotators in coverage.values()))
            campaigns = [payload for _, payload in outputs if payload.get("data")]
            self.assertEqual(len(campaigns), 10)
            self.assertTrue(
                all(len(payload["metadata"]["passage_ids"]) == 1 for payload in campaigns)
            )

    def test_selected_dev_source_is_exact_and_only_td_labels_are_removed(self) -> None:
        source = json.loads(
            (
                PROJECT / "data" / "prepared" / "dev_01" / "source" / "source.json"
            ).read_text(encoding="utf-8")
        )
        expected = [
            ("ted2020-18219_s30-s37", 9),
            ("ted2020-2203_s26-s32", 10),
            ("ted2020-2433_s21-s30", 10),
            ("ted2020-27951_s18-s27", 10),
            ("ted2020-24360_s25-s33", 9),
            ("ted2020-2609_s82-s91", 10),
            ("ted2020-1663_s7-s16", 10),
            ("ted2020-2860_s124-s133", 10),
            ("ted2020-1886_s23-s32", 10),
            ("ted2020-946_s43-s51", 9),
        ]
        passages = [
            passage
            for document in source["documents"]
            for passage in document["passages"]
        ]
        self.assertEqual(
            [(row["passage_id"], row["sentence_count"]) for row in passages],
            expected,
        )
        self.assertEqual(source["source_item_count"], 97)
        changed = [
            sentence
            for passage in passages
            for sentence in passage["sentences"]
            if "raw_source_bp" in sentence
        ]
        self.assertEqual([row["sentence_id"] for row in changed], ["48", "49"])
        for row in changed:
            self.assertTrue(row["raw_source_bp"].startswith("TD:"))
            self.assertFalse(row["source_bp"].startswith("TD:"))
            self.assertEqual(
                row["source_preparation"],
                {"removed_leading_speaker_label": "TD:"},
            )

    def test_duplicate_raw_submission_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest, overrides = self.make_normalization_case(
                root, [("annotator-a", "System-A")]
            )
            raw = overrides["assignment-1"][0]
            overrides["assignment-1"] = [raw, raw]
            with self.assertRaisesRegex(CampaignError, "duplicate raw submission"):
                normalize(dataset, manifest, root / "manifest.json", overrides, "preserve")

    def test_legitimate_repeat_requires_explicit_policy_and_preserves_both(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, manifest, overrides = self.make_normalization_case(
                root, [("annotator-a", "System-A")]
            )
            second = root / "raw-repeat.jsonl"
            self.write_json(
                second,
                self.raw_row(
                    "pearmut-1",
                    200.0,
                    "System-A",
                    "Candidato 1-1.",
                    item_key("d1", "p1", "s1"),
                ),
            )
            overrides["assignment-1"].append(second)
            with self.assertRaisesRegex(CampaignError, "repeated judgment"):
                normalize(dataset, manifest, root / "manifest.json", overrides, "error")
            payload = normalize(dataset, manifest, root / "manifest.json", overrides, "preserve")
            self.assertEqual(payload["record_count"], 2)
            self.assertTrue(
                all(
                    record["provenance"]["repeat_resolution_required"]
                    for record in payload["records"]
                )
            )

    def test_completed_pilot_move_is_byte_identical_and_still_has_40_candidates(self) -> None:
        base = PROJECT / "data" / "prepared" / "pilot_01"
        for line in (base / "checksums.sha256").read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            observed = hashlib.sha256((base / relative).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, relative)
        dataset = json.loads((base / "combined" / "candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(dataset["source_item_count"], 10)
        self.assertEqual(dataset["system_count"], 4)
        self.assertEqual(dataset["candidate_judgment_count"], 40)

    def test_descriptive_disagreement_flags_do_not_collapse_records(self) -> None:
        result = analyze(
            {
                "records": [
                    {
                        "candidate_id": "p::s::m",
                        "system_id": "m",
                        "annotator_id": "a",
                        "score": 100,
                        "error_spans": [],
                        "severity": None,
                        "comment": None,
                    },
                    {
                        "candidate_id": "p::s::m",
                        "system_id": "m",
                        "annotator_id": "b",
                        "score": 60,
                        "error_spans": [
                            {"start_i": 1, "end_i": 3, "severity": "major"}
                        ],
                        "severity": "major",
                        "comment": None,
                    },
                ]
            },
            20,
        )
        row = result["candidates"][0]
        self.assertTrue(row["flags"]["no_error_vs_error"])
        self.assertTrue(row["flags"]["major_vs_no_major"])
        self.assertTrue(row["flags"]["score_difference_above_threshold"])
        self.assertEqual(row["annotation_count"], 2)

    def test_updated_pilot_paths_resolve(self) -> None:
        config, root = load_config(PROJECT / "configs" / "pilot_01.json")
        moved_paths = [
            config["paths"]["source"],
            config["paths"]["full_guidelines"],
            config["paths"]["pearmut_instructions"],
            config["paths"]["candidate_dataset"],
            config["paths"]["candidate_review"],
            *[
                system["output_path"]
                for system in config["systems"]
                if system.get("kind") == "output_file"
            ],
        ]
        for value in moved_paths:
            self.assertTrue(resolve_path(root, value, "pilot path").is_file(), value)


if __name__ == "__main__":
    unittest.main()
