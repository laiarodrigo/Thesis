from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from campaign_common import spreadsheet_label  # noqa: E402
from build_candidate_dataset import technical_flags  # noqa: E402
from build_pearmut_campaign import markdown_instructions_to_html  # noqa: E402


class CampaignPipelineTest(unittest.TestCase):
    def write_json(self, path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def run_script(self, name: str, config: Path, expect_success: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / name), "--config", str(config)],
            capture_output=True,
            text=True,
            check=False,
        )
        if expect_success and result.returncode != 0:
            self.fail(f"{name} failed:\nstdout={result.stdout}\nstderr={result.stderr}")
        return result

    def test_generic_multi_document_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "source_language": "pt-BR",
                "target_language": "pt-PT",
                "documents": [
                    {"document_id": "doc-1", "passages": [{"passage_id": "p-1", "sentences": [
                        {"sentence_id": "a", "position": 1, "source_bp": "Fonte A."},
                        {"sentence_id": "b", "position": 2, "source_bp": "Fonte B."},
                    ]}]},
                    {"document_id": "doc-2", "passages": [{"passage_id": "p-2", "sentences": [
                        {"sentence_id": "x", "position": 1, "source_bp": "Fonte X."},
                    ]}]},
                ],
            }
            generated_rows = [
                {"document_id": "doc-1", "passage_id": "p-1", "sentence_id": "a", "position": 1, "source_bp": "Fonte A.", "raw_model_output": "Saída A.", "cleaned_candidate": "Saída A."},
                {"document_id": "doc-1", "passage_id": "p-1", "sentence_id": "b", "position": 2, "source_bp": "Fonte B.", "raw_model_output": "Saída B.", "cleaned_candidate": "Saída B."},
                {"document_id": "doc-2", "passage_id": "p-2", "sentence_id": "x", "position": 1, "source_bp": "Fonte X.", "raw_model_output": "Saída X.", "cleaned_candidate": "Saída X."},
            ]
            self.write_json(root / "source.json", source)
            self.write_json(root / "model.json", {"system_id": "model-z", "generation_config": {"beams": 3}, "sentences": generated_rows})
            (root / "instructions.md").write_text("# Test instructions\n", encoding="utf-8")
            config = {
                "campaign_id": "generic-test",
                "path_root": ".",
                "paths": {
                    "source": "source.json",
                    "candidate_dataset": "candidates.json",
                    "candidate_review": "review.md",
                    "private_mapping": "mapping.json",
                    "pearmut_instructions": "instructions.md",
                    "pearmut_campaign": "campaign.json",
                },
                "systems": [
                    {"system_id": "copy-z", "display_name": "Copy Z", "kind": "copy"},
                    {"system_id": "model-z", "display_name": "Model Z", "kind": "output_file", "output_path": "model.json"},
                ],
                "anonymization": {"seed": 7, "label_prefix": "Blind"},
                "validation": {"expected_source_sentence_count": 3, "expected_system_count": 2},
                "pearmut": {"protocol": "cESA", "instructions_format": "markdown", "show_model_names": False, "shuffle": False},
            }
            self.write_json(root / "config.json", config)

            self.run_script("build_candidate_dataset.py", root / "config.json")
            self.run_script("anonymize_systems.py", root / "config.json")
            self.run_script("build_pearmut_campaign.py", root / "config.json")

            dataset = json.loads((root / "candidates.json").read_text(encoding="utf-8"))
            mapping = json.loads((root / "mapping.json").read_text(encoding="utf-8"))
            campaign = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
            self.assertEqual(dataset["candidate_judgment_count"], 6)
            self.assertEqual(len(dataset["documents"]), 2)
            self.assertEqual(len(mapping["mappings"]), 2)
            self.assertEqual(len(campaign["data"][0]), 2)
            self.assertEqual(campaign["info"]["instructions"], "<h2>Test instructions</h2>\n")
            self.assertFalse(campaign["info"]["show_model_names"])
            self.assertFalse(campaign["info"]["shuffle"])
            serialized = json.dumps(campaign, ensure_ascii=False)
            self.assertNotIn('"copy-z"', serialized)
            self.assertNotIn('"model-z"', serialized)
            try:
                from pearmut.cli import _validate_item_structure
            except ImportError:
                pass
            else:
                for document in campaign["data"][0]:
                    _validate_item_structure(document)

            config["paths"]["pearmut_campaign"] = "campaign-resumable.json"
            config["pearmut"]["page_layout"] = (
                "one_sentence_per_page_with_full_source_context"
            )
            self.write_json(root / "config.json", config)
            self.run_script("build_pearmut_campaign.py", root / "config.json")

            resumable = json.loads(
                (root / "campaign-resumable.json").read_text(encoding="utf-8")
            )
            pages = resumable["data"][0]
            self.assertEqual(len(pages), 3)
            self.assertEqual([len(page) for page in pages], [2, 2, 1])
            self.assertEqual(
                [sum(bool(segment["tgt"]) for segment in page) for page in pages],
                [1, 1, 1],
            )
            self.assertEqual(
                sum(len(segment["tgt"]) for page in pages for segment in page),
                6,
            )
            self.assertEqual(
                resumable["metadata"],
                {
                    "page_layout": "one_sentence_per_page_with_full_source_context",
                    "passage_ids": ["p-1", "p-2"],
                    "displayed_segment_count": 5,
                    "judged_source_item_count": 3,
                    "candidate_judgment_count": 6,
                },
            )
            for page in pages:
                for segment in page:
                    if not segment["tgt"]:
                        self.assertTrue(segment["context_only"])
                try:
                    from pearmut.cli import _validate_item_structure
                except ImportError:
                    pass
                else:
                    _validate_item_structure(page)

    def test_missing_output_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(root / "source.json", {
                "document_id": "d", "passage_id": "p",
                "sentences": [{"sentence_id": "1", "position": 1, "source_bp": "Fonte."}],
            })
            self.write_json(root / "model.json", {"system_id": "m", "sentences": []})
            self.write_json(root / "config.json", {
                "campaign_id": "failure-test", "path_root": ".",
                "paths": {"source": "source.json", "candidate_dataset": "out.json", "candidate_review": "review.md"},
                "systems": [{"system_id": "m", "kind": "output_file", "output_path": "model.json"}],
            })
            result = self.run_script("build_candidate_dataset.py", root / "config.json", expect_success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ID alignment failure", result.stderr)
            self.assertFalse((root / "out.json").exists())

    def test_anonymous_labels_scale_beyond_z(self) -> None:
        self.assertEqual(spreadsheet_label(0), "A")
        self.assertEqual(spreadsheet_label(25), "Z")
        self.assertEqual(spreadsheet_label(26), "AA")

    def test_wrong_decoder_label_is_not_called_cleanup_failure(self) -> None:
        flags = technical_flags(
            {"control_label_matches_expected": False},
            "Todos os dias eu empilhava os trocados.",
        )
        self.assertEqual(flags, ["unexpected_decoder_control_label"])

    def test_markdown_instructions_render_as_safe_structured_html(self) -> None:
        rendered = markdown_instructions_to_html(
            "# Task\n\nA **good** rewrite:\n\n1. Preserve meaning.\n"
            "2. Use `EP`.\n\n- Minor\n- Major\n"
        )
        self.assertIn("<h2>Task</h2>", rendered)
        self.assertIn("<p>A <strong>good</strong> rewrite:</p>", rendered)
        self.assertIn("<ol>", rendered)
        self.assertIn("<li>Use <code>EP</code>.</li>", rendered)
        self.assertIn("<ul>", rendered)


if __name__ == "__main__":
    unittest.main()
