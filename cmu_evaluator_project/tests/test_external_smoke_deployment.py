from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from normalize_pearmut_annotations import normalize  # noqa: E402
from make_pearmut_launcher_url import make_launcher_url  # noqa: E402
from pearmut_public_proxy import (  # noqa: E402
    BLOCKED_RESEARCH_ROUTES,
    inject_backup_script,
    make_handler,
    route_allowed,
    upstream_request_path,
)
from pearmut_user_backup import (  # noqa: E402
    UserBackupAuthorizationError,
    build_user_backup,
    make_backup_token,
)
from verify_pearmut_submission import verify  # noqa: E402


CAMPAIGN_ID = "ted2020_pilot_external_smoke_01_cesa"
SMOKE_USER = "SMOKE-FRIEND-01"
ROSTER = PROJECT / "data/pearmut/pool_01/private/annotator_roster_PRIVATE.json"
POOL_CONFIG = PROJECT / "configs/pool_01_collection.json"
POOL_MANIFEST = PROJECT / "data/pearmut/pool_01/private/pool_01_manifest_PRIVATE.json"
SMOKE_CAMPAIGN = (
    PROJECT
    / "data/pearmut/external_smoke_01/campaigns"
    / f"{CAMPAIGN_ID}.json"
)
SMOKE_MANIFEST = (
    PROJECT
    / "data/pearmut/external_smoke_01/private/assignment_manifest_PRIVATE.json"
)
PILOT_DATASET = PROJECT / "data/prepared/pilot_01/combined/candidates.json"
PUBLIC_LAUNCHER = REPOSITORY / "docs/cmu-evaluator-launcher/index.html"


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def scalar_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from scalar_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from scalar_strings(child)


class ExternalSmokeDeploymentTest(unittest.TestCase):
    def test_roster_is_private_and_smoke_user_is_not_pool_01(self) -> None:
        roster = load(ROSTER)
        rows = roster["annotators"]
        self.assertTrue(roster["researcher_only"])
        self.assertEqual([row["user_id"] for row in rows], [f"A{i:02d}" for i in range(1, 9)])
        self.assertTrue(all(row["screening_status"] == "pending" for row in rows))
        self.assertNotIn(SMOKE_USER, {row["user_id"] for row in rows})
        for path in (POOL_CONFIG, POOL_MANIFEST):
            self.assertNotIn(SMOKE_USER, path.read_text(encoding="utf-8"))
        ignored = subprocess.run(
            ["git", "check-ignore", str(ROSTER.relative_to(REPOSITORY))],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)

    def test_real_names_are_absent_from_public_campaign_payloads(self) -> None:
        private_names = {row["researcher_name"] for row in load(ROSTER)["annotators"]}
        public_paths = list((PROJECT / "data/pearmut").glob("**/campaigns/*.json"))
        public_paths.append(SMOKE_CAMPAIGN)
        for path in set(public_paths):
            self.assertTrue(private_names.isdisjoint(set(scalar_strings(load(path)))), path)

    def test_smoke_campaign_is_disposable_and_separate(self) -> None:
        campaign = load(SMOKE_CAMPAIGN)
        manifest = load(SMOKE_MANIFEST)
        production_campaign = load(
            PROJECT
            / "data/pearmut/pool_01/campaigns/ted2020_pilot_screening_pool_01_cesa.json"
        )
        self.assertEqual(campaign["campaign_id"], CAMPAIGN_ID)
        self.assertNotEqual(CAMPAIGN_ID, "ted2020_pilot_screening_pool_01_cesa")
        self.assertEqual(campaign["info"]["users"], [SMOKE_USER])
        self.assertEqual(len(campaign["data"]), 1)
        self.assertEqual(campaign["metadata"]["canonical_source_item_count"], 10)
        self.assertEqual(campaign["metadata"]["canonical_candidate_judgment_count"], 40)
        self.assertEqual(campaign["metadata"]["purpose"], "external_web_smoke_test")
        self.assertFalse(campaign["metadata"]["eligible_for_gold"])
        self.assertTrue(campaign["metadata"]["disposable"])
        assignment = manifest["assignments"][0]
        self.assertFalse(assignment["eligible_for_gold"])
        self.assertNotIn("pool_01/campaigns", assignment["campaign_output"])
        def material(task: list) -> list[tuple[str, str, tuple[str, ...]]]:
            return [
                (segment["item_id"], segment["src"], tuple(sorted(segment["tgt"].values())))
                for document in task
                for segment in document
                if segment.get("tgt")
            ]

        self.assertEqual(material(campaign["data"][0]), material(production_campaign["data"][0]))

    def test_smoke_normalization_is_never_gold_eligible(self) -> None:
        campaign = load(SMOKE_CAMPAIGN)
        manifest = load(SMOKE_MANIFEST)
        # Even an accidental later status change cannot override the assignment's
        # permanent smoke-only eligibility policy.
        manifest["annotators"][0]["screening_status"] = "accepted"
        document = campaign["data"][0][0]
        annotations = []
        for segment in document:
            annotations.append(
                {
                    label: {"score": 1, "error_spans": [], "textfield": "garbage"}
                    for label in segment.get("tgt", {})
                }
            )
        row = {
            "user_id": SMOKE_USER,
            "item_i": 0,
            "item": document,
            "annotation": annotations,
            "actions": [
                {"time": 1.0, "action": "load"},
                {"time": 2.0, "action": "submit"},
            ],
            "comment": "disposable",
        }
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.jsonl"
            raw.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            normalized = normalize(
                load(PILOT_DATASET),
                manifest,
                SMOKE_MANIFEST,
                {"external_smoke_01__SMOKE-FRIEND-01": [raw]},
                "preserve",
            )
        self.assertEqual(normalized["record_count"], 4)
        self.assertEqual({record["user_id"] for record in normalized["records"]}, {SMOKE_USER})
        self.assertTrue(
            all(record["eligible_for_gold"] is False for record in normalized["records"])
        )

    def test_public_proxy_allows_only_annotation_surface(self) -> None:
        for path in [
            "/annotate",
            "/annotate?campaign_id=x&user_id=y",
            "/annotate.bundle.js",
            "/annotator-backup.js",
            "/style.css",
            "/favicon.svg",
            "/assets/example.png",
            "/healthz",
        ]:
            self.assertTrue(route_allowed("GET", path), path)
        for path in [
            "/get-next-item",
            "/get-i-item",
            "/log-response",
            "/annotator-backup",
        ]:
            self.assertTrue(route_allowed("POST", path), path)
        for path in BLOCKED_RESEARCH_ROUTES | {"/dashboard/private"}:
            self.assertFalse(route_allowed("GET", path), path)
            self.assertFalse(route_allowed("POST", path), path)
        self.assertFalse(route_allowed("POST", "/annotate"))
        self.assertFalse(route_allowed("GET", "/get-next-item"))
        self.assertFalse(route_allowed("GET", "/annotator-backup"))

    def test_health_cors_is_available_only_for_nonsensitive_health(self) -> None:
        class HealthyResponse:
            status = 200

            @staticmethod
            def read() -> bytes:
                return b"ready"

        connection = Mock()
        connection.getresponse.return_value = HealthyResponse()
        handler_class = make_handler("127.0.0.1", 8001)
        handler = handler_class.__new__(handler_class)
        handler.command = "GET"
        handler.wfile = BytesIO()
        handler.send_response = Mock()
        headers: list[tuple[str, str]] = []
        handler.send_header = lambda key, value: headers.append((key, value))
        handler.end_headers = Mock()

        with patch(
            "pearmut_public_proxy.http.client.HTTPConnection",
            return_value=connection,
        ):
            handler._health()

        handler.send_response.assert_called_once_with(200)
        self.assertIn(("Access-Control-Allow-Origin", "*"), headers)
        self.assertEqual(handler.wfile.getvalue(), b"ok\n")
        proxy_source = (SCRIPTS / "pearmut_public_proxy.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(proxy_source.count('"Access-Control-Allow-Origin"'), 1)

    def test_static_launcher_is_generic_and_generated_links_are_scoped(self) -> None:
        launcher = PUBLIC_LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("https://*.up.railway.app", launcher)
        self.assertIn('new URL("/healthz", origin)', launcher)
        self.assertIn('new URL("/annotate", origin)', launcher)
        self.assertIn('window.location.hash.slice(1)', launcher)
        self.assertIn('destination.hash = new URLSearchParams', launcher)
        self.assertIn("window.location.replace(destination.href)", launcher)
        self.assertNotIn(CAMPAIGN_ID, launcher)
        self.assertNotIn(SMOKE_USER, launcher)
        self.assertNotIn("<script src=", launcher.lower())

        link = make_launcher_url(
            "https://example-production.up.railway.app/",
            CAMPAIGN_ID,
            SMOKE_USER,
            "test-signing-key-that-is-longer-than-thirty-two-characters",
        )
        parsed = urlsplit(link)
        self.assertEqual(parsed.scheme, "https")
        self.assertEqual(parsed.netloc, "laiarodrigo.github.io")
        self.assertEqual(parsed.query, "")
        self.assertEqual(
            parse_qs(parsed.fragment),
            {
                "service": ["https://example-production.up.railway.app"],
                "campaign_id": [CAMPAIGN_ID],
                "user_id": [SMOKE_USER],
                "backup_token": [
                    make_backup_token(
                        "test-signing-key-that-is-longer-than-thirty-two-characters",
                        CAMPAIGN_ID,
                        SMOKE_USER,
                    )
                ],
            },
        )
        for invalid in (
            "http://example-production.up.railway.app",
            "https://evil.example",
            "https://example-production.up.railway.app.evil.example",
            "https://example-production.up.railway.app/dashboard",
        ):
            with self.assertRaises(ValueError):
                make_launcher_url(
                    invalid,
                    CAMPAIGN_ID,
                    SMOKE_USER,
                    "test-signing-key-that-is-longer-than-thirty-two-characters",
                )

    def test_completion_ui_is_injected_without_opening_admin_downloads(self) -> None:
        original = b"<html><body><main>Pearmut</main></body></html>"
        injected = inject_backup_script(original)
        self.assertIn(b'<script defer src="/annotator-backup.js"></script>', injected)
        self.assertEqual(inject_backup_script(injected), injected)
        self.assertEqual(
            upstream_request_path(
                "/annotate?campaign_id=campaign&user_id=A01&backup_token=secret"
            ),
            "/annotate?campaign_id=campaign&user_id=A01",
        )
        self.assertEqual(
            upstream_request_path("/style.css?backup_token=not-a-capability-here"),
            "/style.css?backup_token=not-a-capability-here",
        )
        script = (SCRIPTS / "annotator_backup.js").read_text(encoding="utf-8")
        self.assertIn("Download backup", script)
        self.assertIn("All done", script)
        self.assertIn("window.location.hash.slice(1)", script)
        self.assertNotIn('params.get("backup_token")', script)
        self.assertIn('fetch("/annotator-backup"', script)
        self.assertIn("MutationObserver", script)
        for path in (
            "/download-annotations",
            "/download-progress",
            "/download-campaigns",
            "/export-results",
        ):
            self.assertFalse(route_allowed("GET", path))
            self.assertFalse(route_allowed("POST", path))

    def test_user_backup_is_exactly_scoped_and_capability_bound(self) -> None:
        campaign_id = "ted2020_pilot_screening_pool_01_cesa"
        campaign_source = (
            PROJECT / "data/pearmut/pool_01/campaigns" / f"{campaign_id}.json"
        )
        signing_key = "unit-test-backup-signing-key-that-is-at-least-32-characters"
        a01_row = {
            "user_id": "A01",
            "item_i": 0,
            "annotation": [{"System-A": {"score": 100}}],
            "item": [{"item_id": "safe-public-item"}],
        }
        a02_row = {
            "user_id": "A02",
            "item_i": 0,
            "annotation": [{"System-A": {"score": 20}}],
            "marker": "A02_ONLY_PRIVATE_ROW",
        }
        a01_line = json.dumps(a01_row, ensure_ascii=False).encode("utf-8") + b"\n"
        a02_line = json.dumps(a02_row, ensure_ascii=False).encode("utf-8") + b"\n"
        completed = {
            "progress": ["completed"] * 10,
            "progress_welcome": [],
            "time": 10.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/campaigns").mkdir(parents=True)
            (root / "data/outputs").mkdir(parents=True)
            (root / "data/campaigns" / f"{campaign_id}.json").write_bytes(
                campaign_source.read_bytes()
            )
            (root / "data/outputs" / f"{campaign_id}.jsonl").write_bytes(
                a01_line + a02_line
            )
            (root / "data/progress.json").write_text(
                json.dumps(
                    {
                        campaign_id: {
                            "A01": completed,
                            "A02": completed | {"time": 99.0},
                        }
                    }
                ),
                encoding="utf-8",
            )
            private_marker = "PRIVATE_SYSTEM_MAPPING_MUST_NOT_EXPORT"
            (root / "private_mapping.json").write_text(
                private_marker, encoding="utf-8"
            )

            a01_token = make_backup_token(signing_key, campaign_id, "A01")
            backup = build_user_backup(
                root,
                campaign_id,
                "A01",
                a01_token,
                signing_key,
                exported_at=datetime(2026, 10, 12, 16, 30, tzinfo=timezone.utc),
            )
            self.assertEqual(backup.filename, "A01_pilot_screening_backup.zip")
            with zipfile.ZipFile(BytesIO(backup.content)) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {"annotations.jsonl", "progress.json", "metadata.json"},
                )
                self.assertEqual(archive.read("annotations.jsonl"), a01_line)
                exported_rows = [
                    json.loads(line)
                    for line in archive.read("annotations.jsonl").splitlines()
                ]
                self.assertEqual(exported_rows, [a01_row])
                progress = json.loads(archive.read("progress.json"))
                self.assertEqual(progress, {campaign_id: {"A01": completed}})
                metadata = json.loads(archive.read("metadata.json"))
                self.assertEqual(metadata["campaign_id"], campaign_id)
                self.assertEqual(metadata["user_id"], "A01")
                self.assertEqual(metadata["assignment_id"], "pilot_pool_01__A01")
                self.assertEqual(metadata["passage_id"], "ted2020-2292_s3-s12")
                exported_payload = b"\n".join(
                    archive.read(name) for name in archive.namelist()
                )

            self.assertNotIn(b"A02_ONLY_PRIVATE_ROW", exported_payload)
            self.assertNotIn(private_marker.encode("utf-8"), exported_payload)
            for private_name in {
                row["researcher_name"] for row in load(ROSTER)["annotators"]
            }:
                self.assertNotIn(private_name.encode("utf-8"), exported_payload)
            with self.assertRaises(UserBackupAuthorizationError):
                build_user_backup(root, campaign_id, "A02", a01_token, signing_key)

    def test_smoke_user_backup_is_explicitly_non_gold(self) -> None:
        signing_key = "unit-test-backup-signing-key-that-is-at-least-32-characters"
        smoke_row = {"user_id": SMOKE_USER, "item_i": 0, "annotation": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/campaigns").mkdir(parents=True)
            (root / "data/outputs").mkdir(parents=True)
            runtime_campaign = load(SMOKE_CAMPAIGN)
            runtime_campaign["data"] = {
                SMOKE_USER: runtime_campaign["data"][0]
            }
            (root / "data/campaigns" / f"{CAMPAIGN_ID}.json").write_text(
                json.dumps(runtime_campaign), encoding="utf-8"
            )
            (root / "data/outputs" / f"{CAMPAIGN_ID}.jsonl").write_text(
                json.dumps(smoke_row) + "\n", encoding="utf-8"
            )
            (root / "data/progress.json").write_text(
                json.dumps(
                    {
                        CAMPAIGN_ID: {
                            SMOKE_USER: {"progress": ["completed"] * 10}
                        }
                    }
                ),
                encoding="utf-8",
            )
            backup = build_user_backup(
                root,
                CAMPAIGN_ID,
                SMOKE_USER,
                make_backup_token(signing_key, CAMPAIGN_ID, SMOKE_USER),
                signing_key,
            )
        self.assertFalse(backup.metadata["eligible_for_gold"])
        self.assertEqual(backup.metadata["purpose"], "external_web_smoke_test")
        self.assertEqual(
            backup.filename, "SMOKE-FRIEND-01_external_smoke_test_backup.zip"
        )

    def test_railway_container_contract_is_minimal_and_private_safe(self) -> None:
        dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
        dockerignore = (PROJECT / ".dockerignore").read_text(encoding="utf-8")
        railwayignore = (PROJECT / ".railwayignore").read_text(encoding="utf-8")
        launcher = (SCRIPTS / "start_railway_pearmut.sh").read_text(encoding="utf-8")
        deployment_doc = (
            PROJECT / "deployment/railway/README.md"
        ).read_text(encoding="utf-8")

        self.assertIn('"pearmut==1.1.5"', dockerfile)
        self.assertNotIn("private", dockerfile.lower())
        self.assertNotIn("prepared", dockerfile.lower())
        self.assertTrue(dockerignore.startswith("# The image needs only"))
        self.assertTrue(railwayignore.startswith("# Keep the CLI upload"))
        self.assertNotIn("private", railwayignore.lower())
        self.assertNotIn("prepared", railwayignore.lower())
        self.assertIn('EXPECTED_VOLUME_PATH="/pearmut-volume"', launcher)
        self.assertIn(
            'EXPECTED_PEARMUT_ROOT="$EXPECTED_VOLUME_PATH/pearmut"', launcher
        )
        self.assertIn("/proc/self/mountinfo", launcher)
        self.assertIn("--bind 0.0.0.0 --port \"$PORT\" --allow-public-bind", launcher)
        self.assertIn("--host 127.0.0.1", launcher)
        self.assertIn("PEARMUT_CAMPAIGN_IDS", launcher)
        self.assertIn("PEARMUT_BACKUP_SIGNING_KEY", launcher)
        self.assertNotIn("cloudflared", launcher.lower())
        self.assertIn("scripts/pearmut_user_backup.py", dockerfile)
        self.assertIn("scripts/annotator_backup.js", dockerfile)
        self.assertIn("SMOKE-FRIEND-01", deployment_doc)
        self.assertIn("Do not send A01-A08 links", deployment_doc)

        ignored = subprocess.run(
            [
                "git",
                "check-ignore",
                "cmu_evaluator_project/runtime/hosted_downloads/example",
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0, ignored.stderr)

    def test_persistent_verification_and_backup_exclude_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_path = root / "data/campaigns" / f"{CAMPAIGN_ID}.json"
            output_path = root / "data/outputs" / f"{CAMPAIGN_ID}.jsonl"
            campaign_path.parent.mkdir(parents=True)
            output_path.parent.mkdir(parents=True)
            campaign_path.write_text(json.dumps({"campaign_id": CAMPAIGN_ID}), encoding="utf-8")
            output_path.write_text(json.dumps({"user_id": SMOKE_USER}) + "\n", encoding="utf-8")
            (root / "data/progress.json").write_text(
                json.dumps({CAMPAIGN_ID: {SMOKE_USER: {"progress": ["completed"]}}}),
                encoding="utf-8",
            )
            result = verify(root, CAMPAIGN_ID, SMOKE_USER)
            self.assertEqual(result["submission_count"], 1)
            backup = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "backup_pearmut_campaign.sh"),
                    str(root),
                    CAMPAIGN_ID,
                ],
                cwd=REPOSITORY,
                env=os.environ | {"REPO_ROOT": str(REPOSITORY)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(backup.returncode, 0, backup.stdout + backup.stderr)
            archive = Path(backup.stdout.splitlines()[0])
            with tarfile.open(archive) as handle:
                names = handle.getnames()
            self.assertTrue(any(name.endswith("annotations.jsonl") for name in names))
            self.assertTrue(any(name.endswith("progress.json") for name in names))
            self.assertTrue(any("assignment_manifest_PRIVATE.json" in name for name in names))
            self.assertTrue(any("SMOKE-FRIEND-01.json" in name for name in names))
            self.assertFalse(any("roster" in name.lower() for name in names))

    def test_hosted_download_and_campaign_restore_preserve_other_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volume = root / "volume"
            remote_data = volume / "pearmut/data"
            (remote_data / "campaigns").mkdir(parents=True)
            (remote_data / "outputs").mkdir(parents=True)
            (remote_data / "campaigns" / f"{CAMPAIGN_ID}.json").write_text(
                SMOKE_CAMPAIGN.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (remote_data / "outputs" / f"{CAMPAIGN_ID}.jsonl").write_text(
                json.dumps({"user_id": SMOKE_USER, "item_i": 0}) + "\n",
                encoding="utf-8",
            )
            smoke_progress = {SMOKE_USER: {"progress": ["completed"]}}
            (remote_data / "progress.json").write_text(
                json.dumps({CAMPAIGN_ID: smoke_progress}), encoding="utf-8"
            )

            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_railway = fake_bin / "railway"
            fake_railway.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    import shutil
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    if args[:3] != ["volume", "files", "download"] and args[:3] != ["volume", "files", "upload"]:
                        raise SystemExit(f"unsupported fake Railway command: {args}")
                    volume = Path(os.environ["FAKE_RAILWAY_VOLUME"])
                    if args[2] == "download":
                        source = volume / args[3].lstrip("/")
                        destination = Path(args[4])
                    else:
                        source = Path(args[3])
                        destination = volume / args[4].lstrip("/")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    """
                ),
                encoding="utf-8",
            )
            fake_railway.chmod(0o755)
            downloads = root / "downloads"
            environment = os.environ | {
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "FAKE_RAILWAY_VOLUME": str(volume),
                "PEARMUT_HOSTED_DOWNLOAD_ROOT": str(downloads),
                "REPO_ROOT": str(REPOSITORY),
            }
            download = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "download_hosted_pearmut_campaign.sh"),
                    CAMPAIGN_ID,
                ],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(download.returncode, 0, download.stdout + download.stderr)
            archives = list(downloads.rglob(f"{CAMPAIGN_ID}_*.tar.gz"))
            self.assertEqual(len(archives), 1)
            archive = archives[0]
            self.assertTrue(Path(f"{archive}.sha256").is_file())

            other_campaign = {"someone": {"progress": [None]}}
            (remote_data / "progress.json").write_text(
                json.dumps(
                    {
                        CAMPAIGN_ID: {SMOKE_USER: {"progress": [None]}},
                        "unrelated_campaign": other_campaign,
                    }
                ),
                encoding="utf-8",
            )
            restore = subprocess.run(
                [
                    "bash",
                    str(SCRIPTS / "restore_hosted_pearmut_campaign.sh"),
                    str(archive),
                    CAMPAIGN_ID,
                    "--confirm-service-stopped",
                ],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(restore.returncode, 0, restore.stdout + restore.stderr)
            restored = load(remote_data / "progress.json")
            self.assertEqual(restored[CAMPAIGN_ID], smoke_progress)
            self.assertEqual(restored["unrelated_campaign"], other_campaign)


if __name__ == "__main__":
    unittest.main()
