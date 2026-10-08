"""Release checks: no host absolute paths in recorded metrics or public files.

Run from the project root:  python -m unittest tests.test_public_release -v

Standard library only. Covers src/quality_pilot.py public_path, scrub_text, public_args, the stage
header and the JSON writers that every future stage record goes through, plus a scan of the public
files that ship with the repository.
"""

import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import quality_pilot as qp  # noqa: E402
import report_results as rr  # noqa: E402

POSIX = unittest.skipUnless(os.sep == "/", "POSIX path fixtures")
PILOT_DIR = os.path.join(PROJECT_ROOT, "results", "quality-pilot")
STAGE_FILES = ("baseline.json", "feasibility.json", "train.json", "evaluate.json")

# Evidence files that are tracked and ship publicly. Other evidence logs are local host logs.
PUBLIC_EVIDENCE = ("container-image.txt", "container-tests-quality.log", "official-loader-integration.json",
                   "environment-freeze.txt", "first-upstream-counterexample.json", "base-resource-pilot.json",
                   "unit-tests-release.log")


@POSIX
class TestPublicPath(unittest.TestCase):
    def test_inside_root_is_relative(self):
        self.assertEqual(qp.public_path(os.path.join(qp.PROJECT_ROOT, "models", "qwen2.5-0.5b")),
                         "models/qwen2.5-0.5b")
        self.assertEqual(qp.public_path(qp.PROJECT_ROOT), ".")
        self.assertEqual(qp.public_path(os.path.join(qp.PROJECT_ROOT, "results", "quality-pilot") + "/"),
                         "results/quality-pilot")

    def test_sibling_with_shared_prefix_is_external(self):
        self.assertEqual(qp.public_path(qp.PROJECT_ROOT + "-backup/results/train.json"), "<external>/train.json")

    def test_outside_root_keeps_only_basename(self):
        self.assertEqual(qp.public_path("/Users/someone/.cache/huggingface/hub/x/snapshots/abc"), "<external>/abc")
        self.assertEqual(qp.public_path("/home/someone/data/"), "<external>/data")
        self.assertEqual(qp.public_path("C:\\Users\\someone\\models\\qwen"), "<external>/qwen")

    def test_relative_kept_unless_it_escapes(self):
        self.assertEqual(qp.public_path("results/quality-pilot"), "results/quality-pilot")
        self.assertEqual(qp.public_path("./results//quality-pilot/"), "results/quality-pilot")
        self.assertEqual(qp.public_path("../../Users/someone/x"), "<external>/x")
        self.assertIsNone(qp.public_path(None))
        self.assertEqual(qp.public_path(""), "")


@POSIX
class TestScrubText(unittest.TestCase):
    ROOT = "/opt/work/proj"
    HOME = "/opt/work"

    def test_root_replaced_before_home(self):
        text = 'File "/opt/work/proj/src/quality_pilot.py", line 3'
        self.assertEqual(qp.scrub_text(text, self.ROOT, self.HOME), 'File "<project>/src/quality_pilot.py", line 3')

    def test_home_replaced(self):
        self.assertEqual(qp.scrub_text("cache at /opt/work/.cache/x", self.ROOT, self.HOME), "cache at <home>/.cache/x")

    def test_sibling_prefix_not_replaced(self):
        self.assertEqual(qp.scrub_text("/opt/work/proj2/a", self.ROOT, "/nonexistent-home"), "/opt/work/proj2/a")

    def test_generic_user_homes(self):
        text = "x /Users/alice/a and /home/bob/b and C:\\Users\\carol\\c"
        self.assertEqual(qp.scrub_text(text, self.ROOT, "/nonexistent-home"),
                         "x <home>/a and <home>/b and <home>\\c")

    def test_urls_and_plain_text_unchanged(self):
        for text in ("https://example.org/home/page", "KL(teacher || student), token mean",
                     "models/qwen2.5-0.5b", "results/quality-pilot/train.json"):
            self.assertEqual(qp.scrub_text(text, self.ROOT, "/nonexistent-home"), text)

    def test_find_host_paths_reports_values_and_keys(self):
        obj = {"args": {"model_dir": "/Users/alice/m"}, "ok": ["fine", 1, None, 2.5], "/home/bob/key": 1}
        hits = qp.find_host_paths(obj, self.ROOT, "/nonexistent-home")
        self.assertIn("$.args.model_dir", hits)
        self.assertTrue(any("<key /home/bob/key>" in h for h in hits))
        self.assertEqual(len(hits), 2)


@POSIX
class TestStageRecordSerialization(unittest.TestCase):
    def test_stage_header_args_are_public(self):
        args = qp.build_parser().parse_args(
            ["train", "--out", "/Users/alice/elsewhere/results/qp",
             "--model-dir", os.path.join(qp.PROJECT_ROOT, "models", "qwen2.5-0.5b")])
        h = qp.stage_header("train", args, "abc")
        self.assertEqual(h["args"]["model_dir"], "models/qwen2.5-0.5b")
        self.assertEqual(h["args"]["out"], "<external>/qp")
        self.assertEqual(h["args"]["steps"], 16)
        self.assertNotIn("func", h["args"])
        self.assertEqual(h["args_path_policy"], qp.ARGS_PATH_POLICY)
        self.assertEqual(qp.find_host_paths(h), [])
        self.assertNotIn(qp.PROJECT_ROOT, json.dumps(h))
        cmd = rr.stage_command("train", h["args"])
        self.assertIn("--out <external>/qp", cmd)
        self.assertNotIn("/Users/", cmd)

    def test_default_args_are_public(self):
        for stage in ("baseline", "feasibility", "train", "evaluate"):
            h = qp.stage_header(stage, qp.build_parser().parse_args([stage]), None)
            self.assertEqual(h["args"]["model_dir"], "models/qwen2.5-0.5b", stage)
            self.assertEqual(h["args"]["out"], "results/quality_pilot", stage)
            self.assertEqual(qp.find_host_paths(h), [], stage)

    def test_prepare_data_dir_outside_root(self):
        snap = "/Users/alice/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/" + qp.DATASET_REVISION
        args = qp.build_parser().parse_args(["prepare", "--data-dir", snap])
        self.assertEqual(qp.public_args(args)["data_dir"], "<external>/" + qp.DATASET_REVISION)

    def test_write_json_scrubs_tracebacks_and_keeps_numbers(self):
        home = os.path.expanduser("~")
        rec = {"status": "failed", "error": "RuntimeError: boom",
               "traceback": 'File "%s", line 1' % os.path.join(qp.PROJECT_ROOT, "src", "quality_pilot.py"),
               "cache": os.path.join(home, ".cache", "x"), "n": 3, "x": 0.1, "t": (1, 2), "flag": True}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stage.json")
            qp.write_json(path, rec)
            with open(path, "r", encoding="utf-8") as f:
                back = json.load(f)
            log = os.path.join(d, "steps.jsonl")
            qp.append_jsonl(log, rec)
            with open(log, "r", encoding="utf-8") as f:
                line = json.loads(f.readline())
        for out in (back, line):
            self.assertEqual(qp.find_host_paths(out), [])
            self.assertIn("<project>/src/quality_pilot.py", out["traceback"])
            if len(home) > 1:
                self.assertEqual(out["cache"], "<home>/.cache/x")
            self.assertEqual((out["n"], out["x"], out["t"], out["flag"]), (3, 0.1, [1, 2], True))

    def test_file_record_path_is_public(self):
        rec = qp.file_record(os.path.join(qp.PROJECT_ROOT, "src", "quality_pilot.py"))
        self.assertEqual(rec["path"], "src/quality_pilot.py")
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            self.assertEqual(qp.file_record(f.name)["path"], "<external>/" + os.path.basename(f.name))


class TestTrackedPublicFiles(unittest.TestCase):
    def test_stage_json_sanitized(self):
        for name in STAGE_FILES:
            with open(os.path.join(PILOT_DIR, name), "r", encoding="utf-8") as f:
                rec = json.load(f)
            self.assertEqual(qp.find_host_paths(rec), [], name)
            self.assertEqual(rec["args"]["model_dir"], "models/qwen2.5-0.5b", name)

    def test_public_metadata_lists_sanitized_files(self):
        with open(os.path.join(PILOT_DIR, "public_metadata.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        san = meta["path_sanitization"]
        self.assertEqual(san["fields"], ["args.model_dir"])
        self.assertEqual(sorted(san["files"]), sorted("results/quality-pilot/" + n for n in STAGE_FILES))
        for rec in san["files"].values():
            self.assertEqual(len(rec["pre_sanitization_sha256"]), 64)

    def test_no_user_home_paths_in_public_text(self):
        paths = []
        for top in ("results", "figures", "docs", "patches"):
            for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, top)):
                dirs[:] = [d for d in dirs if d != "checkpoints"]
                paths.extend(os.path.join(root, n) for n in files if not n.endswith(".png"))
        paths.extend(os.path.join(PROJECT_ROOT, n) for n in os.listdir(PROJECT_ROOT)
                     if n.endswith((".md", ".cff")) or n in ("Dockerfile", "requirements.txt", ".gitignore"))
        paths.append(os.path.join(PROJECT_ROOT, ".github", "workflows", "ci.yml"))
        paths.extend(os.path.join(PROJECT_ROOT, "evidence", n) for n in PUBLIC_EVIDENCE)
        checked = 0
        for path in paths:
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    self.assertIsNone(qp.HOST_HOME_RE.search(line), "%s:%d" % (os.path.relpath(path, PROJECT_ROOT), i))
            checked += 1
        self.assertGreater(checked, 10)


class TestDeliverableStyle(unittest.TestCase):
    def test_no_em_dashes(self):
        for rel in ("src/report_results.py", "tests/test_report_results.py", "tests/test_public_release.py",
                    "LICENSE_PROPOSAL.md", "LICENSE_NOTES.md", "README.md", "Dockerfile",
                    "RELEASE_NOTES.md", "CITATION.cff", ".github/workflows/ci.yml",
                    "results/quality-pilot/public_metadata.json"):
            path = os.path.join(PROJECT_ROOT, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                self.assertNotIn(chr(0x2014), f.read(), rel)


if __name__ == "__main__":
    unittest.main()
