# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tim Urista. MIT grant scoped to the files listed in LICENSE, Part 1.
"""Tests for src/report_results.py: metadata-only replay versus private checkpoint validation.

Run from the project root:  python -m unittest tests.test_report_results -v

Standard library only; figures are never drawn here (--no-figures), so matplotlib is not needed.

* Clean-clone replay: only the tracked public files are copied into a temporary tree with no
  checkpoints, weights or evidence logs; --metadata-only must rebuild the recorded numbers and must
  open nothing else.
* Private validation on synthetic checkpoints: the default mode hashes checkpoints against
  train.json and re-derives public_metadata.json; tampering is caught by private validation, and the
  replay statement says that a metadata replay cannot catch it.
* On the owner's host only (skipped in a clean clone): private validation of the real checkpoints
  must agree with the tracked public metadata and with the metadata replay.
* The tracked results/summary.json must equal a fresh metadata replay of the tracked inputs.
"""

import ast
import builtins
import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import quality_pilot as qp  # noqa: E402
import report_results as rr  # noqa: E402

PILOT_DIR = os.path.join(PROJECT_ROOT, "results", "quality-pilot")
PUBLIC_PILOT_FILES = ("manifest.json", "baseline.json", "feasibility.json", "train.json", "evaluate.json",
                      "train_steps.jsonl", "public_metadata.json")
REAL_TRAINED = os.path.join(PILOT_DIR, "checkpoints", "trained", "compressed_layers.safetensors")
STALE = ("results/summary.json does not match a fresh metadata replay of the tracked inputs; regenerate with "
         ".venv/bin/python src/report_results.py --metadata-only")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def run_report(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rr.main(argv)
    return code, out.getvalue(), err.getvalue()


def without(d, *keys):
    return {k: v for k, v in d.items() if k not in keys}


# Cross-platform tolerance for floats DERIVED by report_results.py (sums, means, exp, ratios of the
# recorded values). Different Python versions or libm builds can differ in the last bit or two of
# such results (observed: 13.059324022000347 vs 13.059324022000348 on Python 3.9 vs 3.12), a
# relative change near 1e-16. 1e-12 allows a few thousand ulps of accumulated rounding, stays
# 1000 times tighter than the 1e-9 record checks inside report_results.py, and far below the 4
# decimal places any number is reported with. It does not apply to the recorded measurement JSON
# or to any model-equivalence check; those stay exact or keep their own tolerances.
DERIVED_FLOAT_REL_TOL = 1e-12
DERIVED_FLOAT_ABS_TOL = 1e-12


def json_differences(a, b, where="$"):
    """Paths where two JSON values differ. Floats compare with math.isclose at the derived-float
    tolerance; strings (hashes included), ints, booleans, None, key sets and list lengths compare
    exactly, and a type change (for example int versus float, or bool versus int) is a difference."""
    if isinstance(a, float) and isinstance(b, float) and type(a) is type(b):
        ok = math.isclose(a, b, rel_tol=DERIVED_FLOAT_REL_TOL, abs_tol=DERIVED_FLOAT_ABS_TOL)
        return [] if ok else ["%s: %r != %r" % (where, a, b)]
    if type(a) is not type(b):
        return ["%s: type %s != %s" % (where, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        if set(a) != set(b):
            return ["%s: keys differ %s" % (where, sorted(set(a) ^ set(b)))]
        out = []
        for k in sorted(a):
            out.extend(json_differences(a[k], b[k], "%s.%s" % (where, k)))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return ["%s: length %d != %d" % (where, len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(json_differences(x, y, "%s[%d]" % (where, i)))
        return out
    return [] if a == b else ["%s: %r != %r" % (where, a, b)]


def write_safetensors(path, tensors):
    """Minimal safetensors writer: tensors maps name -> (dtype, shape, raw bytes)."""
    header, data, off = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        data += raw
        off += len(raw)
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + data)


class CloneMixin:
    """A temporary tree holding only the tracked public pilot files."""

    def make_clone(self):
        self.tmp = tempfile.mkdtemp(prefix="littlebit-clone-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pilot = os.path.join(self.tmp, "results", "quality-pilot")
        os.makedirs(self.pilot)
        for name in PUBLIC_PILOT_FILES:
            shutil.copy2(os.path.join(PILOT_DIR, name), self.pilot)
        self.evidence = os.path.join(self.tmp, "evidence")
        os.makedirs(self.evidence)
        self.metadata = os.path.join(self.pilot, "public_metadata.json")
        self.summary = os.path.join(self.tmp, "results", "summary.json")
        self.absent_weights = os.path.join(self.tmp, "models", "absent", "model.safetensors")

    def argv(self, *extra, metadata_only=True, summary=None):
        a = ["--no-figures", "--pilot-dir", self.pilot, "--evidence-dir", self.evidence,
             "--summary", summary or self.summary, "--original-weights", self.absent_weights]
        return (["--metadata-only"] if metadata_only else []) + a + list(extra)


class TestMetadataReplayCleanClone(CloneMixin, unittest.TestCase):
    def setUp(self):
        self.make_clone()

    def test_reads_only_public_files(self):
        real_open = builtins.open
        opened = []

        def spy(file, mode="r", *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and not any(c in mode for c in "wax+"):
                opened.append(os.path.realpath(file))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", spy):
            code, _, err = run_report(self.argv())
        self.assertEqual(code, 0, err)
        allowed = {os.path.realpath(os.path.join(self.pilot, n)) for n in PUBLIC_PILOT_FILES}
        self.assertTrue(opened)
        self.assertTrue(set(opened) <= allowed, sorted(set(opened) - allowed))
        self.assertIn(os.path.realpath(self.metadata), opened)
        self.assertFalse(os.path.exists(os.path.join(self.pilot, "checkpoints")))
        self.assertEqual(os.listdir(self.evidence), [])

    def test_rebuilds_recorded_numbers(self):
        code, out, err = run_report(self.argv())
        self.assertEqual(code, 0, err)
        self.assertIn("metadata_replay", out)
        s = load(self.summary)
        c = s["heldout_test"]["conditions"]
        self.assertEqual("%.4f" % c["original"]["ppl"], "28.8816")
        self.assertEqual("%.4f" % c["initialized_no_qat"]["ppl"], "82426.3372")
        self.assertEqual("%.4f" % c["qat_trained"]["ppl"], "9735.6350")
        self.assertEqual("%.4f" % c["qat_trained_upstream_decoder"]["ppl"], "13558558.4902")
        self.assertEqual(c["qat_trained"]["windows_worse_than_original"], 8)
        self.assertEqual(s["heldout_test"]["qat_minus_initialized"]["windows_improved"], 8)
        st = s["storage"]
        self.assertEqual(st["disk"]["original"]["bytes"], 988097824)
        self.assertEqual(st["disk"]["original"]["embedding_bytes"], 272269312)
        self.assertEqual(st["disk"]["student"]["bytes"], 299683600)
        self.assertEqual(st["disk"]["student"]["frozen_base_embedding_bytes"], 272269312)
        self.assertEqual(st["resident_fp32_unique"]["original_bytes"], 1976131200)
        self.assertEqual(st["resident_fp32_unique"]["student_bytes"], 1234703616)
        self.assertEqual(st["excluded"]["tokenizer_and_config_files_bytes"], 15879995)
        self.assertEqual(round(st["bpw_converted_block_linears"]["advertised_upstream_formula"], 5), 0.52955)
        self.assertEqual(round(st["bpw_converted_block_linears"]["actual_checkpoint_file_fp32_scales"], 4), 0.6094)
        t = s["training"]
        self.assertEqual((t["input_tokens"], t["supervised_next_token_targets"]), (2048, 2032))
        self.assertEqual(t["sign_flips_init_to_trained"], 360930)
        self.assertTrue(t["train_steps_jsonl_matches_step_log"])
        self.assertEqual(round(s["resources"]["train"]["peak_rss_os_gib"], 2), 6.53)
        held = {p["id"]: p["held"] for p in s["preregistered_predictions"]}
        self.assertEqual(held, {"P1": True, "P2": True, "P3": True, "P4": True, "P5": False, "P6": True})
        self.assertEqual((s["tests"]["host"]["ran"], s["tests"]["container"]["ran"]), (95, 95))

    def test_replay_is_labeled_and_keeps_scientific_limits(self):
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 0, err)
        s = load(self.summary)
        r = s["replay"]
        self.assertEqual(r["mode"], rr.MODE_METADATA)
        self.assertEqual(r["private_files_read"], [])
        self.assertEqual(r["private_checks"], [])
        self.assertIn("does not validate the private checkpoints", r["statement"])
        ident = s["identity"]
        for key in ("identity_init", "identity_trained"):
            self.assertIn("one 128-token validation window", ident[key]["compared_inputs"])
            self.assertFalse(ident[key]["official_loader_ran"])
        self.assertIn("synthetic layer file", ident["official_state_dict_loader"]["scope"])
        self.assertIn("not run", ident["official_state_dict_loader"]["full_model_loader"])
        self.assertIn("descriptive only", s["heldout_test"]["statistics"])
        self.assertIn("statistical significance", s["claims"]["not_claimed"])
        self.assertEqual(qp.find_host_paths(s), [])

    def test_missing_public_metadata_fails(self):
        os.remove(self.metadata)
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 2)
        self.assertIn("public_metadata.json", err)
        self.assertFalse(os.path.exists(self.summary))

    def test_metadata_must_agree_with_public_stage_records(self):
        cases = (("trained_compressed_layers", "file_bytes", 27257281, "trained checkpoint size"),
                 ("frozen_base_header", "source_sha256", "0" * 64, "sha256 differs"),
                 ("model_config", "hidden_size", 897, "disagrees with the model config"))
        for section, key, value, msg in cases:
            meta = load(os.path.join(PILOT_DIR, "public_metadata.json"))
            meta[section][key] = value
            dump(self.metadata, meta)
            code, _, err = run_report(self.argv())
            self.assertEqual(code, 2, section)
            self.assertIn(msg, err, section)

    def test_unsanitized_stage_record_rejected(self):
        path = os.path.join(self.pilot, "baseline.json")
        rec = load(path)
        rec["args"]["model_dir"] = "/Users/someone/project/models/qwen2.5-0.5b"
        dump(path, rec)
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 2)
        self.assertIn("host path", err)

    def test_write_flag_refused_in_metadata_mode(self):
        code, _, err = run_report(self.argv("--write-public-metadata"))
        self.assertEqual(code, 2)
        self.assertIn("cannot be combined", err)

    def test_tracked_summary_matches_fresh_replay(self):
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 0, err)
        fresh = load(self.summary)
        tracked = load(os.path.join(PROJECT_ROOT, "results", "summary.json"))
        self.assertEqual(tracked.get("schema"), rr.SCHEMA, STALE)
        self.assertEqual(set(tracked["inputs"]), set(fresh["inputs"]), STALE)
        for key, rec in tracked["inputs"].items():
            self.assertEqual((rec["bytes"], rec["sha256"]), (fresh["inputs"][key]["bytes"], fresh["inputs"][key]["sha256"]),
                             "%s (%s)" % (STALE, key))
        # Exact except for derived floats, which may differ in the last bits across Python versions.
        diffs = json_differences(without(tracked, "inputs", "figures", "replay"),
                                 without(fresh, "inputs", "figures", "replay"))
        self.assertEqual(diffs, [], "%s; first differences: %s" % (STALE, diffs[:5]))
        self.assertEqual(qp.find_host_paths(tracked), [])
        for rec in tracked["figures"]:
            self.assertEqual(sha256(os.path.join(PROJECT_ROOT, rec["path"])), rec["sha256"], "%s (%s)" % (STALE, rec["path"]))


class TestPrivateValidationSynthetic(CloneMixin, unittest.TestCase):
    """Synthetic private files with a tiny config; numbers are meaningless, the checks are the point."""

    def setUp(self):
        self.make_clone()
        os.remove(self.metadata)
        ck = os.path.join(self.pilot, "checkpoints")
        for sub in ("config_and_tokenizer", "init", "trained"):
            os.makedirs(os.path.join(ck, sub))
        cfg = os.path.join(ck, "config_and_tokenizer", "config.json")
        dump(cfg, {"vocab_size": 8, "hidden_size": 4, "torch_dtype": "bfloat16", "tie_word_embeddings": True})
        self.frozen = os.path.join(ck, "frozen_base.safetensors")
        write_safetensors(self.frozen, {"model.embed_tokens.weight": ("BF16", [8, 4], bytes(range(64))),
                                        "model.norm.weight": ("BF16", [4], b"\x01" * 8)})
        init = os.path.join(ck, "init", "compressed_layers.safetensors")
        write_safetensors(init, {"m.U_packed": ("I32", [2], b"\x00" * 8)})
        self.trained = os.path.join(ck, "trained", "compressed_layers.safetensors")
        write_safetensors(self.trained, {"m.U_packed": ("I32", [2], b"\xff" * 8)})
        train_path = os.path.join(self.pilot, "train.json")
        tr = load(train_path)
        c = tr["checkpoints"]
        for rec, path in ((c["frozen_base"], self.frozen), (c["init"], init), (c["trained"], self.trained),
                          (c["config_and_tokenizer"]["config.json"], cfg)):
            rec["bytes"], rec["sha256"] = os.path.getsize(path), sha256(path)
        dump(train_path, tr)
        shutil.copy2(os.path.join(PROJECT_ROOT, "evidence", "official-loader-integration.json"), self.evidence)
        for name, text in (("unit-tests-quality.log", "Ran 3 tests in 0.250s\n\nOK\n"),
                           ("container-tests-quality.log", "Ran 3 tests in 0.300s\n\nOK\n"),
                           ("container-image.txt", "sha256:" + "a" * 64 + " arm64\n")):
            with open(os.path.join(self.evidence, name), "w", encoding="utf-8") as f:
                f.write(text)
        self.all_but_original = [k for k in rr.METADATA_SECTIONS if k != "original_weights_header"]

    def write_metadata(self):
        code, _, err = run_report(self.argv("--write-public-metadata", metadata_only=False))
        self.assertEqual(code, 0, err)
        return load(self.summary)

    def test_round_trip_and_mode_invariance(self):
        first = self.write_metadata()
        self.assertEqual(first["replay"]["mode"], rr.MODE_PRIVATE)
        self.assertEqual(first["replay"]["public_metadata_sections_written"], self.all_but_original)
        self.assertEqual(first["replay"]["public_metadata_sections_not_rechecked"], ["original_weights_header"])
        self.assertTrue(any("sha256 and size equal the train.json record" in c for c in first["replay"]["private_checks"]))
        meta = load(self.metadata)
        self.assertEqual(meta["schema"], rr.PUBLIC_METADATA_SCHEMA)
        self.assertEqual(meta["frozen_base_header"]["embed_tokens"], {"dtype": "BF16", "shape": [8, 4], "bytes": 64})
        self.assertEqual(meta["test_logs"]["host"]["ran"], 3)
        self.assertEqual(qp.find_host_paths(meta), [])

        private_summary = os.path.join(self.tmp, "private.json")
        code, _, err = run_report(self.argv(metadata_only=False, summary=private_summary))
        self.assertEqual(code, 0, err)
        private = load(private_summary)
        self.assertEqual(private["replay"]["public_metadata_sections_verified"], self.all_but_original)

        replay_summary = os.path.join(self.tmp, "replay.json")
        code, _, err = run_report(self.argv(summary=replay_summary))
        self.assertEqual(code, 0, err)
        replay = load(replay_summary)
        self.assertEqual(replay["replay"]["mode"], rr.MODE_METADATA)
        self.assertEqual(without(private, "replay"), without(replay, "replay"))

    def test_changed_checkpoint_caught_only_by_private_validation(self):
        self.write_metadata()
        with open(self.trained, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            f.write(b"\x00")
        code, _, err = run_report(self.argv(metadata_only=False))
        self.assertEqual(code, 2)
        self.assertIn("sha256 differs from train.json record", err)
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 0, err)
        self.assertIn("does not validate the private checkpoints", load(self.summary)["replay"]["statement"])

    def test_metadata_disagreeing_with_private_files_is_fatal(self):
        self.write_metadata()
        meta = load(self.metadata)
        meta["test_logs"]["host"]["ran"] = 4
        dump(self.metadata, meta)
        code, _, err = run_report(self.argv(metadata_only=False))
        self.assertEqual(code, 2)
        self.assertIn("test_logs.host disagrees", err)
        code, _, err = run_report(self.argv())
        self.assertEqual(code, 0, err)

    def test_private_validation_needs_checkpoints(self):
        self.write_metadata()
        shutil.rmtree(os.path.join(self.pilot, "checkpoints"))
        code, _, err = run_report(self.argv(metadata_only=False))
        self.assertEqual(code, 2)
        self.assertIn("--metadata-only", err)


@unittest.skipUnless(os.path.exists(REAL_TRAINED), "private checkpoints not present (clean clone)")
class TestRealPrivateValidation(unittest.TestCase):
    """Owner host only: the real checkpoints must agree with the tracked public metadata."""

    def test_private_validation_agrees_with_metadata_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p_path, m_path = os.path.join(d, "private.json"), os.path.join(d, "replay.json")
            code, _, err = run_report(["--no-figures", "--summary", p_path])
            self.assertEqual(code, 0, err)
            code, _, err = run_report(["--metadata-only", "--no-figures", "--summary", m_path])
            self.assertEqual(code, 0, err)
            private, replay = load(p_path), load(m_path)
        verified = private["replay"]["public_metadata_sections_verified"]
        for key in ("model_config", "frozen_base_header", "trained_compressed_layers"):
            self.assertIn(key, verified)
        self.assertEqual(without(private, "replay"), without(replay, "replay"))


class TestDerivedFloatComparison(unittest.TestCase):
    """The summary comparison tolerates last-bit float variance and nothing else."""

    BASE = {"inputs_sha256": "86833880f0c25fa89a9d24b2b193749104f15d2cd90cf345164be14fe86d98a1",
            "scored_tokens": 1016, "held": True, "note": None,
            "summary": {"mean": 13.059324022000348, "max_abs_dev": 0.0},
            "per_window": [7.698936105137242, 8.714306935127393]}

    def changed(self, path, value):
        obj = json.loads(json.dumps(self.BASE))
        node = obj
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return obj

    def test_last_bit_variance_passes(self):
        self.assertEqual(json_differences(self.BASE, self.BASE), [])
        observed = self.changed(("summary", "mean"), 13.059324022000347)
        self.assertEqual(json_differences(self.BASE, observed), [])
        x = self.BASE["per_window"][1]
        neighbor = struct.unpack("<d", struct.pack("<q", struct.unpack("<q", struct.pack("<d", x))[0] + 2))[0]
        self.assertNotEqual(neighbor, x)
        self.assertEqual(json_differences(self.BASE, self.changed(("per_window", 1), neighbor)), [])
        self.assertEqual(json_differences(self.BASE, self.changed(("summary", "max_abs_dev"), 1e-15)), [])

    def test_exact_fields_and_structure_fail(self):
        cases = {
            "hash": self.changed(("inputs_sha256",), "0" + self.BASE["inputs_sha256"][1:]),
            "count": self.changed(("scored_tokens",), 1017),
            "count as float": self.changed(("scored_tokens",), 1016.0),
            "boolean": self.changed(("held",), False),
            "boolean as int": self.changed(("held",), 1),
            "null": self.changed(("note",), "x"),
            "missing key": {k: v for k, v in self.BASE.items() if k != "note"},
            "list length": self.changed(("per_window",), self.BASE["per_window"][:1]),
        }
        for name, obj in cases.items():
            self.assertNotEqual(json_differences(self.BASE, obj), [], name)

    def test_material_float_change_fails(self):
        mean = self.BASE["summary"]["mean"]
        for value in (13.0594, mean * (1 + 1e-9), mean * (1 + 1e-11)):
            self.assertNotEqual(json_differences(self.BASE, self.changed(("summary", "mean"), value)), [], value)
        self.assertNotEqual(json_differences(self.BASE, self.changed(("summary", "max_abs_dev"), 1e-9)), [])


class TestReportModule(unittest.TestCase):
    def test_top_level_imports(self):
        allowed = {"__future__", "argparse", "copy", "hashlib", "json", "math", "os", "re", "struct", "sys",
                   "typing", "quality_pilot"}
        with open(os.path.join(PROJECT_ROOT, "src", "report_results.py"), "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertIn("quality_pilot", names)
        self.assertTrue(names <= allowed, names - allowed)

    def test_no_subprocess_or_network(self):
        with open(os.path.join(PROJECT_ROOT, "src", "report_results.py"), "r", encoding="utf-8") as f:
            src = f.read()
        for bad in ("import subprocess", "import socket", "urllib", "requests", "http.client", "import torch"):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main()
