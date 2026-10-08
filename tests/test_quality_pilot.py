"""Unit tests for src/quality_pilot.py.

Run from the project root:  python -m unittest tests.test_quality_pilot -v

Standard-library tests cover window freezing, hashing, revision parsing, aggregation, schedule,
feasibility estimate, resource guard and the CLI. Torch tests (skipped without torch) cover the
loss function and the packed-save checks on a single seeded upstream LittleBitLinear; they are
unit tests of functions, never a quality result. Nothing here loads the 0.5B weights.
"""

import ast
import contextlib
import io
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import quality_pilot as qp  # noqa: E402


def _torch():
    try:
        import torch
        return torch
    except Exception:
        return None


TORCH = _torch()
NEEDS_TORCH = unittest.skipUnless(TORCH is not None, "torch not importable")
NEEDS_PY310 = unittest.skipUnless(sys.version_info >= (3, 10), "upstream littlebit.py needs Python >= 3.10")
UPSTREAM_ROOT = os.path.join(PROJECT_ROOT, "upstream", "LittleBit")
NEEDS_UPSTREAM_SOURCE = unittest.skipUnless(
    all(os.path.exists(os.path.join(UPSTREAM_ROOT, *rel.split("/")))
        for rel in ("quantization/utils/binary_packer.py", "quantization/modules/littlebit.py")),
    "upstream/LittleBit checkout not present (not redistributed): upstream-dependent test skipped")


def fake_tokenize(text):
    """Deterministic stand-in tokenizer: one id per character (tests only)."""
    return [ord(c) for c in text]


class TestCliAndImports(unittest.TestCase):
    def test_help_exits_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            qp.main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        for stage in ("prepare", "baseline", "feasibility", "train", "evaluate"):
            self.assertIn(stage, text)

    def test_stage_help_exits_zero(self):
        for stage in ("prepare", "baseline", "feasibility", "train", "evaluate"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
                qp.main([stage, "--help"])
            self.assertEqual(cm.exception.code, 0, stage)

    def test_defaults(self):
        a = qp.build_parser().parse_args(["train"])
        self.assertEqual(a.steps, 16)
        self.assertEqual(a.batch_size, 1)
        self.assertEqual(a.eff_bit, 0.55)
        self.assertTrue(a.residual)
        self.assertEqual(a.threads, 2)
        self.assertEqual(a.device, "cpu")
        self.assertEqual(a.max_rss_gib, 6.0)
        self.assertEqual(a.min_available_gib, 2.0)
        self.assertEqual(a.max_seconds, 900.0)
        self.assertEqual(a.seed, 42)
        self.assertEqual(a.l2l_scale, 0.0)
        self.assertFalse(a.upstream_decoder_variant)
        self.assertFalse(qp.build_parser().parse_args(["train", "--no-residual"]).residual)
        self.assertEqual(qp.config_from_args(a)["use_itq"], False)
        self.assertEqual(qp.config_from_args(a)["quant_func"], "SmoothSign")

    def test_prepare_requires_data_dir(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            qp.build_parser().parse_args(["prepare"])

    def test_device_choices(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            qp.build_parser().parse_args(["baseline", "--device", "cuda"])

    def test_top_level_imports_are_stdlib(self):
        allowed = {"__future__", "argparse", "copy", "gc", "hashlib", "json", "math", "os", "platform",
                   "random", "re", "struct", "sys", "time", "typing"}
        with open(os.path.join(PROJECT_ROOT, "src", "quality_pilot.py"), "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertTrue(names <= allowed, names - allowed)

    def test_no_subprocess_or_network_modules(self):
        with open(os.path.join(PROJECT_ROOT, "src", "quality_pilot.py"), "r", encoding="utf-8") as f:
            src = f.read()
        for bad in ("import subprocess", "import socket", "urllib", "requests", "http.client",
                    "snapshot_download", "hf_hub_download", "load_dataset("):
            self.assertNotIn(bad, src)


class TestWindows(unittest.TestCase):
    def test_build_windows_contiguous_non_overlapping(self):
        ids = list(range(1000))
        ws = qp.build_windows(ids, 4, 128)
        self.assertEqual([w["start"] for w in ws], [0, 128, 256, 384])
        for w in ws:
            self.assertEqual(w["n_tokens"], 128)
            self.assertEqual(w["scored_tokens"], 127)
            self.assertEqual(w["token_ids"], ids[w["start"]:w["end"]])
            self.assertEqual(w["sha256"], qp.hash_tokens(w["token_ids"]))

    def test_build_windows_insufficient(self):
        with self.assertRaises(ValueError):
            qp.build_windows(list(range(100)), 1, 128)

    def test_hash_is_little_endian_int32(self):
        self.assertEqual(qp.token_bytes([1, 256]), b"\x01\x00\x00\x00\x00\x01\x00\x00")
        self.assertEqual(qp.hash_tokens([1, 2]), qp.hash_tokens([1, 2]))
        self.assertNotEqual(qp.hash_tokens([1, 2]), qp.hash_tokens([2, 1]))

    def test_select_prefix_margin_and_rows(self):
        rows = ["abcdefghij"] * 100
        sel = qp.select_prefix_rows(rows, fake_tokenize, needed=200, margin=64, start_rows=4)
        self.assertGreaterEqual(sel["prefix_token_count"], 264)
        self.assertEqual(len(sel["token_ids"]), 200)
        self.assertEqual(sel["row_start"], 0)
        self.assertEqual(sel["row_end"], 32)  # k rows give 12k - 2 tokens: 16 rows 190, 32 rows 382

    def test_select_prefix_too_short(self):
        with self.assertRaises(ValueError):
            qp.select_prefix_rows(["ab"] * 3, fake_tokenize, needed=100, margin=0)

    def test_splits_never_cross(self):
        rows = {"train": ["a" * 50] * 200, "validation": ["b" * 50] * 200, "test": ["c" * 50] * 200}
        out = qp.make_split_windows(rows, fake_tokenize, {"train": 16, "validation": 4, "test": 8}, 128)
        marks = {"train": {ord("a"), ord("\n")}, "validation": {ord("b"), ord("\n")}, "test": {ord("c"), ord("\n")}}
        for split, info in out.items():
            self.assertEqual(len(info["windows"]), qp.PILOT_WINDOWS[split])
            self.assertEqual(info["role"], qp.SPLIT_ROLES[split])
            for w in info["windows"]:
                self.assertTrue(set(w["token_ids"]) <= marks[split], split)

    def test_pilot_sizes_preregistered(self):
        self.assertEqual(qp.PILOT_WINDOWS, {"train": 16, "validation": 4, "test": 8})
        self.assertEqual(qp.SEQ_LEN, 128)
        self.assertEqual(qp.SEED, 42)


class TestProvenance(unittest.TestCase):
    def test_snapshot_revision(self):
        rev = qp.DATASET_REVISION
        p = "/x/datasets--Salesforce--wikitext/snapshots/%s/wikitext-2-raw-v1/test-00000-of-00001.parquet" % rev
        self.assertEqual(qp.parse_snapshot_revision(p), rev)
        self.assertIsNone(qp.parse_snapshot_revision("/data/wikitext-2-raw-v1/test.parquet"))
        self.assertIsNone(qp.parse_snapshot_revision("/x/snapshots/abc/test.parquet"))

    def test_pins(self):
        self.assertEqual(qp.MODEL_REVISION, "060db6499f32faf8b98477b0a26969ef7d8b9987")
        self.assertEqual(qp.DATASET_REVISION, "b08601e04326c79dfdd32d625aee71d232d685c3")
        self.assertEqual(qp.EXPECTED_ROWS, {"train": 36718, "validation": 3760, "test": 4358})

    def test_local_model_revision_metadata(self):
        if not os.path.isdir(os.path.join(qp.DEFAULT_MODEL_DIR, ".cache")):
            self.skipTest("local model metadata not present")
        rev = qp.read_model_revision(qp.DEFAULT_MODEL_DIR)
        self.assertTrue(rev["consistent"])
        self.assertTrue(rev["matches_pinned"])
        self.assertEqual(len(rev["etags"].get("model.safetensors", "")), 64)

    def test_find_split_files_missing(self):
        with self.assertRaises(qp.PreconditionError):
            qp.find_split_files(os.path.join(PROJECT_ROOT, "tests"))


class TestAggregation(unittest.TestCase):
    def test_token_weighted(self):
        agg = qp.token_weighted([127.0, 254.0], [127, 127])
        self.assertAlmostEqual(agg["mean_nll"], 1.5)
        self.assertAlmostEqual(agg["ppl"], 2.718281828459045 ** 1.5)
        self.assertEqual(agg["scored_tokens"], 254)

    def test_token_weighted_not_window_mean(self):
        agg = qp.token_weighted([10.0, 0.0], [10, 30])
        self.assertAlmostEqual(agg["mean_nll"], 0.25)

    def test_token_weighted_empty(self):
        with self.assertRaises(ValueError):
            qp.token_weighted([], [])

    def test_paired_diffs(self):
        a = [{"index": 0, "sha256": "x", "nll_sum": 127.0, "scored_tokens": 127}]
        b = [{"index": 0, "sha256": "x", "nll_sum": 254.0, "scored_tokens": 127}]
        self.assertAlmostEqual(qp.paired_window_diffs(a, b)[0]["mean_nll_diff"], 1.0)
        b[0]["sha256"] = "y"
        with self.assertRaises(ValueError):
            qp.paired_window_diffs(a, b)


class TestScheduleAndEstimate(unittest.TestCase):
    def test_cosine(self):
        self.assertEqual(qp.cosine_lr(4e-5, 0, 16), 4e-5)
        self.assertAlmostEqual(qp.cosine_lr(4e-5, 8, 16), 2e-5)
        self.assertGreater(qp.cosine_lr(4e-5, 15, 16), 0.0)
        with self.assertRaises(ValueError):
            qp.cosine_lr(1.0, 0, 0)

    def test_feasibility_estimate(self):
        est = qp.feasibility_estimate(60.0, [10.0, 20.0], 16, 900.0)
        self.assertAlmostEqual(est["estimated_train_stage_seconds_mean"], 60 + 16 * 15)
        self.assertAlmostEqual(est["estimated_train_stage_seconds_conservative"], 60 + 16 * 20)
        self.assertTrue(est["fits_budget_conservative"])
        self.assertFalse(qp.feasibility_estimate(60.0, [60.0], 16, 900.0)["fits_budget_conservative"])
        with self.assertRaises(ValueError):
            qp.feasibility_estimate(1.0, [], 16, 900.0)


class TestResourceGuard(unittest.TestCase):
    def make(self, rss, avail, t):
        clock = iter(t)
        return qp.ResourceGuard(6 * qp.GIB, 2 * qp.GIB, 900.0, rss_fn=lambda: rss, available_fn=lambda: avail,
                                clock=lambda: next(clock))

    def test_ok(self):
        g = self.make(1 * qp.GIB, 8 * qp.GIB, [0.0, 1.0, 1.0])
        snap = g.check("x")
        self.assertEqual(snap["rss_bytes"], qp.GIB)
        self.assertEqual(g.checks, 1)

    def test_rss_limit(self):
        g = self.make(7 * qp.GIB, 8 * qp.GIB, [0.0, 1.0])
        with self.assertRaisesRegex(qp.GuardStop, "RSS"):
            g.check("x")

    def test_available_limit(self):
        g = self.make(1 * qp.GIB, 1 * qp.GIB, [0.0, 1.0])
        with self.assertRaisesRegex(qp.GuardStop, "available"):
            g.check("x")

    def test_time_limit(self):
        g = self.make(1 * qp.GIB, 8 * qp.GIB, [0.0, 901.0])
        with self.assertRaisesRegex(qp.GuardStop, "elapsed"):
            g.check("x")


class TestClaims(unittest.TestCase):
    def test_claims_block(self):
        c = qp.claims_block(False, "no steps")
        self.assertFalse(c["model_trained"])
        self.assertEqual(c["task_accuracy"], "not measured")
        self.assertEqual(c["throughput"], "not measured")
        self.assertIn("MISSING", c["baselines"]["practical_4bit"])
        self.assertTrue(c["initialization_quality_is_separate_from_training"])

    def test_generation_bounded(self):
        self.assertLessEqual(qp.DEFAULT_MAX_NEW_TOKENS, qp.MAX_NEW_TOKENS_CAP)
        self.assertLessEqual(len(qp.GENERATION_PROMPTS), 5)


class TestDeliverableStyle(unittest.TestCase):
    def test_no_em_dashes(self):
        for rel in ("src/quality_pilot.py", "tests/test_quality_pilot.py", "QUALITY_PREREGISTRATION.md"):
            path = os.path.join(PROJECT_ROOT, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                self.assertNotIn(chr(0x2014), f.read(), rel)


@NEEDS_TORCH
class TestLosses(unittest.TestCase):
    def test_kl_zero_when_equal_and_ce_matches(self):
        import torch
        import torch.nn.functional as F
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, 6, 11, generator=g)
        ids = torch.randint(0, 11, (1, 6), generator=g)
        out = qp.distill_losses(torch, logits, logits.clone(), ids, 1.0, 1.0)
        self.assertLess(abs(float(out["kl"])), 1e-6)
        ce = F.cross_entropy(logits[0, :-1], ids[0, 1:])
        self.assertAlmostEqual(float(out["ce"]), float(ce), places=6)
        self.assertAlmostEqual(float(out["loss"]), float(out["kl"]) + float(out["ce"]), places=6)

    def test_kl_positive_and_teacher_detached(self):
        import torch
        g = torch.Generator().manual_seed(1)
        s = torch.randn(1, 5, 7, generator=g, requires_grad=True)
        t = torch.randn(1, 5, 7, generator=g, requires_grad=True)
        ids = torch.randint(0, 7, (1, 5), generator=g)
        out = qp.distill_losses(torch, s, t, ids, 1.0, 0.0)
        self.assertGreater(float(out["kl"]), 0.0)
        out["loss"].backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)

    def test_l2l_requires_hidden(self):
        import torch
        x = torch.zeros(1, 3, 4)
        with self.assertRaises(ValueError):
            qp.distill_losses(torch, x, x, torch.zeros(1, 3, dtype=torch.long), 1.0, 1.0, l2l_scale=1.0)

    def test_popcount_flips(self):
        import torch
        a = torch.tensor([[0, -1]], dtype=torch.int32)
        b = torch.tensor([[1, 0]], dtype=torch.int32)
        self.assertEqual(qp.popcount_flips(torch, a, b), 1 + 32)

    def test_token_weighted_matches_window_nll_definition(self):
        import torch
        import torch.nn.functional as F

        class Fixed(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = logits

            def forward(self, input_ids, use_cache=False):
                return type("O", (), {"logits": self.logits})()

        g = torch.Generator().manual_seed(2)
        logits = torch.randn(1, 5, 9, generator=g)
        ids = torch.randint(0, 9, (1, 5), generator=g)
        s, n = qp.window_nll(torch, Fixed(logits), ids)
        self.assertEqual(n, 4)
        ref = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
        self.assertAlmostEqual(s, float(ref), places=4)


@NEEDS_TORCH
@NEEDS_PY310
@NEEDS_UPSTREAM_SOURCE
class TestPackedSaveChecks(unittest.TestCase):
    """One seeded upstream LittleBitLinear; checks the save-time decoder guard, not quality."""

    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.up = qp.load_upstream_api()
        torch.manual_seed(0)
        lin = torch.nn.Linear(96, 160, bias=True)
        lin.__class__ = cls.up["LittleBitLinear"]
        lin.__quant_convert__(do_train=True, quant_func=cls.up["SmoothSign"], eff_bit=1.0, residual=True,
                              split_dim=1024, min_split_dim=8)
        cls.mod = lin

    def test_corrected_decoder_passes_and_tensors_named(self):
        import torch
        tensors, checks = qp.pack_module_tensors(torch, "m", self.mod, qp.corrected_decoder())
        self.assertEqual(sorted(checks), ["U", "U_R", "V", "V_R"])
        for f in ("U", "V", "U_R", "V_R"):
            self.assertEqual(tensors["m.%s_packed" % f].dtype, torch.int32)
        for s in ("u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R"):
            self.assertEqual(tensors["m.%s" % s].dtype, torch.float32)
        self.assertIn("m._split_dim_final", tensors)
        self.assertNotIn("m.bias", tensors)

    def test_upstream_decoder_is_refused(self):
        import torch
        with self.assertRaises(RuntimeError):
            qp.pack_module_tensors(torch, "m", self.mod, self.up["binary_unpacker"])

    def test_trainable_selection(self):
        import torch
        model = torch.nn.Sequential(self.mod)
        names = [n for n, _ in qp.trainable_factor_params(model, self.up["LittleBitLinear"])]
        self.assertIn("0.U", names)
        self.assertIn("0.v2_R", names)
        self.assertFalse(any(n.endswith("bias") for n in names))
        self.assertFalse(self.mod.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
