# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted Material under CC BY-NC 4.0 (LICENSES/CC-BY-NC-4.0.txt; LICENSE, Part 2). Licensed
# Material: The LittleBit Project, https://github.com/SamsungLabs/LittleBit, commit
# 933857ed1443b53fc43a875c2cf64249e3c56f0c, CC BY-NC 4.0; methods by Banseok Lee, Dongkyu Kim,
# Youngcheon You and Youngmin Kim. Changes, Copyright (c) 2026 Tim Urista: tests that mirror
# upstream arithmetic and quote short upstream expressions in test data and assertions.
# NonCommercial use only. Not affiliated with or endorsed by the LittleBit authors or Samsung.
"""Deterministic unit tests for the LittleBit deployment audit.

Run from the project root:  python -m unittest discover -s tests -v

Tests in TestUpstream* need torch and are skipped with an explicit reason otherwise. Everything
else is standard library only (ALGORITHM_AUDIT tier).
"""

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import audit  # noqa: E402

TORCH = audit.try_import_torch()
NEEDS_TORCH = unittest.skipUnless(TORCH is not None, "torch not importable: UPSTREAM_IMPORT tier skipped")
NEEDS_PY310 = unittest.skipUnless(sys.version_info >= (3, 10), "upstream littlebit.py needs Python >= 3.10")
# The pinned upstream checkout (upstream/LittleBit, CC BY-NC 4.0) is not redistributed; a clean
# clone, including CI, does not have it.
UPSTREAM_PRESENT = all(os.path.exists(os.path.join(audit.UPSTREAM_ROOT, *rel.split("/")))
                       for rel in ("quantization/utils/binary_packer.py", "quantization/modules/littlebit.py"))
NEEDS_UPSTREAM_SOURCE = unittest.skipUnless(
    UPSTREAM_PRESENT, "upstream/LittleBit checkout not present (not redistributed): upstream-dependent test skipped")


class TestInt32Helpers(unittest.TestCase):
    def test_to_int32_wraps(self):
        self.assertEqual(audit.to_int32(0), 0)
        self.assertEqual(audit.to_int32(2 ** 31 - 1), 2 ** 31 - 1)
        self.assertEqual(audit.to_int32(2 ** 31), -(2 ** 31))
        self.assertEqual(audit.to_int32(2 ** 32 - 1), -1)
        self.assertEqual(audit.to_int32(2 ** 32), 0)
        self.assertEqual(audit.to_int32(-1), -1)

    def test_words_per_row(self):
        expected = {1: 1, 7: 1, 8: 1, 31: 1, 32: 1, 33: 2, 64: 2, 65: 3}
        for n, w in expected.items():
            self.assertEqual(audit.words_per_row(n), w, n)


class TestMirrorPacker(unittest.TestCase):
    def test_known_words(self):
        self.assertEqual(audit.mirror_pack([[1]]), [[0]])
        self.assertEqual(audit.mirror_pack([[-1]]), [[1]])
        self.assertEqual(audit.mirror_pack([[1, -1]]), [[2]])
        self.assertEqual(audit.mirror_pack([[-1] * 32]), [[-1]])
        self.assertEqual(audit.mirror_pack([[1] * 31 + [-1]]), [[-(2 ** 31)]])
        self.assertEqual(audit.mirror_pack([[-1] * 33]), [[-1, 1]])
        self.assertEqual(audit.mirror_pack([[1] * 64]), [[0, 0]])

    def test_padding_is_plus_one(self):
        # Width 7 padded with +1 must equal explicit +1 columns up to 32.
        row = [-1, 1, -1, 1, 1, -1, -1]
        self.assertEqual(audit.mirror_pack([row]), audit.mirror_pack([row + [1] * 25]))

    def test_first_exploratory_counterexample_word(self):
        self.assertEqual(audit.mirror_pack([[1, -1, 1, -1, -1, 1, 1, -1]]), [[154]])

    def test_unvalidated_inputs(self):
        # Upstream does not validate values; 0 packs like +1 (bit 0). Not reachable via
        # pack_param, which binarizes first, so this documents the input contract only.
        self.assertEqual(audit.mirror_pack([[0, -1]]), audit.mirror_pack([[1, -1]]))


class TestReferenceDecoders(unittest.TestCase):
    """H2 under the mirror packer: both independent decoders recover every case exactly."""

    def test_exhaustive_cases_roundtrip(self):
        for case in audit.generate_cases():
            row = case["row"]
            packed = audit.mirror_pack([row])
            shape = (1, len(row))
            self.assertEqual(audit.reference_unpack_shift(packed, shape), [row], case["case_id"])
            self.assertEqual(audit.reference_unpack_bytes(packed, shape), [row], case["case_id"])

    def test_multi_row_matrix(self):
        for width in audit.WIDTHS:
            rows = [audit.byte_tiled_row(width, b) for b in range(256)]
            packed = audit.mirror_pack(rows)
            self.assertEqual(audit.reference_unpack_shift(packed, (256, width)), rows)
            self.assertEqual(audit.reference_unpack_bytes(packed, (256, width)), rows)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            audit.reference_unpack_shift([[0]], (1, 33))


class TestUpstreamUnpackMirror(unittest.TestCase):
    """H1 under the ALGORITHM_AUDIT tier."""

    @classmethod
    def setUpClass(cls):
        cls.rows = audit.run_mirror_cases()
        cls.summary = audit.summarize_cases(cls.rows)

    def test_case_count(self):
        widths = audit.WIDTHS
        expected = sum(256 + w + 6 for w in widths)
        self.assertEqual(len(self.rows), expected)

    def test_prediction_holds_everywhere(self):
        bad = [r["case_id"] for r in self.rows if not r["prediction_matches"]]
        self.assertEqual(bad, [])

    def test_preregistered_byte_counts(self):
        for width, n in audit.PREREGISTERED_EXACT_BYTE_ROWS.items():
            self.assertEqual(self.summary[str(width)]["byte_tiled_exact_rows"], n, width)

    def test_errors_only_flip_minus_to_plus(self):
        for r in self.rows:
            self.assertEqual(r["sign_errors"], r["minus_to_plus_flips"], r["case_id"])

    def test_one_hot_exact_iff_word_start(self):
        for r in self.rows:
            if r["family"] == "one_hot":
                self.assertEqual(r["upstream_exact"], r["param"] % 32 == 0, r["case_id"])

    def test_sign_examples(self):
        by_id = {r["case_id"]: r for r in self.rows}
        for w in audit.WIDTHS:
            self.assertTrue(by_id["sign_w%d_all_plus" % w]["upstream_exact"])
            self.assertTrue(by_id["sign_w%d_minus_at_word_starts" % w]["upstream_exact"])
        self.assertTrue(by_id["sign_w1_all_minus"]["upstream_exact"])
        self.assertEqual(by_id["sign_w64_all_minus"]["sign_errors"], 62)
        self.assertEqual(by_id["sign_w8_all_minus"]["sign_errors"], 7)
        decoded = audit.mirror_unpack_upstream(audit.mirror_pack([[-1] * 64]), (1, 64))[0]
        self.assertEqual([j for j, v in enumerate(decoded) if v == -1], [0, 32])

    def test_minimal_counterexample(self):
        packed = audit.mirror_pack([[1, -1]])
        self.assertEqual(audit.mirror_unpack_upstream(packed, (1, 2)), [[1, 1]])
        self.assertEqual(audit.reference_unpack_shift(packed, (1, 2)), [[1, -1]])

    def test_left_shift_independent_of_sign(self):
        for word in (0, 1, -1, -(2 ** 31), 2 ** 31 - 1, 154):
            decoded = audit.mirror_unpack_upstream([[word]], (1, 32))[0]
            self.assertEqual(decoded[1:], [1] * 31)
            self.assertEqual(decoded[0], -1 if word & 1 else 1)

    def test_reference_decoders_never_fail(self):
        for s in self.summary.values():
            self.assertEqual(s["reference_decoder_failures"], 0)


class TestPriorEvidence(unittest.TestCase):
    def test_mirror_matches_recorded_upstream_run(self):
        res = audit.prior_evidence_check()
        if not res["available"]:
            self.skipTest("evidence file not present: %s" % res["reason"])
        self.assertTrue(res["recorded_commit_matches"])
        self.assertTrue(res["mirror_packed_matches_recorded"])
        self.assertTrue(res["mirror_unpacked_matches_recorded"])
        self.assertTrue(res["reference_recovers_input"])


class TestAccounting(unittest.TestCase):
    def test_min_split_floor_counterexample(self):
        est, s = audit.resolve_split(64, 64, 0.1, residual=False)
        self.assertLess(est, 0)
        self.assertEqual(s, 8)
        self.assertEqual(audit.upstream_eff_bits(64, 64, s, False), 0.78125)

    def test_llama_shapes(self):
        self.assertEqual(audit.resolve_split(4096, 4096, 0.1, False)[1], 184)
        self.assertEqual(audit.resolve_split(4096, 4096, 1.0, True)[1], 1000)

    def test_split_multiple_of_eight(self):
        for _, a, b in audit.ILLUSTRATIVE_LAYERS:
            for eff in audit.ILLUSTRATIVE_TARGETS:
                for res in (False, True):
                    s = audit.resolve_split(a, b, eff, res)[1]
                    self.assertEqual(s % 8, 0)
                    self.assertGreaterEqual(s, 8)

    def test_advertised_decomposition(self):
        for (a, b, s, res) in ((4096, 4096, 184, False), (4096, 11008, 1000, True), (100, 36, 8, True)):
            bd = audit.storage_breakdown(a, b, s, res)
            paths = 2 if res else 1
            self.assertEqual(bd["advertised_sign_bits"], paths * s * (a + b))
            self.assertEqual(bd["advertised_scale_bits"], paths * 16 * (a + b + s))
            self.assertAlmostEqual(bd["advertised_bpw"], audit.upstream_eff_bits(a, b, s, res))

    def test_row_padding(self):
        bd = audit.storage_breakdown(4096, 4096, 184, False)
        self.assertEqual(bd["u_row_padding_bits"], 4096 * 8)
        self.assertEqual(bd["v_row_padding_bits"], 0)
        bd = audit.storage_breakdown(4096, 4096, 192, False)
        self.assertEqual(bd["row_padding_bits"], 0)
        bd = audit.storage_breakdown(100, 36, 8, False)
        self.assertEqual(bd["u_row_padding_bits"], 36 * 24)
        self.assertEqual(bd["v_row_padding_bits"], 8 * 28)

    def test_double_middle_scale_overhead(self):
        for res in (False, True):
            bd = audit.storage_breakdown(4096, 4096, 1000, res)
            paths = 2 if res else 1
            self.assertEqual(bd["scale_overhead_vs_advertised_bits_main_py_bf16"], paths * 16 * 1000)

    def test_resident_vs_packed(self):
        bd = audit.storage_breakdown(4096, 4096, 192, False)
        ratios = bd["resident_over_packed_sign_ratio"]
        self.assertEqual(ratios["int8"], 8.0)
        self.assertEqual(ratios["bf16"], 16.0)
        self.assertEqual(ratios["fp32"], 32.0)
        self.assertGreater(bd["resident_bpw_bf16_factors_main_py_bf16"],
                           bd["serialized_bpw_main_py_bf16"])

    def test_serialized_bytes_main_profile(self):
        a, b, s = 4096, 4096, 192
        bd = audit.storage_breakdown(a, b, s, False)
        expected = (s * a + b * s) // 8 + 32 + (a + b + 2 * s) * 2 + (2 + 2 + 8)
        self.assertEqual(bd["serialized_bytes_main_py_bf16"], expected)

    def test_accounting_table_shape(self):
        rows = audit.accounting_table()
        self.assertEqual(len(rows), len(audit.ILLUSTRATIVE_LAYERS) * len(audit.ILLUSTRATIVE_TARGETS) * 2)
        self.assertTrue(all(r["tier"] == audit.TIER_MIRROR for r in rows))

    def test_phi_split_has_more_scale_elements(self):
        for eff in audit.ILLUSTRATIVE_TARGETS:
            r = audit.phi_split_overhead(eff, False)
            self.assertGreater(r["split_stored_scale_elements_total"], 0)


class TestStaticAndProvenance(unittest.TestCase):
    def test_commit_pinned(self):
        chk = audit.upstream_commit_check()
        if not chk["checks"]:
            self.skipTest("upstream .git metadata not present")
        self.assertTrue(chk["verified"])

    @NEEDS_UPSTREAM_SOURCE
    def test_static_unpack_line_found(self):
        obs = {o["id"]: o for o in audit.static_observations()}
        ev = obs["S1_unpack_left_shift"]["evidence"]
        self.assertEqual(len(ev), 1)
        self.assertIn("<< torch.arange(32", ev[0]["text"])

    @NEEDS_UPSTREAM_SOURCE
    def test_static_split_dim_property(self):
        obs = {o["id"]: o for o in audit.static_observations()}
        texts = [e["text"] for e in obs["S2_split_dim_used_attribute"]["evidence"]]
        self.assertTrue(any("self._split_dim.item()" in t for t in texts))
        self.assertTrue(any('"_split_dim_final"' in t for t in texts))


class TestResultsGenerator(unittest.TestCase):
    def test_build_results_without_upstream(self):
        res = audit.build_results(include_upstream=False)
        self.assertFalse(res["upstream"]["available"])
        self.assertIn(audit.TIER_MIRROR, res["_case_rows"])
        text = json.dumps({k: v for k, v in res.items() if k != "_case_rows"})
        self.assertIn(audit.UPSTREAM_COMMIT, text)
        csv_text = audit.rows_to_csv(res["_case_rows"][audit.TIER_MIRROR])
        header = csv_text.splitlines()[0].split(",")
        self.assertIn("prediction_matches", header)
        self.assertEqual(len(csv_text.splitlines()), 1 + len(res["_case_rows"][audit.TIER_MIRROR]))

    def test_deterministic(self):
        a = audit.rows_to_csv(audit.run_mirror_cases())
        b = audit.rows_to_csv(audit.run_mirror_cases())
        self.assertEqual(a, b)


class TestSignedMultiRowMirror(unittest.TestCase):
    """Direct regression, standard library: multi-row matrices at widths 31/32/33/64 whose words
    include negative int32 values (column 31 is the sign bit)."""

    def test_signed_multirow(self):
        import patched_regression as pr
        for w in pr.SIGNED_WIDTHS:
            rows = pr.signed_multirow_matrix(w)
            shape = (len(rows), w)
            packed = audit.mirror_pack(rows)
            negatives = sum(1 for r in packed for v in r if v < 0)
            if w >= 32:
                self.assertGreater(negatives, 0, w)
            else:
                self.assertEqual(negatives, 0, w)
            self.assertEqual(audit.reference_unpack_shift(packed, shape), rows, w)
            self.assertEqual(audit.reference_unpack_bytes(packed, shape), rows, w)
            decoded = audit.mirror_unpack_upstream(packed, shape)
            for r, d in zip(rows, decoded):
                self.assertEqual(d == r, audit.predicted_upstream_exact(r), (w, r))
            self.assertNotEqual(decoded, rows, w)

    def test_sign_bit_row_alone(self):
        # Column 31 set in every word: the most negative int32 per word.
        for w in (32, 33, 64):
            row = [-1 if j % 32 == 31 else 1 for j in range(w)]
            packed = audit.mirror_pack([row])
            self.assertEqual(packed[0][0], -(2 ** 31))
            self.assertEqual(audit.reference_unpack_shift(packed, (1, w)), [row])
            self.assertEqual(audit.mirror_unpack_upstream(packed, (1, w)), [[1] * w])


@NEEDS_UPSTREAM_SOURCE
class TestPatchFile(unittest.TestCase):
    """The proposed upstream patch is a one-line, two-character change that applies to the pinned
    source text. Standard library only; the upstream file is never written."""

    @classmethod
    def setUpClass(cls):
        import patched_regression as pr
        cls.pr = pr
        cls.patch = pr.read_patch()
        with open(pr.TARGET_PATH, "r", encoding="utf-8") as f:
            cls.source = f.read()

    def test_pinned_source_unmodified(self):
        self.assertEqual(self.pr.sha256_file(self.pr.TARGET_PATH), self.pr.PINNED_TARGET_SHA256)

    def test_one_line_two_characters(self):
        self.assertEqual(len(self.patch["removed"]), 1)
        self.assertEqual(len(self.patch["added"]), 1)
        changes = self.pr.changed_characters(self.patch)
        self.assertEqual([(a, b) for _, a, b in changes], [("<", ">"), ("<", ">")])
        self.assertEqual(changes[1][0], changes[0][0] + 1)

    def test_applies_to_exactly_the_static_line(self):
        patched = self.pr.apply_patch_text(self.source, self.patch)
        old_lines, new_lines = self.source.split("\n"), patched.split("\n")
        self.assertEqual(len(old_lines), len(new_lines))
        diff = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
        self.assertEqual(diff, [87])  # zero-based index of upstream line 88
        obs = {o["id"]: o for o in audit.static_observations()}
        self.assertEqual(old_lines[87].strip(), obs["S1_unpack_left_shift"]["evidence"][0]["text"])
        self.assertIn(">> torch.arange(32", new_lines[87])

    def test_rejects_mismatched_context(self):
        with self.assertRaises(ValueError):
            self.pr.apply_patch_text(self.source.replace("Extract all 32 bits", "Extract bits"), self.patch)


class TestDeliverableStyle(unittest.TestCase):
    def test_no_em_dashes(self):
        files = ["README.md", "PREREGISTRATION.md", "LICENSE_NOTES.md", "requirements.txt",
                 "MODEL_QUALITY_PLAN.md", "Dockerfile", ".devcontainer/devcontainer.json",
                 "patches/binary_unpacker.patch", "docs/medium-draft.md", "docs/research-draft.tex",
                 "src/audit.py", "src/bench.py", "src/patched_regression.py", "tests/test_audit.py"]
        for rel in files:
            path = os.path.join(PROJECT_ROOT, rel)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                self.assertNotIn(chr(0x2014), f.read(), rel)


@NEEDS_TORCH
@NEEDS_UPSTREAM_SOURCE
class TestUpstreamPacker(unittest.TestCase):
    """H2 and H3 against the actual upstream binary_packer / binary_unpacker."""

    @classmethod
    def setUpClass(cls):
        cls.up = audit.load_upstream(include_layer=False)
        cls.rows = audit.run_upstream_cases(cls.up)

    def test_packer_matches_mirror(self):
        self.assertEqual([r["case_id"] for r in self.rows if not r["packer_matches_mirror"]], [])

    def test_unpacker_matches_mirror(self):
        self.assertEqual([r["case_id"] for r in self.rows if not r["unpacker_matches_mirror"]], [])

    def test_reference_recovers_upstream_packed(self):
        self.assertEqual([r["case_id"] for r in self.rows
                          if not (r["reference_shift_exact"] and r["reference_bytes_exact"])], [])

    def test_matrix_equals_rowwise(self):
        pack, unpack = audit.make_torch_backend(self.up)
        for width in audit.WIDTHS:
            rows = [audit.byte_tiled_row(width, b) for b in range(256)]
            packed = pack(rows)
            self.assertEqual(packed, [pack([r])[0] for r in rows])
            self.assertEqual(unpack(packed, (256, width)), audit.mirror_unpack_upstream(packed, (256, width)))

    def test_rejects_non_int8(self):
        import torch
        with self.assertRaises(TypeError):
            self.up["binary_packer"](torch.ones(2, 3, dtype=torch.float32))


@NEEDS_TORCH
@NEEDS_PY310
@NEEDS_UPSTREAM_SOURCE
class TestUpstreamLayerFormulas(unittest.TestCase):
    """Mirror accounting equals the actual upstream static methods."""

    @classmethod
    def setUpClass(cls):
        cls.L = audit.load_upstream(include_layer=True)["LittleBitLinear"]

    def test_formula_parity(self):
        for _, a, b in audit.ILLUSTRATIVE_LAYERS:
            for eff in audit.ILLUSTRATIVE_TARGETS + (None,):
                for res in (False, True):
                    est = self.L._estimate_split_dim(a, b, eff, res)
                    self.assertEqual(est, audit.upstream_estimate_split(a, b, eff, res))
                    s = self.L._finalize_split_dim(est, 1024, 8)
                    self.assertEqual(s, audit.upstream_finalize_split(est, 1024, 8))
                    self.assertEqual(self.L._compute_eff_bits(a, b, s, res), audit.upstream_eff_bits(a, b, s, res))


@NEEDS_TORCH
@NEEDS_UPSTREAM_SOURCE
class TestPatchedUpstream(unittest.TestCase):
    """Actual upstream source with ONLY the one-line patch, executed in a fresh namespace."""

    @classmethod
    def setUpClass(cls):
        import patched_regression as pr
        cls.pr = pr
        cls.patched = pr.load_patched_module()
        cls.up = audit.load_upstream(include_layer=False)
        cls.reg = pr.run_packer_regressions(cls.patched, cls.up)

    def test_isolation(self):
        self.assertTrue(self.patched["upstream_unmodified"])
        self.assertEqual(self.patched["upstream_sha256_before"], self.pr.PINNED_TARGET_SHA256)
        self.assertFalse(self.patched["registered_in_sys_modules"])
        self.assertEqual(self.patched["diff_lines_differing"], 1)
        self.assertNotIn(self.patched["module"].__name__, sys.modules)

    def test_exhaustive_cases_all_exact(self):
        self.assertEqual(self.reg["patched_exact_cases"], self.reg["exhaustive_cases"])
        self.assertIsNone(self.reg["first_patched_failure"])

    def test_unpatched_control_matches_mirror_count(self):
        expected = sum(1 for r in audit.run_mirror_cases() if r["upstream_exact"])
        self.assertEqual(self.reg["unpatched_exact_cases"], expected)

    def test_signed_multirow_31_32_33_64(self):
        for w in self.pr.SIGNED_WIDTHS:
            r = self.reg["signed_multirow"][str(w)]
            self.assertTrue(r["patched_exact"], w)
            self.assertTrue(r["patched_equals_reference"], w)
            self.assertFalse(r["unpatched_exact"], w)
            if w >= 32:
                self.assertGreater(r["negative_int32_words"], 0, w)


@NEEDS_TORCH
@NEEDS_PY310
@NEEDS_UPSTREAM_SOURCE
class TestPatchedResidualLayer(unittest.TestCase):
    """Residual factors U_R and V_R of actual upstream layers through the emulated packed load."""

    @classmethod
    def setUpClass(cls):
        TORCH.set_num_threads(1)
        import patched_regression as pr
        cls.patched = pr.load_patched_module()
        cls.up = audit.load_upstream(include_layer=True)
        cls.cases = [(96, 160, 1.0, True, 0), (256, 256, 1.0, True, 1)]
        cls.fixed = [pr.layer_roundtrip(cls.up, cls.patched["binary_unpacker"], *c) for c in cls.cases]
        cls.control = [pr.layer_roundtrip(cls.up, cls.up["binary_unpacker"], *c) for c in cls.cases]

    def test_patched_all_factors_exact(self):
        for r in self.fixed:
            self.assertEqual(sorted(r["factors"]), ["U", "U_R", "V", "V_R"])
            for name, f in r["factors"].items():
                self.assertTrue(f["exact"], (r["in_features"], r["out_features"], name))
            self.assertEqual(r["missing_keys"], 0)
            self.assertEqual(r["unexpected_keys"], 0)

    def test_patched_outputs_bitwise_equal(self):
        for r in self.fixed:
            self.assertTrue(r["out_bitwise_equal"])
            self.assertEqual(r["out_max_abs"], 0.0)

    def test_unpatched_residual_factors_corrupted(self):
        for r in self.control:
            self.assertLess(r["factors"]["U_R"]["sign_agreement"], 1.0)
            self.assertLess(r["factors"]["V_R"]["sign_agreement"], 1.0)
            self.assertFalse(r["out_bitwise_equal"])


@NEEDS_TORCH
@NEEDS_PY310
class TestBenchSmoke(unittest.TestCase):
    @NEEDS_UPSTREAM_SOURCE
    def test_single_config_runs(self):
        import bench
        res = bench.run_all(shapes=((64, 96),), targets=(1.0,), seeds=(0,),
                            profiles=("raw_state_dict_fp32",), batch=2, warmup=1, iters=3)
        self.assertEqual(res["status"], "ok")
        row = res["rows"][0]
        for key in ("upstream_unpacker_sign_agreement", "reference_decoder_sign_agreement",
                    "fwd_in_memory_median_us", "fwd_dense_matched_p95_us", "h5_pass"):
            self.assertIn(key, row)
        self.assertEqual(row["reference_decoder_missing_keys"], 0)

    def test_percentile(self):
        import bench
        self.assertEqual(bench.percentile_nearest_rank(list(range(1, 101)), 95.0), 95)
        self.assertEqual(bench.percentile_nearest_rank([5.0], 95.0), 5.0)


if __name__ == "__main__":
    unittest.main()
