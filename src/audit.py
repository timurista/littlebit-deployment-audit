"""LittleBit packed-serialization deployment audit.

Original audit code. It does not copy upstream source; it re-expresses the arithmetic of a few
upstream expressions so they can be checked without torch, and, when torch is installed, imports
the actual upstream functions from the pinned local checkout and checks those too.

Evidence tiers (every result row carries one of these labels):

* ALGORITHM_AUDIT: standard-library arithmetic mirror. Tests the algorithm as written. It is
  NOT a model replication and says nothing about shipped tensors or kernels on its own.
* UPSTREAM_IMPORT: actual upstream code imported from upstream/LittleBit (needs torch).

Usage (raw results are produced by a later parent execution, not by this file's author):

    python src/audit.py --out results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import struct
import sys
import types
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

UPSTREAM_COMMIT = "933857ed1443b53fc43a875c2cf64249e3c56f0c"
UPSTREAM_URL = "https://github.com/SamsungLabs/LittleBit"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_ROOT = os.path.join(PROJECT_ROOT, "upstream", "LittleBit")

AUDITED_FILES = (
    "LICENSE",
    "quantization/utils/binary_packer.py",
    "quantization/modules/littlebit.py",
    "quantization/modules/attention.py",
    "quantization/utils/quant_util.py",
    "quantization/functions/binary.py",
    "main.py",
)

TIER_MIRROR = "ALGORITHM_AUDIT"
TIER_UPSTREAM = "UPSTREAM_IMPORT"

WORD_BITS = 32
WIDTHS = (1, 7, 8, 31, 32, 33, 64)
UINT32_MASK = 0xFFFFFFFF

# Preregistered (PREREGISTRATION.md, H1): exact rows of 256 in the byte-tiled family.
PREREGISTERED_EXACT_BYTE_ROWS = {1: 256, 7: 4, 8: 2, 31: 1, 32: 1, 33: 1, 64: 1}

Row = List[int]
Matrix = List[Row]
Packed = List[List[int]]


# ---------------------------------------------------------------------------------------------
# int32 helpers
# ---------------------------------------------------------------------------------------------

def to_int32(value: int) -> int:
    """Reduce any Python int modulo 2**32 and reinterpret as signed int32."""
    value &= UINT32_MASK
    return value - (1 << 32) if value >= (1 << 31) else value


def words_per_row(n_cols: int) -> int:
    return (n_cols + WORD_BITS - 1) // WORD_BITS


# ---------------------------------------------------------------------------------------------
# ALGORITHM_AUDIT tier: arithmetic mirror of the upstream packer and unpacker
# ---------------------------------------------------------------------------------------------

def mirror_pack(rows: Sequence[Sequence[int]]) -> Packed:
    """Mirror of upstream binary_packer arithmetic.

    Pad each row with +1 up to a multiple of 32, map value v to bit (1 - v) // 2 (floor
    division, as upstream does on int8), weight bit k by 2**k, sum modulo 2**32, store int32.
    Upstream computes 2**31 in int32, which wraps to -2**31; that differs from +2**31 by 2**32,
    so the int32 result is identical. Like upstream, values are not validated.
    """
    packed: Packed = []
    for row in rows:
        n_cols = len(row)
        n_words = words_per_row(n_cols)
        padded = list(row) + [1] * (n_words * WORD_BITS - n_cols)
        out_row = []
        for w in range(n_words):
            acc = 0
            for k in range(WORD_BITS):
                bit = (1 - padded[w * WORD_BITS + k]) // 2
                acc += bit * (1 << k)
            out_row.append(to_int32(acc))
        packed.append(out_row)
    return packed


def _check_shape(packed: Packed, shape: Tuple[int, int]) -> Tuple[int, int, int]:
    n_rows, n_cols = shape
    n_words = words_per_row(n_cols)
    if len(packed) != n_rows or any(len(r) != n_words for r in packed):
        raise ValueError("packed shape does not match (n_rows, words_per_row)")
    return n_rows, n_cols, n_words


def mirror_unpack_upstream(packed: Packed, shape: Tuple[int, int]) -> Matrix:
    """Mirror of the upstream unpack line: bit_k = (word << k) & 1, then value = 1 - 2 * bit.

    Only bit 0 of (word << k) is inspected, and a left shift by k >= 1 always clears bit 0, so
    this result does not depend on integer width or sign handling.
    """
    _, n_cols, n_words = _check_shape(packed, shape)
    out: Matrix = []
    for prow in packed:
        bits = []
        for w in range(n_words):
            word = prow[w]
            bits.extend((word << k) & 1 for k in range(WORD_BITS))
        out.append([1 - 2 * b for b in bits[:n_cols]])
    return out


# ---------------------------------------------------------------------------------------------
# Corrected, independent reference decoders
# ---------------------------------------------------------------------------------------------

def reference_unpack_shift(packed: Packed, shape: Tuple[int, int]) -> Matrix:
    """Corrected decoder: bit_k = ((word mod 2**32) >> k) & 1, lsb first."""
    _, n_cols, n_words = _check_shape(packed, shape)
    out: Matrix = []
    for prow in packed:
        bits = []
        for w in range(n_words):
            word = prow[w] & UINT32_MASK
            bits.extend((word >> k) & 1 for k in range(WORD_BITS))
        out.append([-1 if b else 1 for b in bits[:n_cols]])
    return out


def reference_unpack_bytes(packed: Packed, shape: Tuple[int, int]) -> Matrix:
    """Second corrected decoder with no shifts on the word: serialize each int32 little endian
    with struct, then read bit (k % 8) of byte (k // 8) by table lookup."""
    _, n_cols, n_words = _check_shape(packed, shape)
    bit_table = [tuple(1 if (byte & (1 << i)) else 0 for i in range(8)) for byte in range(256)]
    out: Matrix = []
    for prow in packed:
        raw = struct.pack("<%di" % n_words, *prow)
        bits = []
        for byte in raw:
            bits.extend(bit_table[byte])
        out.append([-1 if b else 1 for b in bits[:n_cols]])
    return out


def predicted_upstream_exact(row: Sequence[int]) -> bool:
    """H1: upstream unpack is exact iff every -1 sits at a column j with j % 32 == 0."""
    return all(j % WORD_BITS == 0 for j, v in enumerate(row) if v == -1)


# ---------------------------------------------------------------------------------------------
# Case generation (deterministic)
# ---------------------------------------------------------------------------------------------

def byte_tiled_row(width: int, byte: int) -> Row:
    return [-1 if (byte >> (j % 8)) & 1 else 1 for j in range(width)]


def one_hot_row(width: int, col: int) -> Row:
    return [-1 if j == col else 1 for j in range(width)]


def sign_example_rows(width: int) -> Dict[str, Row]:
    return {
        "all_plus": [1] * width,
        "all_minus": [-1] * width,
        "alternating_minus_first": [-1 if j % 2 == 0 else 1 for j in range(width)],
        "alternating_plus_first": [1 if j % 2 == 0 else -1 for j in range(width)],
        "minus_at_word_starts": [-1 if j % WORD_BITS == 0 else 1 for j in range(width)],
        "minus_at_word_ends": [-1 if j % WORD_BITS == WORD_BITS - 1 else 1 for j in range(width)],
    }


def generate_cases(widths: Iterable[int] = WIDTHS) -> List[dict]:
    cases: List[dict] = []
    for width in widths:
        for byte in range(256):
            cases.append({"case_id": "byte_w%d_b%03d" % (width, byte), "family": "byte_tiled",
                          "width": width, "param": byte, "row": byte_tiled_row(width, byte)})
        for col in range(width):
            cases.append({"case_id": "onehot_w%d_c%d" % (width, col), "family": "one_hot",
                          "width": width, "param": col, "row": one_hot_row(width, col)})
        for name, row in sign_example_rows(width).items():
            cases.append({"case_id": "sign_w%d_%s" % (width, name), "family": "sign_example",
                          "width": width, "param": name, "row": row})
    return cases


# Minimal named counterexamples, used in tests and in the summary.
COUNTEREXAMPLES = (
    {"name": "two_columns_second_negative", "row": [1, -1],
     "note": "smallest row that the upstream unpacker decodes wrongly"},
    {"name": "single_negative", "row": [-1],
     "note": "width 1 always roundtrips, so a width 1 test cannot detect H1"},
    {"name": "all_plus_width_64", "row": [1] * 64,
     "note": "all +1 always roundtrips, so an all-ones test cannot detect H1"},
    {"name": "negative_at_col_32_only", "row": one_hot_row(33, 32),
     "note": "-1 at a word start decodes correctly even in the second word"},
    {"name": "negative_at_col_31", "row": one_hot_row(32, 31),
     "note": "-1 at the sign bit of the int32 word, lost by the upstream unpacker"},
    {"name": "first_exploratory_counterexample", "row": [1, -1, 1, -1, -1, 1, 1, -1],
     "note": "row from evidence/first-upstream-counterexample.json, found before preregistration"},
)

PRIOR_EVIDENCE = os.path.join(PROJECT_ROOT, "evidence", "first-upstream-counterexample.json")


def prior_evidence_check() -> dict:
    """Compare the mirror with the recorded exploratory upstream run (torch 2.6.0). This links the
    ALGORITHM_AUDIT tier to one recorded UPSTREAM_IMPORT observation; it is exploratory evidence,
    not part of the preregistered sweep."""
    try:
        with open(PRIOR_EVIDENCE, "r", encoding="utf-8") as f:
            ev = json.load(f)
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": str(exc)}
    rows = ev["input"]
    shape = (len(rows), len(rows[0]))
    packed = mirror_pack(rows)
    return {
        "available": True, "status": "exploratory, predates preregistration",
        "recorded_commit_matches": ev.get("upstream_commit") == UPSTREAM_COMMIT,
        "recorded_torch": ev.get("torch"),
        "mirror_packed_matches_recorded": packed == ev["packed"],
        "mirror_unpacked_matches_recorded": mirror_unpack_upstream(packed, shape) == ev["upstream_unpacked"],
        "reference_recovers_input": reference_unpack_shift(ev["packed"], shape) == rows,
    }


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

Packer = Callable[[Sequence[Sequence[int]]], Packed]
Unpacker = Callable[[Packed, Tuple[int, int]], Matrix]


def evaluate_case(case: dict, packer: Packer, unpacker: Unpacker, tier: str) -> dict:
    row = case["row"]
    shape = (1, len(row))
    packed = packer([row])
    decoded = unpacker(packed, shape)[0]
    errors = [j for j, (a, b) in enumerate(zip(row, decoded)) if a != b]
    flips_minus_to_plus = sum(1 for j in errors if row[j] == -1 and decoded[j] == 1)
    ref_shift_ok = reference_unpack_shift(packed, shape)[0] == list(row)
    ref_bytes_ok = reference_unpack_bytes(packed, shape)[0] == list(row)
    exact = not errors
    predicted = predicted_upstream_exact(row)
    return {
        "tier": tier,
        "case_id": case["case_id"],
        "family": case["family"],
        "width": case["width"],
        "param": case["param"],
        "n_minus": sum(1 for v in row if v == -1),
        "packed_words_hex": " ".join("%08x" % (w & UINT32_MASK) for w in packed[0]),
        "upstream_exact": exact,
        "sign_errors": len(errors),
        "minus_to_plus_flips": flips_minus_to_plus,
        "first_error_col": errors[0] if errors else -1,
        "reference_shift_exact": ref_shift_ok,
        "reference_bytes_exact": ref_bytes_ok,
        "predicted_upstream_exact": predicted,
        "prediction_matches": predicted == exact,
    }


def summarize_cases(rows: List[dict]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for width in sorted({r["width"] for r in rows}):
        sub = [r for r in rows if r["width"] == width]
        byte_rows = [r for r in sub if r["family"] == "byte_tiled"]
        mismatches = [r for r in sub if not r["prediction_matches"]]
        ref_fail = [r for r in sub if not (r["reference_shift_exact"] and r["reference_bytes_exact"])]
        summary[str(width)] = {
            "cases": len(sub),
            "upstream_exact_cases": sum(1 for r in sub if r["upstream_exact"]),
            "byte_tiled_exact_rows": sum(1 for r in byte_rows if r["upstream_exact"]),
            "byte_tiled_preregistered_exact_rows": PREREGISTERED_EXACT_BYTE_ROWS.get(width),
            "total_sign_errors": sum(r["sign_errors"] for r in sub),
            "total_plus_to_minus_flips": sum(r["sign_errors"] - r["minus_to_plus_flips"] for r in sub),
            "prediction_mismatches": len(mismatches),
            "first_prediction_mismatch": mismatches[0]["case_id"] if mismatches else None,
            "reference_decoder_failures": len(ref_fail),
            "first_reference_failure": ref_fail[0]["case_id"] if ref_fail else None,
        }
    return summary


def run_mirror_cases(cases: Optional[List[dict]] = None) -> List[dict]:
    cases = generate_cases() if cases is None else cases
    return [evaluate_case(c, mirror_pack, mirror_unpack_upstream, TIER_MIRROR) for c in cases]


# ---------------------------------------------------------------------------------------------
# UPSTREAM_IMPORT tier: load actual upstream code from the pinned checkout (torch required)
# ---------------------------------------------------------------------------------------------

def try_import_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception:  # ImportError, or a broken install
        return None


def _load_file_module(qualname: str, relpath: str):
    path = os.path.join(UPSTREAM_ROOT, relpath)
    spec = importlib.util.spec_from_file_location(qualname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream(include_layer: bool = True) -> dict:
    """Import actual upstream code. Returns {"available": False, "reason": ...} without torch.

    binary_packer.py and binary.py only need torch, so they are loaded straight from their files.
    littlebit.py does `from quantization.utils.binary_packer import binary_packer`, and the real
    quantization.utils package __init__ imports quant_util, which needs transformers and
    safetensors. To import the unmodified littlebit.py without those, the three package names are
    temporarily pointed at minimal stand-ins (the stand-in for the leaf is the real upstream
    binary_packer module), then restored. This is recorded in the returned notes.
    """
    torch = try_import_torch()
    if torch is None:
        return {"available": False, "reason": "torch not importable"}
    out = {"available": True, "torch_version": getattr(torch, "__version__", "unknown"), "notes": []}
    packer_mod = _load_file_module("littlebit_audit_upstream.binary_packer",
                                   "quantization/utils/binary_packer.py")
    out["binary_packer"] = packer_mod.binary_packer
    out["binary_unpacker"] = packer_mod.binary_unpacker
    if not include_layer:
        return out
    if sys.version_info < (3, 10):
        out["layer_reason"] = "littlebit.py uses PEP 604 annotations and needs Python >= 3.10"
        return out
    funcs_mod = _load_file_module("littlebit_audit_upstream.binary", "quantization/functions/binary.py")
    out["STEBinary"] = funcs_mod.STEBinary
    out["SmoothSign"] = funcs_mod.SmoothSign
    names = ("quantization", "quantization.utils", "quantization.utils.binary_packer")
    saved = {n: sys.modules.get(n) for n in names}
    try:
        pkg = types.ModuleType("quantization")
        pkg.__path__ = []  # type: ignore[attr-defined]
        sub = types.ModuleType("quantization.utils")
        sub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["quantization"] = pkg
        sys.modules["quantization.utils"] = sub
        sys.modules["quantization.utils.binary_packer"] = packer_mod
        layer_mod = _load_file_module("littlebit_audit_upstream.littlebit",
                                      "quantization/modules/littlebit.py")
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    out["LittleBitLinear"] = layer_mod.LittleBitLinear
    out["notes"].append("littlebit.py imported with temporary package stand-ins; "
                        "quant_util.py (transformers, safetensors) not imported")
    return out


def make_torch_backend(upstream: dict) -> Tuple[Packer, Unpacker]:
    import torch

    def pack(rows: Sequence[Sequence[int]]) -> Packed:
        t = torch.tensor([list(r) for r in rows], dtype=torch.int8)
        return [[int(v) for v in r] for r in upstream["binary_packer"](t).tolist()]

    def unpack(packed: Packed, shape: Tuple[int, int]) -> Matrix:
        t = torch.tensor(packed, dtype=torch.int32)
        return [[int(v) for v in r] for r in upstream["binary_unpacker"](t, tuple(shape)).tolist()]

    return pack, unpack


def run_upstream_cases(upstream: dict, cases: Optional[List[dict]] = None) -> List[dict]:
    cases = generate_cases() if cases is None else cases
    pack, unpack = make_torch_backend(upstream)
    rows = []
    for c in cases:
        r = evaluate_case(c, pack, unpack, TIER_UPSTREAM)
        mirror_words = mirror_pack([c["row"]])[0]
        r["packer_matches_mirror"] = pack([c["row"]])[0] == mirror_words
        r["unpacker_matches_mirror"] = (
            unpack([mirror_words], (1, len(c["row"])))[0]
            == mirror_unpack_upstream([mirror_words], (1, len(c["row"])))[0])
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------
# Rank and storage accounting (mirrors upstream formulas, then decomposes them)
# ---------------------------------------------------------------------------------------------

def upstream_estimate_split(a: int, b: int, eff_bit: Optional[float], residual: bool) -> Optional[float]:
    """Mirror of LittleBitLinear._estimate_split_dim (a = in_features, b = out_features)."""
    if eff_bit is None or a * b == 0:
        return None
    base = a + b + 16
    if residual:
        return (a * b * eff_bit - 32 * (a + b)) / (2 * base)
    return (a * b * eff_bit - 16 * (a + b)) / base


def upstream_finalize_split(split_float: Optional[float], split_default: int, min_split_dim: int) -> int:
    """Mirror of LittleBitLinear._finalize_split_dim."""
    cand = split_float if split_float is not None else split_default
    cand = int(cand) if cand is not None else 0
    cand = (cand // 8) * 8
    if cand == 0:
        cand = min_split_dim
    return max(cand, min_split_dim)


def upstream_eff_bits(a: int, b: int, s: int, residual: bool) -> float:
    """Mirror of LittleBitLinear._compute_eff_bits."""
    if a * b == 0:
        return float("inf")
    return advertised_bits(a, b, s, residual) / (a * b)


def advertised_bits(a: int, b: int, s: int, residual: bool) -> int:
    """Numerator of the upstream eff-bit formula, as an exact integer."""
    if residual:
        return s * 2 * (a + b + 16) + 32 * (a + b)
    return s * (a + b + 16) + 16 * (a + b)


DTYPE_BYTES = {"int8": 1, "fp16": 2, "bf16": 2, "fp32": 4}

# Save profiles. main.py save_artifacts casts every fp32 tensor that is not packed and not a
# shape to bf16 before save_pretrained, which includes scales and the two float buffers.
# raw_state_dict_fp32 models module.state_dict() of an fp32-decomposed layer (hub.py path).
SAVE_PROFILES = {
    "main_py_bf16": {"scale_bytes": 2, "eff_buffer_bytes": 2, "split_buffer_bytes": 8},
    "raw_state_dict_fp32": {"scale_bytes": 4, "eff_buffer_bytes": 4, "split_buffer_bytes": 8},
}
SHAPE_TENSOR_BYTES = 2 * 8  # torch.tensor(param.shape, dtype=torch.long)


def resolve_split(a: int, b: int, eff_bit: Optional[float], residual: bool,
                  split_default: int = 1024, min_split_dim: int = 8,
                  ratio_factor: float = 1.0) -> Tuple[Optional[float], int]:
    est = upstream_estimate_split(a, b, eff_bit, residual)
    if est:  # upstream uses truthiness here, so an estimate of exactly 0.0 is not scaled
        est *= ratio_factor
    return est, upstream_finalize_split(est, split_default, min_split_dim)


def storage_breakdown(a: int, b: int, s: int, residual: bool, bias: bool = False) -> dict:
    """Bit and byte accounting for one LittleBitLinear (a = in_features, b = out_features).

    Bias is excluded by the upstream formula; it is reported separately when bias=True.
    Container overhead (safetensors header, alignment) is excluded.
    """
    paths = 2 if residual else 1
    n = a * b
    sign_bits = paths * s * (a + b)
    adv = advertised_bits(a, b, s, residual)
    adv_scale_bits = adv - sign_bits  # equals paths * 16 * (a + b + s)
    packed_sign_bits = paths * WORD_BITS * (b * words_per_row(s) + s * words_per_row(a))
    padding_bits = packed_sign_bits - sign_bits
    stored_scale_elems = paths * (a + b + 2 * s)  # u1(b), u2(s), v1(s), v2(a) per path
    out = {
        "in_features": a, "out_features": b, "split_dim": s, "residual": residual, "paths": paths,
        "rank_cap": min(a, b), "split_exceeds_rank_cap": s > min(a, b),
        "advertised_bits": adv,
        "advertised_bpw": adv / n,
        "advertised_sign_bits": sign_bits,
        "advertised_scale_bits": adv_scale_bits,
        "packed_sign_bits": packed_sign_bits,
        "row_padding_bits": padding_bits,
        "u_row_padding_bits": paths * b * (words_per_row(s) * WORD_BITS - s),
        "v_row_padding_bits": paths * s * (words_per_row(a) * WORD_BITS - a),
        "stored_scale_elements": stored_scale_elems,
        "double_middle_scale_extra_elements": paths * s,
        "shape_tensor_bytes": 2 * paths * SHAPE_TENSOR_BYTES,
    }
    for name, prof in SAVE_PROFILES.items():
        buffer_bytes = 2 * prof["eff_buffer_bytes"] + prof["split_buffer_bytes"]
        scale_bytes = stored_scale_elems * prof["scale_bytes"]
        bias_bytes = b * prof["scale_bytes"] if bias else 0
        total = packed_sign_bits // 8 + out["shape_tensor_bytes"] + scale_bytes + buffer_bytes + bias_bytes
        out["serialized_bytes_%s" % name] = total
        out["serialized_bpw_%s" % name] = total * 8 / n
        out["scale_overhead_vs_advertised_bits_%s" % name] = scale_bytes * 8 - adv_scale_bits
        # Resident after the upstream packed load: U and V become torch_dtype; scales and
        # buffers keep their saved dtype (quant_util casts params only in the legacy branch).
        for rdt, rbytes in DTYPE_BYTES.items():
            # One resident element per sign, so sign_bits doubles as the element count.
            resident = sign_bits * rbytes + scale_bytes + buffer_bytes + bias_bytes
            out["resident_bytes_%s_factors_%s" % (rdt, name)] = resident
            out["resident_bpw_%s_factors_%s" % (rdt, name)] = resident * 8 / n
    out["resident_over_packed_sign_ratio"] = {
        rdt: (sign_bits * rbytes * 8) / packed_sign_bits for rdt, rbytes in DTYPE_BYTES.items()}
    return out


ILLUSTRATIVE_LAYERS = (
    # (label, in_features, out_features). Public architecture shapes, illustrative only.
    ("tiny_64x64", 64, 64),
    ("tiny_100x36", 100, 36),
    ("llama2_7b_attn_4096x4096", 4096, 4096),
    ("llama2_7b_mlp_up_4096x11008", 4096, 11008),
    ("llama2_7b_mlp_down_11008x4096", 11008, 4096),
    ("phi4_fused_qkv_5120x7680", 5120, 7680),
    ("phi4_split_q_5120x5120", 5120, 5120),
    ("phi4_split_k_5120x1280", 5120, 1280),
    ("phi4_split_v_5120x1280", 5120, 1280),
)
ILLUSTRATIVE_TARGETS = (1.0, 0.55, 0.3, 0.1)


def accounting_table() -> List[dict]:
    rows = []
    for label, a, b in ILLUSTRATIVE_LAYERS:
        for eff in ILLUSTRATIVE_TARGETS:
            for residual in (False, True):
                est, s = resolve_split(a, b, eff, residual)
                bd = storage_breakdown(a, b, s, residual)
                ratios = bd.pop("resident_over_packed_sign_ratio")
                row = {"tier": TIER_MIRROR, "layer": label, "target_bpw": eff,
                       "split_estimate": est, "upstream_eff_bits_actual": upstream_eff_bits(a, b, s, residual),
                       "actual_exceeds_target": upstream_eff_bits(a, b, s, residual) > eff}
                row.update(bd)
                for k, v in ratios.items():
                    row["resident_over_packed_sign_ratio_%s" % k] = v
                rows.append(row)
    return rows


def phi_split_overhead(eff: float, residual: bool) -> dict:
    """attention.py splits Phi qkv_proj into q, k, v Linear layers, each converted separately,
    so each carries its own u1/v2 scale vectors and its own split_dim."""
    fused = storage_breakdown(5120, 7680, resolve_split(5120, 7680, eff, residual)[1], residual)
    parts = [storage_breakdown(5120, o, resolve_split(5120, o, eff, residual)[1], residual)
             for o in (5120, 1280, 1280)]
    return {
        "target_bpw": eff, "residual": residual,
        "fused_advertised_bits": fused["advertised_bits"],
        "split_advertised_bits_total": sum(p["advertised_bits"] for p in parts),
        "fused_stored_scale_elements": fused["stored_scale_elements"],
        "split_stored_scale_elements_total": sum(p["stored_scale_elements"] for p in parts),
        "note": "illustrative; kv_factor=1.0; shapes from public Phi-4 config, verify before citing",
    }


# ---------------------------------------------------------------------------------------------
# Static source observations (located by text search, reported with line numbers)
# ---------------------------------------------------------------------------------------------

def _read_lines(relpath: str) -> List[str]:
    with open(os.path.join(UPSTREAM_ROOT, relpath), "r", encoding="utf-8") as f:
        return f.read().splitlines()


def _find(relpath: str, needle: str) -> List[dict]:
    try:
        lines = _read_lines(relpath)
    except OSError:
        return []
    return [{"file": relpath, "line": i + 1, "text": ln.strip()}
            for i, ln in enumerate(lines) if needle in ln]


def static_observations() -> List[dict]:
    obs = []
    unpack = _find("quantization/utils/binary_packer.py", "unsqueeze(1) <<")
    obs.append({"id": "S1_unpack_left_shift", "evidence": unpack,
                "observation": "unpack extracts bits with a left shift; a right shift is needed "
                               "for lsb-first extraction (see H1)"})
    prop = _find("quantization/modules/littlebit.py", "self._split_dim.item()")
    buf = _find("quantization/modules/littlebit.py", 'register_buffer("_split_dim')
    obs.append({"id": "S2_split_dim_used_attribute", "evidence": prop + buf,
                "observation": "split_dim_used reads self._split_dim, but the registered buffer "
                               "is _split_dim_final; the property would raise AttributeError if called"})
    legacy = _find("quantization/utils/quant_util.py", "Legacy format. Casting")
    binflag = _find("quantization/utils/quant_util.py", "module._binarized = True")
    obs.append({"id": "S3_packed_load_dtype", "evidence": legacy + binflag,
                "observation": "in the packed branch U and V are cast to torch_dtype by the unpack "
                               "step, the cast of other params happens only in the legacy branch, "
                               "and _binarized=True means decoded values are used without re-sign"})
    save = _find("main.py", '"packed" not in k and "shape" not in k')
    obs.append({"id": "S4_save_casts_to_bf16", "evidence": save,
                "observation": "main.py casts fp32 non-packed tensors (scales, float buffers) to bf16 "
                               "before save; basis of the main_py_bf16 profile"})
    return obs


# ---------------------------------------------------------------------------------------------
# Provenance and environment
# ---------------------------------------------------------------------------------------------

def upstream_commit_check() -> dict:
    found = []
    for rel in (".git/packed-refs", ".git/shallow", ".git/refs/heads/main"):
        path = os.path.join(UPSTREAM_ROOT, rel)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        found.append({"file": rel, "contains_pinned_commit": UPSTREAM_COMMIT in text})
    return {"pinned_commit": UPSTREAM_COMMIT, "url": UPSTREAM_URL, "checks": found,
            "verified": any(c["contains_pinned_commit"] for c in found)}


def file_hashes() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for rel in AUDITED_FILES:
        try:
            with open(os.path.join(UPSTREAM_ROOT, rel), "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out[rel] = None
    return out


def environment() -> dict:
    env = {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
           "platform": platform.platform(), "machine": platform.machine(),
           "cpu_count": os.cpu_count()}
    torch = try_import_torch()
    env["torch"] = getattr(torch, "__version__", None) if torch else None
    try:
        import numpy  # noqa: F401
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    return env


# ---------------------------------------------------------------------------------------------
# Results assembly and serialization
# ---------------------------------------------------------------------------------------------

def counterexample_report(packer: Packer = mirror_pack, unpacker: Unpacker = mirror_unpack_upstream) -> List[dict]:
    rep = []
    for ce in COUNTEREXAMPLES:
        row = ce["row"]
        packed = packer([row])
        rep.append({"name": ce["name"], "note": ce["note"], "width": len(row),
                    "input_minus_cols": [j for j, v in enumerate(row) if v == -1],
                    "upstream_decoded_minus_cols": [j for j, v in enumerate(unpacker(packed, (1, len(row)))[0]) if v == -1],
                    "reference_decoded_minus_cols": [j for j, v in enumerate(reference_unpack_shift(packed, (1, len(row)))[0]) if v == -1]})
    return rep


def build_results(include_upstream: bool = True) -> dict:
    mirror_rows = run_mirror_cases()
    results = {
        "schema": "littlebit-deployment-audit/1",
        "primary_question": ("Does packed serialization preserve signs and actual outputs, and "
                             "where do loaded storage and advertised BPW diverge?"),
        "non_claims": ["no novelty claim", "no model quality claims", "no end-to-end decode claims"],
        "provenance": upstream_commit_check(),
        "upstream_file_sha256": file_hashes(),
        "environment": environment(),
        "tiers": {TIER_MIRROR: "standard-library arithmetic mirror; algorithm audit, not model replication",
                  TIER_UPSTREAM: "actual upstream functions imported from the pinned checkout"},
        "mirror": {"tier": TIER_MIRROR, "summary_by_width": summarize_cases(mirror_rows),
                   "counterexamples": counterexample_report()},
        "prior_exploratory_evidence": prior_evidence_check(),
        "static_observations": static_observations(),
        "accounting": accounting_table(),
        "phi_split_overhead": [phi_split_overhead(e, r) for e in ILLUSTRATIVE_TARGETS for r in (False, True)],
        "_case_rows": {TIER_MIRROR: mirror_rows},
    }
    upstream = load_upstream(include_layer=False) if include_upstream else {"available": False, "reason": "disabled"}
    if upstream.get("available"):
        up_rows = run_upstream_cases(upstream)
        pack, unpack = make_torch_backend(upstream)
        results["upstream"] = {
            "tier": TIER_UPSTREAM, "available": True, "torch_version": upstream["torch_version"],
            "summary_by_width": summarize_cases(up_rows),
            "packer_mirror_mismatches": sum(1 for r in up_rows if not r["packer_matches_mirror"]),
            "unpacker_mirror_mismatches": sum(1 for r in up_rows if not r["unpacker_matches_mirror"]),
            "counterexamples": counterexample_report(pack, unpack),
        }
        results["_case_rows"][TIER_UPSTREAM] = up_rows
    else:
        results["upstream"] = {"tier": TIER_UPSTREAM, "available": False,
                               "reason": upstream.get("reason"),
                               "label": "only ALGORITHM_AUDIT evidence was produced in this run"}
    return results


def rows_to_csv(rows: List[dict]) -> str:
    buf = io.StringIO()
    if not rows:
        return ""
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def write_results(results: dict, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    case_rows = results.pop("_case_rows", {})
    path = os.path.join(out_dir, "audit_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    written.append(path)
    for tier, rows in case_rows.items():
        path = os.path.join(out_dir, "unpack_cases_%s.csv" % tier.lower())
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(rows_to_csv(rows))
        written.append(path)
    path = os.path.join(out_dir, "storage_accounting.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(rows_to_csv(results["accounting"]))
    written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results"))
    ap.add_argument("--no-upstream", action="store_true", help="skip the torch UPSTREAM_IMPORT tier")
    args = ap.parse_args(argv)
    results = build_results(include_upstream=not args.no_upstream)
    for p in write_results(results, args.out):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
