# SPDX-License-Identifier: CC-BY-NC-4.0
# Adapted Material under CC BY-NC 4.0 (LICENSES/CC-BY-NC-4.0.txt; LICENSE, Part 2). Licensed
# Material: The LittleBit Project, https://github.com/SamsungLabs/LittleBit, commit
# 933857ed1443b53fc43a875c2cf64249e3c56f0c, CC BY-NC 4.0; methods by Banseok Lee, Dongkyu Kim,
# Youngcheon You and Youngmin Kim. Changes, Copyright (c) 2026 Tim Urista: applies the one-line
# change of patches/binary_unpacker.patch to the upstream source text in memory and tests the
# result; the upstream file is never written and no upstream source is copied into this file.
# NonCommercial use only. Not affiliated with or endorsed by the LittleBit authors or Samsung.
"""Regression evidence for the proposed one-line upstream unpack patch.

Reads the unmodified upstream quantization/utils/binary_packer.py, applies ONLY the single hunk in
patches/binary_unpacker.patch to the source text in memory, and executes the result in a fresh
module object that is never registered in sys.modules and never written to disk. The upstream file
is hashed before and after and must be unchanged.

Tier label: UPSTREAM_PATCHED_IMPORT (actual upstream source plus exactly one changed line).

Scope: the packed load path is emulated (upstream unpack function, cast, load_state_dict with
assign=True, _binarized=True). quant_util.load_quantized_model, the official loader, is NOT run.

    python src/patched_regression.py --out results      # torch required for the regression part
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402

TIER_PATCHED = "UPSTREAM_PATCHED_IMPORT"
PATCH_PATH = os.path.join(audit.PROJECT_ROOT, "patches", "binary_unpacker.patch")
TARGET_REL = "quantization/utils/binary_packer.py"
TARGET_PATH = os.path.join(audit.UPSTREAM_ROOT, TARGET_REL)

# Hash of the pinned, unmodified upstream file as recorded in results/audit_summary.json.
PINNED_TARGET_SHA256 = "4dfb95ce4d1997c6e5cb4d278d6aa90f1ac73ad1383024225ec0b3bb0f96669b"

SIGNED_WIDTHS = (31, 32, 33, 64)
LAYER_CONFIGS = (
    # (in_features, out_features, target_bpw, residual, seed)
    (96, 160, 1.0, False, 0),
    (96, 160, 1.0, True, 0),
    (256, 256, 1.0, True, 1),
    (512, 1024, 0.1, True, 2),
    (1024, 512, 1.0, True, 3),
)


# ---------------------------------------------------------------------------------------------
# Patch parsing and in-memory application (standard library only)
# ---------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_patch(path: str = PATCH_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    headers = [ln for ln in lines if ln.startswith("--- ") or ln.startswith("+++ ")]
    hunks = [i for i, ln in enumerate(lines) if ln.startswith("@@")]
    if len(headers) != 2 or len(hunks) != 1:
        raise ValueError("expected exactly one file and one hunk")
    body = lines[hunks[0] + 1:]
    old_block, new_block, removed, added = [], [], [], []
    for ln in body:
        tag, text = (ln[:1], ln[1:]) if ln else (" ", "")
        if tag == " ":
            old_block.append(text)
            new_block.append(text)
        elif tag == "-":
            old_block.append(text)
            removed.append(text)
        elif tag == "+":
            new_block.append(text)
            added.append(text)
        else:
            raise ValueError("unexpected hunk line: %r" % ln)
    return {"target": headers[0][len("--- a/"):], "hunk_header": lines[hunks[0]],
            "old_block": old_block, "new_block": new_block, "removed": removed, "added": added}


def changed_characters(patch: dict) -> List[Tuple[int, str, str]]:
    old, new = patch["removed"][0], patch["added"][0]
    return [(i, a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b]


def apply_patch_text(source: str, patch: dict) -> str:
    """Strict single-hunk application: the whole old block (context included) must occur exactly
    once, contiguously, and the hunk must change exactly one line."""
    if len(patch["removed"]) != 1 or len(patch["added"]) != 1:
        raise ValueError("patch must change exactly one line")
    old = "\n".join(patch["old_block"])
    new = "\n".join(patch["new_block"])
    if source.count(old) != 1:
        raise ValueError("hunk context does not match the upstream source exactly once")
    return source.replace(old, new, 1)


# ---------------------------------------------------------------------------------------------
# Patched module construction (needs torch to execute)
# ---------------------------------------------------------------------------------------------

def load_patched_module() -> dict:
    hash_before = sha256_file(TARGET_PATH)
    with open(TARGET_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    patch = read_patch()
    if patch["target"] != TARGET_REL:
        raise ValueError("patch targets %s, expected %s" % (patch["target"], TARGET_REL))
    patched = apply_patch_text(source, patch)
    module = types.ModuleType("littlebit_audit_patched_namespace.binary_packer")
    module.__file__ = "<in-memory patched %s>" % TARGET_REL
    exec(compile(patched, module.__file__, "exec"), module.__dict__)
    hash_after = sha256_file(TARGET_PATH)
    return {
        "module": module,
        "binary_packer": module.binary_packer,
        "binary_unpacker": module.binary_unpacker,
        "upstream_sha256_before": hash_before,
        "upstream_sha256_after": hash_after,
        "upstream_unmodified": hash_before == hash_after,
        "registered_in_sys_modules": module.__name__ in sys.modules,
        "changed_line_count": 1,
        "changed_characters": changed_characters(patch),
        "diff_lines_differing": sum(1 for a, b in zip(source.split("\n"), patched.split("\n")) if a != b),
    }


# ---------------------------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------------------------

def signed_multirow_matrix(width: int) -> List[List[int]]:
    """Several distinct rows in one matrix, chosen so that widths >= 32 produce negative int32 words
    (column 31 is the sign bit of each word) and width 33 spans a second word."""
    rows = [
        [-1] * width,
        [1] * width,
        audit.byte_tiled_row(width, 0x80),
        audit.byte_tiled_row(width, 0xFF),
        audit.byte_tiled_row(width, 0x5A),
        [-1 if j % 32 == 31 else 1 for j in range(width)],
        [-1 if j % 2 else 1 for j in range(width)],
    ]
    for col in (0, 1, 30, 31, 32):
        if col < width:
            rows.append(audit.one_hot_row(width, col))
    return rows


def run_packer_regressions(patched: dict, upstream: dict) -> dict:
    import torch
    up_pack, up_unpack = audit.make_torch_backend(upstream)
    _, fix_unpack = audit.make_torch_backend({"binary_packer": upstream["binary_packer"],
                                              "binary_unpacker": patched["binary_unpacker"]})
    cases = audit.generate_cases()
    exact_fixed = exact_unpatched = 0
    first_fixed_failure = None
    for c in cases:
        row = c["row"]
        packed = up_pack([row])
        if fix_unpack(packed, (1, len(row)))[0] == row:
            exact_fixed += 1
        elif first_fixed_failure is None:
            first_fixed_failure = c["case_id"]
        if up_unpack(packed, (1, len(row)))[0] == row:
            exact_unpatched += 1
    multirow = {}
    for width in SIGNED_WIDTHS:
        rows = signed_multirow_matrix(width)
        packed = up_pack(rows)
        t = torch.tensor(packed, dtype=torch.int32)
        multirow[str(width)] = {
            "rows": len(rows),
            "negative_int32_words": int((t < 0).sum()),
            "patched_exact": fix_unpack(packed, (len(rows), width)) == rows,
            "unpatched_exact": up_unpack(packed, (len(rows), width)) == rows,
            "patched_equals_reference": fix_unpack(packed, (len(rows), width))
            == audit.reference_unpack_shift(packed, (len(rows), width)),
        }
    return {"exhaustive_cases": len(cases), "patched_exact_cases": exact_fixed,
            "unpatched_exact_cases": exact_unpatched, "first_patched_failure": first_fixed_failure,
            "signed_multirow": multirow}


def layer_roundtrip(up: dict, unpacker, a: int, b: int, eff: float, residual: bool, seed: int) -> dict:
    """Serialize an actual upstream LittleBitLinear (unmodified upstream packer), roundtrip through
    torch.save/torch.load in memory, decode with `unpacker`, load with assign=True, binarize."""
    import torch
    import bench
    mem = bench.build_layer(up, a, b, eff, residual, seed, do_train=True)
    for p in mem.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, a, generator=torch.Generator().manual_seed(10_000 + seed))
    with torch.no_grad():
        y_mem = mem(x)
    buf = io.BytesIO()
    torch.save(mem.state_dict(), buf)
    buf.seek(0)
    sd = torch.load(buf, map_location="cpu", weights_only=True)
    state = bench.emulate_packed_load(sd, unpacker, torch.float32)
    lm = bench.build_layer(up, a, b, eff, residual, seed, do_train=False)
    missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
    lm._binarized = True
    with torch.no_grad():
        y = lm(x)
    factors = {}
    for name in bench._factor_names(residual):
        orig = mem.quantize(getattr(mem, name).data)
        got = getattr(lm, name).data
        factors[name] = {"shape": list(orig.shape),
                         "sign_agreement": float((orig == got).double().mean()),
                         "exact": bool(torch.equal(orig, got)),
                         "negative_packed_words": int((sd[name + "_packed"] < 0).sum())}
    return {"in_features": a, "out_features": b, "target_bpw": eff, "residual": residual, "seed": seed,
            "split_dim": int(mem.split_dim), "missing_keys": len(missing), "unexpected_keys": len(unexpected),
            "factors": factors, "out_bitwise_equal": bool(torch.equal(y, y_mem)),
            "out_max_abs": float((y - y_mem).abs().max())}


def run_layer_regressions(patched: dict, up: dict) -> List[dict]:
    rows = []
    for (a, b, eff, residual, seed) in LAYER_CONFIGS:
        fixed = layer_roundtrip(up, patched["binary_unpacker"], a, b, eff, residual, seed)
        control = layer_roundtrip(up, up["binary_unpacker"], a, b, eff, residual, seed)
        rows.append({"config": [a, b, eff, residual, seed], "patched": fixed, "unpatched_control": control})
    return rows


def build_report() -> dict:
    patch = read_patch()
    with open(TARGET_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    report = {
        "tier": TIER_PATCHED,
        "official_loader_ran": False,
        "load_path": "emulated: upstream state_dict packing, torch.save/torch.load in memory, "
                     "unpack, cast to float32, load_state_dict(assign=True), _binarized=True",
        "provenance": audit.upstream_commit_check(),
        "patch": {"path": os.path.relpath(PATCH_PATH, audit.PROJECT_ROOT), "hunk": patch["hunk_header"],
                  "removed": patch["removed"], "added": patch["added"],
                  "changed_characters": changed_characters(patch),
                  "applies_to_pinned_source": apply_patch_text(source, patch) != source},
        "environment": audit.environment(),
    }
    torch = audit.try_import_torch()
    if torch is None:
        report["status"] = "skipped_no_torch"
        return report
    torch.set_num_threads(1)
    patched = load_patched_module()
    up = audit.load_upstream(include_layer=True)
    report["isolation"] = {k: patched[k] for k in ("upstream_sha256_before", "upstream_sha256_after",
                                                   "upstream_unmodified", "registered_in_sys_modules",
                                                   "diff_lines_differing")}
    report["packer_regressions"] = run_packer_regressions(patched, up)
    if "LittleBitLinear" in up:
        report["layer_regressions"] = run_layer_regressions(patched, up)
    report["status"] = "ok"
    report["isolation"]["upstream_unmodified_after_all_runs"] = sha256_file(TARGET_PATH) == patched["upstream_sha256_before"]
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Patched-unpacker regression evidence")
    ap.add_argument("--out", default=os.path.join(audit.PROJECT_ROOT, "results"))
    args = ap.parse_args(argv)
    rep = build_report()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "patched_regression.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    print(path)
    print("status: %s" % rep["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
