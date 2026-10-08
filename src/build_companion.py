#!/usr/bin/env python3
# Copyright (c) 2026 Tim Urista. All rights reserved.
# No license is granted for this file (LICENSE, Part 3). It is not one of the seven MIT files.
"""Build the static GitHub Pages companion in docs/ from the recorded v0.1.0 results.

Standard library only. No network, no subprocess, no model execution and no timestamps, so the
same inputs always give byte-identical outputs. Measured files are only read, never written.

Reads:
  results/summary.json                       held-out NLL, storage, BPW, KV cache, OS RSS
  results/unpack_cases_upstream_import.csv   the 2,010 recorded unpack cases (UPSTREAM_IMPORT)
  results/patched_regression.json            patch regression counts and the changed line
  results/quality-pilot/train.json           number of converted modules
  src/companion/index.template.html          page template with {{key}} placeholders
Writes:
  docs/index.html
  docs/site/data/companion.json, docs/site/data/unpack_cases.json
  docs/site/assets/*.svg                     2D charts and the static structure diagram

Every CSV row is re-derived from its packed words before anything is written: the corrected
right-shift decode must reproduce n_minus, and the upstream left-shift expression must reproduce
sign_errors, first_error_col and upstream_exact. A mismatch stops the build.

    python src/build_companion.py           # write the outputs
    python src/build_companion.py --check   # exit 1 if any tracked output differs from a rebuild
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUMMARY = "results/summary.json"
UNPACK_CSV = "results/unpack_cases_upstream_import.csv"
PATCHED = "results/patched_regression.json"
TRAIN = "results/quality-pilot/train.json"
TEMPLATE = "src/companion/index.template.html"

OUT_INDEX = "docs/index.html"
OUT_DATA = "docs/site/data/companion.json"
OUT_CASES = "docs/site/data/unpack_cases.json"
OUT_ASSETS = "docs/site/assets"

COMPANION_VERSION = "0.2.0"
DATA_RELEASE = "v0.1.0"
DATA_COMMIT = "97f075cd1f995e66b693b3c1657fcc2f1a43ed0c"
UPSTREAM_URL = "https://github.com/SamsungLabs/LittleBit"
UPSTREAM_COMMIT = "933857ed1443b53fc43a875c2cf64249e3c56f0c"
REPO_URL = "https://github.com/timurista/littlebit-deployment-audit"
PAGES_URL = "https://timurista.github.io/littlebit-deployment-audit/"
MANUSCRIPT_PDF_URL = REPO_URL + "/releases/download/v0.2.0/littlebit-v0.2.0-manuscript.pdf"
THREE_VERSION = "0.186.1"

WORD_BITS = 32
WIDTHS = (1, 7, 8, 31, 32, 33, 64)
FAMILIES = ("byte_tiled", "one_hot", "sign_example")
ERROR_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63))
GIB = float(1 << 30)

UPSTREAM_FORMULA = "y = ((((x * v2) @ Vq.T) * (v1 * u2)) @ Uq.T) * u1"
FORMULA_NOTE = ("The layer output adds the same expression with the residual factors U_R, V_R and scales "
                "u1_R, u2_R, v1_R, v2_R, and then the bias")

# Palette shared with docs/site/css/site.css.
PAPER = "#FBF8F1"
INK = "#14213D"
MUTED = "#555B69"
RULE = "#D8D1C1"
NAVY = "#14213D"
COBALT = "#2747B8"
TEAL = "#0F7C7A"
TEAL_LIGHT = "#BFDCD9"
SLATE = "#7C88A8"
RUST = "#9C4A1A"
SANS = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

CONDITIONS = (
    ("original", "Original model", "reference, FP32"),
    ("initialized_no_qat", "LittleBit, initialized", "converted, no QAT"),
    ("qat_trained", "LittleBit, 16 QAT steps", "converted, then trained"),
    ("qat_trained_upstream_decoder", "Same QAT weights, defective decoder",
     "decoder defect condition, not a quality result"),
)
SERIES_STYLE = {
    "original": (NAVY, "circle", ""),
    "initialized_no_qat": (SLATE, "triangle", ""),
    "qat_trained": (COBALT, "diamond", ""),
    "qat_trained_upstream_decoder": (RUST, "square", "6 4"),
}
STAGES = ("baseline", "feasibility", "train", "evaluate")


# ---------------------------------------------------------------------------------------------
# Reading and checking the recorded inputs


def read_bytes(rel: str) -> bytes:
    with open(os.path.join(PROJECT_ROOT, rel), "rb") as f:
        return f.read()


def source_record(rel: str, data: bytes, role: str) -> dict:
    return {"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "role": role}


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError("not a recorded boolean: %r" % value)


def corrected_bits(words: Sequence[int], width: int) -> List[int]:
    """Corrected decoder: bit k of the unsigned word is (word >> k) & 1, lsb first."""
    return [(words[j // WORD_BITS] >> (j % WORD_BITS)) & 1 for j in range(width)]


def upstream_bits(words: Sequence[int], width: int) -> List[int]:
    """Shipped expression (word << k) & 1: a left shift by k >= 1 clears bit 0."""
    return [(words[j // WORD_BITS] << (j % WORD_BITS)) & 1 for j in range(width)]


def verify_case(c: dict) -> None:
    cid, width, words = c["case_id"], c["width"], c["words"]
    if len(words) != (width + WORD_BITS - 1) // WORD_BITS:
        raise ValueError("%s: word count does not match width" % cid)
    if any(w < 0 or w > 0xFFFFFFFF for w in words):
        raise ValueError("%s: word outside uint32" % cid)
    true = corrected_bits(words, width)
    up = upstream_bits(words, width)
    if sum(true) != sum(bin(w).count("1") for w in words):
        raise ValueError("%s: set bits beyond the row width" % cid)
    if sum(true) != c["n_minus"]:
        raise ValueError("%s: corrected decode disagrees with n_minus" % cid)
    if any(u == 1 and t == 0 for t, u in zip(true, up)):
        raise ValueError("%s: upstream expression produced a +1 to -1 flip" % cid)
    errors = [j for j in range(width) if true[j] == 1 and up[j] == 0]
    if len(errors) != c["sign_errors"] or c["minus_to_plus_flips"] != c["sign_errors"]:
        raise ValueError("%s: sign_errors not reproduced" % cid)
    if (not errors) != c["upstream_exact"]:
        raise ValueError("%s: upstream_exact not reproduced" % cid)
    if (errors[0] if errors else -1) != c["first_error_col"]:
        raise ValueError("%s: first_error_col not reproduced" % cid)
    predicted = all(j % WORD_BITS == 0 for j in range(width) if true[j])
    if predicted != c["predicted_upstream_exact"] or not c["prediction_matches"]:
        raise ValueError("%s: exactness rule not reproduced" % cid)
    if not (c["reference_shift_exact"] and c["reference_bytes_exact"]):
        raise ValueError("%s: a recorded reference decoder was not exact" % cid)


def parse_cases(text: str) -> List[dict]:
    cases = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["tier"] != "UPSTREAM_IMPORT":
            raise ValueError("unexpected tier %r" % row["tier"])
        if row["family"] not in FAMILIES:
            raise ValueError("unexpected family %r" % row["family"])
        case = {
            "case_id": row["case_id"],
            "family": row["family"],
            "width": int(row["width"]),
            "param": row["param"],
            "n_minus": int(row["n_minus"]),
            "words": [int(w, 16) for w in row["packed_words_hex"].split()],
            "words_hex": row["packed_words_hex"].split(),
            "upstream_exact": parse_bool(row["upstream_exact"]),
            "sign_errors": int(row["sign_errors"]),
            "minus_to_plus_flips": int(row["minus_to_plus_flips"]),
            "first_error_col": int(row["first_error_col"]),
            "reference_shift_exact": parse_bool(row["reference_shift_exact"]),
            "reference_bytes_exact": parse_bool(row["reference_bytes_exact"]),
            "predicted_upstream_exact": parse_bool(row["predicted_upstream_exact"]),
            "prediction_matches": parse_bool(row["prediction_matches"]),
            "packer_matches_mirror": parse_bool(row["packer_matches_mirror"]),
            "unpacker_matches_mirror": parse_bool(row["unpacker_matches_mirror"]),
        }
        verify_case(case)
        cases.append(case)
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case ids")
    return cases


def bin_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else "%d to %d" % (lo, hi)


def error_histogram(cases: Sequence[dict]) -> List[dict]:
    counts = [0] * len(ERROR_BINS)
    for c in cases:
        for i, (lo, hi) in enumerate(ERROR_BINS):
            if lo <= c["sign_errors"] <= hi:
                counts[i] += 1
                break
        else:
            raise ValueError("%s: sign_errors outside every bin" % c["case_id"])
    return [{"lo": lo, "hi": hi, "label": bin_label(lo, hi), "count": n}
            for (lo, hi), n in zip(ERROR_BINS, counts)]


def width_table(cases: Sequence[dict]) -> List[dict]:
    rows = []
    for width in WIDTHS:
        sub = [c for c in cases if c["width"] == width]
        byte = [c for c in sub if c["family"] == "byte_tiled"]
        rows.append({
            "width": width,
            "cases": len(sub),
            "exact_cases": sum(1 for c in sub if c["upstream_exact"]),
            "byte_rows": len(byte),
            "byte_rows_exact": sum(1 for c in byte if c["upstream_exact"]),
            "sign_errors": sum(c["sign_errors"] for c in sub),
        })
    return rows


# ---------------------------------------------------------------------------------------------
# Formatting


def fmt_int(v: int) -> str:
    return "{:,}".format(int(v))


def fmt_fixed(v: float, digits: int) -> str:
    return "{:,.{d}f}".format(float(v), d=digits)


def num(v: float) -> str:
    """Deterministic SVG coordinate: at most two decimals, no trailing zeros."""
    s = "%.2f" % float(v)
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


# ---------------------------------------------------------------------------------------------
# Collecting the numbers


def collect(summary: dict, cases: List[dict], patched: dict, train: dict, sources: List[dict]) -> dict:
    ht = summary["heldout_test"]
    cond = ht["conditions"]
    quality = []
    for key, label, note in CONDITIONS:
        c = cond[key]
        quality.append({
            "key": key, "label": label, "note": note, "tier": c["tier"],
            # Rounded so that libm differences between platforms cannot change the outputs.
            "ppl": c["ppl"], "log10_ppl": round(math.log10(c["ppl"]), 6), "mean_nll": c["mean_nll"],
            "per_window_mean_nll": list(c["per_window_mean_nll"]),
            "scored_tokens": c["scored_tokens"],
            "windows_worse_than_original": c.get("windows_worse_than_original"),
        })
    deltas = list(cond["qat_trained"]["paired_diff_vs_original"])
    if len(deltas) != len(ht["window_sha256"]):
        raise ValueError("paired deltas and window hashes differ in length")

    hist = error_histogram(cases)
    if sum(b["count"] for b in hist) != len(cases):
        raise ValueError("histogram does not cover every case")

    st = summary["storage"]
    disk, res = st["disk"], st["resident_fp32_unique"]
    disk_orig = [("embedding", "Embedding table (frozen, tied head)", disk["original"]["embedding_bytes"], TEAL_LIGHT),
                 ("other", "Other original weights", disk["original"]["non_embedding_bytes"], NAVY)]
    disk_student = [("embedding", "Embedding table (frozen, tied head)", disk["student"]["frozen_base_embedding_bytes"], TEAL_LIGHT),
                    ("norms", "Norms, biases and header of the frozen base", disk["student"]["frozen_base_norms_biases_and_header_bytes"], SLATE),
                    ("compressed", "Compressed LittleBit layers (packed signs, FP32 scales)", disk["student"]["compressed_layers_file_bytes"], COBALT)]
    res_orig = [("embedding", "Embedding table, FP32 (derived from shape)", res["embedding_fp32_bytes_derived"], TEAL_LIGHT),
                ("other", "Other original weights, FP32", res["original_non_embedding_bytes"], NAVY)]
    res_student = [("embedding", "Embedding table, FP32 (derived from shape)", res["embedding_fp32_bytes_derived"], TEAL_LIGHT),
                   ("factors", "LittleBit factors, scales and buffers held as FP32", res["student_littlebit_factor_scale_buffer_bytes"], COBALT),
                   ("other", "Other student tensors", res["student_other_minus_embedding_bytes"], SLATE)]
    for parts, total in ((disk_orig, disk["original"]["bytes"]), (disk_student, disk["student"]["bytes"]),
                         (res_orig, res["original_bytes"]), (res_student, res["student_bytes"])):
        if sum(p[2] for p in parts) != total:
            raise ValueError("storage segments do not add up to the recorded total %d" % total)

    bpw = st["bpw_converted_block_linears"]
    kv = st["kv_cache"]
    rss = []
    for stage in STAGES:
        r = summary["resources"][stage]
        rss.append({"stage": stage, "peak_rss_os_bytes": r["peak_rss_os_bytes"],
                    "peak_rss_os_gib": r["peak_rss_os_bytes"] / GIB,
                    "guard_peak_rss_seen_at_checks_bytes": r["guard_peak_rss_seen_at_checks_bytes"],
                    "guard_threshold_bytes": r["guard_threshold_bytes"],
                    "os_peak_exceeded_guard_threshold": r["os_peak_exceeded_guard_threshold"]})
    pr = patched["packer_regressions"]
    training = summary["training"]
    cfg = summary["pilot"]["littlebit_config"]
    return {
        "schema": "littlebit-companion/1",
        "companion_version": COMPANION_VERSION,
        "companion_release_status": ("v%s: presentation-only release over the immutable v0.1.0 data; scientific content and "
                                     "PDF cleared by independent review" % COMPANION_VERSION),
        "measured_data": {"release": DATA_RELEASE, "commit": DATA_COMMIT,
                          "note": "measured files are read unchanged from the v0.1.0 commit; this companion adds presentation only, no new experiment"},
        "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "license": "CC BY-NC 4.0"},
        "pages_url": PAGES_URL,
        "repository": REPO_URL,
        "three_js_version": THREE_VERSION,
        "sources": sources,
        "unpack": {
            "tier": "UPSTREAM_IMPORT",
            "cases": len(cases),
            "exact_cases": sum(1 for c in cases if c["upstream_exact"]),
            "sign_errors": sum(c["sign_errors"] for c in cases),
            "minus_to_plus_flips": sum(c["minus_to_plus_flips"] for c in cases),
            "plus_to_minus_flips": 0,  # verify_case rejects any +1 to -1 flip, so this is checked, not assumed
            "family_counts": {f: sum(1 for c in cases if c["family"] == f) for f in FAMILIES},
            "histogram": hist,
            "widths": width_table(cases),
            "corrected_decoder_exact_cases": sum(1 for c in cases if c["reference_shift_exact"]),
            "packer_matches_mirror_cases": sum(1 for c in cases if c["packer_matches_mirror"]),
            "unpacker_matches_mirror_cases": sum(1 for c in cases if c["unpacker_matches_mirror"]),
        },
        "patch": {
            "exhaustive_cases": pr["exhaustive_cases"],
            "patched_exact_cases": pr["patched_exact_cases"],
            "unpatched_exact_cases": pr["unpatched_exact_cases"],
            "removed": patched["patch"]["removed"][0].strip(),
            "added": patched["patch"]["added"][0].strip(),
            "official_loader_ran": patched["official_loader_ran"],
        },
        "quality": {
            "tier": "UPSTREAM_IMPORT_REAL_MODEL_PILOT",
            "conditions": quality,
            "qat_minus_original": deltas,
            "window_sha256": list(ht["window_sha256"]),
            "statistics": ht["statistics"],
        },
        "pilot": {
            "model": summary["pilot"]["model"]["repo"],
            "model_revision": summary["pilot"]["model"]["revision"],
            "converted_modules": train["conversion_summary"]["modules"],
            "converted_weight_count": bpw["converted_weight_count"],
            "eff_bit": cfg["eff_bit"], "residual": cfg["residual"], "use_itq": cfg["use_itq"],
            "seed": cfg["seed"], "steps": training["completed_steps"],
            "input_tokens": training["input_tokens"],
            "scored_tokens": cond["original"]["scored_tokens"],
            "test_windows": len(ht["window_sha256"]),
            "device": summary["pilot"]["environment"]["device"],
            "dtype": summary["pilot"]["environment"]["dtype"],
        },
        "storage": {
            "rule": st["rule"],
            "disk": {"original_bytes": disk["original"]["bytes"], "original_dtype": disk["original"]["dtype"],
                     "student_bytes": disk["student"]["bytes"],
                     "original_segments": [{"key": k, "label": l, "bytes": b} for k, l, b, _ in disk_orig],
                     "student_segments": [{"key": k, "label": l, "bytes": b} for k, l, b, _ in disk_student]},
            "resident_fp32": {"original_bytes": res["original_bytes"], "student_bytes": res["student_bytes"],
                              "original_segments": [{"key": k, "label": l, "bytes": b} for k, l, b, _ in res_orig],
                              "student_segments": [{"key": k, "label": l, "bytes": b} for k, l, b, _ in res_student]},
            "kv_cache": {"tokens": kv["tokens"], "original_bytes": kv["original_bytes"],
                         "student_bytes": kv["student_bytes"], "unchanged": kv["unchanged"]},
            "bpw_converted_block_linears": {
                "packed_signs_only": bpw["packed_signs_only"],
                "advertised_upstream_formula": bpw["advertised_upstream_formula"],
                "actual_checkpoint_file_fp32_scales": bpw["actual_checkpoint_file_fp32_scales"],
                "converted_weight_count": bpw["converted_weight_count"],
            },
            "excluded_tokenizer_and_config_bytes": st["excluded"]["tokenizer_and_config_files_bytes"],
            "_segments_for_charts": {"disk_orig": disk_orig, "disk_student": disk_student,
                                     "res_orig": res_orig, "res_student": res_student},
        },
        "resources": {"note": summary["resources"]["note"], "rss": rss},
    }


def public_data(data: dict) -> dict:
    out = json.loads(json.dumps(data))
    out["storage"].pop("_segments_for_charts")
    return out


# ---------------------------------------------------------------------------------------------
# SVG charts. Each quantitative mark carries data-key/data-value so tests can check it directly
# against the recorded files.


def svg_open(w: int, h: int, slug: str, title: str, desc: str) -> List[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" role="img" '
        'aria-labelledby="%s-title %s-desc" font-family="%s" data-chart="%s">\n' % (w, h, w, h, slug, slug, SANS, slug),
        '<title id="%s-title">%s</title>\n' % (slug, esc(title)),
        '<desc id="%s-desc">%s</desc>\n' % (slug, esc(desc)),
        '<rect width="%d" height="%d" fill="%s"/>\n' % (w, h, PAPER),
    ]


def text(x: float, y: float, s: str, size: float = 13, anchor: str = "start", weight: int = 400,
         fill: str = INK, family: str = "", extra: str = "") -> str:
    fam = ' font-family="%s"' % family if family else ""
    return '<text x="%s" y="%s" font-size="%s" text-anchor="%s" font-weight="%d" fill="%s"%s%s>%s</text>\n' % (
        num(x), num(y), num(size), anchor, weight, fill, fam, extra, esc(s))


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = RULE, width: float = 1, dash: str = "") -> str:
    d = ' stroke-dasharray="%s"' % dash if dash else ""
    return '<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s/>\n' % (
        num(x1), num(y1), num(x2), num(y2), stroke, num(width), d)


def header(parts: List[str], title: str, subtitle: str, w: int) -> None:
    parts.append(text(16, 28, title, size=16, weight=700))
    parts.append(text(16, 48, subtitle, size=12, fill=MUTED))
    parts.append(line(16, 58, w - 16, 58, stroke=RULE))


def footer(parts: List[str], note: str, w: int, h: int) -> None:
    parts.append(text(16, h - 12, note, size=11, fill=MUTED))


def hatch_defs(pid: str, color: str) -> str:
    return ('<defs><pattern id="%s" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<rect width="7" height="7" fill="%s"/><line x1="0" y1="0" x2="0" y2="7" stroke="%s" stroke-width="3"/>'
            '</pattern></defs>\n' % (pid, PAPER, color))


def marker(shape: str, x: float, y: float, color: str, attrs: str) -> str:
    if shape == "circle":
        return '<circle cx="%s" cy="%s" r="4.5" fill="%s"%s/>\n' % (num(x), num(y), color, attrs)
    if shape == "square":
        return '<rect x="%s" y="%s" width="9" height="9" fill="%s" stroke="%s"%s/>\n' % (
            num(x - 4.5), num(y - 4.5), PAPER, color, attrs)
    if shape == "diamond":
        pts = "%s,%s %s,%s %s,%s %s,%s" % (num(x), num(y - 6), num(x + 6), num(y), num(x), num(y + 6), num(x - 6), num(y))
        return '<polygon points="%s" fill="%s"%s/>\n' % (pts, color, attrs)
    if shape == "triangle":
        pts = "%s,%s %s,%s %s,%s" % (num(x), num(y - 6), num(x + 5.5), num(y + 4.5), num(x - 5.5), num(y + 4.5))
        return '<polygon points="%s" fill="%s"%s/>\n' % (pts, color, attrs)
    raise ValueError(shape)


def nice_top(maxv: float, target: int = 5) -> Tuple[float, float]:
    raw = maxv / float(target)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            step = m * mag
            break
    return math.ceil(maxv / step) * step, step


def chart_quality(data: dict) -> str:
    w, h = 720, 350
    conds = data["quality"]["conditions"]
    p = svg_open(w, h, "quality-log10-ppl", "Held-out perplexity on a log10 scale",
                 "Horizontal bars of log10 perplexity on the same 8 held-out windows (%s scored tokens): %s." % (
                     fmt_int(conds[0]["scored_tokens"]),
                     "; ".join("%s %s" % (c["label"], fmt_fixed(c["ppl"], 4)) for c in conds)))
    p.append(hatch_defs("defect-hatch", RUST))
    header(p, "Held-out perplexity, log10 scale",
           "Same 8 WikiText-2 test windows, %s scored tokens. Lower is better. Tiny pilot, one seed, one run." % fmt_int(conds[0]["scored_tokens"]), w)
    x0, x1, decades = 260.0, 610.0, 8
    per = (x1 - x0) / decades
    top, row_h = 76.0, 50.0
    axis_y = top + row_h * len(conds) + 2
    for d in range(decades + 1):
        x = x0 + d * per
        p.append(line(x, top - 6, x, axis_y, stroke=RULE))
        p.append('<text x="%s" y="%s" font-size="11" text-anchor="middle" fill="%s">10<tspan dy="-5" font-size="8">%d</tspan></text>\n' % (
            num(x), num(axis_y + 16), MUTED, d))
    p.append(text((x0 + x1) / 2, axis_y + 34, "perplexity (log10 axis; each gridline is 10 times the previous)", size=11, anchor="middle", fill=MUTED))
    for i, c in enumerate(conds):
        y = top + i * row_h
        defect = c["key"] == "qat_trained_upstream_decoder"
        color = SERIES_STYLE[c["key"]][0]
        p.append(text(16, y + 14, c["label"], size=13, weight=600))
        p.append(text(16, y + 30, c["note"], size=11, fill=RUST if defect else MUTED))
        bw = c["log10_ppl"] * per
        fill = "url(#defect-hatch)" if defect else color
        stroke = ' stroke="%s" stroke-width="1.5" stroke-dasharray="5 3"' % RUST if defect else ""
        p.append('<rect x="%s" y="%s" width="%s" height="22" fill="%s"%s data-key="ppl.%s" data-value="%r" data-log10="%r"/>\n' % (
            num(x0), num(y + 2), num(bw), fill, stroke, c["key"], c["ppl"], c["log10_ppl"]))
        p.append(text(x0 + bw + 8, y + 18, fmt_fixed(c["ppl"], 4), size=12, weight=600))
    p.append(line(x0, top - 6, x0, axis_y, stroke=INK, width=1.2))
    footer(p, "Source: results/summary.json (heldout_test). The hatched bar reloads the same QAT weights with the shipped unpacker.", w, h)
    p.append("</svg>\n")
    return "".join(p)


def chart_paired(data: dict) -> str:
    w, h = 720, 400
    conds = data["quality"]["conditions"]
    n = len(conds[0]["per_window_mean_nll"])
    p = svg_open(w, h, "paired-window-nll", "Per-window mean NLL, paired by window",
                 "Mean negative log-likelihood in nats for each of %d held-out windows under four conditions; "
                 "the windows are contiguous segments of one test prefix, not independent draws." % n)
    header(p, "Per-window mean NLL (nats), paired by window",
           "%d contiguous test windows from one prefix: paired, not independent draws. Lower is better." % n, w)
    x0, x1, y0, y1, ymax = 70.0, 520.0, 78.0, 330.0, 18.0
    def X(i: int) -> float:
        return x0 + (x1 - x0) * i / float(n - 1)
    def Y(v: float) -> float:
        return y1 - (y1 - y0) * v / ymax
    for t in range(0, int(ymax) + 1, 3):
        p.append(line(x0, Y(t), x1, Y(t), stroke=RULE))
        p.append(text(x0 - 8, Y(t) + 4, str(t), size=11, anchor="end", fill=MUTED))
    for i in range(n):
        p.append(text(X(i), y1 + 18, "w%d" % i, size=11, anchor="middle", fill=MUTED))
    p.append(text((x0 + x1) / 2, y1 + 38, "test window (identified by SHA-256 of its token ids)", size=11, anchor="middle", fill=MUTED))
    p.append('<text x="22" y="%s" font-size="11" text-anchor="middle" fill="%s" transform="rotate(-90 22 %s)">mean NLL, nats</text>\n' % (
        num((y0 + y1) / 2), MUTED, num((y0 + y1) / 2)))
    for c in conds:
        color, shape, dash = SERIES_STYLE[c["key"]]
        vals = c["per_window_mean_nll"]
        pts = " ".join("%s,%s" % (num(X(i)), num(Y(v))) for i, v in enumerate(vals))
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.6"%s/>\n' % (pts, color, d))
        for i, v in enumerate(vals):
            p.append(marker(shape, X(i), Y(v), color,
                            ' data-key="nll.%s" data-window="%d" data-value="%r"' % (c["key"], i, v)))
    lx, ly = 540.0, 96.0
    for k, c in enumerate(conds):
        color, shape, dash = SERIES_STYLE[c["key"]]
        y = ly + k * 46
        p.append(line(lx, y, lx + 26, y, stroke=color, width=1.6, dash=dash))
        p.append(marker(shape, lx + 13, y, color, ""))
        p.append(text(lx + 34, y + 4, c["label"].replace(", defective decoder", ","), size=11, weight=600))
        p.append(text(lx + 34, y + 19, "defective decoder" if c["key"] == "qat_trained_upstream_decoder" else c["note"],
                      size=10, fill=RUST if c["key"] == "qat_trained_upstream_decoder" else MUTED))
    footer(p, "Source: results/summary.json, per_window_mean_nll. Dashed series: decoder defect condition, not a quality result.", w, h)
    p.append("</svg>\n")
    return "".join(p)


def chart_deltas(data: dict) -> str:
    w, h = 720, 230
    deltas = data["quality"]["qat_minus_original"]
    p = svg_open(w, h, "nll-delta-strip", "Paired NLL increase of the QAT student over the original, all 8 windows",
                 "Dot plot of the %d recorded per-window differences in mean NLL, QAT student minus original: %s nats." % (
                     len(deltas), ", ".join("window %d %.3f" % (i, v) for i, v in enumerate(deltas))))
    header(p, "QAT student minus original: all %d paired window differences" % len(deltas),
           "Each dot is one recorded window (nats). Descriptive only: windows are contiguous, one seed, one run.", w)
    lo = math.floor(min(deltas) * 5) / 5.0
    hi = math.ceil(max(deltas) * 5) / 5.0
    x0, x1, axis_y = 60.0, 660.0, 128.0
    def X(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)
    p.append(line(x0, axis_y, x1, axis_y, stroke=INK, width=1.2))
    ticks = int(round((hi - lo) / 0.2))
    for t in range(ticks + 1):
        v = lo + 0.2 * t
        p.append(line(X(v), axis_y - 4, X(v), axis_y + 4, stroke=INK))
        p.append(text(X(v), axis_y + 62, "%.1f" % v, size=11, anchor="middle", fill=MUTED))
    p.append(text((x0 + x1) / 2, axis_y + 80, "NLL increase over the original model, nats per token", size=11, anchor="middle", fill=MUTED))
    order = sorted(range(len(deltas)), key=lambda i: deltas[i])
    for rank, i in enumerate(order):
        v = deltas[i]
        above = rank % 2 == 0
        ly = axis_y - 22 if above else axis_y + 34
        p.append(line(X(v), axis_y, X(v), ly + (6 if above else -14), stroke=RULE))
        p.append('<circle cx="%s" cy="%s" r="6" fill="%s" stroke="%s" stroke-width="1.5" data-key="delta.qat_minus_original" data-window="%d" data-value="%r"/>\n' % (
            num(X(v)), num(axis_y), COBALT, PAPER, i, v))
        p.append(text(X(v), ly, "w%d  %.3f" % (i, v), size=11, anchor="middle", weight=600))
    footer(p, "Source: results/summary.json, heldout_test.conditions.qat_trained.paired_diff_vs_original (n = %d)." % len(deltas), w, h)
    p.append("</svg>\n")
    return "".join(p)


def chart_histogram(data: dict) -> str:
    w, h = 720, 360
    u = data["unpack"]
    hist = u["histogram"]
    p = svg_open(w, h, "unpack-error-histogram", "Sign errors per recorded unpack case, shipped unpacker",
                 "Histogram of sign errors per case over %s recorded cases through the actual upstream binary_unpacker. %s." % (
                     fmt_int(u["cases"]), "; ".join("%s errors: %s cases" % (b["label"], fmt_int(b["count"])) for b in hist)))
    header(p, "Sign errors per case through the shipped binary_unpacker",
           "All %s recorded cases (UPSTREAM_IMPORT). Bin counts are printed on each bar. Every error is a -1 read back as +1." % fmt_int(u["cases"]), w)
    x0, x1, y0, y1 = 80.0, 690.0, 84.0, 290.0
    top, step = nice_top(max(b["count"] for b in hist))
    def Y(v: float) -> float:
        return y1 - (y1 - y0) * v / top
    t = 0.0
    while t <= top + 1e-9:
        p.append(line(x0, Y(t), x1, Y(t), stroke=RULE))
        p.append(text(x0 - 8, Y(t) + 4, fmt_int(t), size=11, anchor="end", fill=MUTED))
        t += step
    slot = (x1 - x0) / len(hist)
    for i, b in enumerate(hist):
        bx = x0 + i * slot + 8
        bw = slot - 16
        p.append('<rect x="%s" y="%s" width="%s" height="%s" fill="%s" data-bin-lo="%d" data-bin-hi="%d" data-count="%d"/>\n' % (
            num(bx), num(Y(b["count"])), num(bw), num(y1 - Y(b["count"])), TEAL if b["lo"] == 0 else COBALT,
            b["lo"], b["hi"], b["count"]))
        p.append(text(bx + bw / 2, Y(b["count"]) - 6, fmt_int(b["count"]), size=12, anchor="middle", weight=700))
        p.append(text(bx + bw / 2, y1 + 18, b["label"], size=11, anchor="middle", fill=MUTED))
    p.append(line(x0, y1, x1, y1, stroke=INK, width=1.2))
    p.append(text((x0 + x1) / 2, y1 + 38, "sign errors in the case (bins are whole numbers; 0 means the case decodes exactly)", size=11, anchor="middle", fill=MUTED))
    p.append('<text x="24" y="%s" font-size="11" text-anchor="middle" fill="%s" transform="rotate(-90 24 %s)">cases</text>\n' % (
        num((y0 + y1) / 2), MUTED, num((y0 + y1) / 2)))
    footer(p, "Source: results/unpack_cases_upstream_import.csv, column sign_errors. n = %s cases, %s sign errors in total." % (
        fmt_int(u["cases"]), fmt_int(u["sign_errors"])), w, h)
    p.append("</svg>\n")
    return "".join(p)


def chart_stacked(slug: str, title: str, subtitle: str, rows: Sequence[Tuple[str, str, int, list]],
                  notes: Sequence[str], desc: str) -> str:
    # One footer line per note; the canvas (height and viewBox) grows so no line is clipped.
    line_h = 15
    w, h = 720, 300 + line_h * (len(notes) - 1)
    p = svg_open(w, h, slug, title, desc)
    header(p, title, subtitle, w)
    x0, x1 = 16.0, 704.0
    vmax = max(r[2] for r in rows)
    y = 92.0
    for key, label, total, segs in rows:
        p.append(text(x0, y - 8, "%s: %s bytes" % (label, fmt_int(total)), size=13, weight=600))
        x = x0
        for skey, slabel, b, color in segs:
            bw = (x1 - x0) * b / float(vmax)
            p.append('<rect x="%s" y="%s" width="%s" height="26" fill="%s" stroke="%s" stroke-width="0.5" data-key="%s.%s" data-value="%d"><title>%s: %s bytes</title></rect>\n' % (
                num(x), num(y), num(bw), color, PAPER, key, skey, b, esc(slabel), fmt_int(b)))
            x += bw
        y += 70
    ly = y - 12
    seen = []
    for _, _, _, segs in rows:
        for skey, slabel, _, color in segs:
            if (slabel, color) not in seen:
                seen.append((slabel, color))
    for k, (slabel, color) in enumerate(seen):
        lx = x0 + (k % 2) * 344
        yy = ly + (k // 2) * 20
        p.append('<rect x="%s" y="%s" width="12" height="12" fill="%s" stroke="%s"/>\n' % (num(lx), num(yy - 10), color, INK))
        p.append(text(lx + 18, yy, slabel, size=11))
    for i, note in enumerate(notes):
        p.append(text(16, h - 12 - line_h * (len(notes) - 1 - i), note, size=11, fill=MUTED))
    p.append("</svg>\n")
    return "".join(p)


def chart_disk(data: dict) -> str:
    seg = data["storage"]["_segments_for_charts"]
    d = data["storage"]["disk"]
    rows = (("disk.original", "Original, one BF16 file", d["original_bytes"], seg["disk_orig"]),
            ("disk.student", "Pilot student, frozen BF16 base + compressed layers", d["student_bytes"], seg["disk_student"]))
    return chart_stacked(
        "storage-disk", "Checkpoint bytes on disk (stored dtypes)",
        "Weights only. Tokenizer and config files excluded. Not comparable with the resident FP32 panel.",
        rows, ("Source: results/summary.json, storage.disk. The %s-byte norms, biases and header segment is too thin to see." % (
            fmt_int(seg["disk_student"][1][2])),),
        "Stacked bars of checkpoint bytes on disk: original %s bytes, pilot student %s bytes, of which the embedding table is %s bytes in both." % (
            fmt_int(d["original_bytes"]), fmt_int(d["student_bytes"]), fmt_int(seg["disk_orig"][0][2])))


def chart_resident(data: dict) -> str:
    seg = data["storage"]["_segments_for_charts"]
    r = data["storage"]["resident_fp32"]
    kv = data["storage"]["kv_cache"]
    rows = (("resident.original", "Original, resident FP32", r["original_bytes"], seg["res_orig"]),
            ("resident.student", "Pilot student after load, resident FP32", r["student_bytes"], seg["res_student"]))
    return chart_stacked(
        "storage-resident", "Resident unique tensor bytes after loading (FP32)",
        "Tensors only: no KV cache, activations or allocator slack. Latent factors are resident as FP32, not packed bits.",
        rows, ("Source: results/summary.json, storage.resident_fp32_unique.",
               "KV cache (%d tokens, FP32): %s bytes in both, unchanged, not shown." % (
                   kv["tokens"], fmt_int(kv["original_bytes"]))),
        "Stacked bars of resident unique FP32 tensor bytes: original %s bytes, pilot student %s bytes." % (
            fmt_int(r["original_bytes"]), fmt_int(r["student_bytes"])))


def chart_bpw(data: dict) -> str:
    w, h = 720, 250
    b = data["storage"]["bpw_converted_block_linears"]
    rows = (("packed_signs_only", "Packed signs only", "sign bits, no scales", TEAL, 4),
            ("advertised_upstream_formula", "Advertised upstream formula", "what the config targets", NAVY, 5),
            ("actual_checkpoint_file_fp32_scales", "Actual checkpoint file", "packed signs + FP32 scales + header", COBALT, 4))
    p = svg_open(w, h, "bpw", "Bits per weight for the converted block linears",
                 "Bits per weight over %s converted weights: %s." % (
                     fmt_int(b["converted_weight_count"]),
                     "; ".join("%s %s" % (lab, fmt_fixed(b[k], d)) for k, lab, _, _, d in rows)))
    header(p, "Bits per weight, converted block linears only",
           "%s weights in %d modules. The frozen BF16 base file is excluded here." % (
               fmt_int(b["converted_weight_count"]), data["pilot"]["converted_modules"]), w)
    x0, x1, vmax = 250.0, 660.0, 0.7
    top, row_h = 78.0, 44.0
    axis_y = top + row_h * len(rows)
    for k in range(8):
        v = k / 10.0
        x = x0 + (x1 - x0) * v / vmax
        p.append(line(x, top - 6, x, axis_y, stroke=RULE))
        p.append(text(x, axis_y + 16, "%.1f" % v, size=11, anchor="middle", fill=MUTED))
    for i, (key, label, note, color, digits) in enumerate(rows):
        y = top + i * row_h
        p.append(text(16, y + 12, label, size=13, weight=600))
        p.append(text(16, y + 27, note, size=11, fill=MUTED))
        bw = (x1 - x0) * b[key] / vmax
        p.append('<rect x="%s" y="%s" width="%s" height="20" fill="%s" data-key="bpw.%s" data-value="%r"/>\n' % (
            num(x0), num(y + 2), num(bw), color, key, b[key]))
        p.append(text(x0 + bw + 8, y + 17, fmt_fixed(b[key], digits), size=12, weight=600))
    p.append(line(x0, top - 6, x0, axis_y, stroke=INK, width=1.2))
    footer(p, "Source: results/summary.json, storage.bpw_converted_block_linears. Upstream main.py would cast scales to BF16.", w, h)
    p.append("</svg>\n")
    return "".join(p)


def chart_rss(data: dict) -> str:
    w, h = 720, 300
    rss = data["resources"]["rss"]
    guard = rss[0]["guard_threshold_bytes"] / GIB
    p = svg_open(w, h, "os-peak-rss", "Operating system peak RSS per pilot stage",
                 "Peak resident set size reported by the OS for each pilot stage in GiB: %s. Guard threshold %s GiB, sampled at checkpoints." % (
                     "; ".join("%s %.2f" % (r["stage"], r["peak_rss_os_gib"]) for r in rss), num(guard)))
    header(p, "OS peak RSS per pilot stage (GiB, process memory)",
           "A different metric from tensor bytes: whole-process memory, sampled guard. Not comparable with the storage panels.", w)
    x0, x1, vmax = 130.0, 650.0, 7.0
    top, row_h = 92.0, 40.0
    axis_y = top + row_h * len(rss)
    def X(v: float) -> float:
        return x0 + (x1 - x0) * v / vmax
    for k in range(8):
        p.append(line(X(k), top - 6, X(k), axis_y, stroke=RULE))
        p.append(text(X(k), axis_y + 16, str(k), size=11, anchor="middle", fill=MUTED))
    p.append(line(X(guard), top - 14, X(guard), axis_y, stroke=RUST, width=1.5, dash="5 3"))
    p.append(text(X(guard) - 4, top - 18, "%s GiB guard threshold (sampled, not a hard cap)" % num(guard), size=11, anchor="end", fill=RUST))
    for i, r in enumerate(rss):
        y = top + i * row_h
        p.append(text(16, y + 16, r["stage"], size=13, weight=600))
        p.append('<rect x="%s" y="%s" width="%s" height="20" fill="%s" data-key="rss.%s" data-value="%d"/>\n' % (
            num(x0), num(y + 2), num(X(r["peak_rss_os_gib"]) - x0), RUST if r["os_peak_exceeded_guard_threshold"] else SLATE,
            r["stage"], r["peak_rss_os_bytes"]))
        p.append(text(X(r["peak_rss_os_gib"]) + 8, y + 17, "%.2f" % r["peak_rss_os_gib"], size=12, weight=600))
        if r["os_peak_exceeded_guard_threshold"]:
            sx = X(r["guard_peak_rss_seen_at_checks_bytes"] / GIB)
            p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="2" data-key="rss_sampled.%s" data-value="%d"/>\n' % (
                num(sx), num(y - 2), num(sx), num(y + 26), INK, r["stage"], r["guard_peak_rss_seen_at_checks_bytes"]))
    p.append(line(x0, top - 6, x0, axis_y, stroke=INK, width=1.2))
    footer(p, "Source: results/summary.json, resources. Black tick on the train bar: highest value the guard sampled (%.2f GiB)." % (
        max(r["guard_peak_rss_seen_at_checks_bytes"] for r in rss if r["os_peak_exceeded_guard_threshold"]) / GIB
        if any(r["os_peak_exceeded_guard_threshold"] for r in rss) else 0.0), w, h)
    p.append("</svg>\n")
    return "".join(p)


def diagram_dual_path() -> str:
    w, h = 960, 360
    p = svg_open(w, h, "dual-path-diagram", "Structure of the dual-path binary low-rank layer (illustrative)",
                 "Structural diagram, not data. Input x enters a main path and a residual path. Each path computes "
                 "%s with its own binary sign matrices and scale vectors; the two path outputs are added together with the bias to give y. "
                 "No dimension or value in this diagram is measured." % UPSTREAM_FORMULA)
    header(p, "Dual-path binary low-rank layer: structure only",
           "Upstream LittleBitLinear at commit %s. Shapes are symbolic; nothing here is measured." % UPSTREAM_COMMIT[:7], w)
    p.append(text(16, 82, UPSTREAM_FORMULA + "   (main path)", size=13, family=MONO, weight=600))
    p.append(text(16, 100, "+ the same with U_R, V_R, u1_R, u2_R, v1_R, v2_R (residual path)  + bias", size=13, family=MONO))
    p.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker></defs>\n' % INK)
    ops = (("* v2", "scale vector", "length d_in", TEAL),
           ("@ Vq.T", "binary signs, packed", "s x d_in, +1 or -1", COBALT),
           ("* (v1 * u2)", "two scale vectors", "length s", TEAL),
           ("@ Uq.T", "binary signs, packed", "d_out x s, +1 or -1", COBALT),
           ("* u1", "scale vector", "length d_out", TEAL))
    rows = ((150.0, "main path", ""), (262.0, "residual path", "_R"))
    bx0, bw, gap, bh = 130.0, 118.0, 26.0, 44.0
    p.append('<rect x="16" y="%s" width="70" height="%s" rx="6" fill="%s" stroke="%s"/>\n' % (num(186), num(bh), PAPER, INK))
    p.append(text(51, 213, "x", size=15, anchor="middle", weight=700, family=MONO))
    p.append(text(51, 246, "input, d_in", size=10, anchor="middle", fill=MUTED))
    sum_x, sum_y = bx0 + 5 * (bw + gap) + 22, 228.0
    for y, label, suffix in rows:
        p.append(text(bx0, y - 12, label, size=11, weight=700, fill=MUTED))
        p.append('<path d="M86,208 C108,208 108,%s %s,%s" fill="none" stroke="%s" stroke-width="1.3" marker-end="url(#arrow)"/>\n' % (
            num(y + bh / 2), num(bx0 - 2), num(y + bh / 2), INK))
        for i, (op, kind, shape, color) in enumerate(ops):
            x = bx0 + i * (bw + gap)
            label_op = op.replace("v2", "v2" + suffix).replace("Vq", "Vq" + suffix).replace("v1", "v1" + suffix).replace(
                "u2", "u2" + suffix).replace("Uq", "Uq" + suffix).replace("u1", "u1" + suffix)
            p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="6" fill="%s" stroke="%s" stroke-width="1.6"/>\n' % (
                num(x), num(y), num(bw), num(bh), PAPER, color))
            p.append(text(x + bw / 2, y + 20, label_op, size=12, anchor="middle", weight=700, family=MONO))
            p.append(text(x + bw / 2, y + 36, kind, size=10, anchor="middle", fill=MUTED))
            p.append(text(x + bw / 2, y + bh + 14, shape, size=10, anchor="middle", fill=MUTED))
            if i < len(ops) - 1:
                p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.3" marker-end="url(#arrow)"/>\n' % (
                    num(x + bw), num(y + bh / 2), num(x + bw + gap - 2), num(y + bh / 2), INK))
        last_x = bx0 + 4 * (bw + gap) + bw
        p.append('<path d="M%s,%s C%s,%s %s,%s %s,%s" fill="none" stroke="%s" stroke-width="1.3" marker-end="url(#arrow)"/>\n' % (
            num(last_x), num(y + bh / 2), num(sum_x - 8), num(y + bh / 2), num(sum_x - 8), num(sum_y), num(sum_x - 16), num(sum_y), INK))
    p.append('<circle cx="%s" cy="%s" r="14" fill="%s" stroke="%s" stroke-width="1.6"/>\n' % (num(sum_x), num(sum_y), PAPER, INK))
    p.append(text(sum_x, sum_y + 5, "+", size=16, anchor="middle", weight=700))
    p.append(text(sum_x, sum_y + 32, "+ bias", size=11, anchor="middle", family=MONO))
    p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.3" marker-end="url(#arrow)"/>\n' % (
        num(sum_x + 14), num(sum_y), num(sum_x + 44), num(sum_y), INK))
    p.append(text(sum_x + 56, sum_y + 5, "y", size=15, weight=700, family=MONO))
    footer(p, "Structure only: d_in, s and d_out are symbols, not measured sizes. The packed format stores Uq and Vq as 32-bit words.", w, h)
    p.append("</svg>\n")
    return "".join(p)


CHARTS = (
    ("unpack_error_histogram.svg", chart_histogram),
    ("quality_log10_ppl.svg", chart_quality),
    ("paired_window_nll.svg", chart_paired),
    ("nll_delta_strip.svg", chart_deltas),
    ("storage_disk.svg", chart_disk),
    ("storage_resident.svg", chart_resident),
    ("bpw.svg", chart_bpw),
    ("os_peak_rss.svg", chart_rss),
)


# ---------------------------------------------------------------------------------------------
# HTML tables and template values


def table(caption: str, head: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]) -> str:
    out = ['<div class="table-wrap" role="region" tabindex="0" aria-label="%s">\n<table>\n<caption>%s</caption>\n<thead><tr>' % (
        esc(caption), esc(caption))]
    for hcell, isnum in zip(head, numeric):
        out.append('<th scope="col"%s>%s</th>' % (' class="num"' if isnum else "", esc(hcell)))
    out.append("</tr></thead>\n<tbody>\n")
    for r in rows:
        out.append("<tr>")
        for k, (cell, isnum) in enumerate(zip(r, numeric)):
            tag = "th" if k == 0 else "td"
            scope = ' scope="row"' if k == 0 else ""
            out.append('<%s%s%s>%s</%s>' % (tag, scope, ' class="num"' if isnum else "", esc(cell), tag))
        out.append("</tr>\n")
    out.append("</tbody>\n</table>\n</div>")
    return "".join(out)


def template_values(data: dict) -> Dict[str, str]:
    u, q, st, pi, pa = data["unpack"], data["quality"], data["storage"], data["pilot"], data["patch"]
    conds = {c["key"]: c for c in q["conditions"]}
    v: Dict[str, str] = {}
    v["companion_version"] = COMPANION_VERSION
    v["data_release"] = DATA_RELEASE
    v["data_commit"] = DATA_COMMIT
    v["data_commit_short"] = DATA_COMMIT[:7]
    v["upstream_url"] = UPSTREAM_URL
    v["upstream_commit"] = UPSTREAM_COMMIT
    v["upstream_commit_short"] = UPSTREAM_COMMIT[:7]
    v["repo_url"] = REPO_URL
    v["pages_url"] = PAGES_URL
    v["manuscript_pdf_url"] = MANUSCRIPT_PDF_URL
    v["three_version"] = THREE_VERSION
    v["formula"] = UPSTREAM_FORMULA
    v["formula_note"] = FORMULA_NOTE
    v["unpack.cases"] = fmt_int(u["cases"])
    v["unpack.exact"] = fmt_int(u["exact_cases"])
    v["unpack.errors"] = fmt_int(u["sign_errors"])
    v["unpack.byte_tiled"] = fmt_int(u["family_counts"]["byte_tiled"])
    v["unpack.one_hot"] = fmt_int(u["family_counts"]["one_hot"])
    v["unpack.sign_example"] = fmt_int(u["family_counts"]["sign_example"])
    v["unpack.corrected_exact"] = fmt_int(u["corrected_decoder_exact_cases"])
    v["patch.cases"] = fmt_int(pa["exhaustive_cases"])
    v["patch.patched_exact"] = fmt_int(pa["patched_exact_cases"])
    v["patch.unpatched_exact"] = fmt_int(pa["unpatched_exact_cases"])
    v["patch.removed"] = pa["removed"]
    v["patch.added"] = pa["added"]
    for key, c in conds.items():
        v["ppl.%s" % key] = fmt_fixed(c["ppl"], 4)
        v["nll.%s" % key] = fmt_fixed(c["mean_nll"], 4)
    v["ratio.qat"] = "%.0f" % (conds["qat_trained"]["ppl"] / conds["original"]["ppl"])
    d = q["qat_minus_original"]
    v["delta.min"] = "%.2f" % min(d)
    v["delta.max"] = "%.2f" % max(d)
    v["delta.n"] = str(len(d))
    v["pilot.model"] = pi["model"]
    v["pilot.modules"] = str(pi["converted_modules"])
    v["pilot.weights"] = fmt_int(pi["converted_weight_count"])
    v["pilot.steps"] = str(pi["steps"])
    v["pilot.input_tokens"] = fmt_int(pi["input_tokens"])
    v["pilot.scored_tokens"] = fmt_int(pi["scored_tokens"])
    v["pilot.windows"] = str(pi["test_windows"])
    v["pilot.seed"] = str(pi["seed"])
    v["pilot.eff_bit"] = "%g" % pi["eff_bit"]
    v["pilot.itq"] = "true" if pi["use_itq"] else "false"
    v["pilot.device"] = pi["device"].upper()
    v["pilot.dtype"] = pi["dtype"]
    b = st["bpw_converted_block_linears"]
    v["bpw.formula"] = "%.5f" % b["advertised_upstream_formula"]
    v["bpw.file"] = "%.4f" % b["actual_checkpoint_file_fp32_scales"]
    v["bpw.signs"] = "%.4f" % b["packed_signs_only"]
    v["disk.original"] = fmt_int(st["disk"]["original_bytes"])
    v["disk.student"] = fmt_int(st["disk"]["student_bytes"])
    v["resident.original"] = fmt_int(st["resident_fp32"]["original_bytes"])
    v["resident.student"] = fmt_int(st["resident_fp32"]["student_bytes"])
    v["kv.bytes"] = fmt_int(st["kv_cache"]["original_bytes"])
    v["kv.tokens"] = str(st["kv_cache"]["tokens"])
    v["excluded.bytes"] = fmt_int(st["excluded_tokenizer_and_config_bytes"])
    train = [r for r in data["resources"]["rss"] if r["stage"] == "train"][0]
    v["rss.train"] = "%.2f" % train["peak_rss_os_gib"]
    v["rss.guard"] = num(train["guard_threshold_bytes"] / GIB)
    v["rss.sampled"] = "%.2f" % (train["guard_peak_rss_seen_at_checks_bytes"] / GIB)

    v["table.widths"] = table(
        "Recorded unpack cases by row width (shipped binary_unpacker)",
        ("width", "cases", "exact cases", "byte-tiled rows exact", "sign errors"),
        [(str(r["width"]), fmt_int(r["cases"]), fmt_int(r["exact_cases"]),
          "%s of %s" % (fmt_int(r["byte_rows_exact"]), fmt_int(r["byte_rows"])), fmt_int(r["sign_errors"]))
         for r in u["widths"]] + [("all", fmt_int(u["cases"]), fmt_int(u["exact_cases"]),
                                   "%s of %s" % (fmt_int(sum(r["byte_rows_exact"] for r in u["widths"])),
                                                 fmt_int(sum(r["byte_rows"] for r in u["widths"]))),
                                   fmt_int(u["sign_errors"]))],
        (False, True, True, True, True))
    v["table.histogram"] = table(
        "Histogram bins: sign errors per case",
        ("sign errors in the case", "cases"),
        [(bn["label"], fmt_int(bn["count"])) for bn in u["histogram"]] + [("total", fmt_int(u["cases"]))],
        (False, True))
    v["table.quality"] = table(
        "Held-out perplexity and mean NLL, same %s windows" % pi["test_windows"],
        ("condition", "perplexity", "log10 perplexity", "mean NLL (nats)", "windows worse than original"),
        [(c["label"], fmt_fixed(c["ppl"], 4), "%.3f" % c["log10_ppl"], fmt_fixed(c["mean_nll"], 4),
          "reference" if c["key"] == "original" else "%d of %d" % (c["windows_worse_than_original"], pi["test_windows"]))
         for c in q["conditions"]],
        (False, True, True, True, True))
    v["table.paired"] = table(
        "Per-window mean NLL (nats)",
        ["window", "SHA-256 (first 12 hex)"] + [c["label"] for c in q["conditions"]],
        [["w%d" % i, q["window_sha256"][i][:12]] + ["%.4f" % c["per_window_mean_nll"][i] for c in q["conditions"]]
         for i in range(pi["test_windows"])],
        [False, False] + [True] * len(q["conditions"]))
    v["table.deltas"] = table(
        "QAT student minus original, per window (nats)",
        ("window", "NLL increase"),
        [("w%d" % i, "%.4f" % x) for i, x in enumerate(d)],
        (False, True))
    seg = st["_segments_for_charts"]
    v["table.disk"] = table(
        "Checkpoint bytes on disk, stored dtypes",
        ("file part", "original", "pilot student"),
        [("Embedding table (frozen, tied head)", fmt_int(seg["disk_orig"][0][2]), fmt_int(seg["disk_student"][0][2])),
         ("Other original weights", fmt_int(seg["disk_orig"][1][2]), "not stored (replaced)"),
         ("Norms, biases and header of the frozen base", "inside the original file", fmt_int(seg["disk_student"][1][2])),
         ("Compressed LittleBit layers", "none", fmt_int(seg["disk_student"][2][2])),
         ("Total weight bytes", fmt_int(st["disk"]["original_bytes"]), fmt_int(st["disk"]["student_bytes"]))],
        (False, True, True))
    v["table.resident"] = table(
        "Resident unique tensor bytes after loading, FP32",
        ("tensor group", "original", "pilot student"),
        [("Embedding table, FP32 (derived from shape)", fmt_int(seg["res_orig"][0][2]), fmt_int(seg["res_student"][0][2])),
         ("Other original weights, FP32", fmt_int(seg["res_orig"][1][2]), "replaced"),
         ("LittleBit factors, scales and buffers, FP32", "none", fmt_int(seg["res_student"][1][2])),
         ("Other student tensors", "included above", fmt_int(seg["res_student"][2][2])),
         ("Total resident tensor bytes", fmt_int(st["resident_fp32"]["original_bytes"]), fmt_int(st["resident_fp32"]["student_bytes"])),
         ("KV cache, %d tokens, FP32 (not weights)" % st["kv_cache"]["tokens"], fmt_int(st["kv_cache"]["original_bytes"]),
          fmt_int(st["kv_cache"]["student_bytes"]))],
        (False, True, True))
    v["table.bpw"] = table(
        "Bits per weight, converted block linears only",
        ("measure", "bits per weight"),
        [("Packed signs only", "%.4f" % b["packed_signs_only"]),
         ("Advertised upstream formula", "%.5f" % b["advertised_upstream_formula"]),
         ("Actual checkpoint file, FP32 scales", "%.4f" % b["actual_checkpoint_file_fp32_scales"])],
        (False, True))
    v["table.rss"] = table(
        "OS peak RSS per pilot stage",
        ("stage", "OS peak RSS (GiB)", "highest guard sample (GiB)", "above the guard threshold"),
        [(r["stage"], "%.2f" % r["peak_rss_os_gib"], "%.2f" % (r["guard_peak_rss_seen_at_checks_bytes"] / GIB),
          "yes" if r["os_peak_exceeded_guard_threshold"] else "no") for r in data["resources"]["rss"]],
        (False, True, True, False))
    items = []
    for s in data["sources"]:
        items.append('<li><a href="%s/blob/%s/%s"><code>%s</code></a>, %s bytes, sha256 <code class="hash">%s</code></li>' % (
            REPO_URL, DATA_COMMIT, esc(s["path"]), esc(s["path"]), fmt_int(s["bytes"]), s["sha256"]))
    v["sources.list"] = "<ul class=\"sources\">\n" + "\n".join(items) + "\n</ul>"
    return v


PLACEHOLDER = re.compile(r"\{\{([a-z0-9_.]+)\}\}")
RAW_KEYS = ("table.", "sources.")


def render_template(template: str, values: Dict[str, str]) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(values))
    if missing:
        raise KeyError("template placeholders without a value: %s" % ", ".join(missing))
    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        return values[key] if key.startswith(RAW_KEYS) else esc(values[key])
    return PLACEHOLDER.sub(sub, template)


# ---------------------------------------------------------------------------------------------
# Build


def dump_json(obj: object) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n"


def build_outputs() -> Dict[str, bytes]:
    raw = {rel: read_bytes(rel) for rel in (SUMMARY, UNPACK_CSV, PATCHED, TRAIN)}
    sources = [source_record(SUMMARY, raw[SUMMARY], "held-out NLL, storage, BPW, KV cache, OS RSS"),
               source_record(UNPACK_CSV, raw[UNPACK_CSV], "recorded unpack cases"),
               source_record(PATCHED, raw[PATCHED], "patch regression counts"),
               source_record(TRAIN, raw[TRAIN], "converted module count")]
    summary = json.loads(raw[SUMMARY].decode("utf-8"))
    patched = json.loads(raw[PATCHED].decode("utf-8"))
    train = json.loads(raw[TRAIN].decode("utf-8"))
    cases = parse_cases(raw[UNPACK_CSV].decode("utf-8"))
    data = collect(summary, cases, patched, train, sources)

    out: Dict[str, bytes] = {}
    out[OUT_DATA] = dump_json(public_data(data)).encode("utf-8")
    case_doc = {
        "schema": "littlebit-companion-cases/1",
        "source": sources[1],
        "tier": "UPSTREAM_IMPORT",
        "bit_order": "column j is bit (j % 32) of word j // 32; a set bit encodes -1",
        "fields": ["case_id", "family", "width", "param", "n_minus", "packed_words_hex", "upstream_exact",
                   "sign_errors", "first_error_col"],
        "cases": [[c["case_id"], c["family"], c["width"], c["param"], c["n_minus"], c["words_hex"],
                   c["upstream_exact"], c["sign_errors"], c["first_error_col"]] for c in cases],
    }
    out[OUT_CASES] = (json.dumps(case_doc, separators=(",", ":"), sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    for name, fn in CHARTS:
        out[OUT_ASSETS + "/" + name] = fn(data).encode("utf-8")
    out[OUT_ASSETS + "/diagram_dual_path.svg"] = diagram_dual_path().encode("utf-8")
    template = read_bytes(TEMPLATE).decode("utf-8")
    out[OUT_INDEX] = render_template(template, template_values(data)).encode("utf-8")
    return out


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if any output differs")
    args = ap.parse_args(argv)
    outputs = build_outputs()
    stale = []
    for rel in sorted(outputs):
        path = os.path.join(PROJECT_ROOT, rel)
        current = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                current = f.read()
        if current == outputs[rel]:
            continue
        stale.append(rel)
        if not args.check:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(outputs[rel])
    if args.check:
        for rel in stale:
            print("out of date: %s" % rel)
        print("companion outputs: %d checked, %d out of date" % (len(outputs), len(stale)))
        return 1 if stale else 0
    print("companion outputs: %d written or unchanged, %d updated" % (len(outputs), len(stale)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
