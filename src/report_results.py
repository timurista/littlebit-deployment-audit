"""Figures and a summary for the completed LittleBit quality pilot, from recorded JSON only.

Writes:

  figures/paired_window_nll.{png,svg}   per-window held-out NLL, paired by window hash
  figures/heldout_ppl_log.{png,svg}     token-weighted perplexity, log scale
  figures/weight_bytes.{png,svg}        on-disk bytes and unique resident FP32 bytes, separately
  figures/train_ce_kl.{png,svg}         CE and KL per QAT step
  results/summary.json                  every number used in the figures and the docs

Two modes, recorded in summary.json under "replay":

  --metadata-only      METADATA REPLAY. Reads only tracked public files: the stage JSON in
                       results/quality-pilot/ (manifest, baseline, feasibility, train, evaluate),
                       train_steps.jsonl and results/quality-pilot/public_metadata.json. Works from a
                       clean clone without model weights, checkpoints, checkpoint configs or host
                       logs. It recomputes every derived number and checks the public files against
                       each other, but it does NOT validate any private checkpoint: checkpoint byte
                       counts, header facts and test outcomes are taken from public_metadata.json.

  (default)            PRIVATE VALIDATION, owner host only. Additionally hashes the init, trained
                       and frozen-base checkpoints against train.json, parses safetensors headers,
                       reads the checkpoint config, the original weights header (if present) and the
                       host and container test logs, re-derives every public_metadata.json section
                       from those files and fails on any disagreement. --write-public-metadata
                       writes the re-derived sections back.

Apart from "replay", both modes produce the same summary. This script never runs a model, never
opens a network connection and never starts a subprocess. It reads JSON, text logs and
safetensors headers (the 8-byte length plus the JSON header); in private validation it also hashes
the checkpoint files. Top-level imports are standard library only (plus src/quality_pilot.py, whose
top-level imports are standard library only); matplotlib is imported only when figures are drawn.

    .venv/bin/python src/report_results.py --metadata-only               # clean clone
    .venv/bin/python src/report_results.py --metadata-only --no-figures  # summary.json only
    .venv/bin/python src/report_results.py                               # private validation
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import struct
import sys
from typing import Dict, List, Optional, Sequence, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import quality_pilot as qp  # noqa: E402  (standard library only at import time)

PROJECT_ROOT = os.path.dirname(SRC_DIR)
DEFAULT_PILOT_DIR = os.path.join(PROJECT_ROOT, "results", "quality-pilot")
DEFAULT_EVIDENCE_DIR = os.path.join(PROJECT_ROOT, "evidence")
DEFAULT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
DEFAULT_SUMMARY = os.path.join(PROJECT_ROOT, "results", "summary.json")
DEFAULT_ORIGINAL_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "qwen2.5-0.5b", "model.safetensors")
PUBLIC_METADATA_NAME = "public_metadata.json"

SCHEMA = "littlebit-report-summary/2"
PUBLIC_METADATA_SCHEMA = "littlebit-public-metadata/1"
STAGES = ("manifest", "baseline", "feasibility", "train", "evaluate")
GIB = 1024 ** 3
MIB = 1024 ** 2
REL_TOL = 1e-9

MODE_METADATA = "metadata_replay"
MODE_PRIVATE = "private_validation"

# Sections of public_metadata.json, as dotted keys. Only the original weights header is optional.
METADATA_SECTIONS = ("model_config", "original_weights_header", "frozen_base_header",
                     "trained_compressed_layers", "official_state_dict_loader",
                     "test_logs.host", "test_logs.container", "test_logs.container_image")
OPTIONAL_SECTIONS = ("original_weights_header",)
EVIDENCE_FILES = {"official_state_dict_loader": "official-loader-integration.json",
                  "test_logs.host": "unit-tests-quality.log",
                  "test_logs.container": "container-tests-quality.log",
                  "test_logs.container_image": "container-image.txt"}

METADATA_REPLAY_STATEMENT = (
    "metadata replay: rebuilt from tracked public JSON and results/quality-pilot/public_metadata.json "
    "only. No checkpoint, model weight, tokenizer, checkpoint config or host log was read, so this run "
    "does not validate the private checkpoints. Checkpoint byte counts, safetensors header facts and "
    "test outcomes are copied from public_metadata.json, which records the owner's private validation "
    "run; every other number is recomputed and cross-checked across the public files.")
PRIVATE_VALIDATION_STATEMENT = (
    "private validation on the owner's host: checkpoint files hashed and compared with train.json, "
    "safetensors headers parsed, checkpoint config and available logs read, and every re-derived "
    "public_metadata.json section compared with the tracked file. Sections whose private source was "
    "absent are listed as not rechecked.")
IDENTITY_SCOPE = ("signs compared over every packed factor (all modules); logits and next-token loss "
                  "compared on one 128-token validation window (validation window 0) per checkpoint, "
                  "CPU float32, same process")
FULL_MODEL_LOADER = ("quant_util.load_quantized_model not run; the full-model reload is an emulation of its "
                     "packed path")

# Evaluate-stage condition keys, in display order. The defect variant is always drawn separately.
CONDITIONS = (
    ("original", "Original Qwen2.5-0.5B (FP32)"),
    ("initialized_no_qat", "LittleBit initialized, no QAT"),
    ("qat_trained", "LittleBit after 16 QAT steps"),
)
DEFECT = ("qat_trained_upstream_decoder", "Same QAT weights, defective upstream decoder")

PILOT_LABEL = "tiny pilot: 8 test windows x 128 tokens, 1016 scored tokens"
NOT_A_RESULT = "decoder defect variant, not a quality result"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "I32": 4,
               "I64": 8, "BOOL": 1}


class ReportError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------

def rel(path: str) -> str:
    """Display form of a path: project relative, or <external>/<basename>; never absolute."""
    return qp.public_path(path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: str) -> Dict[str, object]:
    return {"path": rel(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise ReportError("missing input: %s" % rel(path))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def close(a: float, b: float, tol: float = REL_TOL) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ReportError(msg)


def describe(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    median = vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    return {"n": n, "mean": sum(vals) / n, "median": median, "min": vals[0], "max": vals[-1]}


def safetensors_header(path: str) -> Dict[str, object]:
    """Parse only the header of a safetensors file: per-tensor dtype, shape and byte span."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        require(0 < n < size, "implausible safetensors header length in %s" % rel(path))
        header = json.loads(f.read(n).decode("utf-8"))
    tensors = {}
    data_end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        count = 1
        for d in info["shape"]:
            count *= d
        width = DTYPE_BYTES.get(info["dtype"])
        if width is not None:
            require(end - start == count * width, "byte span mismatch for %s in %s" % (name, rel(path)))
        tensors[name] = {"dtype": info["dtype"], "shape": info["shape"], "bytes": end - start}
        data_end = max(data_end, end)
    require(8 + n + data_end == size, "safetensors data section does not end at EOF: %s" % rel(path))
    return {"file_bytes": size, "header_bytes": 8 + n, "tensor_bytes": data_end, "tensors": tensors}


def get_section(meta: Optional[Dict], dotted: str):
    node = meta
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_section(meta: Dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = meta
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def assert_public_safe(obj, label: str) -> None:
    hits = qp.find_host_paths(obj)
    require(not hits, "host path in %s at %s; sanitize before release" % (label, ", ".join(hits[:5])))


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------

def load_public(pilot_dir: str, metadata_path: str, need_metadata: bool) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Tracked public inputs only: stage JSON, train_steps.jsonl, public_metadata.json."""
    records: Dict[str, object] = {}
    files: Dict[str, object] = {}
    for stage in STAGES:
        path = os.path.join(pilot_dir, stage + ".json")
        records[stage] = load_json(path)
        files[stage] = file_record(path)
        assert_public_safe(records[stage], rel(path))
    steps_path = os.path.join(pilot_dir, "train_steps.jsonl")
    if os.path.exists(steps_path):
        files["train_steps_jsonl"] = file_record(steps_path)
        with open(steps_path, "r", encoding="utf-8") as f:
            records["train_steps_jsonl"] = [json.loads(line) for line in f if line.strip()]
        assert_public_safe(records["train_steps_jsonl"], rel(steps_path))
    if os.path.exists(metadata_path):
        meta = load_json(metadata_path)
        require(meta.get("schema") == PUBLIC_METADATA_SCHEMA, "unexpected public metadata schema in %s"
                % rel(metadata_path))
        assert_public_safe(meta, rel(metadata_path))
        records["public_metadata"] = meta
        files["public_metadata"] = file_record(metadata_path)
    elif need_metadata:
        raise ReportError("missing input: %s (required for --metadata-only)" % rel(metadata_path))
    return records, files


def check_statuses(records: Dict[str, Dict]) -> None:
    for stage in ("baseline", "feasibility", "train", "evaluate"):
        require(records[stage].get("status") == "completed", "%s stage is not completed" % stage)
    manifest_sha = {records[s].get("manifest_sha256") for s in ("baseline", "feasibility", "train", "evaluate")}
    require(len(manifest_sha) == 1, "stages disagree on manifest_sha256")
    tr = records["train"]
    require(tr["completed_steps"] == tr["args"]["steps"] == 16, "train did not complete 16 steps")
    require(tr["claims"]["model_trained"] is True, "train record does not claim a trained model")
    require(records["evaluate"]["training_record"]["completed_steps"] == 16, "evaluate saw a different step count")


# ----------------------------------------------------------------------------------------------
# Private validation: re-derive public metadata from private files
# ----------------------------------------------------------------------------------------------

def parse_test_log(text: str) -> Dict[str, object]:
    m = re.search(r"^Ran (\d+) tests? in ([0-9.]+)s", text, re.MULTILINE)
    ok = re.search(r"^OK\b", text, re.MULTILINE) is not None
    failed = re.search(r"^FAILED \((.*)\)", text, re.MULTILINE)
    return {"ran": int(m.group(1)) if m else None, "seconds": float(m.group(2)) if m else None,
            "ok": ok and failed is None, "failed_summary": failed.group(1) if failed else None}


def _checked_checkpoint(path: str, record: Dict, label: str, checks: List[str]) -> str:
    require(os.path.exists(path), "private validation needs %s (%s); use --metadata-only without it"
            % (label, rel(path)))
    require(os.path.getsize(path) == record["bytes"], "%s size differs from train.json record" % label)
    digest = sha256_file(path)
    require(digest == record["sha256"], "%s sha256 differs from train.json record" % label)
    checks.append("%s: sha256 and size equal the train.json record" % record["path"])
    return digest


def collect_private(pilot_dir: str, evidence_dir: str, original_weights: str,
                    records: Dict[str, Dict]) -> Tuple[Dict[str, object], List[str]]:
    """Re-derive public metadata sections from private files. Missing optional sources give None."""
    tr = records["train"]
    ev = records["evaluate"]
    ck = tr["checkpoints"]
    ckdir = os.path.join(pilot_dir, "checkpoints")
    sections: Dict[str, object] = {}
    checks: List[str] = []

    cfg_rec = ck["config_and_tokenizer"]["config.json"]
    cfg_path = os.path.join(ckdir, "config_and_tokenizer", "config.json")
    cfg_sha = _checked_checkpoint(cfg_path, cfg_rec, "checkpoint config.json", checks)
    cfg = load_json(cfg_path)
    sections["model_config"] = {"source": cfg_rec["path"], "source_sha256": cfg_sha,
                                "vocab_size": cfg["vocab_size"], "hidden_size": cfg["hidden_size"],
                                "torch_dtype": cfg["torch_dtype"],
                                "tie_word_embeddings": cfg.get("tie_word_embeddings")}

    frozen_path = os.path.join(ckdir, "frozen_base.safetensors")
    frozen_sha = _checked_checkpoint(frozen_path, ck["frozen_base"], "frozen base checkpoint", checks)
    frozen_hdr = safetensors_header(frozen_path)
    emb = frozen_hdr["tensors"]["model.embed_tokens.weight"]
    sections["frozen_base_header"] = {"source": ck["frozen_base"]["path"], "source_sha256": frozen_sha,
                                      "file_bytes": frozen_hdr["file_bytes"],
                                      "embed_tokens": {"dtype": emb["dtype"], "shape": list(emb["shape"]),
                                                       "bytes": emb["bytes"]}}
    checks.append("%s: safetensors header parsed, data section ends at EOF" % ck["frozen_base"]["path"])

    for key, sub in (("init", "init"), ("trained", "trained")):
        if key in ck:
            path = os.path.join(ckdir, sub, "compressed_layers.safetensors")
            digest = _checked_checkpoint(path, ck[key], "%s compressed layers" % key, checks)
            hdr = safetensors_header(path)
            checks.append("%s: safetensors header parsed, data section ends at EOF" % ck[key]["path"])
            if key == "trained":
                sections["trained_compressed_layers"] = {"source": ck[key]["path"], "source_sha256": digest,
                                                         "file_bytes": hdr["file_bytes"]}

    orig = ev["model"]["files"]["model.safetensors"]
    if os.path.exists(original_weights) and os.path.getsize(original_weights) == orig["bytes"]:
        hdr = safetensors_header(original_weights)
        t = hdr["tensors"]["model.embed_tokens.weight"]
        sections["original_weights_header"] = {"source": orig["path"], "file_bytes": hdr["file_bytes"],
                                               "embed_tokens": {"dtype": t["dtype"], "shape": list(t["shape"]),
                                                                "bytes": t["bytes"]}}
        checks.append("%s: safetensors header parsed (tensor data not read)" % orig["path"])
    else:
        sections["original_weights_header"] = None

    path = os.path.join(evidence_dir, EVIDENCE_FILES["official_state_dict_loader"])
    if os.path.exists(path):
        ol = load_json(path)
        sections["official_state_dict_loader"] = {
            "source": "evidence/" + EVIDENCE_FILES["official_state_dict_loader"], "source_sha256": sha256_file(path),
            "scope": ol["scope"], "packed_detected": ol["packed_detected"],
            "original_sign_agreement": ol["original_sign_agreement"],
            "corrected_sign_agreement": ol["corrected_sign_agreement"],
            "corrected_output_bitwise_equal": ol["corrected_output_bitwise_equal"]}
    else:
        sections["official_state_dict_loader"] = None

    for key in ("test_logs.host", "test_logs.container"):
        path = os.path.join(evidence_dir, EVIDENCE_FILES[key])
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                parsed = parse_test_log(f.read())
            parsed.update({"source": "evidence/" + EVIDENCE_FILES[key], "source_sha256": sha256_file(path)})
            sections[key] = parsed
        else:
            sections[key] = None
    path = os.path.join(evidence_dir, EVIDENCE_FILES["test_logs.container_image"])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            image = f.read().split()
        sections["test_logs.container_image"] = {
            "source": "evidence/" + EVIDENCE_FILES["test_logs.container_image"], "source_sha256": sha256_file(path),
            "id": image[0] if image else None, "arch": image[1] if len(image) > 1 else None}
    else:
        sections["test_logs.container_image"] = None
    return sections, checks


def reconcile(private: Dict[str, object], public: Optional[Dict], allow_fill: bool) -> Dict[str, object]:
    """Compare re-derived sections with the tracked public metadata. Any disagreement is fatal."""
    verified, not_rechecked, filled = [], [], []
    effective = copy.deepcopy(public) if public else {"schema": PUBLIC_METADATA_SCHEMA}
    for key in METADATA_SECTIONS:
        priv = private.get(key)
        pub = get_section(public, key)
        if priv is None:
            not_rechecked.append(key)
            continue
        if pub is None:
            require(allow_fill, "public metadata lacks section %s; rerun with --write-public-metadata" % key)
            set_section(effective, key, priv)
            filled.append(key)
            continue
        require(priv == pub, "public metadata section %s disagrees with the private files" % key)
        verified.append(key)
    return {"effective": effective, "verified": verified, "not_rechecked": not_rechecked, "filled": filled}


def require_sections(meta: Dict, label: str) -> None:
    for key in METADATA_SECTIONS:
        if key not in OPTIONAL_SECTIONS:
            require(get_section(meta, key) is not None, "%s lacks required section %s" % (label, key))


# ----------------------------------------------------------------------------------------------
# Held-out quality
# ----------------------------------------------------------------------------------------------

def heldout(records: Dict[str, Dict]) -> Dict[str, object]:
    ev = records["evaluate"]
    manifest_windows = records["manifest"]["splits"]["test"]["windows"]
    hashes = [w["sha256"] for w in manifest_windows]
    out: Dict[str, object] = {"window_sha256": hashes, "conditions": {}}
    keys = [k for k, _ in CONDITIONS] + [DEFECT[0]]
    labels = dict(CONDITIONS + (DEFECT,))
    original_means: List[float] = []
    for key in keys:
        block = ev[key]
        test = block["test"]
        wins = test["windows"]
        require([w["sha256"] for w in wins] == hashes, "%s windows do not match manifest hashes" % key)
        tokens = sum(w["scored_tokens"] for w in wins)
        total = sum(w["nll_sum"] for w in wins)
        require(tokens == test["scored_tokens"], "%s scored token count mismatch" % key)
        require(close(total, test["total_nll"], 1e-9), "%s total NLL mismatch" % key)
        mean = total / tokens
        ppl = math.exp(mean)
        require(close(mean, test["mean_nll"], 1e-12), "%s mean NLL mismatch" % key)
        require(close(ppl, test["ppl"], 1e-9), "%s perplexity mismatch" % key)
        means = [w["nll_sum"] / w["scored_tokens"] for w in wins]
        entry: Dict[str, object] = {
            "label": labels[key],
            "tier": block.get("tier", ev.get("tier")),
            "scored_tokens": tokens,
            "total_nll": total,
            "mean_nll": mean,
            "ppl": ppl,
            "per_window_mean_nll": means,
        }
        if key == "original":
            original_means = means
            baseline_test = records["baseline"]["test"]
            require(close(baseline_test["ppl"], ppl, 1e-12), "baseline and evaluate disagree on original ppl")
            entry["matches_baseline_stage"] = bool(block.get("matches_baseline_stage"))
        else:
            diffs = [m - o for m, o in zip(means, original_means)]
            recorded = [d["mean_nll_diff"] for d in block["paired_vs_original"]]
            require([d["sha256"] for d in block["paired_vs_original"]] == hashes,
                    "%s paired windows out of order" % key)
            max_dev = max(abs(a - b) for a, b in zip(diffs, recorded))
            require(max_dev < 1e-9, "%s recomputed paired differences disagree with record" % key)
            entry["paired_diff_vs_original"] = diffs
            entry["paired_diff_summary"] = describe(diffs)
            entry["windows_worse_than_original"] = sum(1 for d in diffs if d > 0)
            entry["ppl_ratio_vs_original"] = ppl / out["conditions"]["original"]["ppl"]
            entry["max_abs_dev_from_recorded_paired_diff"] = max_dev
            entry["reload"] = block.get("reload", {})
        if key == DEFECT[0]:
            entry["note"] = block.get("note", NOT_A_RESULT)
        out["conditions"][key] = entry
    qat = out["conditions"]["qat_trained"]["per_window_mean_nll"]
    init = out["conditions"]["initialized_no_qat"]["per_window_mean_nll"]
    defect = out["conditions"][DEFECT[0]]["per_window_mean_nll"]
    qat_minus_init = [q - i for q, i in zip(qat, init)]
    defect_minus_qat = [d - q for d, q in zip(defect, qat)]
    out["qat_minus_initialized"] = {"per_window": qat_minus_init, "summary": describe(qat_minus_init),
                                    "windows_improved": sum(1 for d in qat_minus_init if d < 0)}
    out["defect_minus_corrected_same_weights"] = {"per_window": defect_minus_qat,
                                                  "summary": describe(defect_minus_qat),
                                                  "windows_worse": sum(1 for d in defect_minus_qat if d > 0)}
    out["statistics"] = ("descriptive only: 8 contiguous windows from one test-split prefix are not "
                         "independent samples; no confidence intervals or hypothesis tests are reported")
    return out


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------

def training(records: Dict[str, Dict]) -> Dict[str, object]:
    tr = records["train"]
    log = tr["step_log"]
    require([s["step"] for s in log] == list(range(16)), "step log is not steps 0..15")
    jsonl = records.get("train_steps_jsonl")
    jsonl_matches = None
    if jsonl is not None:
        jsonl_matches = [(s["step"], s["ce"], s["kl"]) for s in jsonl] == [(s["step"], s["ce"], s["kl"]) for s in log]
    train_windows = records["manifest"]["splits"]["train"]["windows"]
    by_index = {w["index"]: w for w in train_windows}
    used = [i for s in log for i in s["window_indices"]]
    require(sorted(used) == sorted(by_index), "train windows not each used exactly once")
    for s in log:
        for i, h in zip(s["window_indices"], s["window_sha256"]):
            require(by_index[i]["sha256"] == h, "train window hash mismatch at step %d" % s["step"])
    input_tokens = sum(by_index[i]["n_tokens"] for i in used)
    supervised = sum(by_index[i]["scored_tokens"] for i in used)
    dts = [s["dt_s"] for s in log]
    diff = tr["checkpoint_diff_init_to_trained"]
    return {
        "completed_steps": tr["completed_steps"],
        "planned_steps": tr["args"]["steps"],
        "batch_size": tr["args"]["batch_size"],
        "seq_len": tr["objective"]["seq_len"],
        "input_tokens": input_tokens,
        "supervised_next_token_targets": supervised,
        "objective": tr["objective"],
        "steps": [{"step": s["step"], "ce": s["ce"], "kl": s["kl"], "loss": s["loss"], "lr": s["lr"],
                   "grad_norm_pre_clip": s["grad_norm_pre_clip"], "dt_s": s["dt_s"],
                   "rss_bytes": s["rss_bytes"]} for s in log],
        "step_seconds": describe(dts),
        "train_stage_guard_elapsed_s": tr["guard"]["elapsed_s"],
        "conversion_seconds": tr["conversion_seconds"],
        "train_steps_jsonl_matches_step_log": jsonl_matches,
        "sign_flips_init_to_trained": diff["sign_flips"],
        "sign_flip_fraction": diff["sign_flip_fraction"],
        "scale_relative_l2_change": diff["scale_relative_l2_change"],
        "teacher_preserved": tr["teacher_preserved"],
        "teacher_fingerprint_before_equals_after": tr["teacher_fingerprint_before"] == tr["teacher_fingerprint_after"],
        "dev_validation_ppl": {"initialized_in_memory": tr["dev_validation_init_in_memory"]["ppl"],
                               "trained_in_memory": tr["dev_validation_trained_in_memory"]["ppl"],
                               "original": records["baseline"]["validation"]["ppl"],
                               "note": "validation windows are development only; reported, not used for decisions"},
        "further_training_planned": False,
    }


def identity(records: Dict[str, Dict], meta: Dict) -> Dict[str, object]:
    tr = records["train"]
    out = {}
    for key in ("identity_init", "identity_trained"):
        b = tr[key]
        out[key] = {"device": b["device"], "dtype": "float32", "signs": b["signs"],
                    "sign_agreement": b["sign_agreement"], "logits_bitwise_equal": b["logits_bitwise_equal"],
                    "loss_bitwise_equal": b["loss_bitwise_equal"], "logits_max_abs_diff": b["logits_max_abs_diff"],
                    "decoder": b["reload"]["decoder"], "official_loader_ran": b["reload"]["official_loader_ran"],
                    "missing_keys": b["reload"]["missing_keys"], "compared_inputs": IDENTITY_SCOPE}
    ol = meta["official_state_dict_loader"]
    out["official_state_dict_loader"] = {
        "scope": ol["scope"],
        "packed_detected": ol["packed_detected"],
        "original_decoder_sign_agreement": ol["original_sign_agreement"],
        "corrected_decoder_sign_agreement": ol["corrected_sign_agreement"],
        "corrected_output_bitwise_equal": ol["corrected_output_bitwise_equal"],
        "full_model_loader": FULL_MODEL_LOADER,
    }
    return out


# ----------------------------------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------------------------------

def storage(records: Dict[str, Dict], meta: Dict) -> Dict[str, object]:
    tr = records["train"]
    ev = records["evaluate"]
    ck = tr["checkpoints"]
    cfg = meta["model_config"]
    vocab, hidden = cfg["vocab_size"], cfg["hidden_size"]
    embed_params = vocab * hidden

    original_file = ev["model"]["files"]["model.safetensors"]
    original_bytes = original_file["bytes"]
    ow = meta.get("original_weights_header")
    if ow is not None and ow["file_bytes"] == original_bytes:
        t = ow["embed_tokens"]
        require(t["dtype"] == "BF16", "original embedding is not BF16")
        require(list(t["shape"]) == [vocab, hidden] and t["bytes"] == embed_params * 2,
                "original embedding header disagrees with the model config")
        orig_embed_bytes = t["bytes"]
        orig_embed_source = "safetensors header of %s" % ow["source"]
    else:
        require(cfg["torch_dtype"] == "bfloat16", "unexpected original dtype")
        orig_embed_bytes = embed_params * 2
        orig_embed_source = "derived: vocab_size x hidden_size x 2 bytes (BF16); original weights header not recorded"

    frozen = meta["frozen_base_header"]
    trained = meta["trained_compressed_layers"]
    require(frozen["file_bytes"] == ck["frozen_base"]["bytes"], "frozen base size differs from train record")
    require(trained["file_bytes"] == ck["trained"]["bytes"], "trained checkpoint size differs from record")
    for name, sec in (("frozen_base", frozen), ("trained", trained)):
        if "source_sha256" in sec:
            require(sec["source_sha256"] == ck[name]["sha256"],
                    "public metadata %s sha256 differs from the train.json record" % name)
    frozen_embed = frozen["embed_tokens"]
    require(frozen_embed["dtype"] == "BF16", "frozen embedding not stored as BF16")
    require(list(frozen_embed["shape"]) == [vocab, hidden] and frozen_embed["bytes"] == embed_params * 2,
            "frozen embedding header disagrees with the model config")

    student_disk = ck["frozen_base"]["bytes"] + ck["trained"]["bytes"]
    tokenizer_files = ck["config_and_tokenizer"]
    tokenizer_bytes = sum(v["bytes"] for v in tokenizer_files.values())

    res_orig = ev["original"]["resident_storage"]["total_unique_bytes"]
    res_student = ev["qat_trained"]["resident_storage"]
    embed_fp32 = embed_params * 4
    require(res_student["other_bytes"] >= embed_fp32, "student non-factor bytes smaller than FP32 embedding")

    acc = ck["trained"]
    return {
        "rule": "disk bytes and resident bytes are reported separately and never summed or compared across",
        "excluded": {
            "tokenizer_and_config_files_bytes": tokenizer_bytes,
            "tokenizer_and_config_files": sorted(tokenizer_files),
            "note": "tokenizer, vocab, merges and config files are excluded from every weight byte count",
            "kv_cache": "reported separately below; not weights",
        },
        "disk": {
            "original": {"file": "models/qwen2.5-0.5b/model.safetensors", "dtype": "BF16",
                         "bytes": original_bytes, "embedding_bytes": orig_embed_bytes,
                         "embedding_source": orig_embed_source,
                         "non_embedding_bytes": original_bytes - orig_embed_bytes,
                         "note": "includes the safetensors header; lm_head is tied and not stored"},
            "student": {
                "bytes": student_disk,
                "frozen_base_file_bytes": ck["frozen_base"]["bytes"],
                "frozen_base_embedding_bytes": frozen_embed["bytes"],
                "frozen_base_norms_biases_and_header_bytes": ck["frozen_base"]["bytes"] - frozen_embed["bytes"],
                "frozen_base_dtype": "BF16 (lossless per tensor, as recorded)",
                "compressed_layers_file_bytes": ck["trained"]["bytes"],
                "packed_sign_tensor_bytes": acc["packed_sign_tensor_bytes"],
                "fp32_scale_tensor_bytes": acc["fp32_scale_tensor_bytes"],
                "shape_and_buffer_tensor_bytes": acc["shape_and_buffer_tensor_bytes"],
                "compressed_layers_header_bytes": ck["trained"]["bytes"] - acc["packed_sign_tensor_bytes"]
                - acc["fp32_scale_tensor_bytes"] - acc["shape_and_buffer_tensor_bytes"],
            },
            "student_over_original": student_disk / original_bytes,
        },
        "bpw_converted_block_linears": {
            "converted_weight_count": acc["converted_weight_count"],
            "advertised_upstream_formula": acc["advertised_bpw_upstream_formula"],
            "packed_signs_only": acc["achieved_bpw_packed_signs_only"],
            "actual_checkpoint_file_fp32_scales": acc["achieved_bpw_from_file_bytes"],
            "note": acc["bpw_note"],
        },
        "resident_fp32_unique": {
            "original_bytes": res_orig,
            "student_bytes": res_student["total_unique_bytes"],
            "student_littlebit_factor_scale_buffer_bytes": res_student["littlebit_factor_scale_buffer_bytes"],
            "student_other_bytes": res_student["other_bytes"],
            "embedding_fp32_bytes_derived": embed_fp32,
            "embedding_source": "derived: vocab_size %d x hidden_size %d x 4 bytes; shared by tied lm_head"
                                % (vocab, hidden),
            "student_other_minus_embedding_bytes": res_student["other_bytes"] - embed_fp32,
            "original_non_embedding_bytes": res_orig - embed_fp32,
            "student_over_original": res_student["total_unique_bytes"] / res_orig,
            "note": ev["qat_trained"]["resident_storage"]["note"],
        },
        "kv_cache": {
            "tokens": ev["original"]["kv_cache"]["tokens"],
            "original_bytes": ev["original"]["kv_cache"]["measured_bytes"],
            "student_bytes": ev["qat_trained"]["kv_cache"]["measured_bytes"],
            "unchanged": ev["original"]["kv_cache"]["measured_bytes"] == ev["qat_trained"]["kv_cache"]["measured_bytes"],
            "note": "FP32 KV cache, batch 1; LittleBit does not compress it",
        },
    }


# ----------------------------------------------------------------------------------------------
# Resources, tests, predictions
# ----------------------------------------------------------------------------------------------

def resources(records: Dict[str, Dict]) -> Dict[str, object]:
    out = {}
    for stage in ("baseline", "feasibility", "train", "evaluate"):
        r = records[stage]
        g = r["guard"]
        peak = r["peak_rss_os"]["bytes"]
        out[stage] = {
            "peak_rss_os_bytes": peak,
            "peak_rss_os_gib": peak / GIB,
            "guard_threshold_bytes": g["max_rss_bytes"],
            "guard_peak_rss_seen_at_checks_bytes": g["peak_rss_seen_at_checks"],
            "guard_checks": g["checks"],
            "guard_elapsed_s": g["elapsed_s"],
            "os_peak_exceeded_guard_threshold": peak > g["max_rss_bytes"],
        }
    out["note"] = ("guards sample RSS at checkpoints (per layer, per step, per window); they are not hard "
                   "memory caps, so the OS peak can exceed the threshold between checks without stopping a run")
    return out


def tests_block(meta: Dict) -> Dict[str, object]:
    logs = meta["test_logs"]
    keep = ("ran", "seconds", "ok", "failed_summary", "source")
    return {"host": {k: logs["host"].get(k) for k in keep},
            "container": {k: logs["container"].get(k) for k in keep},
            "container_image": {"id": logs["container_image"]["id"], "arch": logs["container_image"]["arch"]},
            "scope": "test suite as it stood when the pilot finished (95 tests); later suites are reported in README",
            "devcontainer_json": "not present; write refused by the tool permission layer as a sensitive "
                                 "file, including one retry; not retried or bypassed"}


def predictions(held: Dict, train_summary: Dict, store: Dict, res: Dict, ident: Dict) -> List[Dict[str, object]]:
    c = held["conditions"]
    orig = c["original"]["ppl"]
    peak = res["train"]["peak_rss_os_gib"]
    return [
        {"id": "P1", "prediction": "original test perplexity between 8 and 60",
         "observed": orig, "held": 8 <= orig <= 60},
        {"id": "P2", "prediction": "initialized student perplexity at least 10x original",
         "observed_ratio": c["initialized_no_qat"]["ppl_ratio_vs_original"],
         "held": c["initialized_no_qat"]["ppl_ratio_vs_original"] >= 10},
        {"id": "P3", "prediction": "after 16 steps perplexity still at least 5x original",
         "observed_ratio": c["qat_trained"]["ppl_ratio_vs_original"],
         "held": c["qat_trained"]["ppl_ratio_vs_original"] >= 5},
        {"id": "P4", "prediction": "corrected reload reproduces logits bitwise for init and trained; the "
                                   "upstream decoder variant does not",
         "observed": "corrected: bitwise for both, on one 128-token validation window each; upstream variant: "
                     "no identity check recorded, held-out NLL differs from the corrected reload of the same "
                     "weights in 8 of 8 windows",
         "held": ident["identity_init"]["logits_bitwise_equal"] and ident["identity_trained"]["logits_bitwise_equal"],
         "scope_note": "logits compared on one validation window per checkpoint; second half of P4 not tested as "
                       "a bitwise identity check"},
        {"id": "P5", "prediction": "train peak RSS between 4.5 and 6.0 GiB; 16 steps fit in 900 s",
         "observed_peak_rss_gib": peak, "observed_train_stage_s": train_summary["train_stage_guard_elapsed_s"],
         "held": 4.5 <= peak <= 6.0 and train_summary["train_stage_guard_elapsed_s"] <= 900,
         "note": "memory part failed: OS peak RSS exceeded 6.0 GiB; time part held"},
        {"id": "P6", "prediction": "BPW from file bytes exceeds the upstream formula value",
         "observed": [store["bpw_converted_block_linears"]["actual_checkpoint_file_fp32_scales"],
                      store["bpw_converted_block_linears"]["advertised_upstream_formula"]],
         "held": store["bpw_converted_block_linears"]["actual_checkpoint_file_fp32_scales"]
         > store["bpw_converted_block_linears"]["advertised_upstream_formula"]},
    ]


def stage_command(stage: str, args: Dict) -> str:
    """Rebuild a command line from the args block a stage recorded."""
    parts = [".venv/bin/python", "src/quality_pilot.py", stage, "--out", qp.public_path(args["out"])]
    skip = {"stage", "out", "model_dir", "device", "seed"}
    for key in sorted(args):
        if key in skip:
            continue
        val = args[key]
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                parts.append(flag)
            elif key == "residual":
                parts.append("--no-residual")
            continue
        parts.extend([flag, str(val)])
    return " ".join(parts)


def reproduce(records: Dict[str, Dict]) -> Dict[str, object]:
    cmds = [".venv/bin/python src/quality_pilot.py prepare --out results/quality-pilot --data-dir "
            "<local HF cache datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3>"]
    for stage in ("baseline", "feasibility", "train", "evaluate"):
        cmds.append(stage_command(stage, records[stage]["args"]))
    return {
        "note": ("rebuilt from the args recorded by each stage; flags whose values equal defaults are still "
                 "spelled out. Paths are relative to the project root. Requires local model weights and "
                 "dataset parquet files, which are not redistributed."),
        "pilot": cmds,
        "tests": [".venv/bin/python -m unittest discover -s tests -v"],
        "report": [".venv/bin/python src/report_results.py --metadata-only  # metadata replay, clean clone",
                   ".venv/bin/python src/report_results.py  # private validation, needs local checkpoints"],
    }


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def _save(fig, figures_dir: str, stem: str) -> List[Dict[str, object]]:
    out = []
    for ext in ("png", "svg"):
        path = os.path.join(figures_dir, "%s.%s" % (stem, ext))
        if ext == "svg":
            fig.savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
        else:
            fig.savefig(path, format="png", dpi=150, metadata={"Software": None}, bbox_inches="tight")
        out.append(file_record(path))
    return out


def draw_figures(summary: Dict, figures_dir: str) -> List[Dict[str, object]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.hashsalt": "littlebit-report", "font.size": 9,
                                "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(figures_dir, exist_ok=True)
    produced: List[Dict[str, object]] = []
    held = summary["heldout_test"]
    c = held["conditions"]
    idx = list(range(len(held["window_sha256"])))
    colors = {"original": "#1b6ca8", "initialized_no_qat": "#c0392b", "qat_trained": "#e67e22",
              DEFECT[0]: "#7f7f7f"}
    markers = {"original": "o", "initialized_no_qat": "s", "qat_trained": "^", DEFECT[0]: "x"}

    # 1. Paired per-window NLL; the defect variant gets its own panel.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    for w in idx:
        ax1.plot([w] * 3, [c[k]["per_window_mean_nll"][w] for k, _ in CONDITIONS],
                 color="#cccccc", linewidth=0.8, zorder=1)
    for key, label in CONDITIONS:
        ax1.scatter(idx, c[key]["per_window_mean_nll"], label=label, color=colors[key],
                    marker=markers[key], zorder=2)
    ax1.set_xticks(idx)
    ax1.set_xlabel("test window index (same 128-token window across conditions)")
    ax1.set_ylabel("mean next-token NLL (nats, 127 tokens)")
    ax1.set_title("Held-out NLL per window, paired by window hash")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="center right", fontsize=8, frameon=False)
    ax2.plot(idx, c["qat_trained"]["per_window_mean_nll"], marker=markers["qat_trained"],
             color=colors["qat_trained"], label="corrected decoder")
    ax2.plot(idx, c[DEFECT[0]]["per_window_mean_nll"], marker=markers[DEFECT[0]], color=colors[DEFECT[0]],
             linestyle="--", label="upstream decoder (defective)")
    ax2.set_xticks(idx)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("test window index")
    ax2.set_title("Same QAT weights, two decoders\n(%s)" % NOT_A_RESULT, fontsize=9)
    ax2.legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Qwen2.5-0.5B, WikiText-2 test, %s" % PILOT_LABEL, fontsize=9, y=1.02)
    produced += _save(fig, figures_dir, "paired_window_nll")
    plt.close(fig)

    # 2. Log-scale perplexity.
    fig, ax = plt.subplots(figsize=(7, 3.8))
    keys = [k for k, _ in CONDITIONS] + [DEFECT[0]]
    names = ["Original", "Initialized\n(no QAT)", "16-step QAT", "16-step QAT,\ndefective decoder"]
    vals = [c[k]["ppl"] for k in keys]
    bars = ax.bar(names, vals, color=[colors[k] for k in keys], edgecolor="black", linewidth=0.5)
    bars[-1].set_hatch("//")
    bars[-1].set_facecolor("white")
    ax.set_yscale("log")
    ax.set_ylabel("token-weighted perplexity (log scale)")
    ax.set_title("Held-out perplexity, %s" % PILOT_LABEL, fontsize=9)
    for b, v in zip(bars, vals):
        ax.annotate("%.4f" % v, (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.text(3, vals[-1] / 8, NOT_A_RESULT, ha="center", va="top", fontsize=7, style="italic")
    ax.set_ylim(1, max(vals) * 20)
    produced += _save(fig, figures_dir, "heldout_ppl_log")
    plt.close(fig)

    # 3. Disk versus resident bytes, separate panels, embeddings separate.
    st = summary["storage"]
    disk, resident = st["disk"], st["resident_fp32_unique"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    mb = 1e6
    o = disk["original"]
    s = disk["student"]
    ax1.bar(["Original\n(BF16 file)"], [o["embedding_bytes"] / mb], color="#9ecae1", edgecolor="black",
            linewidth=0.5, label="embedding table (frozen, tied head)")
    ax1.bar(["Original\n(BF16 file)"], [o["non_embedding_bytes"] / mb], bottom=[o["embedding_bytes"] / mb],
            color="#1b6ca8", edgecolor="black", linewidth=0.5, label="other original tensors and header")
    base = s["frozen_base_embedding_bytes"] / mb
    ax1.bar(["Student\n(two files)"], [base], color="#9ecae1", edgecolor="black", linewidth=0.5)
    nb = s["frozen_base_norms_biases_and_header_bytes"] / mb
    ax1.bar(["Student\n(two files)"], [nb], bottom=[base], color="#d9d9d9", edgecolor="black", linewidth=0.5,
            label="frozen norms, biases, header (BF16)")
    ax1.bar(["Student\n(two files)"], [s["compressed_layers_file_bytes"] / mb], bottom=[base + nb],
            color="#e67e22", edgecolor="black", linewidth=0.5,
            label="compressed layers: packed signs + FP32 scales")
    ax1.set_ylabel("megabytes (1e6 bytes)")
    ax1.set_title("On disk (stored dtypes)", fontsize=9)
    ax1.legend(fontsize=7, frameon=False, loc="upper right")
    for pos, total in enumerate((o["bytes"], s["bytes"])):
        ax1.annotate("{:,} B".format(total), (pos, total / mb), ha="center", va="bottom", fontsize=7,
                     xytext=(0, 2), textcoords="offset points")
    emb = resident["embedding_fp32_bytes_derived"] / mb
    ax2.bar(["Original"], [emb], color="#9ecae1", edgecolor="black", linewidth=0.5,
            label="embedding table FP32 (frozen, tied head)")
    ax2.bar(["Original"], [resident["original_non_embedding_bytes"] / mb], bottom=[emb], color="#1b6ca8",
            edgecolor="black", linewidth=0.5, label="other original tensors")
    ax2.bar(["Student"], [emb], color="#9ecae1", edgecolor="black", linewidth=0.5)
    other = resident["student_other_minus_embedding_bytes"] / mb
    ax2.bar(["Student"], [other], bottom=[emb], color="#d9d9d9", edgecolor="black", linewidth=0.5,
            label="norms, biases, rotary buffer")
    ax2.bar(["Student"], [resident["student_littlebit_factor_scale_buffer_bytes"] / mb], bottom=[emb + other],
            color="#e67e22", edgecolor="black", linewidth=0.5, label="LittleBit factors, scales, buffers")
    for pos, total in enumerate((resident["original_bytes"], resident["student_bytes"])):
        ax2.annotate("{:,} B".format(total), (pos, total / mb), ha="center", va="bottom", fontsize=7,
                     xytext=(0, 2), textcoords="offset points")
    ax2.set_ylabel("megabytes (1e6 bytes)")
    ax2.set_title("Resident unique tensor storage, FP32 in memory", fontsize=9)
    ax2.legend(fontsize=7, frameon=False, loc="upper right")
    top = max(o["bytes"], resident["original_bytes"]) / mb * 1.35
    ax1.set_ylim(0, o["bytes"] / mb * 1.35)
    ax2.set_ylim(0, top)
    fig.suptitle("Weight bytes only; tokenizer and config files (%s B) and KV cache (%.0f MiB, unchanged) "
                 "excluded; panels use different dtypes and are not comparable"
                 % ("{:,}".format(st["excluded"]["tokenizer_and_config_files_bytes"]),
                    st["kv_cache"]["original_bytes"] / MIB), fontsize=8, y=1.02)
    produced += _save(fig, figures_dir, "weight_bytes")
    plt.close(fig)

    # 4. CE and KL per QAT step.
    steps = summary["training"]["steps"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = [s["step"] for s in steps]
    ax.plot(x, [s["ce"] for s in steps], marker="o", color="#c0392b", label="cross entropy (nats)")
    ax.plot(x, [s["kl"] for s in steps], marker="s", color="#1b6ca8", label="KL(teacher || student), token mean")
    ax.axhline(c["original"]["mean_nll"], color="#555555", linestyle=":", linewidth=1,
               label="original model held-out test NLL (reference only)")
    ax.set_xticks(x)
    ax.set_xlabel("QAT step (batch 1, one 128-token train window per step, each used once)")
    ax.set_ylabel("loss")
    ax.set_ylim(bottom=0)
    ax.set_title("16-step QAT on 2048 input tokens (2032 supervised targets); single run, seed 42", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    produced += _save(fig, figures_dir, "train_ce_kl")
    plt.close(fig)
    return produced


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------

def resolve_metadata(pilot_dir: str, evidence_dir: str, original_weights: str, metadata_path: str,
                     metadata_only: bool, write_metadata: bool) -> Tuple[Dict[str, Dict], Dict[str, object], Dict, Dict]:
    """Load public inputs and decide which metadata the summary uses, with the replay record."""
    records, files = load_public(pilot_dir, metadata_path, need_metadata=metadata_only)
    check_statuses(records)
    public = records.get("public_metadata")
    if metadata_only:
        require_sections(public, rel(metadata_path))
        replay = {"mode": MODE_METADATA, "statement": METADATA_REPLAY_STATEMENT,
                  "files_read": sorted(v["path"] for v in files.values()),
                  "private_files_read": [], "private_checks": [],
                  "public_metadata_sections_used": [k for k in METADATA_SECTIONS if get_section(public, k) is not None]}
        return records, files, public, replay
    private, checks = collect_private(pilot_dir, evidence_dir, original_weights, records)
    rc = reconcile(private, public, allow_fill=write_metadata)
    meta = rc["effective"]
    require_sections(meta, "private validation")
    if write_metadata:
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
        assert_public_safe(meta, rel(metadata_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
        records["public_metadata"] = meta
        files["public_metadata"] = file_record(metadata_path)
    replay = {"mode": MODE_PRIVATE, "statement": PRIVATE_VALIDATION_STATEMENT, "private_checks": checks,
              "public_metadata_sections_verified": rc["verified"],
              "public_metadata_sections_not_rechecked": rc["not_rechecked"],
              "public_metadata_sections_written": rc["filled"] if write_metadata else []}
    return records, files, meta, replay


def build_summary(records: Dict[str, Dict], files: Dict[str, object], meta: Dict, replay: Dict) -> Dict[str, object]:
    held = heldout(records)
    train_summary = training(records)
    ident = identity(records, meta)
    store = storage(records, meta)
    res = resources(records)
    man = records["manifest"]
    tr = records["train"]
    summary: Dict[str, object] = {
        "schema": SCHEMA,
        "generated_by": "src/report_results.py (reads recorded JSON only; no model execution, network or subprocess)",
        "replay": replay,
        "inputs": files,
        "pilot": {
            "tier": records["evaluate"]["tier"],
            "model": {"repo": man["model"]["repo"], "revision": man["model"]["revision"]["pinned"],
                      "license": man["model"]["license"],
                      "note": "Qwen2.5-0.5B is smaller than, and outside, the model table of the LittleBit papers"},
            "dataset": {"repo": man["dataset"]["repo"], "config": man["dataset"]["config"],
                        "revision": man["dataset"]["revision_pinned"],
                        "license_note": man["dataset"]["license_note"]},
            "upstream": {"url": records["evaluate"]["upstream"]["provenance"]["url"],
                         "commit": records["evaluate"]["upstream"]["provenance"]["pinned_commit"],
                         "license": "CC BY-NC 4.0"},
            "littlebit_config": tr["config"],
            "windows": {split: {"n": man["splits"][split]["n_windows"], "seq_len": man["splits"][split]["seq_len"],
                                "tokens": man["splits"][split]["tokens_used"],
                                "scored_tokens": sum(w["scored_tokens"] for w in man["splits"][split]["windows"]),
                                "role": man["splits"][split]["role"]}
                        for split in ("train", "validation", "test")},
            "environment": records["evaluate"]["environment"],
        },
        "heldout_test": held,
        "training": train_summary,
        "identity": ident,
        "storage": store,
        "resources": res,
        "tests": tests_block(meta),
        "preregistered_predictions": predictions(held, train_summary, store, res, ident),
        "missing_baselines": [
            "validated practical 4-bit quantization of the same model",
            "smaller same-family model",
            "task accuracy",
            "statistical analysis (confidence intervals, multiple seeds, more windows)",
            "end-to-end latency or throughput",
        ],
        "claims": {
            "negative_result": ("at %.5f advertised BPW in the converted blocks, %d QAT steps on %d input tokens "
                                "left held-out perplexity %.1f times the original model on %d scored tokens"
                                % (store["bpw_converted_block_linears"]["advertised_upstream_formula"],
                                   train_summary["completed_steps"], train_summary["input_tokens"],
                                   held["conditions"]["qat_trained"]["ppl_ratio_vs_original"],
                                   held["conditions"]["qat_trained"]["scored_tokens"])),
            "not_claimed": [
                "novelty",
                "replication of any published LittleBit result",
                "practical usefulness of the compressed model",
                "statistical significance",
                "anything about longer training, other models or other bit widths",
            ],
            "model_trained_basis": tr["claims"]["model_trained_basis"],
            "no_further_training_planned": True,
            "no_license_agreement_accepted_no_cloud_spend": True,
            "publication": ("public GitHub repository with release v0.1.0 authorized by the owner; no Medium "
                            "publication, no arXiv submission; no license granted (all rights reserved on "
                            "original portions, CC BY-NC 4.0 on adapted portions)"),
        },
        "reproduce": reproduce(records),
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--metadata-only", action="store_true",
                   help="metadata replay from tracked public files only; does not validate private checkpoints")
    p.add_argument("--pilot-dir", default=DEFAULT_PILOT_DIR)
    p.add_argument("--public-metadata", default=None,
                   help="public metadata JSON (default: <pilot-dir>/%s)" % PUBLIC_METADATA_NAME)
    p.add_argument("--write-public-metadata", action="store_true",
                   help="private validation only: write re-derived sections into the public metadata file")
    p.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    p.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    p.add_argument("--original-weights", default=DEFAULT_ORIGINAL_WEIGHTS,
                   help="private validation only, optional; only its safetensors header is read")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    metadata_path = a.public_metadata or os.path.join(a.pilot_dir, PUBLIC_METADATA_NAME)
    try:
        require(not (a.metadata_only and a.write_public_metadata),
                "--write-public-metadata needs private validation; it cannot be combined with --metadata-only")
        records, files, meta, replay = resolve_metadata(a.pilot_dir, a.evidence_dir, a.original_weights,
                                                        metadata_path, a.metadata_only, a.write_public_metadata)
        summary = build_summary(records, files, meta, replay)
        summary["figures"] = [] if a.no_figures else draw_figures(summary, a.figures_dir)
        assert_public_safe(summary, "summary")
    except ReportError as e:
        print("report_results: %s" % e, file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.summary)), exist_ok=True)
    with open(a.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    c = summary["heldout_test"]["conditions"]
    print("wrote %s (%s)" % (rel(a.summary), summary["replay"]["mode"]))
    for key in ("original", "initialized_no_qat", "qat_trained", DEFECT[0]):
        print("  %-30s ppl %.4f" % (key, c[key]["ppl"]))
    for rec in summary["figures"]:
        print("  %s" % rec["path"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
