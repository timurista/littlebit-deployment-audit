# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Tim Urista. MIT grant scoped to the files listed in LICENSE, Part 1.
"""Bounded local quality pilot: actual upstream LittleBitLinear on a real pretrained model.

This is a wiring and resource pilot on Qwen/Qwen2.5-0.5B with tiny frozen WikiText-2 windows. It is
NOT a statistical benchmark, NOT a paper replication, and NOT a throughput measurement. See
QUALITY_PREREGISTRATION.md for the fixed configuration, predictions, stopping rules and departures
from upstream.

Stages (run in order, each writes JSON into --out):

  prepare      read local WikiText-2 raw parquet files, tokenize with the native tokenizer, freeze
               train (QAT only), validation (development only) and test (held out) windows
  baseline     original model NLL on validation and test windows, fixed generation prompts
  feasibility  real upstream conversion plus 2 real optimizer steps, then an elapsed estimate
  train        full bounded QAT run, init and trained packed checkpoints, reload identity check
  evaluate     held-out NLL of original, initialized and trained students after packed reload

Every stage runs offline: local files only, no network, no subprocesses. The module imports only
the standard library at import time, so --help works without torch.

    .venv311/bin/python src/quality_pilot.py --help
    .venv311/bin/python src/quality_pilot.py prepare --data-dir PATH_TO_LOCAL_WIKITEXT
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import struct
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "qwen2.5-0.5b")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "results", "quality_pilot")

SCHEMA = "littlebit-quality-pilot/1"
TIER_PILOT = "UPSTREAM_IMPORT_REAL_MODEL_PILOT"
TIER_DEFECT = "UPSTREAM_DECODER_DEFECT_VARIANT"

MODEL_REPO = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
MODEL_LICENSE = "Apache-2.0, ungated (verified by the project owner before this pilot)"

DATASET_REPO = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EXPECTED_ROWS = {"train": 36718, "validation": 3760, "test": 4358}
SPLIT_ROLES = {"train": "qat_training_only", "validation": "development_only",
               "test": "frozen_heldout"}
SPLITS = ("train", "validation", "test")
PILOT_WINDOWS = {"train": 16, "validation": 4, "test": 8}
SEQ_LEN = 128
SEED = 42
JOIN_RULE = "\n\n"
TOKENIZE_MARGIN = 64  # trailing prefix tokens never used, so the prefix cut cannot change a window

DEFAULT_STEPS = 16
DEFAULT_BATCH = 1
DEFAULT_EFF_BIT = 0.55
DEFAULT_LR = 4e-5  # upstream README training commands
DEFAULT_THREADS = 2
DEFAULT_MAX_RSS_GIB = 6.0
DEFAULT_MIN_AVAILABLE_GIB = 2.0
DEFAULT_MAX_SECONDS = 900.0
DEFAULT_MAX_NEW_TOKENS = 20
MAX_NEW_TOKENS_CAP = 32
GIB = 1024 ** 3

FACTOR_NAMES = ("U", "V", "U_R", "V_R")
SCALE_NAMES = ("u1", "u2", "v1", "v2", "u1_R", "u2_R", "v1_R", "v2_R")
LBL_BUFFERS = ("_eff_bit_target", "_split_dim_final", "_eff_bit_actual")

GENERATION_PROMPTS = (
    "The capital of France is",
    "In 1990 , the population of the city was",
    "Water boils at a temperature of",
)

NON_CLAIMS = (
    "wiring and resource pilot only, not a statistical benchmark",
    "not a replication of any published LittleBit result",
    "no end-to-end throughput claim; operator microbenchmarks are not end-to-end",
    "task accuracy not measured; generation outputs are shown, never judged",
    "no validated 4-bit baseline exists in this project yet (MISSING)",
    "a model is called trained only when a completed step log and a checkpoint with sha256 exist",
)

MISSING_BASELINES = {
    "practical_4bit": "MISSING: no validated 4-bit backend in this project yet",
    "smaller_same_family_model": "MISSING: not part of this pilot",
}


class GuardStop(RuntimeError):
    """Raised when a resource guard limit is hit. The run stops and records why."""


class PreconditionError(RuntimeError):
    """A required input (file, revision, prior stage output) is missing or does not match."""


# ---------------------------------------------------------------------------------------------
# Standard library helpers (unit tested without torch)
# ---------------------------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def file_record(path: str) -> dict:
    return {"path": public_path(path), "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


# ---------------------------------------------------------------------------------------------
# Public-safe serialization: recorded metrics never carry host absolute paths
# ---------------------------------------------------------------------------------------------

PATH_ARGS = ("out", "model_dir", "data_dir")
EXTERNAL_PREFIX = "<external>/"
PROJECT_TOKEN = "<project>"
HOME_TOKEN = "<home>"
ARGS_PATH_POLICY = ("path arguments are recorded relative to the project root, or as <external>/<basename> "
                    "when outside it; other strings have the project root and home directory replaced")
# A user home directory on macOS, Linux or Windows, with the user name segment. The lookbehind
# keeps URL paths such as https://example.org/home/page intact.
HOST_HOME_RE = re.compile(r"(?<![\w.])(?:/Users|/home|[A-Za-z]:[\\/]Users)[\\/][^\\/\s\"':,;)\]}]+")
_PATH_END = r"(?=$|[\\/\s\"':,;)\]}])"


def _last_component(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]+", path) if p]
    return parts[-1] if parts else ""


def public_path(path, root: str = PROJECT_ROOT) -> Optional[str]:
    """A path as it may appear in a public record: relative to the project root, or
    <external>/<basename> when it lies outside the root. Never an absolute host path."""
    if path is None:
        return None
    s = os.fspath(path)
    if s == "":
        return s
    root_abs = os.path.normpath(os.path.abspath(root))
    if os.path.isabs(s) or HOST_HOME_RE.match(s):
        norm = os.path.normpath(s)
        if os.path.isabs(norm):
            if norm == root_abs:
                return "."
            if norm.startswith(root_abs.rstrip(os.sep) + os.sep):
                return os.path.relpath(norm, root_abs).replace(os.sep, "/")
        return EXTERNAL_PREFIX + _last_component(s)
    norm = os.path.normpath(s)
    if norm == os.pardir or norm.startswith(os.pardir + os.sep):
        return EXTERNAL_PREFIX + _last_component(s)
    return norm.replace(os.sep, "/")


def scrub_text(text: str, root: str = PROJECT_ROOT, home: Optional[str] = None) -> str:
    """Replace the project root, the home directory and any user home path inside free text."""
    root_abs = os.path.normpath(os.path.abspath(root))
    home = os.path.expanduser("~") if home is None else home
    for prefix, token in ((root_abs, PROJECT_TOKEN), (os.path.normpath(home) if home else "", HOME_TOKEN)):
        if len(prefix) > 1:
            text = re.sub(re.escape(prefix) + _PATH_END, token, text)
    return HOST_HOME_RE.sub(HOME_TOKEN, text)


def scrub_host_paths(obj, root: str = PROJECT_ROOT, home: Optional[str] = None):
    """Recursively scrub every string (keys included) of a JSON-like object."""
    if isinstance(obj, str):
        return scrub_text(obj, root, home)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {(scrub_text(k, root, home) if isinstance(k, str) else k): scrub_host_paths(v, root, home)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_host_paths(v, root, home) for v in obj]
    return scrub_text(str(obj), root, home)


def find_host_paths(obj, root: str = PROJECT_ROOT, home: Optional[str] = None, where: str = "$") -> List[str]:
    """JSON pointers of every string the scrubber would change, that is, every host path left in obj."""
    found: List[str] = []
    if isinstance(obj, str):
        if scrub_text(obj, root, home) != obj:
            found.append(where)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and scrub_text(k, root, home) != k:
                found.append("%s.<key %s>" % (where, k))
            found.extend(find_host_paths(v, root, home, "%s.%s" % (where, k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found.extend(find_host_paths(v, root, home, "%s[%d]" % (where, i)))
    return found


def public_args(args) -> dict:
    """The parsed CLI arguments as recorded in a stage header, with no host absolute paths."""
    out = {}
    for k, v in vars(args).items():
        if k == "func":
            continue
        out[k] = public_path(v) if (k in PATH_ARGS and v is not None) else scrub_host_paths(v)
    return out


def token_bytes(ids: Sequence[int]) -> bytes:
    """Token ids as little-endian int32, the byte form every token hash is taken over."""
    return struct.pack("<%di" % len(ids), *ids)


def hash_tokens(ids: Sequence[int]) -> str:
    return sha256_bytes(token_bytes(ids))


def build_windows(ids: Sequence[int], n_windows: int, seq_len: int) -> List[dict]:
    """Contiguous, non-overlapping windows starting at token 0 of one split's stream."""
    need = n_windows * seq_len
    if n_windows <= 0 or seq_len <= 1:
        raise ValueError("need n_windows >= 1 and seq_len >= 2")
    if len(ids) < need:
        raise ValueError("stream has %d tokens, %d windows of %d need %d" % (len(ids), n_windows, seq_len, need))
    out = []
    for w in range(n_windows):
        start, end = w * seq_len, (w + 1) * seq_len
        chunk = [int(t) for t in ids[start:end]]
        out.append({"index": w, "start": start, "end": end, "n_tokens": len(chunk),
                    "scored_tokens": len(chunk) - 1, "sha256": hash_tokens(chunk), "token_ids": chunk})
    return out


def select_prefix_rows(rows: Sequence[str], tokenize: Callable[[str], List[int]], needed: int,
                       margin: int = TOKENIZE_MARGIN, start_rows: int = 64) -> dict:
    """Smallest doubling prefix rows[0:k] whose joined text tokenizes to >= needed + margin tokens.

    Only the first `needed` tokens are used. The margin keeps used tokens away from the cut point,
    where tokenizing a prefix could merge differently than tokenizing the whole split.
    """
    k = max(1, min(start_rows, len(rows)))
    while True:
        text = JOIN_RULE.join(rows[:k])
        ids = list(tokenize(text))
        if len(ids) >= needed + margin:
            return {"row_start": 0, "row_end": k, "prefix_text_sha256": sha256_bytes(text.encode("utf-8")),
                    "prefix_token_count": len(ids), "tokens_used": needed, "margin_tokens": margin,
                    "token_ids": ids[:needed]}
        if k >= len(rows):
            raise ValueError("split has only %d tokens, need %d plus margin %d" % (len(ids), needed, margin))
        k = min(len(rows), k * 2)


def make_split_windows(rows_by_split: Dict[str, Sequence[str]], tokenize: Callable[[str], List[int]],
                       counts: Dict[str, int], seq_len: int = SEQ_LEN) -> Dict[str, dict]:
    """Each split is tokenized on its own, so no window can contain tokens of another split."""
    out = {}
    for split in SPLITS:
        if split not in counts:
            continue
        sel = select_prefix_rows(rows_by_split[split], tokenize, counts[split] * seq_len)
        ids = sel.pop("token_ids")
        windows = build_windows(ids, counts[split], seq_len)
        sel.update({"role": SPLIT_ROLES[split], "n_windows": counts[split], "seq_len": seq_len,
                    "stream_sha256": hash_tokens(ids), "windows": windows})
        out[split] = sel
    return out


SNAPSHOT_RE = re.compile(r"snapshots[\\/]+([0-9a-f]{40})(?:[\\/]|$)")


def parse_snapshot_revision(path: str) -> Optional[str]:
    """Revision commit from a Hugging Face cache path (.../snapshots/<40 hex>/...), if present."""
    m = SNAPSHOT_RE.search(path)
    return m.group(1) if m else None


def read_model_revision(model_dir: str) -> dict:
    """Read the revision lines that huggingface_hub wrote next to each downloaded file."""
    meta_dir = os.path.join(model_dir, ".cache", "huggingface", "download")
    revs: Dict[str, str] = {}
    etags: Dict[str, str] = {}
    if os.path.isdir(meta_dir):
        for name in sorted(os.listdir(meta_dir)):
            if not name.endswith(".metadata"):
                continue
            with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            if lines:
                revs[name[: -len(".metadata")]] = lines[0].strip()
            if len(lines) > 1:
                etags[name[: -len(".metadata")]] = lines[1].strip()
    values = set(revs.values())
    return {"pinned": MODEL_REVISION, "per_file": revs, "etags": etags,
            "consistent": len(values) == 1, "matches_pinned": values == {MODEL_REVISION}}


def token_weighted(nll_sums: Sequence[float], counts: Sequence[int]) -> dict:
    total_nll = float(sum(nll_sums))
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("no scored tokens")
    mean = total_nll / total
    return {"total_nll": total_nll, "scored_tokens": total, "mean_nll": mean,
            "ppl": math.exp(mean) if mean < 700 else float("inf")}


def paired_window_diffs(a: Sequence[dict], b: Sequence[dict]) -> List[dict]:
    """Per-window (b minus a) mean NLL differences on identical windows (checked by sha256)."""
    out = []
    for x, y in zip(a, b):
        if x["sha256"] != y["sha256"]:
            raise ValueError("windows differ: %s vs %s" % (x["sha256"], y["sha256"]))
        out.append({"index": x["index"], "sha256": x["sha256"],
                    "mean_nll_diff": y["nll_sum"] / y["scored_tokens"] - x["nll_sum"] / x["scored_tokens"]})
    if len(a) != len(b):
        raise ValueError("window count mismatch")
    return out


def cosine_lr(base: float, step: int, total: int) -> float:
    """Cosine decay from base at step 0 toward 0, no warmup (declared departure from upstream)."""
    if total <= 0:
        raise ValueError("total must be positive")
    return base * 0.5 * (1.0 + math.cos(math.pi * step / total))


def feasibility_estimate(conversion_s: float, step_times: Sequence[float], steps: int,
                         budget_s: float) -> dict:
    if not step_times:
        raise ValueError("no measured steps")
    mean_s = sum(step_times) / len(step_times)
    max_s = max(step_times)
    est_mean = conversion_s + steps * mean_s
    est_max = conversion_s + steps * max_s
    return {"measured_steps": len(step_times), "step_seconds": list(step_times),
            "conversion_seconds": conversion_s, "planned_steps": steps,
            "estimated_train_stage_seconds_mean": est_mean,
            "estimated_train_stage_seconds_conservative": est_max,
            "budget_seconds": budget_s, "fits_budget_conservative": est_max <= budget_s,
            "note": "estimate from actual measured step times; excludes checkpoint save and identity checks"}


class ResourceGuard:
    """Stops a run when RSS, system available memory, or wall clock crosses a limit."""

    def __init__(self, max_rss_bytes: int, min_available_bytes: int, max_seconds: float,
                 rss_fn: Optional[Callable[[], int]] = None,
                 available_fn: Optional[Callable[[], int]] = None,
                 clock: Callable[[], float] = time.monotonic,
                 extra_fn: Optional[Callable[[], dict]] = None):
        self.max_rss_bytes = int(max_rss_bytes)
        self.min_available_bytes = int(min_available_bytes)
        self.max_seconds = float(max_seconds)
        self._rss_fn = rss_fn
        self._available_fn = available_fn
        self._clock = clock
        self._extra_fn = extra_fn
        self.start = clock()
        self.peak_rss = 0
        self.min_available = None
        self.checks = 0

    def _rss(self) -> int:
        if self._rss_fn is None:
            import psutil
            self._rss_fn = lambda p=psutil.Process(): int(p.memory_info().rss)
        return int(self._rss_fn())

    def _available(self) -> int:
        if self._available_fn is None:
            import psutil
            self._available_fn = lambda: int(psutil.virtual_memory().available)
        return int(self._available_fn())

    def elapsed(self) -> float:
        return self._clock() - self.start

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def snapshot(self) -> dict:
        snap = {"rss_bytes": self._rss(), "available_bytes": self._available(),
                "elapsed_s": self.elapsed()}
        if self._extra_fn is not None:
            snap.update(self._extra_fn())
        return snap

    def check(self, where: str) -> dict:
        snap = self.snapshot()
        self.checks += 1
        self.peak_rss = max(self.peak_rss, snap["rss_bytes"])
        if self.min_available is None or snap["available_bytes"] < self.min_available:
            self.min_available = snap["available_bytes"]
        if snap["rss_bytes"] > self.max_rss_bytes:
            raise GuardStop("process RSS %d bytes exceeds limit %d at %s" % (snap["rss_bytes"], self.max_rss_bytes, where))
        if snap["available_bytes"] < self.min_available_bytes:
            raise GuardStop("system available memory %d bytes below %d at %s" % (snap["available_bytes"], self.min_available_bytes, where))
        if snap["elapsed_s"] > self.max_seconds:
            raise GuardStop("elapsed %.1f s exceeds %.1f s at %s" % (snap["elapsed_s"], self.max_seconds, where))
        return snap

    def summary(self) -> dict:
        return {"max_rss_bytes": self.max_rss_bytes, "min_available_bytes": self.min_available_bytes,
                "max_seconds": self.max_seconds, "checks": self.checks, "peak_rss_seen_at_checks": self.peak_rss,
                "min_available_seen_at_checks": self.min_available, "elapsed_s": self.elapsed()}


def write_json(path: str, obj) -> str:
    """Every recorded JSON passes through scrub_host_paths, including error tracebacks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(scrub_host_paths(obj), f, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str, obj) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(scrub_host_paths(obj), sort_keys=True, default=str) + "\n")
        f.flush()


def peak_rss_os() -> dict:
    """Process lifetime peak RSS from getrusage. macOS reports bytes, Linux reports KiB."""
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return {"ru_maxrss_raw": None}
    unit = "bytes" if sys.platform == "darwin" else "KiB"
    return {"ru_maxrss_raw": raw, "unit": unit,
            "bytes": raw if unit == "bytes" else raw * 1024}


def package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in ("torch", "transformers", "datasets", "psutil", "safetensors", "tokenizers",
                 "pyarrow", "numpy", "huggingface_hub"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = None
    return out


def base_environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": package_versions()}


def force_offline_env() -> dict:
    """Set before transformers is imported, so no hub lookup can happen."""
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    for k in keys:
        os.environ[k] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return {k: os.environ[k] for k in keys}


def config_from_args(args) -> dict:
    return {"eff_bit": args.eff_bit, "residual": args.residual, "split_dim_default": 1024,
            "min_split_dim": 8, "kv_factor": 1.0, "use_itq": False, "quant_func": "SmoothSign",
            "quant_mod": "LittleBitLinear", "seed": args.seed, "seq_len": SEQ_LEN}


def claims_block(trained: bool, reason: str) -> dict:
    return {"model_trained": trained, "model_trained_basis": reason,
            "initialization_quality_is_separate_from_training": True,
            "task_accuracy": "not measured", "throughput": "not measured",
            "statistical_significance": "not assessed; pilot sizes are fixed and tiny",
            "baselines": dict(MISSING_BASELINES), "non_claims": list(NON_CLAIMS)}


# ---------------------------------------------------------------------------------------------
# Torch side (lazy imports only)
# ---------------------------------------------------------------------------------------------

def setup_torch(args):
    force_offline_env()
    import torch
    torch.set_num_threads(int(args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = resolve_device(torch, args.device)
    return torch, device


def resolve_device(torch, name: str):
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "") not in ("", "0"):
            raise PreconditionError("PYTORCH_ENABLE_MPS_FALLBACK is set; refusing silent CPU fallback")
        if not torch.backends.mps.is_available():
            raise PreconditionError("MPS requested but not available; refusing to fall back to CPU")
        return torch.device("mps")
    raise PreconditionError("unsupported device %r (cpu or mps)" % name)


def sync(torch, device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def mps_memory(torch, device) -> dict:
    if device.type != "mps":
        return {}
    return {"mps_current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())}


def make_guard(args, torch=None, device=None) -> ResourceGuard:
    extra = (lambda: mps_memory(torch, device)) if (torch is not None and device is not None
                                                     and device.type == "mps") else None
    return ResourceGuard(int(args.max_rss_gib * GIB), int(args.min_available_gib * GIB),
                         float(args.max_seconds), extra_fn=extra)


def torch_environment(torch, device, args) -> dict:
    env = base_environment()
    env.update({"torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "requested_threads": args.threads, "device": str(device),
                "mps_available": bool(torch.backends.mps.is_available()),
                "cuda_available": bool(torch.cuda.is_available()), "dtype": "float32",
                "attn_implementation": "sdpa", "offline_env": force_offline_env(),
                "conversion_device": "cpu"})
    return env


def load_upstream_api() -> dict:
    sys.path.insert(0, SRC_DIR)
    import audit
    up = audit.load_upstream(include_layer=True)
    if not up.get("available") or "LittleBitLinear" not in up:
        raise PreconditionError("actual upstream LittleBitLinear not importable: %s"
                                % (up.get("reason") or up.get("layer_reason")))
    up["provenance"] = audit.upstream_commit_check()
    up["file_sha256"] = audit.file_hashes()
    return up


def corrected_decoder():
    sys.path.insert(0, SRC_DIR)
    import bench
    return bench.torch_reference_unpack


def model_provenance(model_dir: str, verify_weights: bool = True) -> dict:
    rev = read_model_revision(model_dir)
    if not rev["matches_pinned"]:
        raise PreconditionError("model revision metadata does not match pinned %s: %s" % (MODEL_REVISION, rev["per_file"]))
    out = {"repo": MODEL_REPO, "license": MODEL_LICENSE, "revision": rev,
           "model_dir": public_path(model_dir), "files": {}}
    for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "LICENSE"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            out["files"][name] = file_record(path)
    weights = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights):
        raise PreconditionError("model.safetensors not found in %s" % model_dir)
    if verify_weights:
        rec = file_record(weights)
        etag = rev["etags"].get("model.safetensors")
        rec["hub_lfs_sha256"] = etag
        rec["matches_hub_lfs_sha256"] = (etag == rec["sha256"])
        if not rec["matches_hub_lfs_sha256"]:
            raise PreconditionError("model.safetensors sha256 %s does not match recorded %s" % (rec["sha256"], etag))
        out["files"]["model.safetensors"] = rec
    return out


def load_tokenizer(model_dir: str):
    force_offline_env()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def tokenizer_record(tok) -> dict:
    return {"class": type(tok).__name__, "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id, "add_bos_token": getattr(tok, "add_bos_token", None),
            "add_eos_token": getattr(tok, "add_eos_token", None), "add_special_tokens_argument": "default (True)",
            "bos_eos_override": "none", "vocab_size": len(tok)}


def load_base_model(torch, model_dir: str):
    force_offline_env()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True,
                                                 attn_implementation="sdpa")
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def params_fingerprint(torch, model) -> str:
    """sha256 over every named parameter (name, dtype, shape, raw bytes), in name order."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        t = p.detach()
        if t.device.type != "cpu":
            t = t.to("cpu")
        t = t.contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.view(torch.uint8).numpy() if t.dtype == torch.bfloat16 else t.numpy())
    return h.hexdigest()


def lbl_modules(model, LBL) -> List[Tuple[str, object]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LBL)]


def block_linear_targets(model) -> List[Tuple[int, str]]:
    """Exact nn.Linear modules inside the transformer blocks. lm_head and embeddings are outside."""
    import torch.nn as nn
    out = []
    for li, layer in enumerate(model.model.layers):
        for name, mod in layer.named_modules():
            if type(mod) is nn.Linear:
                out.append((li, name))
    return out


def clone_student(teacher):
    """Deep copy of the module tree that shares every parameter tensor with the frozen teacher.

    Conversion never writes into a shared tensor: upstream reads weight.data and then drops the
    student module's reference. Trainable factors are new tensors owned only by the student.
    """
    memo = {id(p): p for p in teacher.parameters()}
    return copy.deepcopy(teacher, memo)


def convert_modules(torch, model, up: dict, cfg: dict, do_train: bool, guard: Optional[ResourceGuard] = None,
                    milestones: Optional[list] = None, log: Callable[[str], None] = print) -> List[dict]:
    """Reclass block nn.Linear modules to actual upstream LittleBitLinear (patch_inst style)."""
    LBL = up["LittleBitLinear"]
    targets = block_linear_targets(model)
    n_layers = len(model.model.layers)
    records = []
    t0 = time.monotonic()
    if do_train:
        torch.manual_seed(cfg["seed"])
    with torch.no_grad():
        for li in range(n_layers):
            layer = model.model.layers[li]
            for (lj, name) in [t for t in targets if t[0] == li]:
                if guard is not None:
                    guard.check("convert layer %d %s" % (li, name))
                mod = layer.get_submodule(name)
                ts = time.monotonic()
                mod.__class__ = LBL
                mod.__quant_convert__(do_train=do_train, quant_func=up["SmoothSign"], split_dim=cfg["split_dim_default"],
                                      eff_bit=cfg["eff_bit"], residual=cfg["residual"],
                                      ratio_factor=cfg["kv_factor"], min_split_dim=cfg["min_split_dim"],
                                      use_itq=False)
                records.append({"module": "model.layers.%d.%s" % (li, name), "in_features": mod.in_features,
                                "out_features": mod.out_features, "split_dim": int(mod.split_dim),
                                "eff_bit_actual": float(mod.eff_bit_actual), "has_bias": mod.bias is not None,
                                "seconds": time.monotonic() - ts})
            if milestones is not None:
                elapsed = time.monotonic() - t0
                projected = elapsed / (li + 1) * n_layers
                m = {"layers_done": li + 1, "layers_total": n_layers, "elapsed_s": elapsed,
                     "projected_total_s": projected}
                if guard is not None:
                    m.update(guard.snapshot())
                milestones.append(m)
                log("convert milestone %d/%d elapsed %.1fs projected %.1fs" % (li + 1, n_layers, elapsed, projected))
                if guard is not None and do_train and (projected - elapsed) > guard.remaining():
                    raise GuardStop("projected SVD initialization %.1fs exceeds remaining budget %.1fs"
                                    % (projected - elapsed, guard.remaining()))
    return records


def trainable_factor_params(model, LBL) -> List[Tuple[str, object]]:
    """Freeze everything, then unfreeze only LittleBit factors and scales (not bias, norms, embeddings, head)."""
    import torch.nn as nn
    for p in model.parameters():
        p.requires_grad_(False)
    out = []
    for mname, m in lbl_modules(model, LBL):
        for pname in FACTOR_NAMES + SCALE_NAMES:
            p = getattr(m, pname, None)
            if p is not None and isinstance(p, nn.Parameter):
                p.requires_grad_(True)
                out.append(("%s.%s" % (mname, pname), p))
    return out


def window_tensor(torch, windows: Sequence[dict], device):
    return torch.tensor([w["token_ids"] for w in windows], dtype=torch.long, device=device)


def window_nll(torch, model, ids) -> Tuple[float, int]:
    """Token-weighted NLL of tokens 1..T-1 given their prefix; the first token is never scored."""
    import torch.nn.functional as F
    logits = model(input_ids=ids, use_cache=False).logits.float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return float(nll.double().sum().item()), int(tgt.numel())


def eval_windows(torch, model, windows: Sequence[dict], device, guard: Optional[ResourceGuard], label: str) -> dict:
    rows = []
    model.eval()
    with torch.no_grad():
        for w in windows:
            if guard is not None:
                guard.check("eval %s window %d" % (label, w["index"]))
            if hash_tokens(w["token_ids"]) != w["sha256"]:
                raise PreconditionError("window %s/%d token hash mismatch" % (label, w["index"]))
            t = time.perf_counter()
            s, n = window_nll(torch, model, window_tensor(torch, [w], device))
            sync(torch, device)
            rows.append({"index": w["index"], "sha256": w["sha256"], "nll_sum": s, "scored_tokens": n,
                         "mean_nll": s / n, "wall_s": time.perf_counter() - t})
    agg = token_weighted([r["nll_sum"] for r in rows], [r["scored_tokens"] for r in rows])
    agg.update({"windows": rows, "label": label, "wall_s_note": "descriptive pilot wall clock, not throughput"})
    return agg


def distill_losses(torch, s_logits, t_logits, ids, kd_weight: float, ce_weight: float, l2l_scale: float = 0.0,
                   s_hidden=None, t_hidden=None) -> dict:
    """KL(teacher || student) per position, token mean, plus next-token CE, plus optional hidden MSE."""
    import torch.nn.functional as F
    s = s_logits.float()
    t = t_logits.detach().float()
    t_logp = F.log_softmax(t, dim=-1)
    s_logp = F.log_softmax(s, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean()
    ce = F.cross_entropy(s[:, :-1].reshape(-1, s.size(-1)), ids[:, 1:].reshape(-1))
    l2l = s.new_zeros(())
    if l2l_scale > 0:
        if s_hidden is None or t_hidden is None:
            raise ValueError("l2l_scale > 0 needs hidden states")
        for a, b in zip(s_hidden[1:], t_hidden[1:]):
            l2l = l2l + F.mse_loss(a.float(), b.detach().float())
    loss = kd_weight * kl + ce_weight * ce + l2l_scale * l2l
    return {"loss": loss, "kl": kl, "ce": ce, "l2l": l2l}


def tensor_storage_bytes(tensors: Iterable) -> int:
    seen = {}
    for t in tensors:
        if t is None or t.device.type == "meta":
            continue
        st = t.untyped_storage()
        seen[(t.device.type, st.data_ptr())] = st.nbytes()
    return int(sum(seen.values()))


def resident_storage(model, LBL) -> dict:
    """Unique resident tensor storage, deduplicated by storage pointer; tied tensors count once."""
    comp, other = [], []
    lbl_ids = set()
    for _, m in lbl_modules(model, LBL):
        for pname in FACTOR_NAMES + SCALE_NAMES + LBL_BUFFERS:
            t = getattr(m, pname, None)
            if t is not None:
                comp.append(t)
                lbl_ids.add(id(t))
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in lbl_ids:
            other.append(t)
    return {"littlebit_factor_scale_buffer_bytes": tensor_storage_bytes(comp),
            "other_bytes": tensor_storage_bytes(other),
            "total_unique_bytes": tensor_storage_bytes(comp + other),
            "note": "resident tensors only; excludes KV cache, activations, allocator slack"}


def kv_cache_bytes(torch, model, ids) -> dict:
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    cache = out.past_key_values
    tensors = []
    if hasattr(cache, "key_cache"):
        tensors = list(cache.key_cache) + list(cache.value_cache)
    else:
        for layer in cache:
            tensors.extend(layer)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    formula = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * ids.shape[1] * 4
    return {"tokens": int(ids.shape[1]), "batch": int(ids.shape[0]),
            "measured_bytes": int(sum(t.numel() * t.element_size() for t in tensors)),
            "formula_bytes_fp32": formula, "note": "KV cache is separate from weights; LittleBit does not compress it"}


def generate_fixed(torch, model, tok, device, max_new_tokens: int) -> List[dict]:
    max_new_tokens = min(int(max_new_tokens), MAX_NEW_TOKENS_CAP)
    out = []
    model.eval()
    for prompt in GENERATION_PROMPTS:
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new_tokens, do_sample=False,
                               num_beams=1, use_cache=True,
                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new = [int(x) for x in g[0, ids.shape[1]:].tolist()]
        out.append({"prompt": prompt, "prompt_token_ids": [int(x) for x in ids[0].tolist()],
                    "output_token_ids": new, "continuation": tok.decode(new, skip_special_tokens=False),
                    "decoding": "greedy", "max_new_tokens": max_new_tokens, "judged": False})
    return out


# ---------------------------------------------------------------------------------------------
# Checkpoint save and packed reload
# ---------------------------------------------------------------------------------------------

def save_frozen_base(torch, student, LBL, path: str, meta: dict) -> dict:
    """Every non-LittleBit parameter (embedding, norms, biases). Stored bf16 only when lossless."""
    from safetensors.torch import save_file
    lbl_ids = set()
    for _, m in lbl_modules(student, LBL):
        for pname in FACTOR_NAMES + SCALE_NAMES:
            p = getattr(m, pname, None)
            if p is not None:
                lbl_ids.add(id(p))
    tensors, dtypes = {}, {}
    for name, p in student.named_parameters():  # tied lm_head.weight is deduplicated here
        if id(p) in lbl_ids:
            continue
        t = p.detach().to("cpu").contiguous()
        b = t.to(torch.bfloat16)
        if torch.equal(b.to(torch.float32), t):
            tensors[name], dtypes[name] = b.clone(), "bfloat16_lossless"
        else:
            tensors[name], dtypes[name] = t.clone(), "float32"
    save_file(tensors, path, metadata={k: str(v) for k, v in meta.items()})
    rec = file_record(path)
    rec.update({"tensors": len(tensors), "stored_dtypes": dtypes,
                "tied_not_stored": ["lm_head.weight (tied to model.embed_tokens.weight)"]})
    return rec


def pack_module_tensors(torch, name: str, m, decoder) -> Tuple[dict, dict]:
    """Upstream pack_weights() (upstream binary_packer) plus fp32 scales and buffers of one module.

    Before anything is saved, the corrected decoder must recover quantize(factor) exactly.
    """
    packed = m.pack_weights()
    tensors, checks = {}, {}
    for fname in FACTOR_NAMES:
        if (fname + "_packed") not in packed:
            continue
        p = packed[fname + "_packed"].detach().to("cpu").contiguous()
        shape_t = packed[fname + "_shape"].detach().to("cpu").contiguous()
        ref = m.quantize(getattr(m, fname).data).to(torch.int8).to("cpu")
        dec = decoder(p, tuple(int(x) for x in shape_t.tolist()))
        if not torch.equal(dec, ref):
            raise RuntimeError("corrected decoder does not recover %s.%s; refusing to save" % (name, fname))
        tensors["%s.%s_packed" % (name, fname)] = p
        tensors["%s.%s_shape" % (name, fname)] = shape_t
        checks[fname] = int(ref.numel())
    for sname in SCALE_NAMES:
        t = getattr(m, sname, None)
        if t is not None:
            tensors["%s.%s" % (name, sname)] = t.detach().to("cpu", torch.float32).contiguous().clone()
    for bname in LBL_BUFFERS:
        tensors["%s.%s" % (name, bname)] = getattr(m, bname).detach().to("cpu").contiguous().clone()
    return tensors, checks


def save_compressed(torch, student, LBL, path: str, meta: dict) -> dict:
    from safetensors.torch import save_file
    decoder = corrected_decoder()
    tensors, n_signs, n_weights, adv_bits = {}, 0, 0, 0.0
    for name, m in lbl_modules(student, LBL):
        t, checks = pack_module_tensors(torch, name, m, decoder)
        tensors.update(t)
        n_signs += sum(checks.values())
        n_weights += m.in_features * m.out_features
        adv_bits += float(m.eff_bit_actual) * m.in_features * m.out_features
    save_file(tensors, path, metadata={k: str(v) for k, v in meta.items()})
    rec = file_record(path)
    packed_bytes = sum(v.numel() * v.element_size() for k, v in tensors.items() if k.endswith("_packed"))
    scale_bytes = sum(v.numel() * v.element_size() for k, v in tensors.items()
                      if k.rsplit(".", 1)[-1] in SCALE_NAMES)
    other_bytes = sum(v.numel() * v.element_size() for k, v in tensors.items()) - packed_bytes - scale_bytes
    rec.update({"tensors": len(tensors), "sign_count": n_signs, "converted_weight_count": n_weights,
                "packed_sign_tensor_bytes": packed_bytes, "fp32_scale_tensor_bytes": scale_bytes,
                "shape_and_buffer_tensor_bytes": other_bytes,
                "advertised_bpw_upstream_formula": adv_bits / n_weights,
                "achieved_bpw_from_file_bytes": rec["bytes"] * 8 / n_weights,
                "achieved_bpw_packed_signs_only": packed_bytes * 8 / n_weights,
                "bpw_note": "converted block linears only; biases are in the frozen base file; scales are fp32 here, "
                            "upstream main.py would cast them to bf16"})
    return rec


def build_skeleton(torch, model_dir: str, up: dict, cfg: dict):
    force_offline_env()
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen2 import modeling_qwen2
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, attn_implementation="sdpa")
    convert_modules(torch, model, up, cfg, do_train=False)
    model.model.rotary_emb = modeling_qwen2.Qwen2RotaryEmbedding(config=model.config, device=torch.device("cpu"))
    return model


def load_packed_student(torch, model_dir: str, frozen_path: str, compressed_path: str, up: dict, cfg: dict,
                        decoder, decoder_name: str):
    """Emulated packed load (official quant_util loader not run): decode, cast to fp32,
    load_state_dict(assign=True), tie weights, _binarized=True. Fails on any leftover meta tensor."""
    from safetensors.torch import load_file
    LBL = up["LittleBitLinear"]
    model = build_skeleton(torch, model_dir, up, cfg)
    comp = load_file(compressed_path)
    frozen = load_file(frozen_path)
    state = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in frozen.items()}
    for k, v in comp.items():
        if not (k.endswith("_packed") or k.endswith("_shape")):
            state[k] = v
    for k in sorted(k for k in comp if k.endswith("_packed")):
        base = k[: -len("_packed")]
        shape = tuple(int(x) for x in comp[base + "_shape"].tolist())
        state[base] = decoder(comp[k], shape).to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    model.tie_weights()
    bad_missing = [k for k in missing if k != "lm_head.weight"]
    if bad_missing or unexpected:
        raise RuntimeError("reload key mismatch: missing %s unexpected %s" % (bad_missing[:5], unexpected[:5]))
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise RuntimeError("lm_head is not tied to embed_tokens after reload")
    for name, m in lbl_modules(model, LBL):
        m._binarized = True
        if int(m._split_dim_final.item()) != int(m.split_dim):
            raise RuntimeError("%s split_dim %d differs from saved %d" % (name, m.split_dim, int(m._split_dim_final.item())))
    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    if meta_left:
        raise RuntimeError("meta tensors left after reload (not zero-filled): %s" % meta_left[:5])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    info = {"decoder": decoder_name, "official_loader_ran": False,
            "load_path": "safetensors load_file, decode packed signs, cast fp32, load_state_dict(assign=True), "
                         "tie_weights, _binarized=True",
            "missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    return model, info


def sign_agreement(torch, mem_model, reloaded, LBL) -> dict:
    agree = total = m2p = 0
    rel = dict(lbl_modules(reloaded, LBL))
    for name, m in lbl_modules(mem_model, LBL):
        r = rel[name]
        for fname in FACTOR_NAMES:
            p = getattr(m, fname, None)
            if p is None:
                continue
            a = m.quantize(p.data).to("cpu")
            b = getattr(r, fname).data.to("cpu")
            agree += int((a == b).sum())
            total += a.numel()
            m2p += int(((a == -1) & (b == 1)).sum())
    return {"sign_agreement": agree / total if total else None, "signs": total, "minus_to_plus": m2p}


def identity_check(torch, mem_model, reloaded, ids, LBL) -> dict:
    """Same input, CPU fp32, in-memory student versus its packed reload."""
    import torch.nn.functional as F
    mem_model.eval()
    with torch.no_grad():
        la = mem_model(input_ids=ids, use_cache=False).logits
        lb = reloaded(input_ids=ids, use_cache=False).logits
        ce_a = F.cross_entropy(la[:, :-1].reshape(-1, la.size(-1)).float(), ids[:, 1:].reshape(-1))
        ce_b = F.cross_entropy(lb[:, :-1].reshape(-1, lb.size(-1)).float(), ids[:, 1:].reshape(-1))
    out = {"logits_bitwise_equal": bool(torch.equal(la, lb)),
           "logits_max_abs_diff": float((la - lb).abs().max()),
           "loss_in_memory": float(ce_a), "loss_reloaded": float(ce_b),
           "loss_bitwise_equal": bool(torch.equal(ce_a, ce_b)), "device": str(ids.device)}
    out.update(sign_agreement(torch, mem_model, reloaded, LBL))
    return out


def popcount_flips(torch, a, b) -> int:
    x = (a.contiguous() ^ b.contiguous()).view(torch.uint8).to(torch.long)
    table = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.long)
    return int(table[x].sum())


def checkpoint_diff(torch, init_path: str, trained_path: str) -> dict:
    from safetensors.torch import load_file
    a, b = load_file(init_path), load_file(trained_path)
    flips = signs = 0
    num = den = 0.0
    for k in sorted(a):
        if k.endswith("_packed"):
            shape = [int(x) for x in a[k[: -len("_packed")] + "_shape"].tolist()]
            flips += popcount_flips(torch, a[k], b[k])
            signs += shape[0] * shape[1]
        elif k.rsplit(".", 1)[-1] in SCALE_NAMES:
            num += float(((b[k].double() - a[k].double()) ** 2).sum())
            den += float((a[k].double() ** 2).sum())
    return {"sign_flips": flips, "signs": signs, "sign_flip_fraction": flips / signs if signs else None,
            "scale_relative_l2_change": math.sqrt(num / den) if den else None,
            "note": "padding bits are +1 in both files and never counted as flips"}


# ---------------------------------------------------------------------------------------------
# Stage plumbing
# ---------------------------------------------------------------------------------------------

def out_paths(out: str) -> dict:
    ck = os.path.join(out, "checkpoints")
    return {"manifest": os.path.join(out, "manifest.json"), "windows": os.path.join(out, "windows.json"),
            "baseline": os.path.join(out, "baseline.json"), "feasibility": os.path.join(out, "feasibility.json"),
            "train": os.path.join(out, "train.json"), "steps": os.path.join(out, "train_steps.jsonl"),
            "evaluate": os.path.join(out, "evaluate.json"), "ckpt": ck,
            "frozen": os.path.join(ck, "frozen_base.safetensors"),
            "init": os.path.join(ck, "init", "compressed_layers.safetensors"),
            "trained": os.path.join(ck, "trained", "compressed_layers.safetensors"),
            "config_dir": os.path.join(ck, "config_and_tokenizer"),
            "ckpt_manifest": os.path.join(ck, "checkpoint_manifest.json")}


def load_frozen_windows(out: str) -> Tuple[dict, dict, str]:
    paths = out_paths(out)
    if not (os.path.exists(paths["manifest"]) and os.path.exists(paths["windows"])):
        raise PreconditionError("run the prepare stage first (manifest.json and windows.json in %s)" % out)
    manifest = read_json(paths["manifest"])
    if sha256_file(paths["windows"]) != manifest["windows_json_sha256"]:
        raise PreconditionError("windows.json does not match the frozen manifest hash")
    windows = read_json(paths["windows"])
    for split in SPLITS:
        ws = windows[split]
        if len(ws) != manifest["splits"][split]["n_windows"]:
            raise PreconditionError("window count changed for %s" % split)
        for w, mw in zip(ws, manifest["splits"][split]["windows"]):
            if hash_tokens(w["token_ids"]) != mw["sha256"] or w["sha256"] != mw["sha256"]:
                raise PreconditionError("window %s/%d does not match manifest" % (split, w["index"]))
    return manifest, windows, sha256_file(paths["manifest"])


def stage_header(stage: str, args, manifest_sha: Optional[str] = None) -> dict:
    return {"schema": SCHEMA, "stage": stage, "tier": TIER_PILOT, "started_unix": time.time(),
            "manifest_sha256": manifest_sha, "args": public_args(args), "args_path_policy": ARGS_PATH_POLICY}


def record_failure(res: dict, exc: BaseException) -> None:
    import traceback
    res["status"] = "failed"
    res["error"] = "%s: %s" % (type(exc).__name__, exc)
    res["traceback"] = traceback.format_exc()


def finish(result: dict, path: str, guard: Optional[ResourceGuard]) -> int:
    result["finished_unix"] = time.time()
    result["peak_rss_os"] = peak_rss_os()
    if guard is not None:
        result["guard"] = guard.summary()
    write_json(path, result)
    print(public_path(path))
    print("status: %s" % result.get("status"))
    if result.get("error"):
        print("error: %s" % result["error"], file=sys.stderr)
    return {"completed": 0, "failed": 1, "stopped_by_guard": 3}.get(result.get("status"), 2)


def find_split_files(data_dir: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {s: [] for s in SPLITS}
    for root, _dirs, files in os.walk(data_dir):
        rel = os.path.relpath(root, data_dir)
        in_cfg = DATASET_CONFIG in rel.split(os.sep) or os.path.basename(root) == DATASET_CONFIG
        for f in files:
            for s in SPLITS:
                if f.startswith(s + "-") and f.endswith(".parquet") and (in_cfg or root == data_dir):
                    found[s].append(os.path.join(root, f))
    for s in SPLITS:
        found[s].sort()
        if not found[s]:
            raise PreconditionError("no %s-*.parquet for %s under %s" % (s, DATASET_CONFIG, data_dir))
    return found


# ---------------------------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------------------------

def stage_prepare(args) -> int:
    paths = out_paths(args.out)
    if os.path.exists(paths["manifest"]):
        raise PreconditionError("manifest already frozen at %s; use a new --out for a new experiment" % paths["manifest"])
    force_offline_env()
    files = find_split_files(args.data_dir)
    # HF cache snapshot entries are symlinks into blobs/, so read the revision from the link path.
    revs = {parse_snapshot_revision(os.path.abspath(p)) for ps in files.values() for p in ps}
    if revs == {DATASET_REVISION}:
        rev_status = "verified_from_hf_cache_snapshot_path"
    elif revs == {None} and args.accept_unverified_dataset_revision:
        rev_status = "UNVERIFIED_no_snapshot_path_owner_override"
    else:
        raise PreconditionError("dataset revision evidence %s does not equal pinned %s (no override for mismatches; "
                                "--accept-unverified-dataset-revision only covers paths without a snapshot)"
                                % (sorted(str(r) for r in revs), DATASET_REVISION))
    import pyarrow.parquet as pq
    rows: Dict[str, List[str]] = {}
    file_recs = {}
    for s in SPLITS:
        rows[s] = []
        file_recs[s] = []
        for p in files[s]:
            rows[s].extend(pq.read_table(p, columns=["text"]).column("text").to_pylist())
            file_recs[s].append(file_record(p))
        if len(rows[s]) != EXPECTED_ROWS[s]:
            raise PreconditionError("%s has %d rows, expected %d" % (s, len(rows[s]), EXPECTED_ROWS[s]))
    model_prov = model_provenance(args.model_dir, verify_weights=False)
    tok = load_tokenizer(args.model_dir)
    tokenize = lambda text: tok(text)["input_ids"]  # native defaults, no BOS/EOS override
    splits = make_split_windows(rows, tokenize, PILOT_WINDOWS, SEQ_LEN)
    special_ids = set(tok.all_special_ids)
    windows_out = {}
    for s in SPLITS:
        info = splits[s]
        info["parquet_files"] = file_recs[s]
        info["rows_total"] = len(rows[s])
        info["special_tokens_in_windows"] = sum(1 for w in info["windows"] for t in w["token_ids"] if t in special_ids)
        info["first_token_is_bos"] = (tok.bos_token_id is not None
                                      and info["windows"][0]["token_ids"][0] == tok.bos_token_id)
        windows_out[s] = info["windows"]
        info["windows"] = [{k: v for k, v in w.items() if k != "token_ids"} for w in info["windows"]]
    write_json(paths["windows"], windows_out)
    manifest = {
        "schema": SCHEMA, "stage": "prepare", "frozen": True, "created_unix": time.time(),
        "purpose": "wiring and resource pilot windows; not a statistical benchmark",
        "dataset": {"repo": DATASET_REPO, "config": DATASET_CONFIG, "revision_pinned": DATASET_REVISION,
                    "revision_evidence": sorted(str(r) for r in revs), "revision_status": rev_status,
                    "license_note": "card metadata cc-by-sa-3.0 and gfdl; card text says CC BY-SA 4.0",
                    "source": "local parquet files only, read with pyarrow; no network"},
        "split_policy": {"train": "QAT training only", "validation": "development only",
                         "test": "frozen held-out, evaluated in the evaluate stage"},
        "window_policy": {"seq_len": SEQ_LEN, "counts": PILOT_WINDOWS, "join_rule": repr(JOIN_RULE),
                          "selection": "contiguous non-overlapping windows from token 0 of each split's own "
                                       "row prefix; splits are tokenized separately, so no window crosses splits",
                          "tokenize_margin_tokens": TOKENIZE_MARGIN,
                          "scoring": "first token of each window is not scored", "seed": args.seed,
                          "seed_use": "window selection is deterministic; seed drives SVD init and train order"},
        "tokenizer": tokenizer_record(tok), "model": model_prov, "splits": splits,
        "windows_json_sha256": sha256_file(paths["windows"]),
    }
    write_json(paths["manifest"], manifest)
    print(public_path(paths["manifest"]))
    print("status: completed")
    return 0


def stage_baseline(args) -> int:
    paths = out_paths(args.out)
    manifest, windows, msha = load_frozen_windows(args.out)
    torch, device = setup_torch(args)
    guard = make_guard(args, torch, device)
    res = stage_header("baseline", args, msha)
    res["environment"] = torch_environment(torch, device, args)
    try:
        guard.check("start")
        res["model"] = model_provenance(args.model_dir)
        t = time.perf_counter()
        model = load_base_model(torch, args.model_dir).to(device)
        sync(torch, device)
        res["load_seconds"] = time.perf_counter() - t
        res["resident_storage"] = {"total_unique_bytes": tensor_storage_bytes(list(model.parameters()) + list(model.buffers()))}
        guard.check("model loaded")
        res["validation"] = eval_windows(torch, model, windows["validation"], device, guard, "validation")
        res["test"] = eval_windows(torch, model, windows["test"], device, guard, "test")
        res["kv_cache"] = kv_cache_bytes(torch, model, window_tensor(torch, windows["test"][:1], device))
        tok = load_tokenizer(args.model_dir)
        res["generation"] = generate_fixed(torch, model, tok, device, args.max_new_tokens)
        res["status"] = "completed"
    except GuardStop as e:
        res["status"], res["stop_reason"] = "stopped_by_guard", str(e)
    except PreconditionError:
        raise
    except Exception as e:  # recorded, never reported as completed
        record_failure(res, e)
    res["claims"] = claims_block(False, "baseline stage trains nothing")
    return finish(res, paths["baseline"], guard)


def build_teacher_student(torch, args, up, guard, res):
    cfg = config_from_args(args)
    t = time.perf_counter()
    teacher = load_base_model(torch, args.model_dir)
    res["teacher_load_seconds"] = time.perf_counter() - t
    guard.check("teacher loaded")
    res["teacher_fingerprint_before"] = params_fingerprint(torch, teacher)
    student = clone_student(teacher)
    milestones = []
    res["conversion_milestones"] = milestones
    t = time.perf_counter()
    records = convert_modules(torch, student, up, cfg, do_train=True, guard=guard, milestones=milestones)
    res["conversion_seconds"] = time.perf_counter() - t
    res["converted_modules"] = records
    LBL = up["LittleBitLinear"]
    trainable = trainable_factor_params(student, LBL)
    teacher_ids = {id(p) for p in teacher.parameters()}
    shared = [n for n, p in trainable if id(p) in teacher_ids]
    if shared:
        raise RuntimeError("trainable tensors shared with the teacher: %s" % shared[:5])
    n_w = sum(r["in_features"] * r["out_features"] for r in records)
    res["conversion_summary"] = {
        "modules": len(records), "converted_weight_count": n_w,
        "advertised_bpw_upstream_formula": sum(r["eff_bit_actual"] * r["in_features"] * r["out_features"]
                                               for r in records) / n_w,
        "trainable_tensors": len(trainable), "trainable_parameters": sum(p.numel() for _, p in trainable),
        "frozen": "embeddings, tied lm_head, norms and biases frozen; only LittleBit factors and scales train",
        "teacher_sharing": "student shares frozen tensors with the teacher by reference; no trainable tensor is shared",
        "skipped": "lm_head and embed_tokens are outside transformer blocks and are never converted"}
    res["config"] = cfg
    return teacher, student, trainable, cfg


def train_steps(torch, args, teacher, student, trainable, train_windows, device, guard, steps: int,
                log_path: Optional[str], step_log: list) -> List[float]:
    params = [p for _, p in trainable]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False)
    order = list(range(len(train_windows)))
    random.Random(args.seed).shuffle(order)
    if steps * args.batch_size > len(order):
        raise PreconditionError("%d steps x batch %d exceeds %d frozen train windows; no silent reuse"
                                % (steps, args.batch_size, len(order)))
    want_hidden = args.l2l_scale > 0
    student.train()
    teacher.eval()
    times = []
    for step in range(steps):
        guard.check("before step %d" % step)
        if times and guard.remaining() < max(times):
            raise GuardStop("remaining %.1fs is less than the slowest step %.1fs before step %d"
                            % (guard.remaining(), max(times), step))
        idx = order[step * args.batch_size:(step + 1) * args.batch_size]
        ids = window_tensor(torch, [train_windows[i] for i in idx], device)
        lr_t = cosine_lr(args.lr, step, steps)
        for g in opt.param_groups:
            g["lr"] = lr_t
        t0 = time.perf_counter()
        with torch.no_grad():
            t_out = teacher(input_ids=ids, use_cache=False, output_hidden_states=want_hidden)
        s_out = student(input_ids=ids, use_cache=False, output_hidden_states=want_hidden)
        losses = distill_losses(torch, s_out.logits, t_out.logits, ids, args.kd_weight, args.ce_weight,
                                args.l2l_scale, s_out.hidden_states if want_hidden else None,
                                t_out.hidden_states if want_hidden else None)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip if args.grad_clip > 0 else float("inf"))
        opt.step()
        sync(torch, device)
        dt = time.perf_counter() - t0
        times.append(dt)
        rec = {"step": step, "window_indices": idx, "window_sha256": [train_windows[i]["sha256"] for i in idx],
               "lr": lr_t, "loss": float(losses["loss"]), "kl": float(losses["kl"]), "ce": float(losses["ce"]),
               "l2l": float(losses["l2l"]), "grad_norm_pre_clip": float(gn), "dt_s": dt, "seed": args.seed}
        rec.update(guard.snapshot())
        step_log.append(rec)
        if log_path:
            append_jsonl(log_path, rec)
        print("step %d loss %.4f kl %.4f ce %.4f gn %.3f dt %.2fs rss %.2f GiB"
              % (step, rec["loss"], rec["kl"], rec["ce"], rec["grad_norm_pre_clip"], dt, rec["rss_bytes"] / GIB))
        del t_out, s_out, losses
        if not all(math.isfinite(rec[k]) for k in ("loss", "kl", "ce", "grad_norm_pre_clip")):
            raise GuardStop("non-finite loss or gradient norm at step %d" % step)
        guard.check("after step %d" % step)
    del opt
    gc.collect()
    student.eval()
    return times


def stage_feasibility(args) -> int:
    paths = out_paths(args.out)
    manifest, windows, msha = load_frozen_windows(args.out)
    torch, device = setup_torch(args)
    guard = make_guard(args, torch, device)
    res = stage_header("feasibility", args, msha)
    res["environment"] = torch_environment(torch, device, args)
    res["note"] = "two real optimizer steps on a real conversion; nothing is saved and nothing is trained"
    try:
        guard.check("start")
        up = load_upstream_api()
        res["upstream"] = {"provenance": up["provenance"], "file_sha256": up["file_sha256"], "notes": up.get("notes")}
        res["model"] = model_provenance(args.model_dir)
        teacher, student, trainable, cfg = build_teacher_student(torch, args, up, guard, res)
        teacher.to(device)
        student.to(device)
        step_log = []
        times = train_steps(torch, args, teacher, student, trainable, windows["train"], device, guard, 2, None, step_log)
        res["step_log"] = step_log
        res["estimate"] = feasibility_estimate(res["conversion_seconds"], times, args.steps, args.max_seconds)
        res["status"] = "completed"
    except GuardStop as e:
        res["status"], res["stop_reason"] = "stopped_by_guard", str(e)
    except PreconditionError:
        raise
    except Exception as e:  # recorded, never reported as completed
        record_failure(res, e)
    res["claims"] = claims_block(False, "feasibility runs 2 steps only and saves no checkpoint")
    return finish(res, paths["feasibility"], guard)


def save_config_and_tokenizer(model_dir: str, dest: str, cfg: dict) -> dict:
    from transformers import AutoConfig
    os.makedirs(dest, exist_ok=True)
    AutoConfig.from_pretrained(model_dir, local_files_only=True).save_pretrained(dest)
    load_tokenizer(model_dir).save_pretrained(dest)
    write_json(os.path.join(dest, "littlebit_config.json"), cfg)
    return {name: file_record(os.path.join(dest, name)) for name in sorted(os.listdir(dest))}


def stage_train(args) -> int:
    paths = out_paths(args.out)
    manifest, windows, msha = load_frozen_windows(args.out)
    if os.path.exists(paths["train"]) or os.path.exists(paths["ckpt"]):
        raise PreconditionError("train outputs already exist in %s; use a new --out" % args.out)
    torch, device = setup_torch(args)
    guard = make_guard(args, torch, device)
    res = stage_header("train", args, msha)
    res["environment"] = torch_environment(torch, device, args)
    res["objective"] = {"kd": "KL(teacher || student), teacher logits detached, token mean", "kd_weight": args.kd_weight,
                        "ce_weight": args.ce_weight, "l2l_scale": args.l2l_scale,
                        "optimizer": "AdamW betas (0.9, 0.999) eps 1e-8 weight_decay 0", "lr": args.lr,
                        "schedule": "cosine, no warmup", "grad_clip": args.grad_clip,
                        "steps": args.steps, "batch_size": args.batch_size, "seq_len": SEQ_LEN}
    step_log: list = []
    res["step_log_path"] = public_path(paths["steps"])
    completed_steps = 0
    try:
        guard.check("start")
        up = load_upstream_api()
        LBL = up["LittleBitLinear"]
        res["upstream"] = {"provenance": up["provenance"], "file_sha256": up["file_sha256"], "notes": up.get("notes")}
        res["model"] = model_provenance(args.model_dir)
        teacher, student, trainable, cfg = build_teacher_student(torch, args, up, guard, res)
        os.makedirs(os.path.dirname(paths["init"]), exist_ok=True)
        meta = {"schema": SCHEMA, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "upstream_commit": up["provenance"]["pinned_commit"], "manifest_sha256": msha,
                "littlebit": json.dumps(cfg, sort_keys=True)}
        ckpt = {"frozen_base": save_frozen_base(torch, student, LBL, paths["frozen"], meta)}
        ckpt["init"] = save_compressed(torch, student, LBL, paths["init"], dict(meta, state="initialized_no_qat"))
        ckpt["config_and_tokenizer"] = save_config_and_tokenizer(args.model_dir, paths["config_dir"], cfg)
        write_json(paths["ckpt_manifest"], ckpt)
        guard.check("init checkpoint saved")
        dec = corrected_decoder()
        dev_ids = window_tensor(torch, windows["validation"][:1], torch.device("cpu"))
        reloaded, info = load_packed_student(torch, args.model_dir, paths["frozen"], paths["init"], up, cfg, dec,
                                             "corrected_reference_decoder")
        res["identity_init"] = dict(identity_check(torch, student, reloaded, dev_ids, LBL), reload=info)
        del reloaded
        gc.collect()
        guard.check("init identity checked")
        teacher.to(device)
        student.to(device)
        res["dev_validation_init_in_memory"] = eval_windows(torch, student, windows["validation"], device, guard,
                                                            "validation_init")
        times = train_steps(torch, args, teacher, student, trainable, windows["train"], device, guard, args.steps,
                            paths["steps"], step_log)
        completed_steps = len(times)
        res["dev_validation_trained_in_memory"] = eval_windows(torch, student, windows["validation"], device, guard,
                                                               "validation_trained")
        teacher.to("cpu")
        student.to("cpu")
        res["teacher_fingerprint_after"] = params_fingerprint(torch, teacher)
        res["teacher_preserved"] = res["teacher_fingerprint_after"] == res["teacher_fingerprint_before"]
        if not res["teacher_preserved"]:
            raise RuntimeError("frozen teacher parameters changed during training")
        os.makedirs(os.path.dirname(paths["trained"]), exist_ok=True)
        ckpt["trained"] = save_compressed(torch, student, LBL, paths["trained"],
                                          dict(meta, state="qat_%d_steps" % completed_steps))
        write_json(paths["ckpt_manifest"], ckpt)
        reloaded, info = load_packed_student(torch, args.model_dir, paths["frozen"], paths["trained"], up, cfg, dec,
                                             "corrected_reference_decoder")
        res["identity_trained"] = dict(identity_check(torch, student, reloaded, dev_ids, LBL), reload=info)
        del reloaded
        gc.collect()
        if args.upstream_decoder_variant:
            bad, info = load_packed_student(torch, args.model_dir, paths["frozen"], paths["trained"], up, cfg,
                                            up["binary_unpacker"], "upstream_binary_unpacker")
            res["upstream_decoder_variant_trained"] = dict(identity_check(torch, student, bad, dev_ids, LBL),
                                                           reload=info, tier=TIER_DEFECT,
                                                           note="in-memory only, never saved; not a quality result")
            del bad
            gc.collect()
        res["checkpoint_diff_init_to_trained"] = checkpoint_diff(torch, paths["init"], paths["trained"])
        res["resident_storage_student_in_memory"] = resident_storage(student, LBL)
        res["checkpoints"] = ckpt
        res["status"] = "completed"
    except GuardStop as e:
        res["status"], res["stop_reason"] = "stopped_by_guard", str(e)
    except PreconditionError:
        raise
    except Exception as e:  # recorded, never reported as completed
        record_failure(res, e)
    res["step_log"] = step_log
    res["completed_steps"] = len(step_log)
    trained = (res.get("status") == "completed" and len(step_log) == args.steps
               and "trained" in res.get("checkpoints", {}))
    res["claims"] = claims_block(trained, "completed %d of %d logged steps; trained checkpoint %s"
                                 % (len(step_log), args.steps, "saved" if trained else "absent"))
    return finish(res, paths["train"], guard)


def verify_checkpoint_files(paths: dict, ckpt: dict) -> dict:
    out = {}
    for key, path in (("frozen_base", paths["frozen"]), ("init", paths["init"]), ("trained", paths["trained"])):
        if key not in ckpt:
            continue
        actual = sha256_file(path)
        out[key] = {"sha256": actual, "bytes": os.path.getsize(path), "matches_train_record": actual == ckpt[key]["sha256"]}
        if actual != ckpt[key]["sha256"]:
            raise PreconditionError("%s checkpoint sha256 changed since training" % key)
    return out


def stage_evaluate(args) -> int:
    paths = out_paths(args.out)
    manifest, windows, msha = load_frozen_windows(args.out)
    if not os.path.exists(paths["ckpt_manifest"]):
        raise PreconditionError("no checkpoint manifest; run the train stage first")
    ckpt = read_json(paths["ckpt_manifest"])
    train_res = read_json(paths["train"]) if os.path.exists(paths["train"]) else {}
    torch, device = setup_torch(args)
    guard = make_guard(args, torch, device)
    res = stage_header("evaluate", args, msha)
    res["environment"] = torch_environment(torch, device, args)
    res["heldout_split"] = "test"
    try:
        guard.check("start")
        up = load_upstream_api()
        LBL = up["LittleBitLinear"]
        res["upstream"] = {"provenance": up["provenance"], "file_sha256": up["file_sha256"]}
        res["model"] = model_provenance(args.model_dir)
        res["checkpoint_files"] = verify_checkpoint_files(paths, ckpt)
        cfg = read_json(os.path.join(paths["config_dir"], "littlebit_config.json"))
        tok = load_tokenizer(args.model_dir)
        test = windows["test"]
        first = window_tensor(torch, test[:1], device)

        base = load_base_model(torch, args.model_dir).to(device)
        inv_freq_ref = base.model.rotary_emb.inv_freq.detach().to("cpu").clone()
        res["original"] = {"test": eval_windows(torch, base, test, device, guard, "test_original"),
                           "resident_storage": {"total_unique_bytes": tensor_storage_bytes(
                               list(base.parameters()) + list(base.buffers()))},
                           "kv_cache": kv_cache_bytes(torch, base, first),
                           "generation": generate_fixed(torch, base, tok, device, args.max_new_tokens)}
        if os.path.exists(paths["baseline"]):
            bl = read_json(paths["baseline"])
            if bl.get("status") == "completed":
                res["original"]["matches_baseline_stage"] = all(
                    a["nll_sum"] == b["nll_sum"] for a, b in zip(bl["test"]["windows"], res["original"]["test"]["windows"]))
        del base
        gc.collect()

        dec = corrected_decoder()
        variants = [("initialized_no_qat", "init", dec, "corrected_reference_decoder", TIER_PILOT)]
        if "trained" in ckpt:
            variants.append(("qat_trained", "trained", dec, "corrected_reference_decoder", TIER_PILOT))
            if args.upstream_decoder_variant:
                variants.append(("qat_trained_upstream_decoder", "trained", up["binary_unpacker"],
                                 "upstream_binary_unpacker", TIER_DEFECT))
        else:
            res["qat_trained"] = {"status": "not available: no trained checkpoint"}
        for label, ckey, decoder, dname, tier in variants:
            guard.check("load %s" % label)
            model, info = load_packed_student(torch, args.model_dir, paths["frozen"], paths[ckey], up, cfg,
                                              decoder, dname)
            info["rotary_inv_freq_equals_original"] = bool(torch.equal(model.model.rotary_emb.inv_freq, inv_freq_ref))
            model.to(device)
            entry = {"tier": tier, "reload": info, "checkpoint": res["checkpoint_files"][ckey],
                     "checkpoint_accounting": {k: v for k, v in ckpt[ckey].items() if k != "path"},
                     "test": eval_windows(torch, model, test, device, guard, "test_" + label),
                     "resident_storage": resident_storage(model, LBL)}
            entry["paired_vs_original"] = paired_window_diffs(res["original"]["test"]["windows"], entry["test"]["windows"])
            if tier == TIER_PILOT:
                entry["kv_cache"] = kv_cache_bytes(torch, model, first)
                entry["generation"] = generate_fixed(torch, model, tok, device, args.max_new_tokens)
            else:
                entry["note"] = "decoder defect variant, in memory only, never saved; not a quality result"
            res[label] = entry
            del model
            gc.collect()
        res["training_record"] = {"status": train_res.get("status"), "completed_steps": train_res.get("completed_steps"),
                                  "planned_steps": (train_res.get("args") or {}).get("steps"),
                                  "claims": train_res.get("claims")}
        res["status"] = "completed"
    except GuardStop as e:
        res["status"], res["stop_reason"] = "stopped_by_guard", str(e)
    except PreconditionError:
        raise
    except Exception as e:  # recorded, never reported as completed
        record_failure(res, e)
    trained = bool((train_res.get("claims") or {}).get("model_trained")) and "qat_trained" in res \
        and "test" in res.get("qat_trained", {})
    res["claims"] = claims_block(trained, "trained claim inherited from train.json step log and checkpoint sha256")
    return finish(res, paths["evaluate"], guard)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: %(default)s)")
    common.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="local pinned checkpoint directory")
    common.add_argument("--seed", type=int, default=SEED)
    common.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="cpu (default) or mps float32; no silent fallback")
    common.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch intra-op threads")
    common.add_argument("--max-rss-gib", type=float, default=DEFAULT_MAX_RSS_GIB, help="stop if process RSS exceeds this")
    common.add_argument("--min-available-gib", type=float, default=DEFAULT_MIN_AVAILABLE_GIB,
                        help="stop if system available memory drops below this")
    common.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="wall clock limit per stage")
    common.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="greedy tokens per fixed prompt (capped at %d)" % MAX_NEW_TOKENS_CAP)

    qat = argparse.ArgumentParser(add_help=False)
    qat.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    qat.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    qat.add_argument("--lr", type=float, default=DEFAULT_LR)
    qat.add_argument("--eff-bit", type=float, default=DEFAULT_EFF_BIT, help="upstream eff_bit target")
    qat.add_argument("--residual", dest="residual", action="store_true", default=True)
    qat.add_argument("--no-residual", dest="residual", action="store_false")
    qat.add_argument("--kd-weight", type=float, default=1.0)
    qat.add_argument("--ce-weight", type=float, default=1.0)
    qat.add_argument("--l2l-scale", type=float, default=0.0,
                     help="optional upstream-style intermediate hidden-state MSE (upstream default 10.0; off here)")
    qat.add_argument("--grad-clip", type=float, default=1.0)
    qat.add_argument("--upstream-decoder-variant", action="store_true",
                     help="also reload with the defective upstream unpacker, labeled separately, never saved")

    p = argparse.ArgumentParser(
        prog="quality_pilot.py",
        description="Bounded local LittleBit quality pilot on Qwen2.5-0.5B (wiring and resource pilot, "
                    "not a benchmark). Offline, local files only.",
        epilog="Stages in order: prepare, baseline, feasibility, train, evaluate. Exit codes: 0 completed, "
               "1 failed (recorded in the stage JSON), 2 precondition failure, 3 stopped by resource guard.")
    sub = p.add_subparsers(dest="stage", required=True)
    sp = sub.add_parser("prepare", parents=[common], help="freeze WikiText-2 windows from local parquet files")
    sp.add_argument("--data-dir", required=True, help="local directory with wikitext-2-raw-v1 {split}-*.parquet")
    sp.add_argument("--accept-unverified-dataset-revision", action="store_true",
                    help="allow files without an HF snapshot path; recorded as UNVERIFIED in the manifest")
    sp.set_defaults(func=stage_prepare)
    sp = sub.add_parser("baseline", parents=[common], help="original model NLL and fixed generations")
    sp.set_defaults(func=stage_baseline)
    sp = sub.add_parser("feasibility", parents=[common, qat], help="real conversion plus 2 real steps, then estimate")
    sp.set_defaults(func=stage_feasibility)
    sp = sub.add_parser("train", parents=[common, qat], help="bounded QAT run with checkpoints and identity check")
    sp.set_defaults(func=stage_train)
    sp = sub.add_parser("evaluate", parents=[common, qat], help="held-out NLL after packed reload")
    sp.set_defaults(func=stage_evaluate)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_new_tokens", 0) > MAX_NEW_TOKENS_CAP:
        args.max_new_tokens = MAX_NEW_TOKENS_CAP
    try:
        return args.func(args)
    except PreconditionError as e:
        print("precondition failed: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
