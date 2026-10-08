"""Matched microbenchmark and serialization-fidelity runner for actual upstream LittleBitLinear.

UPSTREAM_IMPORT tier only: requires torch. Without torch it writes a record saying so and exits 0.

For each deterministic configuration it:
  1. builds an actual upstream LittleBitLinear from a seeded nn.Linear (do_train=True init),
  2. computes a dense reconstructed matched baseline from the same binarized factors and scales,
  3. serializes with the upstream state_dict override (packed U/V) into an in-memory buffer,
  4. emulates the upstream packed load path twice: once with the upstream binary_unpacker and
     once with the corrected reference decoder, then load_state_dict(assign=True) into an empty
     upstream module and sets _binarized = True, as quant_util.load_quantized_model does,
  5. reports sign agreement, output fidelity, byte counts, and median/p95 timings.

Load emulation note: quant_util._load_and_process_state_dict reads safetensors from disk and
needs transformers; it is not called. Only its unpack step (upstream binary_unpacker followed
by a cast to torch_dtype) is reproduced. Synthetic layers carry no quality meaning.

    python src/bench.py --out results
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402

DEFAULT_SHAPES = ((256, 256), (512, 1024), (1024, 512))
DEFAULT_TARGETS = (1.0, 0.1)
DEFAULT_SEEDS = (0, 1, 2)
H5_REL_FRO_TOL = 1e-6


def percentile_nearest_rank(samples: List[float], pct: float) -> float:
    ordered = sorted(samples)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


def time_call(fn: Callable[[], object], warmup: int, iters: int, sync: Callable[[], None]) -> dict:
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        sync()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    return {"median_us": statistics.median(samples), "p95_us": percentile_nearest_rank(samples, 95.0),
            "min_us": min(samples), "iters": iters, "warmup": warmup}


def torch_reference_unpack(packed, shape):
    """Corrected decoder in torch: mask to uint32 range in int64, right shift, lsb first."""
    import torch
    n_rows, n_cols = (int(shape[0]), int(shape[1]))
    shifts = torch.arange(32, dtype=torch.int64, device=packed.device)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    bits = (words.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(n_rows, -1)[:, :n_cols]
    return (1 - 2 * bits).to(torch.int8)


def _factor_names(residual: bool) -> List[str]:
    return ["U", "V", "U_R", "V_R"] if residual else ["U", "V"]


def build_layer(up: dict, a: int, b: int, eff: float, residual: bool, seed: int, do_train: bool):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    lin = nn.Linear(a, b, bias=False)
    lin.__class__ = up["LittleBitLinear"]
    lin.__quant_convert__(do_train=do_train, quant_func=up["STEBinary"], eff_bit=eff,
                          residual=residual, split_dim=1024, min_split_dim=8)
    lin.eval()
    return lin


def dense_reconstruction(m):
    """Same reconstruction formula the upstream module uses to form its residual target."""
    import torch
    with torch.no_grad():
        def path(U, V, u1, u2, v1, v2):
            return (m.quantize(U) * (u1.t() @ u2)) @ (m.quantize(V) * (v1.t() @ v2))
        W = path(m.U, m.V, m.u1, m.u2, m.v1, m.v2)
        if m.residual:
            W = W + path(m.U_R, m.V_R, m.u1_R, m.u2_R, m.v1_R, m.v2_R)
    return W


def apply_save_profile(sd: dict, profile: str) -> dict:
    import torch
    if profile == "raw_state_dict_fp32":
        return dict(sd)
    if profile == "main_py_bf16":  # main.py save_artifacts cast rule
        return {k: (v.to(torch.bfloat16) if ("packed" not in k and "shape" not in k
                                              and v.dtype == torch.float32) else v)
                for k, v in sd.items()}
    raise ValueError(profile)


def emulate_packed_load(sd: dict, decoder, load_dtype) -> dict:
    out = {}
    names = {k[: -len("_packed")] for k in sd if k.endswith("_packed")}
    for k, v in sd.items():
        if not (k.endswith("_packed") or k.endswith("_shape")):
            out[k] = v
    for name in names:
        shape = tuple(sd[name + "_shape"].tolist())
        out[name] = decoder(sd[name + "_packed"], shape).to(load_dtype)
    return out


def tensor_bytes(tensors) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors))


def run_config(up: dict, a: int, b: int, eff: float, residual: bool, seed: int, profile: str,
               batch: int, warmup: int, iters: int) -> dict:
    import torch
    sync = (lambda: None)
    mem = build_layer(up, a, b, eff, residual, seed, do_train=True)
    for p in mem.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(10_000 + seed)
    x = torch.randn(batch, a, generator=gen, dtype=torch.float32)

    with torch.no_grad():
        y_mem = mem(x)
        W = dense_reconstruction(mem)
        y_dense = x @ W.t()

    sd = mem.state_dict()
    buf = io.BytesIO()
    torch.save(apply_save_profile(sd, profile), buf)
    serialized_bytes = buf.tell()
    buf.seek(0)
    sd_loaded = torch.load(buf, map_location="cpu", weights_only=True)
    packed_keys = [k for k in sd_loaded if k.endswith("_packed")]

    row = {"tier": audit.TIER_UPSTREAM, "in_features": a, "out_features": b, "target_bpw": eff,
           "residual": residual, "seed": seed, "save_profile": profile, "batch": batch,
           "split_dim": int(mem.split_dim), "upstream_eff_bits_actual": float(mem.eff_bit_actual),
           "serialized_container_bytes": serialized_bytes,
           "serialized_tensor_bytes": tensor_bytes(sd_loaded.values()),
           "packed_sign_bytes": tensor_bytes(sd_loaded[k] for k in packed_keys),
           "dense_vs_mem_max_abs": float((y_dense - y_mem).abs().max()),
           "dense_vs_mem_rel_fro": float((y_dense - y_mem).norm() / y_mem.norm())}

    decoders = {"upstream_unpacker": up["binary_unpacker"], "reference_decoder": torch_reference_unpack}
    loaded_modules = {}
    for dname, decoder in decoders.items():
        state = emulate_packed_load(sd_loaded, decoder, torch.float32)
        lm = build_layer(up, a, b, eff, residual, seed, do_train=False)
        missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
        lm._binarized = True  # as load_quantized_model does for packed checkpoints
        loaded_modules[dname] = lm
        with torch.no_grad():
            y = lm(x)
        agree, total, m2p = 0, 0, 0
        for fname in _factor_names(residual):
            orig = mem.quantize(getattr(mem, fname).data)
            got = getattr(lm, fname).data
            agree += int((orig == got).sum())
            total += orig.numel()
            m2p += int(((orig == -1) & (got == 1)).sum())
        row.update({
            "%s_missing_keys" % dname: len(missing),
            "%s_unexpected_keys" % dname: len(unexpected),
            "%s_sign_agreement" % dname: agree / total,
            "%s_minus_to_plus" % dname: m2p,
            "%s_out_max_abs" % dname: float((y - y_mem).abs().max()),
            "%s_out_rel_fro" % dname: float((y - y_mem).norm() / y_mem.norm()),
            "%s_out_bitwise_equal" % dname: bool(torch.equal(y, y_mem)),
            "%s_resident_param_bytes" % dname: tensor_bytes(lm.parameters()),
            "%s_resident_buffer_bytes" % dname: tensor_bytes(lm.buffers()),
        })
    row["h5_applicable"] = profile == "raw_state_dict_fp32"
    row["h5_pass"] = (row["reference_decoder_sign_agreement"] == 1.0
                      and row["reference_decoder_out_rel_fro"] <= H5_REL_FRO_TOL) if row["h5_applicable"] else None
    row["h4_outputs_changed"] = row["upstream_unpacker_out_max_abs"] > 0.0

    # Timings (descriptive only): same x, same dtype, same thread count.
    loaded = loaded_modules["reference_decoder"]
    with torch.no_grad():
        timings = {
            "fwd_in_memory": time_call(lambda: mem(x), warmup, iters, sync),
            "fwd_loaded_binarized": time_call(lambda: loaded(x), warmup, iters, sync),
            "fwd_dense_matched": time_call(lambda: x @ W.t(), warmup, iters, sync),
        }
    u_packed, u_shape = sd_loaded["U_packed"], tuple(sd_loaded["U_shape"].tolist())
    timings["unpack_U_upstream"] = time_call(lambda: up["binary_unpacker"](u_packed, u_shape), warmup, iters, sync)
    timings["unpack_U_reference"] = time_call(lambda: torch_reference_unpack(u_packed, u_shape), warmup, iters, sync)
    for tname, t in timings.items():
        row["%s_median_us" % tname] = t["median_us"]
        row["%s_p95_us" % tname] = t["p95_us"]
    return row


def bench_environment(threads: int) -> dict:
    import torch
    env = audit.environment()
    env.update({"torch_num_threads": torch.get_num_threads(), "requested_threads": threads,
                "cuda_available": bool(torch.cuda.is_available()), "device_used": "cpu",
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "timer": "time.perf_counter_ns", "percentile": "nearest-rank"})
    return env


def run_all(shapes=DEFAULT_SHAPES, targets=DEFAULT_TARGETS, seeds=DEFAULT_SEEDS,
            profiles=("raw_state_dict_fp32", "main_py_bf16"), batch: int = 8, warmup: int = 5,
            iters: int = 50, threads: int = 1) -> dict:
    torch = audit.try_import_torch()
    if torch is None:
        return {"status": "skipped_no_torch", "tier": audit.TIER_UPSTREAM,
                "label": "no UPSTREAM_IMPORT evidence produced; nothing here is a measurement",
                "environment": audit.environment()}
    up = audit.load_upstream(include_layer=True)
    if "LittleBitLinear" not in up:
        return {"status": "skipped_layer_unavailable", "reason": up.get("layer_reason"),
                "environment": audit.environment()}
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = []
    for (a, b) in shapes:
        for eff in targets:
            for residual in (False, True):
                for seed in seeds:
                    for profile in profiles:
                        rows.append(run_config(up, a, b, eff, residual, seed, profile, batch, warmup, iters))
    h5_rows = [r for r in rows if r["h5_applicable"]]
    return {
        "status": "ok", "tier": audit.TIER_UPSTREAM,
        "provenance": audit.upstream_commit_check(),
        "upstream_file_sha256": audit.file_hashes(),
        "upstream_import_notes": up.get("notes", []),
        "environment": bench_environment(threads),
        "settings": {"shapes": [list(s) for s in shapes], "targets": list(targets), "seeds": list(seeds),
                     "profiles": list(profiles), "batch": batch, "warmup": warmup, "iters": iters,
                     "h5_rel_fro_tol": H5_REL_FRO_TOL},
        "hypotheses": {
            "H4_all_outputs_changed": all(r["h4_outputs_changed"] for r in rows),
            "H4_all_sign_agreement_below_1": all(r["upstream_unpacker_sign_agreement"] < 1.0 for r in rows),
            "H5_all_pass": all(r["h5_pass"] for r in h5_rows) if h5_rows else None,
        },
        "rows": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LittleBit matched microbenchmark (torch required)")
    ap.add_argument("--out", default=os.path.join(audit.PROJECT_ROOT, "results"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)
    res = run_all(batch=args.batch, warmup=args.warmup, iters=args.iters, threads=args.threads)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "bench.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(path)
    if res.get("rows"):
        path = os.path.join(args.out, "bench.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(audit.rows_to_csv(res["rows"]))
        print(path)
    else:
        print("status: %s" % res.get("status"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
