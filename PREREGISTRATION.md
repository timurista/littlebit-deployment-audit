# Preregistration: LittleBit packed-serialization deployment audit

Written: 2026-10-08, before the full audit sweep. The shift-direction hypothesis and first eight-element counterexample were already discovered and executed independently, as recorded in evidence/first-upstream-counterexample.json. H1 is an exploratory defect confirmation, not a blind preregistered discovery. Remaining predictions precede the full sweep.
Author of audit: Tim Urista, with Claude as implementation assistant.
Upstream under audit: https://github.com/SamsungLabs/LittleBit at commit
`933857ed1443b53fc43a875c2cf64249e3c56f0c` (local shallow checkout in `upstream/LittleBit`).

## Scope and non-claims

* This is a deployment and serialization audit of existing published code. It makes no novelty
  claim. The LittleBit and LittleBit-2 methods are the work of their authors (see LICENSE_NOTES.md),
  and an independent check already identified the current upstream as LittleBit-2.
* No claims about model quality (perplexity, zero-shot accuracy), no end-to-end decode or
  generation claims, and no throughput claims for full models. Synthetic layers in the
  microbenchmark exist only to test whether a serialize, load, forward cycle preserves outputs.
* Two evidence tiers are kept separate in every output:
  * `ALGORITHM_AUDIT`: standard-library arithmetic mirror of the upstream expressions. This
    tests the algorithm as written, not the shipped tensors or kernels. It is not a model
    replication.
  * `UPSTREAM_IMPORT`: the actual upstream `binary_packer`, `binary_unpacker`, and
    `LittleBitLinear` imported from the pinned checkout. Only available if `torch` is installed.

## Primary question

Does the upstream packed serialization preserve the binary signs and the actual layer outputs,
and where do loaded (resident) storage and the advertised bits per weight (BPW) diverge?

## Files inspected

`quantization/utils/binary_packer.py`, `quantization/modules/littlebit.py`,
`quantization/modules/attention.py`, `quantization/utils/quant_util.py`, `LICENSE`, plus
`quantization/functions/binary.py` and `main.py` (save path) for context.

## Hypotheses and exact predictions

H1 (unpack shift direction). The upstream unpack expression is
`(word_data.unsqueeze(1) << torch.arange(32)) & 1`. Because a left shift by k >= 1 always clears
bit 0, this yields `word & 1` at in-word position 0 and 0 at positions 1..31. Prediction: a row
decodes exactly if and only if every -1 entry lies at a column index j with j % 32 == 0.
All other -1 entries decode as +1. Signs are only ever flipped from -1 to +1, never the reverse.

H1 numeric predictions, byte-tiled family (column j is -1 iff bit (j % 8) of the byte is set),
all 256 byte values per width, count of rows decoded exactly by the upstream unpacker:

| width | 1 | 7 | 8 | 31 | 32 | 33 | 64 |
|---|---|---|---|---|---|---|---|
| exact rows of 256 | 256 | 4 | 2 | 1 | 1 | 1 | 1 |

One-hot family (single -1 at column j): exact iff j % 32 == 0.
Sign examples: all +1 decodes exactly at every width (a naive all-ones test passes). All -1 at
width 64 decodes to -1 only at columns 0 and 32.

H2 (packer correctness). The upstream packer (lsb first, -1 maps to bit 1, +1 padding maps to
bit 0, int32 two's complement words) is correct. Prediction: both independent reference
decoders (a mask-and-right-shift decoder and a little-endian byte decoder built on `struct`)
recover every original row exactly, for every case above, from the mirror packer and, if torch
is available, from the actual upstream packer.

H3 (mirror fidelity). If torch is available, the actual upstream packer output equals the mirror
packer output word for word, and the actual upstream unpacker output equals the mirror unpacker
output element for element, on every case. Failure of H3 invalidates the ALGORITHM_AUDIT tier
for the failing component and is reported, not suppressed.

H4 (layer outputs, torch only). For actual `LittleBitLinear` layers built with deterministic
seeds, emulating the upstream packed load path (upstream `binary_unpacker`, `assign=True`
load, `_binarized = True`) changes outputs relative to the in-memory module. Prediction: sign
agreement of loaded U and V with the original binarized factors is below 1.0 and outputs differ
(max absolute difference > 0) for every tested configuration.

H5 (corrected decoder, torch only). Replacing only the decode step with the corrected reference
decoder restores outputs. Prediction: sign agreement exactly 1.0 and outputs bitwise equal to the
in-memory module on CPU float32 with a fixed thread count. If not bitwise equal, report the max
absolute difference; any value above 1e-6 relative Frobenius error counts as a failure of H5.

H6 (storage accounting). Using the upstream formulas mirrored from `_estimate_split_dim`,
`_finalize_split_dim`, `_compute_eff_bits`:
* Advertised bits per path count s(a+b) sign bits plus 16(a+b+s) scale bits, that is, one fused
  middle scale vector of length s. The module actually stores two middle vectors (u2 and v1), so
  stored scale bits exceed the advertised scale bits by 16 s per path at 16-bit scales
  (double-path residual mode: 32 s per layer).
* Packed sign storage pads each row of U (s columns) and V (a columns) to a multiple of 32 bits.
  Since s is rounded to a multiple of 8, not 32, U padding is nonzero whenever s % 32 != 0.
* Each packed tensor adds a 16 byte int64 shape tensor; each layer adds three buffers.
* After the upstream packed load, U and V are resident as `torch_dtype` (bf16 or fp32), so the
  resident sign storage is 16x (bf16) or 32x (fp32) the packed sign bits. int8 would be 8x.
* When the min_split_dim floor binds, actual eff bits exceed the target. Prediction for a 64 x 64
  layer at target 0.1 BPW, non residual: s = 8, actual 0.78125 BPW.
* Illustrative 4096 x 4096 shapes: target 0.1 non residual gives s = 184; target 1.0 residual
  gives s = 1000.

## Analysis plan

* Per case: exact roundtrip flag, number of sign errors, first error column, prediction match.
* Per width: count of exact rows, total sign errors, count of prediction mismatches (expected 0).
* Accounting table per layer config: target BPW, s, upstream actual BPW, serialized BPW under two
  save profiles (main.py bf16 save, raw fp32 state_dict), resident BPW for int8, fp16/bf16, fp32
  factors, padding bits, double-path scale overhead bits.
* Microbenchmark: median and p95 forward latency (nearest-rank) for the in-memory module, the
  loaded module, and a dense reconstructed matched baseline using the same quantized factors,
  same inputs, same dtype, same thread count. Timings are descriptive only.

## Decision rules

* H1 is supported if all predictions in the H1 table and families hold under the mirror; it is
  confirmed for shipped code only if H3 also holds under torch.
* Any prediction mismatch is reported verbatim with the first counterexample.
* Raw results are generated later by a separate parent execution. Nothing in this document was
  adjusted after seeing results. Any later deviation must be recorded in a dated addendum below.

## Addenda

### 2026-10-08, after results were received

* Results were generated by a separate parent execution (results/audit_summary.json,
  results/bench.json). Outcomes against the predictions above, without changing them: H1 table and
  families held (0 prediction mismatches); H2 held (0 reference decoder failures); H3 held (0
  packer and 0 unpacker mirror mismatches); H4 held in 72 of 72 configurations; H5 held in 36 of 36
  raw fp32 configurations (bitwise equal). The 36 main.py bf16 profile rows were never part of H5.
* results/audit_summary.json came from the audit.py revision before the prior-evidence block was
  added. Regeneration is pending.
* Post hoc, not preregistered: the one-line patch (patches/binary_unpacker.patch), its regression
  runner (src/patched_regression.py), and direct regression tests for signed multi-row matrices at
  widths 31/32/33/64 and for residual factors U_R and V_R. These are confirmatory engineering
  checks of a fix and are labeled as post hoc wherever reported.
* Model quality remains out of scope here; any pretrained-model pilot gets its own preregistration
  following MODEL_QUALITY_PLAN.md.
