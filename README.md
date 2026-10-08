# LittleBit deployment audit

A narrow audit of the packed serialization path and storage accounting in
[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit) at commit
`933857ed1443b53fc43a875c2cf64249e3c56f0c` (current upstream, includes LittleBit-2).

Primary question: does packed serialization preserve signs and actual outputs, and where do
loaded storage and advertised BPW diverge? Secondary, added later: what does a tiny, bounded,
local pilot on a real pretrained model show (section 5)?

## Scope, read this first

* Sections 1 to 4 are component level: seeded random-weight layers, no model quality.
* Section 5 is a **tiny real-model pilot with a negative quality result**: Qwen2.5-0.5B, 16 QAT
  steps on 2,048 input tokens, scored on 1,016 held-out tokens. It is a wiring and resource pilot,
  not a benchmark, not a replication of any published LittleBit number, and not evidence that the
  compressed model is useful. Qwen2.5-0.5B is smaller than, and outside, the model table of the
  LittleBit papers.
* Loader: the official upstream state-dict loading function in `quant_util.py`
  (`_load_and_process_state_dict`) was imported and executed, unmodified, on a **synthetic
  single-layer file only** (evidence/official-loader-integration.json). The full-model official
  loader `quant_util.load_quantized_model` was **not** run; the benchmark and the pilot emulate its
  packed path: upstream packing, the unpack function (or a corrected decoder), cast to float32,
  `load_state_dict(assign=True)`, `_binarized = True`.
* Full-model reload identity (section 5) compared signs over every packed factor, but logits and
  loss on **one 128-token validation window** per checkpoint only.
* No novelty claim. The defect was first seen in an exploratory run
  (evidence/first-upstream-counterexample.json) before the preregistration, so H1 is an
  exploratory defect confirmation. No claim is made about the papers or their reported results.
* **Rights: no license is granted.** The original code and text of this repository are copyright
  the author, all rights reserved; publication on GitHub does not grant a license.
  LICENSE_PROPOSAL.md proposes terms but has not been enacted. Portions that adapt or quote
  upstream LittleBit code remain under CC BY-NC 4.0 (attribution, NonCommercial only).
* Upstream code is CC BY-NC 4.0, Qwen2.5-0.5B is Apache-2.0, WikiText-2 is CC BY-SA; model
  weights and dataset files are not redistributed. See LICENSE_NOTES.md.
* AI assistance: the code, tests and drafts in this repository were written with an AI coding
  assistant (Claude, Anthropic). The pilot stages and test suites were executed by an automated AI
  coding workflow on the author's Mac, not run by hand; the author's review of the outputs is
  pending. Numbers in this README come from the recorded JSON and logs listed below.

## Measured results

Run environment: host Mac mini (Apple M4), macOS 26.0.1 arm64, torch 2.6.0, numpy 2.2.4, CPU.
Python as recorded in each result file: 3.11.15 for the benchmark (results/bench.json), 3.12.14
for the exhaustive audit (results/audit_summary.json), the patch regression and the quality pilot.
Sections 1 to 4 use 1 torch thread with deterministic algorithms on; the pilot uses 2 threads.
Executed by a separate parent process, not by the assistant that wrote the code.

### 1. Exhaustive unpack sweep against actual upstream code (UPSTREAM_IMPORT)

Raw: [results/audit_summary.json](results/audit_summary.json),
[results/unpack_cases_upstream_import.csv](results/unpack_cases_upstream_import.csv),
[results/unpack_cases_algorithm_audit.csv](results/unpack_cases_algorithm_audit.csv).

2,010 cases (every byte value tiled across widths 1, 7, 8, 31, 32, 33, 64, every single -1
position, and fixed +1/-1 patterns), through the actual upstream `binary_packer` and
`binary_unpacker`.

| width | 1 | 7 | 8 | 31 | 32 | 33 | 64 |
|---|---|---|---|---|---|---|---|
| byte-tiled rows exact, of 256 (preregistered) | 256 (256) | 4 (4) | 2 (2) | 1 (1) | 1 (1) | 1 (1) | 1 (1) |
| sign errors, all cases | 0 | 786 | 917 | 3,930 | 4,062 | 4,062 | 8,124 |

* 296 of 2,010 cases decode exactly; 21,881 signs are wrong in total. Every error is a -1 decoded
  as +1; zero +1 to -1 flips.
* Prediction mismatches: 0. A row decodes exactly if and only if all of its -1 entries sit at
  columns j with j % 32 == 0.
* Both corrected reference decoders recover every case from the upstream packer: the packer is
  correct; the unpacker is not.
* Mirror versus upstream: 0 packer mismatches, 0 unpacker mismatches, so the standard-library
  mirror reproduces the shipped functions on every case.
* Cause, upstream `quantization/utils/binary_packer.py` line 88:
  `bits = (word_data.unsqueeze(1) << torch.arange(32, ...)) & 1`. A left shift by k >= 1 clears
  bit 0, so only in-word position 0 can come back as -1.

Note: this audit_summary.json was produced by the audit.py revision before the
`prior_exploratory_evidence` block and the sixth named counterexample were added; regenerate it
to include them. The unit test that checks the mirror against the recorded exploratory run passed.

### 2. Actual upstream layers through the emulated packed load (UPSTREAM_IMPORT)

Raw: [results/bench.json](results/bench.json), [results/bench.csv](results/bench.csv).

72 configurations: shapes 256x256, 512x1024, 1024x512 (in x out); target 1.0 and 0.1 BPW;
residual off and on; seeds 0, 1, 2; two save profiles (raw fp32 `state_dict`, and the
`main.py` rule that casts non-packed fp32 tensors to bf16).

| decoder | sign agreement with in-memory factors | layer output vs in-memory |
|---|---|---|
| upstream `binary_unpacker` | 0.514 to 0.553 (all 72 below 1.0) | changed in **72 of 72**; relative Frobenius error 1.75 to 22.2 |
| corrected decoder, raw fp32 profile | 1.0 in 36 of 36 | **bitwise equal in 36 of 36** |
| corrected decoder, main.py bf16 profile | 1.0 in 36 of 36 | relative error 0.0029 to 0.0055 |

* The bf16-profile residual comes from bf16 rounding of saved scale vectors, not from signs.
* Dense reconstructed matched baseline (same binarized factors and scales, same input):
  max absolute difference to the in-memory module at most 1.97e-6, relative at most 9.8e-7.
* Missing and unexpected keys: 0 in every load.
* Measured resident bytes match the accounting. Example 256x256, s = 104, fp32 load: 215,872
  parameter bytes resident versus 7,424 packed sign bytes (about 29x).

Timings (median, 20 iterations after 3 warmup, 1 thread; p95 is the 19th of 20 ordered samples).
Descriptive only, small synthetic layers, no end-to-end or model-level speed claim:
in-memory forward 26 to 449 us (it re-binarizes U and V on every call), loaded binarized forward
12 to 76 us, dense matched 6.3 to 53 us. The corrected unpack was not uniformly faster than the
upstream unpack (the same order of magnitude in both directions).

### 3. Storage accounting (ALGORITHM_AUDIT, formula level)

Raw: [results/storage_accounting.csv](results/storage_accounting.csv) and the `accounting`
block of audit_summary.json. Illustrative shapes, mirrored upstream formulas.

| layer, target | split s | advertised BPW | serialized BPW (main.py bf16) | resident BPW, bf16 factors | resident BPW, fp32 factors |
|---|---|---|---|---|---|
| 4096x4096, 1.0, plain | 2024 | 0.998 | 1.006 | 15.82 | 31.64 |
| 4096x4096, 1.0, residual | 1000 | 0.994 | 1.008 | 15.64 | 31.27 |
| 64x64, 0.1, plain | 8 | 0.781 | 1.273 | 4.59 | 8.59 |
| 64x64, 0.1, residual | 8 | 1.563 | 2.523 | 9.15 | 17.15 |

* The formula counts one middle scale vector of length s per path; the module stores two (u2
  and v1): +16 s bits per path at 16-bit scales.
* U rows (s columns) are padded to 32 bits; s is a multiple of 8, so padding is nonzero whenever
  s % 32 != 0.
* When the min_split_dim floor binds, actual BPW exceeds the target (64x64 at target 0.1:
  0.78125 plain, 1.5625 residual).
* After the upstream packed load, U and V are resident at torch_dtype: about 16x (bf16) or 32x
  (fp32) the packed sign bytes for large layers.

### 4. Unit tests

[evidence/unit-tests.log](evidence/unit-tests.log): first run, 43 ran, 1 failed
(`test_no_em_dashes` contained the literal character it scans for; fixed with `chr(0x2014)`, test
kept). [evidence/unit-tests-final.log](evidence/unit-tests-final.log): independent rerun after
the fix and the new regression tests, **56 ran, all passed**.

Current suite (audit plus quality pilot tests):
[evidence/unit-tests-quality.log](evidence/unit-tests-quality.log), host, **95 ran, OK**;
[evidence/container-tests-quality.log](evidence/container-tests-quality.log), Docker image
`sha256:e2001739838437f9e33226e856b2dc4caa47aae1269791455b3699773c818588` (arm64,
[evidence/container-image.txt](evidence/container-image.txt)), **95 ran, OK**. These tests check
functions and bookkeeping; they do not measure model quality.

Release revision: the full local suite now has 129 tests. At the time of writing, private
validation of `public_metadata.json` against the real checkpoints succeeded on the owner's host,
and the 129-test suite and the metadata replay were being run independently; their logs are the
record, not this sentence. New tests: `tests/test_public_release.py` (host path sanitization of stage
records and a scan of the public files) and `tests/test_report_results.py` (metadata-only replay
from a clean-clone copy that must open only the public files and rebuild the recorded numbers;
private validation on synthetic checkpoints, including tampering that only private validation
catches; on the owner's host, private validation of the real checkpoints; and a check that the
tracked `results/summary.json` equals a fresh replay). The 95 count above predates them. In a
clean clone (and in CI) without torch, the upstream checkout or the checkpoints, the tests that
need them skip with a stated reason.

### 5. Real-model quality pilot: negative result (UPSTREAM_IMPORT_REAL_MODEL_PILOT)

Raw: [results/quality-pilot/](results/quality-pilot/) (`manifest.json`, `baseline.json`,
`feasibility.json`, `train.json`, `evaluate.json`, `train_steps.jsonl`, `public_metadata.json`),
preregistration [QUALITY_PREREGISTRATION.md](QUALITY_PREREGISTRATION.md), derived summary
[results/summary.json](results/summary.json) and figures in [figures/](figures/), both produced by
`src/report_results.py` from the recorded JSON only (render log: evidence/report-render.log). The
figures exist; see "Public release" below for how to regenerate them from a clean clone and what
that replay does and does not check.

Setup. Qwen/Qwen2.5-0.5B (revision `060db649`, Apache-2.0), WikiText-2 raw (revision `b08601e0`),
splits tokenized separately. Every `nn.Linear` in the 24 transformer blocks (168 modules,
357,826,560 weights) converted to actual upstream `LittleBitLinear`, `eff_bit` 0.55, residual,
SVD initialization. Embeddings, tied head, norms and biases frozen. QAT: 16 steps, batch 1, one
128-token train window per step, each window used once: **2,048 input tokens, 2,032 supervised
next-token targets**. Loss: next-token cross entropy plus KL(teacher || student), weight 1.0 each,
AdamW lr 4e-5 cosine. CPU, float32, 2 threads, seed 42, one run. Held-out: **8 test windows of 128
tokens, 1,016 scored tokens**, each student scored after a packed save and reload.

| Condition (same 8 test windows) | Perplexity | Mean NLL | Windows worse than original |
|---|---|---|---|
| Original model | 28.8816 | 3.3632 | (reference) |
| LittleBit, initialized, no QAT | 82,426.3372 | 11.3197 | 8 of 8 |
| LittleBit, 16 QAT steps | 9,735.6350 | 9.1835 | 8 of 8 |
| Same QAT weights, defective upstream decoder | 13,558,558.4902 | 16.4225 | 8 of 8 |

The last row is the decoder defect variant, evaluated in memory only. It is not a quality result
for any model; it shows what the shipped unpacker does to these weights.

* Paired per-window NLL increase over the original: initialized +7.23 to +8.71 nats, QAT +5.13 to
  +6.51 nats. QAT lowered NLL against the initialized student in 8 of 8 windows (by 1.81 to 2.52
  nats), yet perplexity stayed about 337 times the original (initialized about 2,854 times).
* Statistics: descriptive only. Eight contiguous windows from one test prefix are not independent
  samples, there is one seed and one run, and no confidence interval or test is reported.
* Fixed greedy generations are recorded in evaluate.json and not judged.
* Preregistered predictions: P1 (original perplexity 8 to 60), P2 (initialized at least 10 times
  worse), P3 (still at least 5 times worse after 16 steps), P6 (file BPW above formula BPW) held.
  P4 held for the corrected decoder (below; logits compared on one validation window). P5 failed
  on memory (below) and held on time.

Reload identity (corrected decoder, same process, CPU float32): for both the initialized and the
trained checkpoint, 171,294,720 of 171,294,720 signs agree, and logits and loss are **bitwise
identical** to the in-memory student **on the one 128-token validation window that was compared**
(validation window 0). Identity was not checked on other inputs; the held-out test windows were
scored only after reload. Teacher parameter fingerprint unchanged by training. Sign flips between
initialized and trained checkpoints: 360,930 (0.21 percent).

Storage. Disk and resident numbers use different dtypes and are never mixed. Tokenizer, vocab,
merges and config files (15,879,995 bytes in the checkpoint directory) are excluded from all
weight counts; the KV cache is separate.

| Quantity | Original | Pilot student |
|---|---|---|
| Disk, weights (stored dtypes) | 988,097,824 B, one BF16 file | 299,683,600 B = frozen base 272,426,320 B (BF16) + compressed layers 27,257,280 B |
| of which embedding table (frozen, tied head) | 272,269,312 B | 272,269,312 B |
| Resident, unique tensor storage, FP32 | 1,976,131,200 B | 1,234,703,616 B |
| of which embedding table, FP32 (derived from shape) | 544,538,624 B | 544,538,624 B |
| KV cache, 128 tokens, FP32 | 3 MiB | 3 MiB (unchanged) |

* Converted block linears only: advertised upstream formula **0.52955 BPW**; packed signs alone
  0.4967 BPW; actual compressed checkpoint file with FP32 scales **0.6094 BPW**.
* The frozen embedding table is about 91 percent of the student's disk bytes. After loading, the
  latent factors are resident as FP32, not packed bits.

Resources. Train stage 82.36 s wall clock (conversion 8.98 s; steps 2.53 to 5.32 s). OS peak RSS of
the train stage was **6.53 GiB, above the 6 GiB guard threshold**; the guard samples RSS at
checkpoints (highest sampled value 5.34 GiB), so it did not stop the run. Guards are sampled, not
hard caps. No further training is planned.

Why it matters and where it applies. The result says that this conversion of a 0.5B model to
about 0.53 advertised BPW does not come close to the original model after a 16-step budget on this
CPU setup, and that a pilot of this size cannot say anything about the methods at their published
training budgets. It is useful as a worked, reproducible harness: frozen hashed windows, paired
per-window losses, separate initialization and trained results, a bitwise reload identity check,
and byte accounting that keeps disk, resident and KV memory apart.

Missing, not measured: a validated 4-bit baseline, a smaller same-family model, task accuracy,
statistics over seeds or more windows, end-to-end latency or throughput.

## Proposed upstream fix

[patches/binary_unpacker.patch](patches/binary_unpacker.patch): one line, `<<` to `>>`. The
upstream checkout stays unmodified. `src/patched_regression.py` reads the pinned source text,
applies only that hunk in memory (strict context match, exactly one changed line), executes it in
a fresh module object that is never registered in `sys.modules`, checks the upstream file hash
before and after, and runs:

* the 2,010 exhaustive cases (expected: all exact; unpatched control expected: 296 exact);
* multi-row signed matrices at widths 31, 32, 33, 64 (negative int32 words at width >= 32);
* actual upstream layers including residual factors U_R and V_R, with the unpatched decoder as a
  control.

Matching tests: `TestPatchFile`, `TestSignedMultiRowMirror` (standard library),
`TestPatchedUpstream`, `TestPatchedResidualLayer` (torch).

Measured ([results/patched_regression.json](results/patched_regression.json), post hoc,
Python 3.12.14, torch 2.6.0, upstream file hash identical before and after, patched module never
registered in `sys.modules`):

* Exhaustive sign cases: **2,010 of 2,010 exact with the patch**, versus 296 of 2,010 with the
  original unpacker.
* Signed multi-row matrices at widths 31, 32, 33, 64 (6, 6 and 11 negative int32 words at widths
  32, 33, 64): exact and equal to the reference decoder with the patch; not exact without it.
* 5 actual upstream layer configurations, 4 of them residual: every factor including `U_R` and
  `V_R` exact and outputs bitwise equal with the patch; unpatched control sign agreement 0.496 to
  0.569 and outputs changed in all 5.
* Load path still emulated; the official loader was not run.

## Containers

* A standard-library audit ran in Docker (OrbStack), offline, 1 CPU, 1 GB, read-only
  (evidence/container-audit.log, mirror tier only).
* `Dockerfile` (pinned CPU torch 2.6.0, numpy 2.2.4, non-root, no credentials, offline at run
  time): built and tested; image `sha256:e2001739...c818588`, arm64; 95 tests ran, OK
  (evidence/container-tests-quality.log). The quality pilot itself ran on the host, not in the
  container. The tests added in the release revision have not been run in the container yet.
* `.devcontainer/devcontainer.json`: not present. The write was refused by the tool permission
  layer as a sensitive file, including one authorized retry, and was not retried or bypassed. It
  needs the owner to create the file or approve the write.

## Status

| Stage | What | Status |
|---|---|---|
| 0 | Exploratory upstream counterexample | done, predates preregistration |
| 1 | Preregistration | written; addendum 2026-10-08; quality pilot preregistration separate |
| 2 | Audit code and tests | done; 95 of 95 tests passed on host and in the Docker image before the release revision; release suite of 129 tests run independently, see its log |
| 3 | Raw component results | done (sections 1 to 3) |
| 4 | Patch regression | done, post hoc: 2,010 of 2,010 patched vs 296 of 2,010 original |
| 5 | Real pretrained model pilot | done, negative result (section 5); no further training planned |
| 6 | Report figures and summary | run (evidence/report-render.log); figures and summary.json exist; must be regenerated with `--metadata-only` after the path sanitization (see "Public release") |

Authorized by the owner: this repository as a public GitHub repository
(timurista/littlebit-deployment-audit) with release v0.1.0 (RELEASE_NOTES.md). Not done and not
planned here: accepting any license agreement, cloud spend, Medium publication, arXiv submission,
enacting a license, or sending the patch upstream. Those need separate owner decisions.

## Running

```
timeout 600 .venv/bin/python -m unittest discover -s tests -v
timeout 900 python src/audit.py --out results
timeout 1800 python src/bench.py --out results --iters 20 --warmup 3 --threads 1
timeout 900 python src/patched_regression.py --out results
```

Quality pilot stages, as recorded in each stage's `args` block (model weights and dataset parquet
files must be present locally; nothing is downloaded):

```
.venv/bin/python src/quality_pilot.py prepare --out results/quality-pilot --data-dir <local wikitext snapshot dir>
.venv/bin/python src/quality_pilot.py baseline --out results/quality-pilot --max-new-tokens 8 --max-seconds 120.0 --threads 2
.venv/bin/python src/quality_pilot.py feasibility --out results/quality-pilot --max-new-tokens 20 --max-seconds 300.0 --threads 2
.venv/bin/python src/quality_pilot.py train --out results/quality-pilot --max-new-tokens 8 --max-seconds 300.0 --threads 2
.venv/bin/python src/quality_pilot.py evaluate --out results/quality-pilot --max-new-tokens 8 --max-seconds 180.0 --threads 2 --upstream-decoder-variant
```

All other flags were at their defaults (`--steps 16 --eff-bit 0.55 --lr 4e-05 --residual`, guards
6 GiB RSS and 2 GiB available). `results/summary.json` lists the full rebuilt command lines.

Figures and summary (no model execution, network or subprocess; matplotlib 3.10.0 in `.venv`):

```
.venv/bin/python src/report_results.py --metadata-only   # metadata replay, works from a clean clone
.venv/bin/python src/report_results.py                   # private validation, owner host only
```

Both write `figures/paired_window_nll`, `figures/heldout_ppl_log`, `figures/weight_bytes` and
`figures/train_ce_kl` as PNG and SVG, plus `results/summary.json`. The script was run once before
the release revision (evidence/report-render.log); the figures in `figures/` come from that run.

## Public release

**Metadata replay versus private checkpoint validation.** These are different checks, and
`results/summary.json` records which one produced it under `replay.mode`.

* `--metadata-only` (`metadata_replay`) reads only tracked public files: the stage JSON in
  `results/quality-pilot/`, `train_steps.jsonl` and `results/quality-pilot/public_metadata.json`.
  It needs no model weights, checkpoints, checkpoint config, tokenizer files or host logs. It
  recomputes every derived number (perplexities, paired differences, byte and BPW accounting,
  predictions) and checks the public files against each other (window hashes, token counts,
  totals, checkpoint sizes and sha256 values between `public_metadata.json` and `train.json`). It
  does **not** validate any checkpoint: checkpoint sizes, the embedding header facts and the test
  outcomes are copied from `public_metadata.json`.
* The default mode (`private_validation`) runs only where the local checkpoints exist. It also
  hashes the frozen base, init and trained checkpoints and the checkpoint `config.json` against
  `train.json`, parses their safetensors headers, reads the original weights header and the test
  logs when present, re-derives every `public_metadata.json` section and fails on any
  disagreement. Apart from `replay`, both modes produce the same summary;
  `tests/test_report_results.py` checks this.

**Path sanitization.** The `args.model_dir` field of `baseline.json`, `feasibility.json`,
`train.json` and `evaluate.json` held an absolute host path. It now reads `models/qwen2.5-0.5b`;
no other field and no measured number changed (`manifest.json` was not edited, so every recorded
`manifest_sha256` still holds). The pre-sanitization sha256 values are listed in
`public_metadata.json` (`path_sanitization`) and in evidence/quality-artifact-hashes.json; the
original files are kept in the owner's private local backup outside the public export. Future
stage runs record arguments through `public_args` and write every record through
`scrub_host_paths` (`src/quality_pilot.py`): path arguments become project relative or
`<external>/<basename>`, and the project root and home directory in any other string (error
tracebacks included) become `<project>` and `<home>`.

Because the four stage files changed, the input hashes in `results/summary.json` must be
regenerated. Order for the owner's host:

```
.venv/bin/python src/report_results.py --no-figures --summary /tmp/littlebit-private-summary.json
.venv/bin/python src/report_results.py --metadata-only
timeout 900 .venv/bin/python -m unittest discover -s tests -v
```

The first command is the private validation of `public_metadata.json` against the real
checkpoints; it writes its summary outside the repository. The second rewrites the public
`results/summary.json` and the figures from public files only, so a clean clone reproduces them.
Until the second command runs, `TestMetadataReplayCleanClone.test_tracked_summary_matches_fresh_replay`
fails by design. Host logs under `evidence/` that are not tracked (for example
`evidence/model-download.log`, `evidence/report-render.log`, `evidence/docker-info.json`) still
contain absolute host paths and are not part of the public export.

License: none granted; see LICENSE_NOTES.md. A license for the original harness code is proposed,
not adopted, in LICENSE_PROPOSAL.md.

**Release, CI and citation.**

* Release v0.1.0: RELEASE_NOTES.md. Expected URL, to verify once it exists:
  https://github.com/timurista/littlebit-deployment-audit/releases/tag/v0.1.0. The release commit
  hash is filled in after the clean-root commit. No DOI.
* CI: `.github/workflows/ci.yml`, one job on Python 3.12 with the standard library only. It runs
  the unit tests, where torch-dependent tests, tests that need the upstream checkout and tests that
  need the private checkpoints skip with a stated reason, and then the `--metadata-only
  --no-figures` replay. Read-only token, no secrets, no deployment, no downloads, no training.
  Actions are pinned to full commit SHAs (checkout v4, setup-python v5).
* Citation: CITATION.cff (repository URL to verify; no DOI).

Drafts: docs/medium-draft.md, docs/research-draft.tex (review-ready pilot write-up, not a
submission).
