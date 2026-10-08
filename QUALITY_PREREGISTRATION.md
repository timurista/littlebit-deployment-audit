# Quality pilot preregistration (wiring and resource pilot)

Status 2026-10-08: written before model-quality evaluation and training. The deterministic prepare stage and 95 unit tests had already run before this document was written; no model loss had been measured. The separate base resource check is recorded in evidence/base-resource-pilot.json. Code: `src/quality_pilot.py`, tests: `tests/test_quality_pilot.py`.

## What this pilot is and is not

* It is a bounded, local, offline wiring and resource pilot: actual upstream `LittleBitLinear`
  (SamsungLabs/LittleBit commit `933857ed1443b53fc43a875c2cf64249e3c56f0c`, imported through
  `src/audit.py` `load_upstream`) on a real pretrained checkpoint, measured on tiny frozen windows.
* It is not a statistical benchmark, not a replication of any published LittleBit number, not a
  throughput measurement, and not a task accuracy evaluation.
* No random or tiny model is substituted for the real checkpoint. Small seeded layers appear only in
  unit tests of functions.

## Fixed inputs

| Item | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, Apache-2.0, ungated (verified by the owner), local at `models/qwen2.5-0.5b` |
| Weight check | sha256 of `model.safetensors` must equal the hub LFS sha256 recorded in `.cache/huggingface/download/model.safetensors.metadata`; revision lines must all equal the pin |
| Dataset | Salesforce/wikitext, config `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, read from local parquet files only |
| Dataset revision check | taken from an HF cache `snapshots/<revision>/` path; files without such a path need `--accept-unverified-dataset-revision` and are then labeled UNVERIFIED in the manifest |
| Row counts | train 36,718, validation 3,760, test 4,358 (hard check) |
| Split roles | train: QAT only. validation: development only (logged, no automatic decisions). test: frozen held-out, scored only in `evaluate` |
| Windows | 16 train, 4 validation, 8 test, 128 tokens each, contiguous and non-overlapping from token 0 of each split's own row prefix, documents joined with `"\n\n"` |
| Tokenizer | native Qwen tokenizer, default `add_special_tokens`, no BOS or EOS id override; ids and whether the first token is BOS are recorded |
| Manifest | source revision, parquet sha256, row ranges, prefix text sha256, token counts, per-window and per-stream sha256 of little-endian int32 ids; written once, never edited; a new manifest is a new experiment |
| Seed | 42 (svd_lowrank initialization and train window order) |

## Fixed QAT configuration (train stage defaults)

* Conversion: every exact `nn.Linear` inside `model.layers` (q, k, v, o, gate, up, down) reclassed to
  upstream `LittleBitLinear` and `__quant_convert__(do_train=True)`: `eff_bit=0.55`, `residual=True`,
  `split_dim=1024` default, `min_split_dim=8`, `kv_factor=1.0`, `use_itq=False` (original SVD
  initialization), `quant_func=SmoothSign`. `lm_head` and `embed_tokens` are never converted.
* Teacher: the original model, float32, frozen, parameters fingerprinted (sha256 over all parameter
  bytes) before conversion and after training; the run fails if they differ.
* Student: a deep copy that shares every frozen tensor with the teacher by reference; conversion
  drops the student's references to the original weights. Trainable: only LittleBit factors
  (U, V, U_R, V_R latent values) and scales (u1, u2, v1, v2 and residual copies). Frozen:
  embeddings, tied head, norms, biases.
* Objective: KL(teacher || student) per position with teacher logits detached, token mean, weight
  1.0, plus next-token cross entropy, weight 1.0. Optional upstream-style intermediate hidden-state
  MSE (`--l2l-scale`) is off by default.
* Optimizer: AdamW, lr 4e-5 (upstream README), betas (0.9, 0.999), eps 1e-8, weight decay 0, cosine
  decay without warmup, gradient clip 1.0. 16 steps, batch 1, sequence 128, each train window used
  exactly once in a seed-42 order. No silent window reuse.
* Device: CPU float32, 2 threads by default. Optional `--device mps` float32 with synchronization;
  MPS unavailability or `PYTORCH_ENABLE_MPS_FALLBACK` set is a hard error (no silent fallback).
  Conversion and identity checks always run on CPU.

## Declared departures from upstream

1. CPU or MPS float32; upstream trains in bf16 with DeepSpeed and `GPUtil`.
2. KL is averaged over positions; upstream uses `kl_div(..., reduction="batchmean")` on 3D logits,
   which sums over positions. The loss scale differs by a factor of the sequence length.
3. An added cross entropy term (upstream is distillation only).
4. Intermediate hidden-state MSE off by default (upstream `l2l_loss_scale` 10.0 with bf16 autocast).
5. Norms and biases frozen (upstream freezes only lm_head and embeddings).
6. No gradient checkpointing; no warmup (upstream HF warmup ratio 0.03 would make step 0 lr 0).
7. Checkpoint format: own safetensors files, not `save_pretrained`. Packed signs come from upstream
   `pack_weights` (upstream `binary_packer`); scales are stored float32 (upstream `main.py` casts
   them to bf16); frozen base tensors in a separate file, stored bf16 only where the bf16 round trip
   is exact per tensor.
8. Reload is emulated (decode, cast float32, `load_state_dict(assign=True)`, `tie_weights`,
   `_binarized=True`); the official `quant_util.load_quantized_model` is not run. Any meta tensor left
   after reload is an error, never zero-filled.
9. Values decoded by the defective upstream `binary_unpacker` are never saved; the optional upstream
   decoder variant is evaluated in memory only and labeled `UPSTREAM_DECODER_DEFECT_VARIANT`.

## Resource guards and stopping rules

* Stop if process RSS exceeds 6 GiB, if system available memory falls below 2 GiB, or if the stage
  wall clock exceeds 900 s. Checked before and after every layer conversion, every train step, every
  evaluation window.
* Conversion logs a milestone per transformer layer (elapsed, projected total, RSS) and stops if the
  projected remaining SVD initialization exceeds the remaining budget.
* A step is not started if the remaining budget is shorter than the slowest step so far.
* A non-finite loss or gradient norm stops the run.
* Stopped runs keep their partial step log and are reported as `stopped_by_guard`; they are never
  reported as trained.
* `feasibility` runs only 2 real steps and estimates the train stage from measured times
  (conversion plus 16 times the mean and the slowest step). A conservative estimate above 900 s means
  the train stage is not launched without the owner's sign-off.

## Measurements

* Held-out NLL: per test window, sum of next-token NLL over tokens 2..128 (first token never
  scored), plus token-weighted mean NLL and perplexity, for the original model, the initialized
  student (no QAT) and the QAT student, each student evaluated after packed reload with the
  corrected decoder. Raw per-window sums, counts and window hashes are written; paired per-window
  differences against the original are listed without confidence intervals.
* Initialization quality and trained quality are reported separately.
* Identity: on one validation window, in-memory student versus its packed reload (same process, CPU
  float32): bitwise logits equality, max abs difference, loss, sign agreement; for both the init and
  the trained checkpoint.
* Storage: sha256 and actual bytes of every checkpoint file, packed sign bytes, float32 scale bytes,
  upstream formula BPW next to BPW from actual file bytes, unique resident tensor storage
  (deduplicated by storage pointer), KV cache bytes reported separately.
* Resources: per-step dt, RSS, available memory, MPS allocated memory, process peak RSS.
* Fixed generation: 3 fixed prompts, greedy, 20 new tokens (cap 32), outputs shown, never judged.

## Predictions (written before any run)

* P1 Original model on the 8 test windows: token-weighted perplexity between 8 and 60 (loose
  sanity range for 128-token context; not a hypothesis test).
* P2 Initialized student at 0.55 BPW without QAT: perplexity at least 10 times the original.
* P3 After 16 steps: not expected to recover meaningful quality; perplexity still at least 5 times
  the original. Any change is a pilot observation, not evidence about LittleBit.
* P4 Corrected-decoder reload reproduces in-memory logits bitwise on CPU float32 with fixed threads,
  for both init and trained checkpoints. The upstream decoder variant does not.
* P5 Peak RSS between 4.5 and 6.0 GiB (teacher float32 about 2 GB, latent factors plus gradients plus
  AdamW states about 2.7 GB); the 6 GiB guard may trip, which would be a recorded negative
  feasibility result. 16 steps fit in 900 s on the M4 CPU with 2 threads.
* P6 BPW from actual file bytes exceeds the upstream formula value (float32 scales, two middle scale
  vectors per path, row padding, shapes, buffers, container header).

## Claims policy

* "Trained" requires a completed step log for all planned steps plus a trained checkpoint with its
  sha256. Otherwise only initialization results exist.
* Task accuracy: not measured. Throughput: not measured; operator microbenchmarks
  (`src/bench.py`) are not end-to-end.
* Practical 4-bit baseline: MISSING, no validated 4-bit backend yet. Smaller same-family model
  baseline: MISSING in this pilot.
* Negative or null results are reported the same way as positive ones.

## Commands

```
.venv311/bin/python src/quality_pilot.py --help
.venv311/bin/python -m unittest tests.test_quality_pilot -v
.venv311/bin/python src/quality_pilot.py prepare --data-dir <local wikitext snapshot dir>
.venv311/bin/python src/quality_pilot.py baseline
.venv311/bin/python src/quality_pilot.py feasibility
.venv311/bin/python src/quality_pilot.py train --upstream-decoder-variant
.venv311/bin/python src/quality_pilot.py evaluate --upstream-decoder-variant
```

The WikiText-2 parquet files are not in this workspace. They must be placed locally by the owner
(for example an HF cache `datasets--Salesforce--wikitext/snapshots/<revision>/` tree); this code
never downloads anything.
