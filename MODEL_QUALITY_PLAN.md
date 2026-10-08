# Model quality plan: bounded, local-first pilot on a real pretrained model

Status 2026-10-08: **no model quality has been measured.** The pretrained model and the dataset
are downloaded locally (below), and one base-model resource check has run. Everything else
measured so far concerns the packed serialization component (see README.md).

## Local setup (recorded facts)

* Model: Qwen/Qwen2.5-0.5B, verified ungated, Apache-2.0, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`, downloaded to `models/qwen2.5-0.5b`.
* Data: WikiText-2 raw, dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, downloaded
  to `data/wikitext2-raw` with train, validation and test kept as separate splits.
* Quality environment: `.venv/bin/python`, Python 3.12.14, same packages as the audit environment
  (torch 2.6.0, numpy 2.2.4). The older Python 3.11 environment lacked `lzma`, which is why a
  separate interpreter is used.
* Base resource pilot (evidence/base-resource-pilot.json): float32, CPU, 2 threads, a 128-token
  input, load 5.57 s, one forward pass 0.657 s, process RSS 2.25 GB, available memory 3.49 GB at
  measurement, finite logits. This is a resource measurement only, not held-out quality: the input
  was not a frozen evaluation window and no loss was computed.

## Requirements set by the project owner

* Real pretrained model, not random toy layers.
* Local first: Mac mini M4, 10 CPU cores, 16 GB unified memory (evidence/READINESS.md). No cloud
  spending, no paid provisioning.
* Runs inside the pinned CPU dev container (Dockerfile) once that image is built and validated.
  Today only a standard-library audit has run in a container; the dev container has not been
  built.

## Candidate models and license gates

| Model | Upstream support | License status | Decision |
|---|---|---|---|
| facebook/opt-350m | OPT family listed in upstream README | Model card `license: other`; OPT-175B License Agreement (evidence/opt-license.md) is accept-by-use, non-commercial research only | Blocked: owner has not accepted the agreement, so no download |
| facebook/opt-1.3b | as above | as above | Blocked for the same reason; also heavier for CPU QAT |
| Qwen/Qwen2.5-0.5B | Qwen2.5 listed in upstream README | Verified ungated, Apache-2.0, revision `060db6499f32faf8b98477b0a26969ef7d8b9987` | Selected; downloaded to `models/qwen2.5-0.5b` |

Rules: record repository id, revision commit hash, license text hash, and file sha256 values before
first use. Any model whose license requires acceptance waits for the owner's explicit decision.

## Data: WikiText-2 (raw v1)

* Pinned revision `b08601e04326c79dfdd32d625aee71d232d685c3`, local copy `data/wikitext2-raw`,
  splits stored separately.
* Splits from evidence/wikitext-card.md: train 36,718, validation 3,760, test 4,358 rows. The card
  metadata says CC BY-SA 3.0 while its text says CC BY-SA 4.0; record both and attribute.
* Train split: QAT only. Validation split: hyperparameter choice, early stopping, all
  development decisions. Test split: evaluated once per frozen configuration at the end.
* Native tokenizer of the chosen model, default special-token behavior, **no BOS or EOS id
  override**. (The upstream eval loader reassigns bos/eos ids to 1/2 when they differ; this pilot
  deliberately does not, and records the native ids and whether a BOS token is prepended.)
* Documents joined with "\n\n" as in upstream and common practice; join rule recorded.
* Frozen held-out windows: fixed sequence length (512 to start, CPU bound), non-overlapping stride,
  window start offsets, token counts per window and total, and a sha256 of the token id stream,
  written to a manifest **before** any compressed model is evaluated. Manifests are never edited
  afterwards; a changed manifest is a new experiment.

## Stages and gates

P0. Provenance. License checks above, dataset revision, manifests, container image digest.

P1. Original model baseline. Per-window loss on validation windows, wall time and peak memory
measured (not estimated).

P2. Identity before speed. For the same compressed weights (LittleBit initialization, no QAT):
in-memory module versus packed, serialized, reloaded module.
* Patched decoder: logits and per-window loss must be identical (bitwise on CPU fp32 with fixed
  threads, otherwise a stated tolerance fixed in advance).
* Unpatched upstream decoder: report the logits and loss divergence on the real model. This is a
  model-level measurement of the serialization defect, still not a quality claim for LittleBit.
* State whether the official loader (`quant_util.load_quantized_model`) or an emulation ran. It
  needs transformers and safetensors; until it is run inside the container, report "emulated".
* No timing is reported for any configuration that fails identity.

P3. Cost measurement before prolonged QAT. Upstream `main.py` requires DeepSpeed, GPUtil and
`bf16=True`, so a CPU QAT harness must either be validated or written (and then labeled as this
project's implementation, not upstream's). Measure seconds per optimizer step, peak RSS, and
teacher plus student memory (upstream uses knowledge distillation) for a few steps. Extrapolate a
budget and get owner sign-off before any run longer than the agreed cap.

P4. Bounded QAT pilot. Every trained claim requires the training log, the config, the seed, and a
checkpoint with its sha256. No log plus checkpoint, no trained claim.

P5. Baselines, all on the same frozen windows:
* the original model (P1);
* a smaller model of the same family, at roughly matched resident bytes;
* a practical 4-bit baseline (for example round-to-nearest group-wise int4) only after its own
  implementation passes the same identity and loss checks.

## Reporting

* Paired per-window losses (compressed minus original on identical windows), mean difference with
  bootstrap confidence interval, and token-weighted perplexity, with token counts.
* Worst windows and concrete failure examples (text span, original next-token loss, compressed
  next-token loss).
* Measured bytes: file bytes on disk, unique resident tensor bytes, peak RSS, achieved bits per
  weight from actual bytes, next to the advertised formula value.
* Speed only after identity holds, with median and p95, thread count and environment.
* Negative or null results are reported the same way as positive ones.

## Explicit non-goals for the pilot

Reproducing published LittleBit numbers, large models, GPU runs, any claim of originality, and any
claim about the papers' results.
