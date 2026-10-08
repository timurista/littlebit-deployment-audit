# Release notes

## v0.1.0

* Repository: https://github.com/timurista/littlebit-deployment-audit
* Release: https://github.com/timurista/littlebit-deployment-audit/releases/tag/v0.1.0
* Release commit: recorded in the build manifest attached to the release.
* DOI: none assigned.

### What this release is

A deployment audit of existing research code, SamsungLabs/LittleBit at commit
`933857ed1443b53fc43a875c2cf64249e3c56f0c`, plus a tiny wiring pilot on a real model. No novelty
claim, no replication of any published LittleBit result, not affiliated with or endorsed by the
authors or Samsung.

### Central findings (component level)

* **Unpacker defect.** The shipped `binary_unpacker` reads bit k with a left shift, so a -1 can
  only come back at in-word position 0. Exhaustive sweep through the actual upstream functions:
  2,010 cases, 296 exact, 21,881 signs flipped, every one from -1 to +1. The packer is correct.
  The defect was first seen in an exploratory run before preregistration.
* **One-line fix.** `patches/binary_unpacker.patch` (`<<` to `>>`), applied in memory only:
  2,010 of 2,010 cases exact, 5 actual upstream layer configurations (4 residual) exact with
  bitwise equal outputs. Not yet sent upstream.
* **Seeded upstream layers through an emulated packed load** (72 configurations): shipped decoder
  sign agreement 0.514 to 0.553 and outputs changed in 72 of 72; corrected decoder, raw fp32
  profile, bitwise equal outputs in 36 of 36.
* **Storage accounting.** Advertised BPW, serialized BPW and resident BPW are reported separately.
  The upstream formula counts one middle scale vector per path where the module stores two, rows
  are padded to 32 bits, the `min_split_dim` floor can push actual BPW above the target, and after
  loading the factors are resident at float width (about 16x or 32x the packed sign bytes for
  large layers).
* **Official loader scope.** The official state-dict loading function in `quant_util.py` ran,
  unmodified, on a synthetic single-layer file only. The full-model loader
  `quant_util.load_quantized_model` was not run; full-model reloads are emulated.

### Tiny wiring pilot (harness check, not a quality benchmark)

Qwen2.5-0.5B (outside the papers' model table), all 168 block linears converted at about 0.53
advertised BPW, 16 QAT steps on 2,048 training tokens, one seed, one run, CPU float32, 8 held-out
windows (1,016 scored tokens). It exercised conversion, training, packed save, reload and byte
accounting end to end on a real model.

* Held-out perplexity: original 28.8816, initialized 82,426.3372, after 16 steps 9,735.6350; the
  same trained weights through the shipped decoder 13,558,558.4902 (defect illustration, not a
  quality result). Descriptive only; eight contiguous windows are not independent samples.
* Reload identity with the corrected decoder: all 171,294,720 signs agree; logits and loss bitwise
  equal on the one 128-token validation window compared per checkpoint.
* Bytes for the converted block linears: formula 0.52955 BPW, packed signs alone 0.4967, actual
  checkpoint file with fp32 scales 0.6094.
* Train-stage OS peak RSS 6.53 GiB, above the sampled 6 GiB guard threshold.
* This budget and model say nothing about LittleBit at its published settings. No further
  training is planned.

### Reproducibility in this release

* `results/` holds the recorded stage JSON; `results/summary.json` and `figures/` are regenerated
  by `python src/report_results.py --metadata-only` from tracked public files only. That replay
  does not validate any checkpoint; the private validation (checkpoints hashed against
  `train.json`, headers re-derived) ran on the owner's host, where the checkpoints stay.
* Model weights, pilot checkpoints, dataset files, frozen window token ids and the upstream
  checkout are not redistributed.
* Absolute host paths were removed from four stage records (`args.model_dir` only; see
  `results/quality-pilot/public_metadata.json`). No measured number changed.
* CI (`.github/workflows/ci.yml`): Python 3.12, standard library only; torch-dependent,
  upstream-checkout and private-checkpoint tests skip; runs the metadata-only replay.

Validation status at the time of writing:

* Private validation of `public_metadata.json` against the real checkpoints succeeded on the
  owner's host.
* The full local suite (129 tests, with torch and the upstream checkout present) passed earlier on
  the host.
* Final clean-export validation (tests and metadata replay from the exported tree, where the
  upstream checkout and checkpoints are absent and the tests that need them skip) is pending. Its
  outcome is recorded in the release QA asset attached to the release, not here.

### Licensing status

* **The license proposal (LICENSE_PROPOSAL.md) is not enacted.** It is a proposal only.
* **No license is granted by this release.** The original portions of this repository are
  copyright the author, all rights reserved. `LICENSE_PROPOSAL.md` is a proposal only and has not
  been enacted.
* Portions that adapt, re-express or quote SamsungLabs/LittleBit code (see LICENSE_PROPOSAL.md)
  remain under CC BY-NC 4.0: attribution required, NonCommercial use only, changes indicated.
* Third-party terms are unchanged: upstream LittleBit code CC BY-NC 4.0; Qwen2.5-0.5B Apache-2.0
  (not redistributed); WikiText-2 CC BY-SA (no text included). See LICENSE_NOTES.md.

### Not in this release

No Medium publication and no arXiv submission; `docs/` holds review drafts only. No validated
4-bit baseline, smaller same-family model, task accuracy, statistics over seeds, or end-to-end
latency and throughput.
