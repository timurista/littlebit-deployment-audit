# License proposal for the audit repository

Status: **proposal only, not enacted.** No LICENSE file, SPDX header or relicensing has been
added. Until the owner decides, LICENSE_NOTES.md stays the governing statement: all rights are
reserved for the original audit code, subject to the upstream terms for any adapted portions.
Not legal advice; a qualified reviewer should confirm the classification before release.

## Principle

* MIT is proposed only for files that are clearly original harness code: written for this audit,
  copying no upstream source and not re-expressing upstream algorithms or formulas.
* Anything that adapts, re-expresses or quotes code from SamsungLabs/LittleBit is conservatively
  treated as Adapted Material under CC BY-NC 4.0, the upstream license, and keeps its
  attribution and NonCommercial condition. Nothing in this proposal relicenses upstream code or
  any derived portion.
* Apache-2.0 notices are preserved wherever Apache-2.0 material is involved (listed below).
  Nothing here removes, shortens or replaces them.

## Proposed MIT: clearly original harness files

| File | Why it is treated as original |
|---|---|
| `src/quality_pilot.py` | Pilot orchestration, window freezing, guards, accounting, public-safe serialization. Imports the unmodified upstream module at run time from the user's own checkout; no upstream source is copied. It names upstream parameters and their default values (`split_dim` 1024, `min_split_dim` 8, `SmoothSign`) to configure that module. |
| `src/report_results.py` | Summary and figures from recorded JSON; metadata replay and private validation. |
| `tests/test_quality_pilot.py` | Tests of the pilot helpers. The torch tests drive the upstream module, they do not copy it. |
| `tests/test_public_release.py` | Path sanitization and public file scan tests. |
| `tests/test_report_results.py` | Replay and private validation tests. |
| `Dockerfile`, `requirements.txt` | Environment definition. |

Reviewer check before enacting: confirm that none of these files contains a copied upstream line;
the static scans that look for upstream expressions are in `src/audit.py`, not in these files.

## Conservatively CC BY-NC 4.0: derived or quoting portions

| File | Upstream relation | Proposed handling |
|---|---|---|
| `patches/binary_unpacker.patch` | A modification of upstream `quantization/utils/binary_packer.py`, including upstream context lines. Adapted Material. | CC BY-NC 4.0 with attribution, a link to the license and an indication of the change (one line, `<<` to `>>`). |
| `src/audit.py` | The ALGORITHM_AUDIT tier re-expresses upstream arithmetic (`mirror_pack`, `mirror_unpack_upstream`, `upstream_estimate_split`, `upstream_finalize_split`, `upstream_eff_bits`, `advertised_bits`, `storage_breakdown`) and searches for quoted upstream lines (`static_observations`). | Whole file CC BY-NC 4.0 unless the owner later moves the mirror functions into a separate module; the remaining harness code could then be proposed as MIT. |
| `src/bench.py` | `torch_reference_unpack` is a corrected re-expression of the upstream unpack expression; the rest is original benchmark harness. | Whole file CC BY-NC 4.0 unless split as above. |
| `src/patched_regression.py` | Reads the pinned upstream source and applies the patch hunk in memory; its logic is defined by the upstream file and the derived patch. | CC BY-NC 4.0 unless a review confirms it contains no upstream text. |
| `tests/test_audit.py` | Mirrors and quotes short upstream expressions in test data and strings. | CC BY-NC 4.0. |

For all of these, keep the attribution block from LICENSE_NOTES.md (creators, license link,
repository link, commit `933857ed1443b53fc43a875c2cf64249e3c56f0c`) and the statement that the
audit is not affiliated with or endorsed by Samsung or the authors.

## Third-party material: terms unchanged, not relicensed

* Upstream LittleBit code (CC BY-NC 4.0): not redistributed; users need their own checkout.
* Upstream `quantization/modules/attention.py` carries an Apache-2.0 header (Copyright 2024
  Microsoft and the HuggingFace Inc. team). It is not copied into this repository. If any part of
  it is ever redistributed, its header and the Apache-2.0 license text must be kept verbatim.
* Qwen/Qwen2.5-0.5B (Apache-2.0): weights, config and tokenizer files are not redistributed. The
  local checkpoint directory `results/quality-pilot/checkpoints/config_and_tokenizer/` holds copies
  of Qwen files; if any of them is ever shared, include the Apache-2.0 license text and any NOTICE
  shipped with the model, and state that the files were saved by this pilot.
* WikiText-2 (CC BY-SA, see LICENSE_NOTES.md): no text is published; window token ids stay local.

## Not covered by this proposal

* Prose and documents (README.md, LICENSE_NOTES.md, PREREGISTRATION.md,
  QUALITY_PREREGISTRATION.md, MODEL_QUALITY_PLAN.md, `docs/`): they quote short upstream
  expressions for commentary and identification of the defect. A documentation license is a
  separate owner decision; MIT is not proposed for them.
* Result data (`results/`, `figures/`, `evidence/`): numbers and logs produced by running
  NonCommercial upstream code on Apache-2.0 weights and CC BY-SA data. Whether these outputs carry
  any of those terms is an open question; until the owner decides, share them with the attribution
  and NonCommercial caution of LICENSE_NOTES.md.

## Steps if the owner adopts this (not done here)

1. Choose the copyright holder line and year for the MIT text.
2. Add a top-level LICENSE that applies MIT only to the files listed as proposed MIT, and states
   that the files listed as CC BY-NC 4.0 remain under CC BY-NC 4.0.
3. Add SPDX headers (`MIT` or `CC-BY-NC-4.0`) to each source file to match.
4. Keep LICENSE_NOTES.md and the attribution and non-endorsement statements.
5. Optionally split the upstream mirror functions out of `src/audit.py` and `src/bench.py` so that
   more of the harness can be offered under MIT.
