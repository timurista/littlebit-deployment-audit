# Licensing and attribution notes

Not legal advice. These notes record what was checked and how this audit relates to the
upstream material.

## Upstream material

* Project: The LittleBit Project, https://github.com/SamsungLabs/LittleBit
* Commit audited: `933857ed1443b53fc43a875c2cf64249e3c56f0c` (shallow checkout in `upstream/LittleBit`)
* License of the code: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0), full text in `upstream/LittleBit/LICENSE`,
  https://creativecommons.org/licenses/by-nc/4.0/
* Papers (cite these when referring to the methods):
  * LittleBit: Ultra Low-Bit Quantization via Latent Factorization. Banseok Lee, Dongkyu Kim,
    Youngcheon You, Youngmin Kim. NeurIPS 2025. arXiv:2506.13771.
  * LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry
    Alignment. Banseok Lee, Youngmin Kim. ICML 2026. arXiv:2603.00042.
* Per `evidence/READINESS.md`, the original paper HTML is CC BY-NC-ND 4.0. No paper text is
  reproduced here.
* `quantization/modules/attention.py` carries its own header stating it is derived from
  Apache-2.0 code (Copyright 2024 Microsoft and the HuggingFace Inc. team). This audit only reads
  that file to describe its effect on accounting; nothing from it is copied.

## Model and dataset used by the quality pilot

* Model: Qwen/Qwen2.5-0.5B, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, Apache License
  2.0, ungated (license file `models/qwen2.5-0.5b/LICENSE`, sha256 `832dd9e0...`, recorded in
  `results/quality-pilot/manifest.json`). No license agreement or gated-access form was accepted
  for this project.
* Dataset: Salesforce/wikitext, config `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. The dataset card metadata lists cc-by-sa-3.0 and
  GFDL, while the card text says CC BY-SA 4.0 (recorded in the manifest). WikiText is derived from
  Wikipedia; attribution and share-alike terms apply to the text itself. Only window hashes and
  token counts are recorded here, not text.
* Upstream code: CC BY-NC 4.0 (above). The pilot imports unmodified upstream modules at run time.

## Weights, checkpoints and data are not redistributed

* `models/`, `cache/`, `data/` and every file under `results/quality-pilot/checkpoints/` are local
  only and listed in `.gitignore`. The pilot checkpoints are derivatives of the Qwen weights,
  produced with NonCommercial-licensed upstream code; they are not redistributed. Anyone
  reproducing the pilot needs their own copies of the model, the dataset and the upstream code.
* Result JSON files contain metrics, hashes, token ids of three fixed prompts and short greedy
  continuations, but no dataset text and no weights. `results/quality-pilot/windows.json` holds
  the token ids of the frozen WikiText windows, which decode to dataset text; it is local only and
  listed in `.gitignore`; the public manifest keeps only window hashes and counts.
* `results/quality-pilot/public_metadata.json` holds aggregate header facts copied from the
  local checkpoints, the checkpoint config and the original weights header (file sizes, embedding
  dtype, shape and byte count, sha256 values) plus parsed test outcomes. It contains no weights,
  tokenizer files or dataset text.
* The figures and `results/summary.json` contain only numbers derived from those JSON files.

## What this audit contains

* `src/audit.py`, `src/bench.py`, `src/patched_regression.py`, `src/quality_pilot.py`,
  `src/report_results.py` and the tests are original code written for this audit with AI
  assistance (Claude, Anthropic). They do not copy upstream source files.
* The ALGORITHM_AUDIT tier re-expresses the arithmetic of a few upstream expressions (the
  packer weighting, the unpack shift expression, and three split-dim and eff-bit formulas) in
  plain Python so they can be tested without torch. Short expressions are quoted in comments,
  docs, and test strings for the purpose of commentary and identification of the defect.
  Whether re-expressing a formula is an Adapted Material question under CC BY-NC 4.0 has not
  been resolved here. To stay on the safe side, treat this audit as NonCommercial research and
  keep this attribution with it.
* The UPSTREAM_IMPORT tier loads unmodified upstream files at runtime from the local checkout.
  The upstream code is not redistributed by this audit; anyone running it needs their own copy.
* `quantization/utils/quant_util.py`: the scripts in `src/` do not import it; `src/audit.py` reads
  it for static observations only. It was, however, imported and executed, unmodified, in a
  separate integration run (noted in `evidence/READINESS.md`): its official state-dict loading
  function (`_load_and_process_state_dict`) was run on a synthetic single-layer safetensors file,
  and the result is recorded in
  `evidence/official-loader-integration.json` (corrected decoder: sign agreement 1.0 and bitwise
  equal layer output; shipped decoder: sign agreement 0.53 to 0.59). The full-model loader
  `quant_util.load_quantized_model` was not run. No part of `quant_util.py` is copied here.

## Obligations if this audit is shared

* Keep this file and the attribution above (creators, link to the license, link to the
  repository, commit hash).
* Indicate that the audit is not affiliated with or endorsed by Samsung or the authors
  (CC BY-NC 4.0 Section 2(a)(6)).
* NonCommercial use only for anything that includes or adapts upstream material.

## License of the original audit code

Not yet chosen by the author. Until a license is chosen, all rights are reserved by the author
for the original audit code, subject to the upstream terms above for any adapted portions.
`LICENSE_PROPOSAL.md` is a proposal for the owner to decide on; it does not license or relicense
anything. Publishing this repository on GitHub (release v0.1.0) does not grant a license either:
apart from what GitHub's terms of service allow (viewing and forking on GitHub), no permission to
copy, modify or redistribute the original portions is given until a license is enacted, and the
CC BY-NC 4.0 obligations above continue to apply to every adapted portion.
