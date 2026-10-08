# Bounded next experiment, pending review

The existing 16-step run is wiring and resource evidence, not a trained compression assessment. No new run is authorized by this proposal.

Freeze seeded document-disjoint training and validation windows from the official splits before optimization. Reserve the test split until the stopping decision. Start with 64 steps, then extend to at most 256 only if gradients are finite, validation NLL improves, and the memory guard remains comfortable. Evaluate validation every 32 steps. Declare convergence only when relative validation NLL improvement is below 1% across three successive checkpoints, with no significant regression. Hitting the cap without convergence means undertrained, not a negative verdict on LittleBit.

Use the same native tokenizer and token-weighted evaluation for original Qwen, initialized factors, trained factors, and corrected reloaded factors. Require exact sign agreement and CPU logit round-trip checks. Add a validated local 4-bit baseline before comparative utility claims; a smaller pretrained model is a separate family comparison, not a matched ablation. Declare any differences from the paper objective. A paper-faithful intermediate-loss configuration needs its own two-step memory/time feasibility test before its budget estimate is trusted.

Measured current 128-token, batch-one CPU steps took 2.53 to 5.32 seconds. A 256-step continuation is approximately 11 to 23 minutes plus conversion, validation and saving; reserve a 30-minute total cap. This extrapolation applies only to the current objective and shape. Use at least 64 held-out windows for the next diagnostic evaluation, and ultimately the full standard test set before benchmark claims. Never tune against test losses.

Observed OS peak was 6.53 GiB on the 16 GiB M4 Mac, exceeding the sampled 6 GiB guard. Expect roughly 6.5 to 8 GiB and require a fresh available-memory check and a continuous external process monitor. Sampling is not a hard memory cap. The current Docker engine has about 3.9 GiB and cannot contain this run as configured. Changing engine memory or the rejected devcontainer configuration requires separate approval; no such change was made.

Report actual checkpoint bytes, resident tensors, sampled and OS peak RSS, unchanged KV cache, prefill and decode latency separately. Theoretical 11.6x potential speedup is not a measured acceleration. Keep all seeds, windows provenance, loss trajectories, checkpoints hashes and failures.
