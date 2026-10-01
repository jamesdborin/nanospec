# Historical numerical validation — 29 September 2026

These measurements describe the architecture and objectives before the CE + KL
simplification on 30 September. They do not validate the current loss or checkpoints.

These are recorded comparison experiments, not a unit or end-to-end test suite.
The full JSON reports in [`reports/`]() contain all training ranks,
per-step measurements, configured tolerances, source hashes, model revisions,
CUDA device names, CUDA peer-access checks, and active NVLink/NCCL topology.

## Module refactor

DSPark and the Qwen teacher now use registered PyTorch module trees. The manual
derivatives, AdamW arithmetic and sharding layout are preserved. Earlier reports
are archived in [`reports/pre-modules/`](pre-modules/).

A local replay against the previous implementation produced **exactly equal**
losses, hidden states, raw gradients and updated parameters for three steps in
both FP32 and BF16, under both the upstream and KL objectives. A four-layer
teacher replay with FP8 projections, two 70-token sequences, full attention and
gated delta attention also produced exactly equal captured states and logits.
Restoring an old checkpoint at step two and taking step three produced exactly
equal parameters, FP32 masters and Adam moments. Parameter placement and the
legacy flat layout were checked for every rank in 1/2/4-way sharding.
These local measurements used an RTX 3060 and are recorded in the
[module preservation report](module-refactor-local.json).

The 1/2/4-GPU TorchSpec comparisons below were rerun successfully on eight H100s
with the module implementation. The full five-layer draft comparison and a
four-GPU resume from an old checkpoint also passed on H100. No unit or
end-to-end test suite was added.

## Fixed-hidden-state comparisons on eight H100 80GB GPUs

Each draft starts from identical seed-42 weights. Both implementations receive
the same synthetic hidden states, token IDs, masks and anchor RNG state. The
three-step experiments compare forward activations, raw numerator gradients,
globally normalized and clipped gradient norms, AdamW updates, and updated
parameters. Different ranks have different supervision masks to exercise global
weighted-token normalization. The reference uses the installed pinned TorchSpec,
its BF16Optimizer, and its FSDP2 FULL_SHARD setup for BF16 runs.

| Training GPUs | Precision | Maximum relative loss difference | Minimum gradient cosine | Maximum parameter relative L2 | Report |
| --- | --- | --- | --- | --- | --- |
| 1 | FP32 | 0% | 0.9999999979 | 4.78e-8 | [JSON](draft-tiny-1-float32-torchspec-H100.json) |
| 2 | BF16 | 0.01072% | 0.99964031 | 5.11e-4 | [JSON](draft-tiny-2-bfloat16-torchspec-H100.json) |
| 4 | BF16 | 0.00946% | 0.99970085 | 5.61e-4 | [JSON](draft-tiny-4-bfloat16-torchspec-H100.json) |

The FP32 run has exactly equal reported losses: **1.9866549 → 1.9560004 →
1.9271874**. Explicit attention and derivatives change fusion and BF16 rounding;
the BF16 result is numerical agreement, not bitwise equivalence.

## Full teacher and training comparisons

The teacher is the actual pinned `Qwen/Qwen3.5-27B-FP8` checkpoint: all 64 decoder
layers, 5120-wide states and the full 248,320-token vocabulary. The independent
oracle is Hugging Face Transformers 5.3.0 using its own FP8 linear kernels,
eager full attention and native delta-rule fallback. No nanospec inference
functions are substituted into the oracle.

The full-teacher experiment uses fixed random token IDs of length 32, two draft
anchors, identical initial draft weights and a constant learning rate of `1e-5`.
The native-teacher activations feed nanospec, while Hugging Face activations feed
TorchSpec. Captured decoder outputs are layers 1, 16, 31, 46 and 61; final
normalized states and logits are also compared.

### Default five-layer draft on four H100 training GPUs

The full default architecture, **1,752,500,737 trainable parameters**, passes
the same comparison with four training GPUs and four native-teacher GPUs.
Hugging Face also runs on the last inference GPU as the independent oracle.
The minimum mean teacher hidden-state cosine across ranks and captured outputs
is **0.999076**, the minimum individual-token cosine is **0.995449**, and
the minimum mean logit cosine is **0.999717**.

| Step | nanospec loss | TorchSpec loss |
| --- | --- | --- |
| 1 | 3.7259264 | 3.7288537 |
| 2 | 3.7406931 | 3.7367184 |
| 3 | 3.4832444 | 3.4800241 |

The maximum relative loss difference is **0.10637%**, minimum gradient cosine
is **0.9992317**, and maximum updated-parameter relative L2 is **1.298e-4**.
The [full five-layer H100 report](full-qwen-4-bfloat16-torchspec-H100.json)
contains every rank's results and the common initial checkpoint location.

A supplementary [five-layer H200 run](full-qwen-4-bfloat16-torchspec-H200.json)
also passed, with maximum relative loss difference **0.14222%** and minimum
gradient cosine **0.9992383**. The H100 run predates a reference-adapter change
that detaches the frozen embedding before loading it into TorchSpec; the H200
run includes it. All nanospec runtime source hashes match both reports.

### Smaller draft diagnostic before the module refactor

The completed H100 run uses the smaller `qwen-parity` draft:
one 5120-wide layer, 175,826,209 trainable parameters, with the complete teacher
and vocabulary. Its worst mean hidden-state cosine across captured layers,
final normalization and ranks is **0.999055**; the worst individual token cosine
is **0.995569**. The worst mean logit cosine is **0.999717**.

| Step | nanospec loss | TorchSpec loss |
| --- | --- | --- |
| 1 | 3.9152672 | 3.9180396 |
| 2 | 3.5497193 | 3.5534801 |
| 3 | 3.3361762 | 3.3365688 |

The maximum loss difference is **0.10584%**, minimum gradient cosine is
**0.9993184**, and maximum updated-parameter relative L2 is **1.35e-4**.
See the [full H100 report](full-qwen-parity-2-bfloat16-torchspec-H100.json).
The [supplementary H200 report](full-qwen-parity-2-bfloat16-torchspec-H200.json)
also passes and records the same teacher cosine measurements.

## Requested KL objective and checkpoints

TorchSpec's original loss uses L1 distribution matching. Its comparison runs
therefore use `--objective torchspec`; the requested KL objective is the default
for nanospec training and is a separate run.

Before the module refactor, on four H100 training GPUs, `qwen-parity` completed
three KL updates with losses
**5.53241, 6.17675, 5.76457** on different dummy data each step. All gradients
were finite, all updates completed, and the checkpoint was saved. See the
[H100 KL training report](train-qwen-parity-4-bfloat16-kl-H100.json).

Also before the refactor, the full default five-layer draft, with
**1,752,500,737 trainable parameters**,
completed three KL updates on four H200 training GPUs while the other four GPUs
ran the full teacher. Losses were **5.30368, 5.53487, 5.28159** on different dummy
data each step. All gradients were finite and all optimizer updates completed.
The [report](train-qwen-4-bfloat16-kl-H200.json) records the saved
model/master-weight/Adam checkpoint in the Modal volume.

The module implementation restored the old three-step `qwen-parity` checkpoint
on four H100 training GPUs, advanced to step four with a finite KL loss of
**5.8915882**, and saved a new checkpoint generation. See the
[H100 resume report](train-qwen-parity-4-bfloat16-kl-H100-resume.json).
A supplementary [H200 resume](train-qwen-parity-4-bfloat16-kl-H200-resume.json)
also succeeded, with loss **5.8920727**.
Completed generations are not overwritten; `metadata.json` is written after
all rank files are complete. Resume requires the same configuration, dtype,
objective and GPU count. Keep the same seed, token data and sequence length to
continue the same data stream.

## Scope and reproducibility

- Teacher weights, TorchSpec and Transformers revisions are pinned in the code
  and reports. Exact commands are in the [README](../README.md).
- H200 results are supplementary and are explicitly labeled. H100 jobs use
  Modal's `H100!:8` request to prevent automatic hardware upgrades.
- The native teacher preserves FP8 storage and activation quantization but uses
  FP32 projection accumulation; these runs do not measure FP8 tensor-core speed.
- Short dummy sequences establish numerical agreement. They do not establish
  useful speculative acceptance, convergence on a real dataset, long-context
  memory limits, or a speedrun score.
- An earlier `3e-4` full-teacher probe produced similarly rising losses in both
  implementations and exceeded the parameter-relative-L2 tolerance. Its
  [report is retained](attempts/full-qwen-parity-lr3e-4-H200.json). The
  final run uses `1e-5`; its acceptance tolerances were not relaxed.
