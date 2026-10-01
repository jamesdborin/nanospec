# Numerical validation — 30 September 2026

These are recorded comparison experiments, not a unit or end-to-end test suite.
The current model has a frozen LM head and a trainable Markov head. Its only
objective is **0.1 CE + 0.9 KL(target ‖ draft)**.

The reference uses the installed pinned TorchSpec backbone, Markov head,
FlexAttention and optimizer. Its loss tail is an independent autograd CE + KL
implementation in `experiments/reference.py`. Both implementations start from
the same weights and use the same inputs and anchor RNG state. The comparison
measures losses, activations, raw gradients and updated parameters over three
steps. No nanospec derivative is used by the reference.

## Communication/computation overlap

The double-buffered schedule is compared directly with `train(overlap=False)`
using the full five-layer draft, BF16, 128 tokens and eight anchors on a Modal
8×H100 NVLink node. Both schedules start from identical weights and receive the
same fixed synthetic teacher tensors, tokens and anchor RNG state on each rank.
The three-step comparison hashes every hidden-state, reduced-gradient, parameter,
FP32-master and Adam-moment element, and compares losses and optimizer metrics.

| Training GPUs | Serial median | Overlap median | Step-time reduction | Numerical result | Report |
| --- | --- | --- | --- | --- | --- |
| 2 | 79.67 ms | 72.89 ms | 8.51% | Bitwise identical on every rank | [JSON](reports/overlap-qwen-2-bfloat16-ce-kl-H100.json) |
| 4 | 65.82 ms | 58.11 ms | 11.70% | Bitwise identical on every rank | [JSON](reports/overlap-qwen-4-bfloat16-ce-kl-H100.json) |

Timing includes draft forward, backward and AdamW; verifier inference and its
transfers are excluded. Four windows run in serial/overlap/overlap/serial order,
each with three warmups and six measured steps. Each sample uses synchronized
wall time and the maximum across training ranks. These are short measurements
for this workload, not an end-to-end throughput claim.

A separate CUDA profiler step confirms real kernel overlap: on four ranks,
12 of 14 all-gathers and six of seven reduce-scatters intersect compute kernels
on another stream. Summed intersections are 5.93 ms and 4.33 ms respectively.
The JSON retains per-collective durations, intersections, all per-rank hashes,
timing samples, source hashes and hardware topology. Full Chrome traces are
stored in the `nanospec-artifacts` Modal volume at the recorded trace paths.
Peak-memory samples include the persistent overlap buffers in both timing modes
and should not be interpreted as an independent serial-memory baseline.

The one-GPU fallback also preserves the previous implementation exactly across
the 24-step local replay described below. No unit or end-to-end suite was added.

## Explicit training schedule

Verifier inference, GPU transfers, draft forward, loss, backward and optimizer
execution now run through one top-level `train()` function. Weight containers
have no architecture forward/backward methods.

A replay against the previous class-based implementation produced **exactly
equal** losses, hidden states, raw gradients, parameters, FP32 masters and both
Adam moments over **24 steps**: three steps each in FP32 and BF16 for the default
draft, supplied teacher logits, omitted feature normalization, and omitted Markov
head. A four-layer FP8 verifier replay covering full attention and gated delta
attention on two 70-token sequences also matched exactly, including all captured
states, final normalization and logits.

The integrated optimizer call and checkpoint resume inside the new schedule
matched the previous implementation exactly over three steps. These one-off
RTX 3060 measurements are in the [schedule preservation report](reports/explicit-train-local.json).

The full **1,752,495,360-parameter**, five-layer draft then passed three steps on
four H100 training GPUs, while four other H100s ran the full 27B FP8 verifier.
Hugging Face provided independent verifier states to the TorchSpec-based draft.
Every training rank passed the existing tolerances; none were relaxed.

| Step | nanospec CE + KL | Reference CE + KL |
| --- | --- | --- |
| 1 | 5.1970716 | 5.2024398 |
| 2 | 4.4126735 | 4.4174132 |
| 3 | 6.1664872 | 6.2010474 |

Maximum relative loss difference was **0.55733%**, minimum gradient cosine
**0.9987750**, and maximum updated-parameter relative L2 **1.481e-4**. Across all
training ranks, the verifier's minimum mean hidden-state cosine against Hugging
Face was **0.9990760** and minimum mean logit cosine **0.9997168**. The
[full H100 report](reports/full-qwen-4-bfloat16-ce-kl-H100.json) records the common
initial checkpoint, all ranks, NVLink topology and source hashes from before the
communication-overlap change. These short random-token runs measure numerical agreement,
not convergence or speculative acceptance.

## CE + KL baseline comparisons before moving the execution schedule

All three-step comparisons passed. The local runs use a tiny draft with two
32-token sequences and unequal supervision masks. The Modal run uses eight H100
80GB GPUs, four training ranks, fixed synthetic teacher states, and explicit
parameter/gradient sharding against TorchSpec's FSDP2 reference.

| Hardware / training GPUs | Precision | Maximum relative loss difference | Minimum gradient cosine | Maximum parameter relative L2 | Report |
| --- | --- | --- | --- | --- | --- |
| RTX 3060 / 1 | FP32 | 0% | 0.9999999996 | 2.63e-8 | [JSON](reports/draft-tiny-1-float32-ce-kl-RTX3060.json) |
| RTX 3060 / 1 | BF16 | 0.02303% | 0.99998338 | 2.67e-4 | [JSON](reports/draft-tiny-1-bfloat16-ce-kl-RTX3060.json) |
| H100 / 4 | BF16 | 0.01263% | 0.99997264 | 3.14e-4 | [JSON](reports/draft-tiny-4-bfloat16-ce-kl-H100.json) |

The H100 loss trajectory was **1.0700260 → 1.0519366 → 1.0339181**, versus
**1.0700622 → 1.0519704 → 1.0340487** for the independent CE + KL reference.
The reports retain the measured source hashes. A subsequent normalization-weight
initializer cleanup was checked to produce identical tensors in FP32 and BF16.

Additional one-off checks confirmed that the removed configuration fields and
CLI objective selector are rejected, registered model parameters do not require
autograd, and new checkpoint metadata round-trips with only CE/KL loss weights.
Lint, formatting and compilation checks pass. No test suite was added.

These short comparisons validate the CE + KL objective and reduced head shard.
Their source hashes predate moving execution into the top-level training function.

## Historical results

The [29 September record](reports/2026-09-29-validation.md) preserves the original
and module-refactor measurements, including the full 27B teacher comparison to
Hugging Face and the five-layer draft runs on H100. Those reports apply to the
source hashes they record. They predate the removal of the confidence head and
L1/confidence losses and do not validate the current training objective.

Old checkpoints and configurations are unsupported. New checkpoints use the
smaller head shard and CE + KL objective metadata, with no compatibility path.
