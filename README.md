# nanospec

A small, explicit PyTorch starting point for a speculative-draft training
speedrun on one **8 × H100 SXM NVLink** node. The target is
`Qwen/Qwen3.5-27B-FP8`; the draft uses **DSPark**'s Qwen-style backbone and Markov head.

Models and layers store weights in ordinary `nn.Module`, `nn.ModuleList`,
`nn.Linear`, `nn.Embedding`, `nn.RMSNorm`, `nn.Conv1d`, and registered
`nn.Parameter` objects. Parameters have `requires_grad=False`. The top-level
[`train()`](nanospec/train.py) function lays out verifier inference, NVLink
transfers, draft forward, CE + KL, draft backward, and the optimizer call in order.
Weight containers have no architecture forward/backward methods; `ops` provides
the tensor operations and handwritten local derivatives.
The runtime has no autograd graph, `Tensor.backward`, `autograd.grad`, automatic
FSDP, `torch.optim`, Transformers model execution, or inference engine.
`experiments/compare.py` imports the reference implementations solely to measure
numerical agreement. There is deliberately no unit or end-to-end test suite.

See [VALIDATION.md](VALIDATION.md) for measured agreement, hardware evidence,
loss trajectories and checkpoint results.

## Read the implementation

| File | Responsibility |
| --- | --- |
| `nanospec/train.py` | One visible verifier → transfer → draft → loss → reverse → optimizer schedule, plus CLI setup |
| `nanospec/dspark.py` | DSPark weight tree, checkpoint layout, anchor sampling and masks |
| `nanospec/layers.py` | Draft layer weight containers and named activation buffers |
| `nanospec/ops.py` | Tensor operations: projections, norms, RoPE, attention, FP8, delta rule and explicit derivatives |
| `nanospec/objective.py` | CE + KL loss, named loss metrics and explicit derivatives |
| `nanospec/shard.py` | Parameter all-gathers, gradient reduce-scatters, FP32 master weights/moments, clipping and AdamW |
| `nanospec/qwen.py` | Qwen weight containers and FP8 checkpoint loading |
| `nanospec/distributed.py` | GPU ownership, verifier partition metadata and CUDA broadcast buffers |
| `nanospec/checkpoint.py` | Explicit per-rank optimizer/model checkpoints and resume |
| `experiments/compare.py` | Installed TorchSpec + Hugging Face numerical comparison runs |
| `experiments/reference.py` | TorchSpec backbone and Markov head with an independent autograd CE + KL objective |
| `experiments/overlap.py` | Paired serial/overlap timing, bitwise state comparisons and a CUDA profiler trace |
| `modal_app.py` | Machine provisioning and `torchrun` launch |

Training ranks are `0..N-1`, with **N = 1, 2 or 4**. The remaining GPUs own
contiguous partitions of the 64-layer teacher. Training waits for all inference
to finish. Captured decoder outputs, final normalized hidden states and teacher
logits travel as CUDA buffers through NCCL. No hidden-state CPU copies or
inference/training overlap occur.

For N > 1, trainable parameters, gradients, master weights and Adam moments are
sharded explicitly. Two reusable weight buffers alternate between the current
unit and the next prefetched unit. A communication stream gathers unit `i+1`
while the compute stream executes unit `i`; backward uses the same scheme in
reverse order. Completed gradients are reduce-scattered asynchronously while
the preceding layer computes, using two separate gradient buffers.

CUDA events guard weight readiness, the last compute-side read before weight
buffer reuse, and reduction completion before gradient buffer reuse. The compute
stream waits for all reductions before clipping, AdamW or gradient inspection.
Optimizer states stay sharded. The buffers are sized for the largest unit, so
overlap trades additional persistent GPU memory for less exposed communication.
Saved activation dataclasses hold activations,
never gathered weight views. N = 1 uses the same arithmetic with local tensors
and no sharding collectives. Frozen target embeddings and LM heads are replicated
on training GPUs. The optimizer normalizes by the **globally pooled weighted
token count**, including unequal supervision masks across ranks.

The training code reads like `ops.linear(n, attn.q_proj.weight)` and
`ops.linear_backward(dy, saved, weight, weight.grad)`. The layer loops show norms,
projections, rotary embeddings, attention, gates and residuals directly. Each
shard binds gathered weight views to registered parameters and binds a gradient
buffer to `Parameter.grad` before the handwritten backward pass. For N > 1,
parameter storage is empty
between gathers; the registered module hierarchy and parameter identities persist.
`model.state_dict()` collectively exports full CPU weights, including the frozen
embedding/head. `model.draft_state_dict()` exports only TorchSpec's draft weights.
All training ranks must participate in either export when sharded. Full state
loading, `model.to(...)`, and optimizer checkpoint resume preserve shard ownership;
optimizer masters and moments remain FP32.

`train(ids, drafter, objective, verifier=teacher, optimizer=optimizer)` runs a
complete synchronous step. Numerical comparisons call the same function with a
`TeacherBatch` of fixed tensors and omit the optimizer to inspect raw gradients.
The verifier-only form, `train(ids, verifier=teacher)`, executes the identical
inference and transfer operations for comparison with Hugging Face.
`overlap=False` selects the serial sharding schedule for paired measurements;
the default overlaps draft communication when using two or four training GPUs.

## Objective and reference pins

Training and comparisons use **0.1 CE + 0.9 KL(target ‖ draft)**. Cross-entropy
supervises the next token in the training sequence; KL matches the teacher's
distribution. Both use the combined frozen LM-head and trainable Markov logits,
with position weights `exp(-slot / gamma)` (`gamma=4` by default).

Each training step also logs `accept_len`, DSPark-style simulated acceptance
length. For each slot, `a = sum_v min(softmax(target)_v, softmax(draft)_v)` is
the analytical rejection-sampling acceptance probability. Each block contributes
`1 + sum_k prod_{i=0..k} (a_i * valid_i)`, including the verifier bonus token.
The metric pools length sums and valid-block counts across training ranks;
blocks without supervision are excluded (an entirely empty batch reports zero).
All slots predict a next token, including slot 0. Binary supervision validity,
not exponentially decayed loss weights, determines the accepted prefix.
This is a teacher-forced training proxy, not measured decoding acceptance or
greedy token accuracy. `TrainResult.metrics` exposes the same metric when
calling `train()` directly.

The comparison builds DSPark's retained components directly from installed
TorchSpec: its DFlash backbone and vanilla Markov head. TorchSpec still owns
anchor sampling, masking, FlexAttention, automatic differentiation, FSDP and
the reference optimizer. An independent autograd CE + KL loss replaces its
upstream training objective; this is not a comparison to the upstream loss.

Every block slot predicts `anchor + slot + 1`, the Markov head sees the
teacher-forced previous token, and a supervision gap truncates the rest of the
block. Context attention is strictly before the anchor; draft attention is
bidirectional within its own block. The target embedding and LM head are frozen.
The mask embedding is Qwen's `<|fim_pad|>` (248063).

- TorchSpec: `076a738fdd7e64be1495338f055e8188784633c4`.
- Transformers: `5.3.0`; PyTorch: `2.10.0`.
- Qwen checkpoint: `709d2d2a46f204254221f583f45e00f93b578c9e`.
- Captured target decoder outputs: `[1, 16, 31, 46, 61]` (zero based).
  These correspond to Hugging Face `hidden_states[i + 1]`, not `hidden_states[i]`.

## Run on Modal

Compare serial and overlapped draft training on the same H100 allocation:

```bash
modal run modal_app.py --mode overlap --config qwen --train-gpus 4 --dtype bfloat16 --steps 6
```

This uses fixed teacher tensors, 128-token sequences and eight draft anchors.
It checks three updates bitwise (including raw gradients and Adam state), then
times complete draft forward/backward/optimizer steps in serial/overlap/overlap/
serial order, with three warmups per window. Inference is excluded to isolate
the changed phase. A separate CUDA profiler step records actual kernel overlap.
The JSON report and trace are saved in the Modal volume; the report is also
downloaded into `reports/`.

Authenticate the Modal CLI, then download the teacher into the persistent
`nanospec-artifacts` volume using a CPU container:

```bash
modal run modal_app.py --mode cache
```

`modal run modal_app.py --mode sequence --steps 3` runs the three draft-sharding
comparisons, full-teacher comparisons with the small and five-layer drafts,
and a KL training run in sequence,
inside one eight-GPU allocation. The default requests **`H100!:8`**, explicitly
disabling Modal's automatic H200 upgrade. `--gpu H200` is available for
supplementary diagnostics; report filenames identify the actual GPU type.

Compare fixed-hidden-state draft training with installed TorchSpec. Each run
allocates eight H100s, using the selected number for training:

```bash
modal run modal_app.py --mode draft --train-gpus 1 --dtype float32 --steps 3
modal run modal_app.py --mode draft --train-gpus 2 --dtype bfloat16 --steps 3
modal run modal_app.py --mode draft --train-gpus 4 --dtype bfloat16 --steps 3
```

Run the actual 27B FP8 teacher against Hugging Face, then train both drafts from
the same saved initial checkpoint on the same fixed dummy token IDs:

```bash
modal run modal_app.py --mode full --config qwen-parity --train-gpus 2 --dtype bfloat16 --steps 3
```

Both implementations receive identical seed-42 initial weights; the comparison
saves that common state as a checkpoint. The learning rate is `3e-4` for the tiny
arithmetic comparisons
and `1e-5` for the full teacher comparison. Override it with `--learning-rate`.

The Qwen comparison uses the full 64-layer teacher, full 248,320-token vocabulary,
5120-wide target states and frozen target embedding/head. `qwen-parity` uses a
smaller one-layer DSPark backbone to make iteration inexpensive. `qwen` selects
the five-layer, 5120-wide draft with 15,360 intermediate width and rank-256 Markov
head. `tiny` is only for isolated draft arithmetic comparisons.

Run the full five-layer comparison on four H100 training GPUs:

```bash
modal run modal_app.py --mode full --config qwen --train-gpus 4 --dtype bfloat16 --steps 3
```

Train with KL and write a resumable checkpoint:

```bash
modal run modal_app.py --mode train --config qwen --train-gpus 4 --dtype bfloat16 --steps 3
```

Reports are downloaded to `reports/`; initial checkpoints and training checkpoints
remain in the Modal volume. The report records numerical tolerances, rank-specific
losses, gradient/parameter errors, cosine similarities and hardware evidence.
For resume, pass the saved checkpoint directory as `--checkpoint`. Resume currently
requires the same dtype, configuration, objective and training GPU count.
Checkpoints and configurations from before the CE + KL simplification are
incompatible and must be regenerated; no migration or compatibility mode is provided.
Keep the same seed, sequence length and token data to continue the same data stream.
Each save creates a new checkpoint generation; its exact directory is in the report.

Add `--checkpoint-every N` to `--mode train` (or `python -m nanospec.train`) to save
resumable drafter weights and optimizer shards every N completed training steps.
The default `0` disables periodic saves; negative values are rejected. Scheduling
uses absolute steps after resume: resuming step 3 with N=2 saves at steps 4, 6, etc.
The final step is always saved, only once if it is also a scheduled step. Rank 0
logs each saved directory; training reports list all saves as `checkpoints`
(`step`, `path`) and keep `checkpoint` as the final directory. These are the existing
resumable model/optimizer shards, not an inference-engine export.

On an equivalent local node, install the core package with `pip install -e .`,
cache the pinned model, then use
`torchrun --standalone --nproc_per_node=8 -m nanospec.train`. The CLI also
accepts `--tokens /path/to/tokens.pt` (an integer tensor `[samples, seq_len]`),
`--seq-len`, `--anchors`, `--learning-rate` (training default `1e-5`), and `--seed`.
Without a token file it generates dummy data. A fixed dataset, tokenizer pipeline,
validation acceptance target and speedrun scoring rules have not yet been chosen.

## Numerical and performance boundaries

The reverse pass is fully explicit, including attention probabilities and their
derivatives. It matches the reference's mathematical operations and casting
boundaries, but it does **not** claim bitwise identity with fused FlexAttention
backward, fused SiLU derivatives or fused AdamW. Comparisons measure those
differences in FP32 and BF16; the FP32 reference run changes TorchSpec's fixed
BF16 policy to FP32 for a more sensitive diagnostic.

The Qwen path retains FP8 checkpoint weights and dynamic per-token, 128-channel
activation quantization. It expands weight scales and computes each projection
in FP32 before casting to BF16. This is a correctness implementation, **not an
FP8 tensor-core speed benchmark**. Hugging Face uses its own FP8 linear kernels
as the independent oracle. Text-only, equal-length, unpadded sequences are
supported; vision inputs, a KV cache and cached generation are outside this first
training path.

Attention matrices and vocabulary distributions are explicit allocations. Memory
therefore grows with context length, anchor count and vocabulary size; the initial
CLI defaults use short sequences and two anchors. No speed claims are made yet.

See `THIRD_PARTY_NOTICES.md` for source attribution and licenses.
