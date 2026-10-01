# Sources and licenses

The DSPark architecture, anchor sampling, masking, label alignment, and original
loss conventions in `nanospec/dspark.py`, `nanospec/layers.py`, `nanospec/train.py`,
`nanospec/ops.py` and `nanospec/objective.py` are adapted from
[LightSeek TorchSpec](https://github.com/lightseekorg/TorchSpec/tree/076a738fdd7e64be1495338f055e8188784633c4),
Copyright (c) 2026 LightSeek Foundation, under the MIT license included in
`licenses/TorchSpec-MIT.txt`. TorchSpec credits DeepSeek's MIT-licensed
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) for DSPark's Markov head and
original loss conventions (Copyright (c) 2026 The DeepSpec Authors;
`licenses/DeepSpec-MIT.txt`). The reverse schedule, explicit sharding,
and explicit optimizer are nanospec implementations.

`experiments/reference.py` reuses TorchSpec's installed backbone and Markov
implementation and adapts its label alignment and position weighting to the
CE + KL objective. It uses autograd only as a numerical reference.

The Qwen3.5 weight layout, architecture and chunked delta-rule arithmetic in
`nanospec/qwen.py`, `nanospec/train.py` and `nanospec/ops.py` are adapted and simplified from
[Hugging Face Transformers v5.3.0](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
Copyright 2025 The Qwen Team and The HuggingFace Inc. team, under
Apache 2.0 (`licenses/Transformers-Apache-2.0.txt`). Changes include explicit
checkpoint tensors, synchronous pipeline partitioning, text-only inference,
and a plain PyTorch FP8 quantize/dequantize projection path.

Model checkpoint weights retain their upstream license. They are downloaded
into a Modal volume and are not distributed in this repository.
