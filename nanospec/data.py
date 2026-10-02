"""Pinned, streaming OpenPerfectBlend chat tokens for the training CLI."""

from collections import deque

import torch

DATASET = "mlabonne/open-perfectblend"
DATASET_REVISION = "af60f3c18201652a83a93f46fcfee1b646ba3df7"


def open_perfectblend_batches(tokenizer_path, batch_size, seq_len, *, examples=None, start_step=0):
    """Pack chats in source order, cycling without padding and replaying on resume.

    Only one streamed row and the pending token buffer are held in memory. The
    first ``examples`` source rows are selected before any empty rows are skipped.
    All chat tokens (including prompts and delimiters) are supervised.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if batch_size < 1 or seq_len < 2 or start_step < 0:
        raise ValueError("Need positive batch_size, seq_len >= 2 and start_step >= 0")
    if examples is not None and examples < 1:
        raise ValueError("dataset-examples must be positive")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    source = load_dataset(
        DATASET, name="default", split="train", revision=DATASET_REVISION, streaming=True
    )
    if examples is not None:
        source = source.take(examples)
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    pending = deque()
    size = batch_size * seq_len
    skip = start_step * size
    while True:
        token_count = 0
        for index, row in enumerate(source):
            try:
                messages = [
                    {"role": roles[m["from"]], "content": m["value"]}
                    for m in row["conversations"] or []
                ]
                if any(not isinstance(m["content"], str) for m in messages):
                    raise ValueError("message content must be text")
                if not any(m["role"] == "user" for m in messages):
                    continue
                tokens = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_dict=False
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid OpenPerfectBlend conversation at row {index}") from error
            token_count += len(tokens)
            dropped = min(skip, len(tokens))
            skip -= dropped
            pending.extend(tokens[dropped:])
            while len(pending) >= size:
                yield torch.tensor(
                    [pending.popleft() for _ in range(size)], dtype=torch.long
                ).reshape(batch_size, seq_len)
        if token_count == 0:
            raise ValueError("OpenPerfectBlend selection contains no usable chat tokens")
