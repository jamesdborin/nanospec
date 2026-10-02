"""Small, append-only JSONL journal shared by training and other metric producers."""

import json
import math
from pathlib import Path


def append_metrics(path, row):
    """Append and flush one scalar-valued row without overwriting resumed history.

    An interrupted final record is separated from the next one so readers can skip
    it. Non-finite floats become JSON null, including a skipped optimizer's norm.
    Only one process should write a given journal (rank zero during training).
    """
    row = {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in row.items()
    }
    encoded = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell():
            stream.seek(-1, 2)
            if stream.read(1) != b"\n":
                stream.write(b"\n")
        stream.write(encoded)
        stream.flush()
