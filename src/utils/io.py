"""
Utilities for reading and writing representations.jsonl records and numpy arrays.

Each record in representations.jsonl has the schema:
{
    "conversation_id": str,
    "model": str,
    "attack_framework": str,       # crescendo | actorattack | xteaming | benign
    "verdict": str,                # jailbroken | refusal | near_miss | benign
    "harm_category": str,
    "objective_pair_id": int,      # links jailbroken and benign conversations on the same objective
    "extraction_design": str,      # A | B
    "k_or_t": int,                 # k (Design A) or t (Design B)
    "layer": int,
    "aggregation": str,            # mean_pool | token_level
    "array_path": str              # path to .npy file
}
"""

import json
import numpy as np
from pathlib import Path
from typing import Iterator


def iter_records(jsonl_path: str | Path) -> Iterator[dict]:
    """Iterate over records in representations.jsonl."""
    with open(jsonl_path) as f:
        for line in f:
            yield json.loads(line)


def load_array(record: dict) -> np.ndarray:
    """Load the numpy array referenced by a record."""
    return np.load(record["array_path"])


def write_record(record: dict, array: np.ndarray, jsonl_path: str | Path, arrays_dir: str | Path) -> None:
    """Save a numpy array to disk and append its record to the jsonl file."""
    arrays_dir = Path(arrays_dir)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    array_path = arrays_dir / f"{record['conversation_id']}_design{record['extraction_design']}_k{record['k_or_t']}_layer{record['layer']}_{record['aggregation']}.npy"
    np.save(array_path, array)

    record["array_path"] = str(array_path)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def filter_records(jsonl_path: str | Path, **kwargs) -> list[dict]:
    """Return records matching all provided key=value filters."""
    return [
        r for r in iter_records(jsonl_path)
        if all(r.get(k) == v for k, v in kwargs.items())
    ]
