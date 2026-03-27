"""
Filter and index generated conversations by verdict.

Reads raw conversation files and produces a filtered manifest at
data/processed/conversations_manifest.jsonl that downstream extraction
can iterate over. Verdicts are preserved — filtering to jailbroken-only
or other subsets happens at analysis time, not here.
"""

from pathlib import Path


def build_manifest(
    conversations_dir: str | Path,
    output_path: str | Path = "data/processed/conversations_manifest.jsonl",
) -> None:
    """
    Walk conversations_dir, validate each file, and write a manifest
    with one entry per conversation including all metadata fields.
    """
    raise NotImplementedError


def load_manifest(manifest_path: str | Path) -> list[dict]:
    raise NotImplementedError
