"""
Main extraction orchestrator.

Loads each conversation from the manifest, runs both Design A and Design B
extraction in the same forward passes where possible, and writes results to
representations.jsonl via src/utils/io.py.
"""

import yaml
from pathlib import Path
from tqdm import tqdm

from src.utils.io import write_record
from src.utils.models import load_model_and_tokenizer
from src.data_generation.filter import load_manifest
from src.extraction.design_a import extract_design_a
from src.extraction.design_b import extract_design_b


def run_extraction(
    manifest_path: str | Path,
    model_shortname: str,
    config_path: str | Path = "configs/extraction.yaml",
    output_jsonl: str | Path = "data/representations/representations.jsonl",
    arrays_dir: str | Path = "data/representations/arrays/",
) -> None:
    """Extract hidden states for all conversations in the manifest for one model."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_and_tokenizer(model_shortname)
    conversations = load_manifest(manifest_path)

    for conv in tqdm(conversations, desc=f"Extracting [{model_shortname}]"):
        results_a = extract_design_a(conv, model, tokenizer, **cfg["extraction"])
        results_b = extract_design_b(conv, model, tokenizer, **cfg["extraction"])

        for result in results_a + results_b:
            array = result.pop("array")
            write_record(result, array, output_jsonl, arrays_dir)
