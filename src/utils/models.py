"""
Utilities for loading models and tokenizers from HuggingFace.
Handles device placement and dtype configuration per models.yaml.
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str | Path = "configs/models.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_model_config(shortname: str, config_path: str | Path = "configs/models.yaml") -> dict:
    """Return the config dict for a model by its shortname."""
    cfg = load_config(config_path)
    for m in cfg["models"]:
        if m["shortname"] == shortname:
            return m
    raise ValueError(f"Model '{shortname}' not found in {config_path}")


def load_model_and_tokenizer(shortname: str, device: str = "cuda"):
    """Load a model and tokenizer by shortname. Returns (model, tokenizer)."""
    cfg = get_model_config(shortname)
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"],
        torch_dtype=dtype,
        device_map=device,
        output_hidden_states=True,
    )
    model.eval()
    return model, tokenizer
