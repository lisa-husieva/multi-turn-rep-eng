"""
Design B extraction: r_t at full context.

For each conversation, extract hidden states from each turn's own response r_t
at full context up to that turn:
  - r_1 with context p_1
  - r_2 with context p_1, r_1, p_2
  - ...
  - r_n with context p_1, r_1, ..., p_n

This is the more natural operationalization of trajectory — tracking how the
model represents its own responses as the conversation evolves.
"""

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer


def extract_design_b(
    conversation: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layers: list[int],
    token_level_layers: list[int],
) -> list[dict]:
    """
    Run Design B extraction for one conversation.

    Returns a list of result dicts, one per (t, layer, aggregation),
    each containing the numpy array and metadata for write_record().
    """
    raise NotImplementedError
