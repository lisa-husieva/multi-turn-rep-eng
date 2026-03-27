"""
Design A extraction: r_n at varying k.

For each conversation, extract hidden states at the final response tokens r_n
while varying the number of context turns k from 1 to k_max. Replicates the
Bullwinkel et al. design.

At each k, the input to the model is the concatenated sequence of the final k
turns: x_k = p_{n-k+1} ∘ r_{n-k+1} ∘ ... ∘ p_n ∘ r_n.
Hidden states are extracted at the positions corresponding to r_n tokens.
"""

import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer


def extract_design_a(
    conversation: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layers: list[int],
    token_level_layers: list[int],
) -> list[dict]:
    """
    Run Design A extraction for one conversation.

    Returns a list of result dicts, one per (k, layer, aggregation),
    each containing the numpy array and metadata for write_record().
    """
    raise NotImplementedError
