"""
Hidden state extraction runner — shared logic for notebook 03.

Extracts hidden states at the Zhao et al. token positions (t_inst, t_post_inst)
across four extraction conditions:

    1. Full-context trajectory (per conv, per turn k = 1..n)
    2. No-context           (per conv, per turn k; system + user_k only)
    3. Compressed            (per conv; full conversation flattened into one user msg)
    4. Single-turn baseline  (per JBB goal; no attack framing)

All conditions write `(N, n_layers, hidden_dim)` float16 arrays as `h_inst.npy`
and `h_post_inst.npy`, plus a `metadata.parquet` alongside.
"""

import json
import sys
import threading
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── Conversation parsing ─────────────────────────────────────────────────────

def get_accepted_turns(conv):
    """
    Return list of accepted (non-rolled-back) turn pairs.
    Each element: (user_content, asst_content, asst_turn_dict).

    Crescendo has rolled_back / is_refusal flags on retried turns;
    ActorAttack and X-Teaming lack these fields (all turns accepted).
    """
    turns = conv.get("turns", [])
    by_idx = {}
    for t in turns:
        by_idx.setdefault(t["turn_idx"], {})[t["role"]] = t

    accepted = []
    for turn_idx in sorted(by_idx):
        pair = by_idx[turn_idx]
        user_t = pair.get("user")
        asst_t = pair.get("assistant")
        if not user_t or not asst_t:
            continue
        if user_t.get("rolled_back", False) or asst_t.get("rolled_back", False):
            continue
        accepted.append((user_t["content"], asst_t["content"], asst_t))
    return accepted


# ─── Message builders (one per condition) ─────────────────────────────────────

def build_fullcontext_messages(conv, k):
    """Condition 1: turns 1..k with prior assistant replies, ending at user_k."""
    accepted = get_accepted_turns(conv)
    system_prompt = conv.get("target_system_prompt", "")
    if not accepted or k > len(accepted):
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for user_c, asst_c, _ in accepted[: k - 1]:
        messages.append({"role": "user", "content": user_c})
        messages.append({"role": "assistant", "content": asst_c})
    messages.append({"role": "user", "content": accepted[k - 1][0]})
    return messages


def build_nocontext_messages(conv, k):
    """Condition 2: system prompt + user_k only — no prior history."""
    accepted = get_accepted_turns(conv)
    system_prompt = conv.get("target_system_prompt", "")
    if not accepted or k > len(accepted):
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": accepted[k - 1][0]})
    return messages


def build_compressed_messages(conv):
    """Condition 3: all turns concatenated into one user msg with plain-text prefixes."""
    accepted = get_accepted_turns(conv)
    system_prompt = conv.get("target_system_prompt", "")
    if not accepted:
        return None

    parts = [f"User: {u}\nAssistant: {a}" for u, a, _ in accepted[:-1]]
    parts.append(f"User: {accepted[-1][0]}")
    compressed = "\n\n".join(parts)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": compressed})
    return messages


# ─── Token positions & forward pass ───────────────────────────────────────────

def get_positions(tokenizer, input_ids, user_end_token="<|eot_id|>"):
    """
    For Llama-3.x: `<|eot_id|>` closes every user turn.

        t_post_inst = position of the last `<|eot_id|>` in the sequence
                      (closes the final user message)
        t_inst      = t_post_inst - 1
                      (last content token of the final user message)
    """
    eot_id = tokenizer.convert_tokens_to_ids(user_end_token)
    eot_pos = (input_ids[0] == eot_id).nonzero(as_tuple=True)[0]
    t_post_inst = eot_pos[-1].item()
    t_inst = t_post_inst - 1
    return t_inst, t_post_inst


@torch.no_grad()
def extract_at_positions(model, tokenizer, messages, layer_indices,
                         dtype=torch.bfloat16, user_end_token="<|eot_id|>"):
    """
    One forward pass. Returns (h_inst, h_post, t_inst, t_post_inst, seq_len).
    On OOM, returns (None, None, t_inst, t_post_inst, seq_len).
    """
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=False
    ).to(model.device)

    t_inst, t_post_inst = get_positions(tokenizer, input_ids, user_end_token)
    seq_len = input_ids.shape[1]

    try:
        with torch.autocast(device_type="cuda", dtype=dtype):
            outputs = model(input_ids, output_hidden_states=True)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, t_inst, t_post_inst, seq_len

    h_inst = np.stack([
        outputs.hidden_states[l][0, t_inst, :].cpu().to(torch.float16).numpy()
        for l in layer_indices
    ])
    h_post = np.stack([
        outputs.hidden_states[l][0, t_post_inst, :].cpu().to(torch.float16).numpy()
        for l in layer_indices
    ])

    del outputs, input_ids
    torch.cuda.empty_cache()
    return h_inst, h_post, t_inst, t_post_inst, seq_len


# ─── Save + resume helpers ────────────────────────────────────────────────────

def save_arrays(save_dir, h_inst_list, h_post_list, meta_list,
                layer_indices=None, n_transformer_layers_total=None):
    """Concatenate and save arrays + metadata parquet.

    If `layer_indices` (1-indexed transformer-block IDs saved) and
    `n_transformer_layers_total` are given, also writes `layer_indices.json`
    so the shared analysis notebooks can read the per-model sweep.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    h_inst = np.concatenate(h_inst_list, axis=0).astype(np.float16)
    h_post = np.concatenate(h_post_list, axis=0).astype(np.float16)
    meta = pd.concat(meta_list, ignore_index=True)
    np.save(str(save_dir / "h_inst.npy"), h_inst)
    np.save(str(save_dir / "h_post_inst.npy"), h_post)
    meta.to_parquet(save_dir / "metadata.parquet", index=False)

    if layer_indices is not None and n_transformer_layers_total is not None:
        mapping = {
            "n_transformer_layers_original": int(n_transformer_layers_total),
            "n_sweep": len(layer_indices),
            "saved_positions_0_indexed": list(range(len(layer_indices))),
            "transformer_layers_1_indexed": [int(l) for l in layer_indices],
            "labels": [f"L{int(l)}" for l in layer_indices],
        }
        (save_dir / "layer_indices.json").write_text(json.dumps(mapping, indent=2))

    print(f"  Saved → {save_dir}")
    print(f"  h_inst:      {h_inst.shape}  ({h_inst.nbytes / 1e9:.2f} GB)")
    print(f"  h_post_inst: {h_post.shape}  ({h_post.nbytes / 1e9:.2f} GB)")
    print(f"  rows:        {len(meta)}")


def load_done(save_dir, key_cols):
    """Load existing metadata (if any) and return resume key set + prior arrays."""
    save_dir = Path(save_dir)
    meta_path = save_dir / "metadata.parquet"
    if not meta_path.exists():
        return set(), [], [], []
    existing_meta = pd.read_parquet(meta_path)
    done = set(zip(*[existing_meta[c] for c in key_cols]))
    eh_inst = [np.load(str(save_dir / "h_inst.npy"))]
    eh_post = [np.load(str(save_dir / "h_post_inst.npy"))]
    return done, eh_inst, eh_post, [existing_meta]


# ─── Per-condition unit iterators ─────────────────────────────────────────────

def iter_trajectory_units(conv):
    """Yield (unit_meta, messages) per turn k for Condition 1."""
    accepted = get_accepted_turns(conv)
    n = len(accepted)
    for k in range(1, n + 1):
        asst = accepted[k - 1][2]
        unit_meta = {
            "turn_k": k,
            "n_accepted_turns": n,
            "is_refusal": bool(asst.get("is_refusal", False)),
            "judge_success": bool(asst.get("judge_success", False)),
        }
        yield unit_meta, build_fullcontext_messages(conv, k)


def iter_nocontext_units(conv):
    """Yield (unit_meta, messages) per turn k for Condition 2."""
    accepted = get_accepted_turns(conv)
    n = len(accepted)
    for k in range(1, n + 1):
        asst = accepted[k - 1][2]
        unit_meta = {
            "turn_k": k,
            "n_accepted_turns": n,
            "is_refusal": bool(asst.get("is_refusal", False)),
            "judge_success": bool(asst.get("judge_success", False)),
        }
        yield unit_meta, build_nocontext_messages(conv, k)


def iter_compressed_units(conv):
    """Yield a single (unit_meta, messages) per conv for Condition 3."""
    accepted = get_accepted_turns(conv)
    if not accepted:
        return
    yield {"n_accepted_turns": len(accepted)}, build_compressed_messages(conv)


def make_base_meta_fn(framework, split):
    """Returns a callable (conv, fpath) -> dict of conversation-level metadata."""
    def fn(conv, fpath):
        return {
            "conversation_id": conv.get("conversation_id", fpath.stem),
            "pair_id": conv["objective_pair_id"],
            "goal_type": conv.get("goal_type", split),
            "framework": framework,
            "attempt": conv.get("attempt", 1),
            "attack_success": bool(conv.get("attack_success", False)),
        }
    return fn


# ─── Main extraction loops ────────────────────────────────────────────────────

def run_condition(
    *,
    files,
    iter_units,
    base_meta_fn,
    save_dir,
    model_id,
    logical_gpu_ids,
    layer_indices,
    dtype,
    resume_key_cols,
    desc,
    user_end_token="<|eot_id|>",
    n_transformer_layers_total=None,
):
    """
    Multi-GPU parallel extraction over conversation files for one condition.

    iter_units(conv) yields (unit_meta, messages) pairs — unit_meta supplies
        any condition-specific resume key fields (e.g. turn_k).
    base_meta_fn(conv, fpath) returns conversation-level metadata including
        the pair_id and attempt that together with unit_meta form the full
        resume key.
    resume_key_cols: tuple of metadata columns that uniquely identify each row.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    done, eh_inst, eh_post, emeta = load_done(save_dir, resume_key_cols)
    if done:
        print(f"Resuming: {len(done)} entries already present in {save_dir.name}")

    # Count total units (one fast pass).
    total = 0
    for fp in files:
        conv = json.loads(fp.read_text())
        total += sum(1 for _ in iter_units(conv))
    print(f"Total units: {total}   Remaining: {total - len(done)}")

    n_gpus = len(logical_gpu_ids)
    chunks = [files[i::n_gpus] for i in range(n_gpus)]

    all_h_inst = list(eh_inst)
    all_h_post = list(eh_post)
    all_meta = list(emeta)

    pbar = tqdm(total=total - len(done), desc=desc,
                file=sys.stdout, dynamic_ncols=True)
    lock = threading.Lock()

    def worker(gpu_id, chunk):
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        print(f"GPU {gpu_id}: loading {model_id} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        # Load on CPU first, then move explicitly. `device_map={"": device}` routes
        # through accelerate's meta-tensor fastpath, which breaks on models with
        # tied embeddings (Gemma-2): `lm_head.weight` stays on `meta` and the
        # dispatch raises "Cannot copy out of meta tensor".
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        mdl = mdl.to(device)
        mdl.eval()
        print(f"GPU {gpu_id}: ready, {len(chunk)} files", flush=True)

        b_inst, b_post, b_meta = [], [], []
        for fpath in chunk:
            conv = json.loads(fpath.read_text())
            base_meta = base_meta_fn(conv, fpath)

            for unit_meta, messages in iter_units(conv):
                full_meta = {**base_meta, **unit_meta}
                key = tuple(full_meta[c] for c in resume_key_cols)
                if key in done:
                    continue
                if messages is None:
                    with lock:
                        pbar.update(1)
                    continue

                h_inst, h_post, t_inst, t_post, seq_len = extract_at_positions(
                    mdl, tok, messages, layer_indices, dtype, user_end_token
                )
                if h_inst is None:
                    print(f"  OOM: {fpath.name} key={key} seq_len={seq_len}",
                          flush=True)
                    with lock:
                        pbar.update(1)
                    continue

                b_inst.append(h_inst)
                b_post.append(h_post)
                b_meta.append({
                    **full_meta,
                    "seq_len": seq_len,
                    "t_inst": t_inst,
                    "t_post_inst": t_post,
                    "fname": fpath.name,
                })
                with lock:
                    pbar.update(1)

        with lock:
            if b_inst:
                all_h_inst.append(np.stack(b_inst))
                all_h_post.append(np.stack(b_post))
                all_meta.append(pd.DataFrame(b_meta))

    threads = [threading.Thread(target=worker, args=(g, c))
               for g, c in zip(logical_gpu_ids, chunks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pbar.close()

    save_arrays(save_dir, all_h_inst, all_h_post, all_meta,
                layer_indices=layer_indices,
                n_transformer_layers_total=n_transformer_layers_total)


def run_single_turn_baseline(
    *,
    goal_type,
    goals_df,
    system_prompt,
    save_dir,
    model_id,
    logical_gpu_ids,
    layer_indices,
    dtype,
    user_end_token="<|eot_id|>",
    n_transformer_layers_total=None,
):
    """Extract hidden states for raw JBB goals (Condition 4). Single-GPU, fast."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{logical_gpu_ids[0]}"
    print(f"Loading {model_id} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    # See matching comment in run_condition — avoid the meta-tensor fastpath
    # to keep tied-weight models (e.g. Gemma-2) loading cleanly.
    mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    mdl = mdl.to(device)
    mdl.eval()

    h_inst_list, h_post_list, meta_list = [], [], []
    for _, row in tqdm(goals_df.iterrows(), total=len(goals_df),
                       desc=f"single_turn_{goal_type}",
                       file=sys.stdout):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row["Goal"]})
        h_inst, h_post, t_inst, t_post, seq_len = extract_at_positions(
            mdl, tok, messages, layer_indices, dtype, user_end_token
        )
        h_inst_list.append(h_inst)
        h_post_list.append(h_post)
        meta_list.append({
            "pair_id": int(row["Index"]),
            "goal_type": goal_type,
            "behavior": row["Behavior"],
            "category": row["Category"],
            "seq_len": seq_len,
            "t_inst": t_inst,
            "t_post_inst": t_post,
        })

    save_arrays(
        save_dir,
        [np.stack(h_inst_list)],
        [np.stack(h_post_list)],
        [pd.DataFrame(meta_list)],
        layer_indices=layer_indices,
        n_transformer_layers_total=n_transformer_layers_total,
    )
