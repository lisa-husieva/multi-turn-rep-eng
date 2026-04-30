# Experiment: Context Accumulation and the Displacement of Harmful/Benign Representations in Multi-Turn Jailbreaks

## Overview

This document describes the design, methodology, data requirements, and implementation details for a study of how context accumulation in multi-turn jailbreak attacks affects the internal representation of harmfulness in large language models.

---

## Research Question

**Does context accumulation in multi-turn jailbreaks systematically displace the harmful/benign distinction in representation space, and does this displacement explain why single-turn detection methods fail?**

Three sub-questions that build on each other:

1. Does the harmful/benign direction change as context accumulates turn by turn, and is this change driven by context rather than just message content?
2. Does this displacement cause fixed-direction detection methods to degrade, while adaptive methods remain robust?
3. At turns where displacement is largest, what is the model doing behaviorally — and does this connect displacement to jailbreak success?

---

## Background and Motivation

### Why single-turn methods fail on multi-turn attacks

Single-turn jailbreak defenses — including circuit breakers (Zou et al., 2024) and Latent Guard (Zhao et al., 2025) — assume that a stable "harmfulness direction" exists in the model's representation space, and that monitoring or blocking representations along this direction is sufficient to detect harmful inputs. These methods work well against single-turn attacks but have been shown to fail against multi-turn attacks like Crescendo (Bullwinkel et al., 2025).

The key intuition is that in multi-turn attacks, no single message is explicitly harmful. The attacker gradually steers the conversation toward a harmful goal over several turns, and by the final turn the model has accumulated enough context to comply, but each individual message appeared benign. This means that any defense anchored to a fixed "harmful" direction — learned from single-turn data — may not see the accumulated danger.

### Connection to prior work

**Zhao et al. (2025) — "LLMs Encode Harmfulness and Refusal Separately"**  
Identified two token positions that encode different safety-relevant signals in single-turn settings:
- `t_inst`: the last token of the user's instruction — encodes **harmfulness** (whether the instruction is harmful, independent of how the model responds)
- `t_post_inst`: the end-of-turn special token (e.g., `<|eot_id|>`) — encodes **refusal** (whether the model will refuse, independent of whether the instruction is actually harmful)

This separation is causal: steering along the harmfulness direction causes the model to internally perceive a harmless instruction as harmful, while steering along the refusal direction elicits refusal without changing the model's internal belief about harmfulness.

**Bullwinkel et al. (2025) — "A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks"**  
Applied representation engineering to Crescendo attacks, but with two important limitations:
1. Their probe was trained on **single-turn data** and applied to multi-turn outputs — they measured "does this look like a single-turn harmful response" rather than studying what happens to representations turn by turn
2. They extracted representations from **the response tokens of the final jailbroken response** — not from input-side positions like Zhao et al.'s `t_inst` / `t_post_inst`

**This work** directly extends Zhao et al.'s positions to multi-turn settings and studies what happens to the harmful/benign distinction *at each turn* as context accumulates. This is the gap neither prior work filled.

---

## Hypotheses

**H1 (Displacement):** The harmful/benign direction `v_harmful(k)` — computed from hidden states at turn k — has decreasing cosine similarity with the single-turn baseline direction `v_harmful(ST)` as k increases. This displacement is driven by context accumulation, not just by message content.

**H2 (Detection failure):** A classifier anchored to the single-turn baseline direction degrades in AUROC as turns accumulate, while a per-turn adaptive classifier maintains performance. The gap between these two is the practical cost of displacement.

**H3 (Behavioral correlation):** Turns where displacement is largest correspond to the model's most cooperative behavioral responses, connecting geometric displacement to jailbreak success.

---

## Experimental Setup

### Model

- **Target:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **System prompt:** Standard helpful assistant system prompt (consistent across all conversations in the dataset)

### Attack frameworks

- **Crescendo** (primary): Gradual escalation through seemingly benign steps. Longest conversations, clearest trajectory signal from preliminary results. Analyze first.
- **ActorAttack** (secondary): Different attack strategy, used for replication and comparison.
- **X-Teaming** (secondary): Third framework for cross-framework generalization.

### Dataset

- **JBB benchmark**: 100 harmful goals + 100 benign goals, matched by topic into 100 pairs
- **Attempts per goal**: 20 per framework
- **Sample sizes**: Crescendo ~3970 conversations, ActorAttack ~4000, X-Teaming ~4000
- **Labels**: goal-level (`harmful` / `benign`), conversation-level success (`attack_success`), turn-level refusal (`is_refusal`), turn-level success (`judge_success`)

### Token positions

Following Zhao et al., hidden states are extracted at two positions per turn:

- **`h_inst`** (`t_inst`): last token of the k-th attacker (user) message, just before the closing `<|eot_id|>`. In Zhao et al., this position encodes harmfulness.
- **`h_post`** (`t_post_inst`): the `<|eot_id|>` token that closes the k-th attacker message. In Zhao et al., this position encodes refusal.

**Important caveat:** Whether these positions encode the same things in multi-turn settings is an open empirical question. The single-turn interpretation assumes the model processes the instruction in isolation. In multi-turn settings, self-attention means these positions absorb information from the entire prior context. The positions are used here to enable direct comparison with Zhao et al., while explicitly acknowledging that the multi-turn interpretation may differ.

### Layers

Based on preliminary MLP probe results showing peak performance around layers 14–20:
- Save layers **14, 20, and 32** for flexibility
- Primary analysis on layer 20; sensitivity check on 14 and 32

---

## Data Collection: The K-Turn Extraction

This is the central new data collection task. It enables the controlled comparison between fixed-direction and per-turn-adaptive classifiers.

### Core idea

For each conversation of length n, instead of extracting hidden states once with the full conversation in context, extract n times — once for each prefix length k = 1, 2, ..., n. At prefix k, the model has seen only the first k attacker messages and k-1 model responses. This allows us to study how the representation of the same semantic content (a harmful vs. benign goal) changes as context accumulates.

### What a prefix-k input looks like

For a conversation with turns `[(user_1, assistant_1), (user_2, assistant_2), ..., (user_n, assistant_n)]`, the prefix-k input is:

```
[system prompt]
[user_1] [assistant_1]
[user_2] [assistant_2]
...
[user_{k-1}] [assistant_{k-1}]
[user_k]                          ← stops here, NO assistant_k
```

The sequence ends after `user_k`'s message, before the assistant's response at turn k. This mirrors Zhao et al.'s setup: we extract at `t_inst` (last token of the user message) and `t_post_inst` (end-of-turn token), which are both part of the user turn before any response is generated.

### What we extract

At each prefix k, we extract:
- `h_inst`: hidden state at the last token of `user_k`'s message (token immediately before `<|eot_id|>`)
- `h_post`: hidden state at the `<|eot_id|>` token that closes `user_k`'s turn
- At layers 14, 20, 32

Each extracted vector has shape `(4096,)` for Llama-3.1-8B.

### Three extraction conditions

The experiment requires three distinct extraction conditions for each conversation turn:

**Condition 1: Full-context extraction** (k-turn prefix)  
Input = turns 1 through k, as described above. This is the primary condition. The hidden state at turn k reflects both the content of `user_k`'s message AND the accumulated influence of all prior turns via self-attention.

**Condition 2: No-context extraction** (single-message)  
Input = only `user_k`'s message, with no prior history. Same token positions extracted. The hidden state reflects only the content of the k-th message in isolation, without any contextual influence.

The difference between Condition 1 and Condition 2 for the same (conversation, k) is the **pure effect of context accumulation**. Anything that differs between them cannot be due to the message content — it must be due to the accumulated history.

**Condition 3: Compressed single-turn**  
Input = the entire multi-turn conversation concatenated into a single user message, passed as one prompt with no turn structure. One extraction per conversation (not per turn). This tests whether the context effect is driven by semantic content alone (in which case compressed should match full-context) or whether the turn structure and role-based attention patterns matter (in which case they should differ). Bullwinkel et al. found representations were nearly identical for compressed vs. multi-turn — this is an important replication check.

**Condition 4: Single-turn baseline**  
Input = the original JBB goal passed directly as a single-turn prompt, with no multi-turn framing. One extraction per goal (not per conversation). This produces `v_harmful(ST)`, the anchor direction representing what any single-turn defense would be trained on.

---

## Implementation Details

### Handling the conversation JSON

Each conversation is stored as a JSON with a `turns` list alternating `role: "user"` and `role: "assistant"`. The `turn_idx` field is shared between user and assistant entries at the same turn.

**Critical:** Exclude any turn with `rolled_back: true` when building prefixes. Rolled-back turns were not part of the conversation the model actually experienced — including them would misrepresent the context the model had at each step.

```python
def get_non_rolled_back_turns(turns):
    """Return only turns that were not rolled back, preserving order."""
    return [t for t in turns if not t.get("rolled_back", False)]

def get_prefix_messages(turns, k, system_prompt):
    """
    Build the message list for prefix k.
    
    turns: filtered list (rolled-back excluded), alternating user/assistant
    k: include the first k user messages (1-indexed)
       assistant responses are included for turns 1..k-1 only
    """
    messages = [{"role": "system", "content": system_prompt}]
    user_turn_count = 0

    for turn in turns:
        if turn["role"] == "user":
            user_turn_count += 1
            messages.append({"role": "user", "content": turn["content"]})
            if user_turn_count == k:
                break  # stop before including assistant_k
        elif turn["role"] == "assistant":
            if user_turn_count < k:
                messages.append({"role": "assistant", "content": turn["content"]})

    return messages
```

### Applying the chat template

Use `tokenizer.apply_chat_template` with `add_generation_prompt=False`. This closes the final user turn with `<|eot_id|>` without adding the assistant's opening tokens, leaving `<|eot_id|>` as the last token in the sequence.

```python
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=False,
    return_tensors="pt"
)
# input_ids shape: (1, seq_len)

tpost_inst_idx = input_ids.shape[1] - 1   # <|eot_id|>
tinst_idx      = input_ids.shape[1] - 2   # last token of user_k message
```

**Caveat:** This assumes the chat template ends the user turn with exactly one special token (`<|eot_id|>`). Verify this for your exact `transformers` version before running at scale. If the template adds additional tokens after `<|eot_id|>` (e.g., a newline or BOS), the indices above will be wrong.

### Extracting hidden states efficiently

Use forward hooks rather than `output_hidden_states=True` to avoid storing all 33 layer activations in memory simultaneously.

```python
LAYERS_TO_SAVE = [14, 20, 32]

def extract_with_hooks(model, input_ids, tinst_idx, tpost_inst_idx, layers):
    hidden_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output[0] shape: (batch, seq_len, hidden_dim)
            hidden_cache[layer_idx] = output[0].detach().cpu().float()
        return hook

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        hooks.append(h)

    with torch.no_grad():
        model(input_ids.to(model.device))

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx in layers:
        acts = hidden_cache[layer_idx]  # (1, seq_len, 4096)
        result[f"h_inst_L{layer_idx}"]  = acts[0, tinst_idx, :].numpy()
        result[f"h_post_L{layer_idx}"] = acts[0, tpost_inst_idx, :].numpy()

    return result
```

### Storage format

Save one file per conversation to keep the number of files manageable and allow partial reruns:

```python
# Flattened key structure for npz:
# turns_{k}_h_inst_L14, turns_{k}_h_post_L14, etc.

save_dict = {
    "conversation_id": conv["conversation_id"],
    "objective_pair_id": conv["objective_pair_id"],
    "goal_type": conv["goal_type"],
    "attack_framework": conv["attack_framework"],
    "attack_success": conv["attack_success"],
    "jailbreak_turn": conv.get("jailbreak_turn", -1),
    "executed_turns": conv["executed_turns"],
}

for k, turn_data in extractions.items():
    for key, arr in turn_data.items():
        save_dict[f"turn_{k}_{key}"] = arr
    # Also save per-turn behavioral labels
    save_dict[f"turn_{k}_is_refusal"] = turn_labels[k]["is_refusal"]
    save_dict[f"turn_{k}_judge_success"] = turn_labels[k]["judge_success"]

np.savez_compressed(f"extractions/{conv['conversation_id']}.npz", **save_dict)
```

Estimated storage: ~48KB per conversation × ~4000 conversations per framework × 3 frameworks ≈ **~600MB total** (comfortable).

### Parallelizing across 4 GPUs

Split the conversation list into 4 shards and run one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python extract.py --framework crescendo --shard 0 --n_shards 4 &
CUDA_VISIBLE_DEVICES=1 python extract.py --framework crescendo --shard 1 --n_shards 4 &
CUDA_VISIBLE_DEVICES=2 python extract.py --framework crescendo --shard 2 --n_shards 4 &
CUDA_VISIBLE_DEVICES=3 python extract.py --framework crescendo --shard 3 --n_shards 4 &
wait
```

Each process loads the model independently. No inter-GPU communication needed. Expected runtime: ~3–4 hours for Crescendo full-context extraction on 4 GPUs.

---

## Caveats and Potential Failure Modes

### 1. Token position assumptions may not hold in multi-turn

In Zhao et al., `t_inst` encodes harmfulness because the model processes a single user instruction and the last token of that instruction aggregates information about the full request. In multi-turn settings, `t_inst` at turn k still aggregates information about `user_k`'s message, but via self-attention it also aggregates information about all prior turns. This means the multi-turn `t_inst` hidden state is not a clean signal about the k-th message alone — it is a signal about the k-th message *given the full prior context*. This is arguably the right thing to measure (it is what the model is actually computing), but it should not be interpreted the same way as the single-turn `t_inst`.

**Mitigation:** The no-context condition (Condition 2) isolates the message-only signal. Comparing full-context to no-context extractions at the same turn gives a direct measure of how much the prior context is influencing the hidden state.

### 2. Chat template behavior with `add_generation_prompt=False`

Different versions of `transformers` may handle `add_generation_prompt=False` differently. In some versions, the template may append a newline or other token after the closing `<|eot_id|>`, shifting `tpost_inst_idx`. Always verify empirically.

**Mitigation:** After tokenizing, decode `input_ids[0, -1]` and `input_ids[0, -2]` and confirm they are `<|eot_id|>` and the last substantive token of the user message respectively.

### 3. Rolled-back turns

The Crescendo framework supports backtracking — if an attack turn doesn't progress, it is rolled back and a different turn is tried. Rolled-back turns have `rolled_back: True` in the JSON. Including these in the prefix would give the model context it didn't actually have during the attack, invalidating the comparison with behavioral labels.

**Mitigation:** Always filter to `rolled_back: False` before constructing prefixes. The `executed_turns` field gives the expected number of non-rolled-back turns — use this as a check.

### 4. Variable conversation lengths across frameworks

Crescendo conversations vary in length (1–10 turns). ActorAttack and X-Teaming are more concentrated. This means the k-sweep plots will have different support on the x-axis per framework, and averaging across conversations at turn k=8 includes only the subset of conversations long enough to reach turn 8.

**Mitigation:** Report the number of conversations contributing to each turn-k estimate. For turn-level analyses, condition on conversations of at least length k. Do not average across frameworks at fixed k without noting sample size differences.

### 5. The harmful/benign direction conflates message content and context

`v_harmful(k)` is computed as the mean difference between harmful-goal and benign-goal hidden states at turn k. This direction captures whatever distinguishes those two sets of conversations at that point, which includes (a) the semantic content of turn k's message, (b) the accumulated context from prior turns, and (c) any structural regularities that differ between harmful and benign conversations (e.g., attacker strategy patterns, topic distributions).

**Mitigation:** The no-context vs. full-context comparison (Condition 1 vs. Condition 2) directly isolates the contribution of accumulated context. Topic is addressed by the JBB pairing structure: each harmful/benign pair shares a topic. Using `objective_pair_id` to hold out entire topic pairs during cross-validation prevents the direction from being confounded by topic in the AUROC evaluation.

### 6. The "rotation" claim requires careful wording

Observing that `cos(v_harmful(k), v_harmful(1))` decreases does not unambiguously imply that the direction is rotating. It could alternatively be:
- The signal weakening (direction becoming less separable, not just pointing elsewhere)
- The direction rotating (signal remains strong but in a different subspace)
- Both simultaneously

These are distinguishable: if the signal is merely weakening, the per-turn AUROC should also drop. If the direction is rotating, the per-turn AUROC remains high even as the fixed-direction AUROC drops. The gap between fixed and adaptive classifiers in Figure 2 is the key diagnostic.

**Preferred framing:** "Context accumulation systematically *displaces* the harmful/benign direction" rather than "rotates." Displacement is weaker and more defensible — it makes no claim about the path the direction takes, only that it ends up somewhere different.

---

## Sanity Checks

Run these on 5–10 conversations before launching the full extraction. Do not proceed to full-scale extraction until all checks pass.

### Check 1: Token position verification

```python
# After tokenizing, decode the last two tokens and confirm
last_token_id   = input_ids[0, -1].item()
second_last_id  = input_ids[0, -2].item()

last_token_str   = tokenizer.decode([last_token_id])
second_last_str  = tokenizer.decode([second_last_id])

print(f"tpost_inst token: '{last_token_str}'")     # should be '<|eot_id|>'
print(f"tinst token:      '{second_last_str}'")    # should be last word/subword of user_k message

# Also check the full decoded sequence ends correctly
decoded = tokenizer.decode(input_ids[0])
print(decoded[-200:])  # visually inspect the end of the sequence
```

**Expected:** `tpost_inst` decodes to `<|eot_id|>`. `tinst` decodes to the final token of the user message (e.g., a punctuation mark or the last word). The decoded sequence should end with the user message followed by `<|eot_id|>` and nothing else.

### Check 2: Hidden state shape

```python
for key, arr in extracted.items():
    assert arr.shape == (4096,), f"Unexpected shape for {key}: {arr.shape}"
print("All shapes correct.")
```

### Check 3: Prefix construction correctness

Visually inspect one conversation at each prefix length k = 1, 2, n:

```python
for k in [1, 2, conv["executed_turns"]]:
    messages = get_prefix_messages(turns, k, system_prompt)
    print(f"\n--- Prefix k={k} ---")
    for m in messages:
        role = m["role"]
        content = m["content"][:100]  # first 100 chars
        print(f"[{role}]: {content}...")
    print(f"Total messages: {len(messages)}")
    # Check: should have 1 system + k user + (k-1) assistant = 2k messages
    expected = 1 + k + (k - 1)
    assert len(messages) == expected, f"Expected {expected} messages, got {len(messages)}"
```

### Check 4: Consistency with existing extractions

You already have hidden states extracted from full conversations. For k = n (the final prefix), your new k-turn extraction should produce the same hidden states as your existing full-conversation extraction (since they use the same context).

```python
# Compare h_inst_L20 at k=n between new extraction and existing extraction
existing = load_existing_extraction(conversation_id)
new = extract_full_context_prefix(conversation, k=n)

cos_sim = np.dot(existing, new) / (np.linalg.norm(existing) * np.linalg.norm(new))
print(f"Cosine similarity between existing and new (k=n): {cos_sim:.6f}")
# Expected: > 0.9999. If not, the prefix construction or token position is wrong.
```

### Check 5: No-context vs full-context are actually different

The no-context extraction (Condition 2) should differ from the full-context extraction (Condition 1) for k > 1. If they are identical, context is not influencing the hidden states, which would suggest an implementation error.

```python
for k in [1, 3, 5]:
    h_full    = extract_full_context(conversation, k)["h_inst_L20"]
    h_nocontext = extract_no_context(conversation, k)["h_inst_L20"]
    
    cos = np.dot(h_full, h_nocontext) / (
        np.linalg.norm(h_full) * np.linalg.norm(h_nocontext)
    )
    print(f"k={k}: cos(full_context, no_context) = {cos:.4f}")

# Expected pattern:
# k=1: should be ~1.0 (no prior context to differ on)
# k=3: should be noticeably < 1.0 (context has accumulated)
# k=5: should be even further from 1.0
# If k=3 and k=5 are also ~1.0, something is wrong with context inclusion.
```

### Check 6: Harmful and benign directions are separable

Before committing to the full extraction, verify that `v_harmful(k)` at a mid-layer (e.g., layer 20) actually discriminates between harmful and benign conversations. Compute it on a small subsample (e.g., 50 harmful + 50 benign conversations) and check AUROC.

```python
# For a small pilot:
projections_harmful = [np.dot(h, v_harmful) for h in harmful_hidden_states]
projections_benign  = [np.dot(h, v_harmful) for h in benign_hidden_states]

from sklearn.metrics import roc_auc_score
labels = [1]*len(projections_harmful) + [0]*len(projections_benign)
scores = projections_harmful + projections_benign
auroc  = roc_auc_score(labels, scores)
print(f"Pilot AUROC: {auroc:.3f}")
# Expected: > 0.65 at minimum. If near 0.5, the direction is not separating the classes.
```

### Check 7: Rolled-back turns are correctly excluded

```python
# Verify that executed_turns matches the number of non-rolled-back user turns
filtered_turns = get_non_rolled_back_turns(conv["turns"])
n_user_turns = sum(1 for t in filtered_turns if t["role"] == "user")
assert n_user_turns == conv["executed_turns"], (
    f"Mismatch: executed_turns={conv['executed_turns']}, "
    f"actual non-rolled-back user turns={n_user_turns}"
)
```

---

## Analysis Plan

### Direction computation

For each framework and each turn k, compute:

```
v_harmful(k) = normalize(
    mean(h_inst at turn k | goal_type=harmful) -
    mean(h_inst at turn k | goal_type=benign)
)
```

Computed separately for full-context and no-context conditions. Use conversations 1–16 (out of 20 attempts per goal) for direction computation; reserve attempts 17–20 for evaluation.

### Figure 1 — Direction displacement across turns

For each framework, plot turn k on the x-axis against:
- `cos(v_harmful_full(k), v_harmful(ST))` — drift from single-turn baseline
- `cos(v_harmful_full(k), v_harmful_no_context(k))` — contribution of context accumulation
- Compressed single-turn as a horizontal reference line

Repeat for both `h_inst` and `h_post` positions (two rows of panels).

### Figure 2 — Detection failure and recovery

At each turn k, evaluate two classifiers on held-out attempts (17–20):

- **Fixed:** project onto `v_harmful(ST)` (single-turn baseline direction)
- **Adaptive:** project onto `v_harmful_full(k)` (turn-specific direction)

Report AUROC at each k per framework. The widening gap between fixed and adaptive is the central result.

### Supporting analysis — Behavioral correlation

For each turn k and each conversation, plot the projection onto `v_harmful_full(k)` against the behavioral label (`is_refusal`, `judge_success`). Ask whether compliant turns cluster at lower harmfulness projections than refused turns. Run at the final turn only for the refusal direction comparison with Zhao et al.

---

## Priority Order for Data Collection

1. **Crescendo full-context extraction** (Conditions 1) — primary framework, run first
2. **Crescendo no-context extraction** (Condition 2) — needed for controlled comparison
3. **Single-turn baseline** (Condition 4) — needed for anchor direction
4. **Compressed single-turn** (Condition 3) — important control, can run in parallel with 3
5. **ActorAttack and X-Teaming** (Conditions 1 and 2) — replication, run after Crescendo analysis is confirmed

---

## References

- Zhao et al. (2025). *LLMs Encode Harmfulness and Refusal Separately.* arXiv:2507.11878
- Bullwinkel et al. (2025). *A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks.* arXiv:2507.02956
- Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405
- Zou et al. (2024). *Improving Alignment and Robustness with Circuit Breakers.* arXiv:2406.04313
- Chao et al. (2024). *Jailbreaking Black Box Large Language Models in Twenty Queries (JBB).* arXiv:2310.08419
- Russinovich et al. (2024). *Great, Now Write an Article About That: The Crescendo Multi-Turn Jailbreak Attack.*
