# Circuit Breakers Integration — Plan

Mechanistic extension of the displacement framework: use the per-turn harmfulness direction `v_harmful(k)` to explain *why* circuit breakers (Zou et al., 2024) fail against multi-turn attacks, despite being designed as a representation-level defense.

## 1. Motivation

Circuit breakers are the canonical single-turn representation-level defense. Zou et al. fine-tune the target model so that representations at chosen mid-layers are pushed toward a direction orthogonal to their single-turn harmfulness axis — the "rerouting" direction. Bullwinkel et al. (2025) showed empirically that Crescendo still bypasses them, but did not characterize *why*. This project's displacement finding — that `v_harmful(k)` rotates away from the single-turn baseline `v_harmful(ST)` as context accumulates — provides a natural candidate explanation:

> Circuit breakers target `v_harmful(ST)`, but the multi-turn attack's effective harmfulness axis is `v_harmful(k)`. By late turns these are nearly orthogonal, so the defense's intervention is pointed at a stale direction and misses what actually fires.

This plan turns that intuition into a measurable claim and a section of the paper.

## 2. Hypotheses

**H-CB1 (Rerouting direction mismatch).** The circuit-breakers-trained model's "rerouting direction" at its intervention layers is aligned with `v_harmful(ST)` of the base model (high cosine), but only weakly aligned with `v_harmful_full(k)` at later turns (low cosine).

**H-CB2 (Residual displacement).** On a circuit-breakers-trained model, the displacement pattern still holds: `cos(v_full(k), v_ST)` drops with k on the trained model, similarly to the base model. The defense does not close the geometric gap.

**H-CB3 (Residual multi-turn jailbreak rate).** Crescendo/ActorAttack/X-Teaming ASR on the CB model remains substantially above chance and is correlated with the magnitude of residual displacement.

**H-CB4 (Single-turn ASR drop without multi-turn ASR drop).** CB training drops single-turn JBB ASR sharply (confirming it works in the regime it was trained for) but leaves multi-turn ASR largely intact.

If all four hold, the framing for the paper is: *"displacement geometrically explains the known failure mode of circuit breakers against multi-turn attacks."*

## 3. Required artifacts

### Target model
- `GraySwanAI/Llama-3-8B-Instruct-RR` (public CB release; base = Llama-3-8B-Instruct)
- Base control: `meta-llama/Meta-Llama-3-8B-Instruct` (not 3.1, to match)

### Data
- Full 4-condition extraction on both models:
  - CB model: conversations + trajectories + no-context + compressed + single-turn baseline
  - Base Llama-3 model: same (control)
- 20 attempts × 100 JBB × 3 frameworks per (model, split)

### Metadata about the defense
- Which layers CB training intervenes at (needed to locate the rerouting direction). Documented in Zou et al.'s paper and repo — typically a contiguous mid-layer block.
- The rerouting loss formulation (determines the expected direction).

## 4. Methodology

### 4.1 Data generation
- New notebook folder: `notebooks/llama_3_8B_RR_instruct/` with `01_data_generation.ipynb` and `03_extract_hidden_states.ipynb`
- Base control: `notebooks/llama_3_8B_instruct/` (new, parallel)
- Data layout: `data/llama_3_RR/` and `data/llama_3_base/`
- System prompt: the same "helpful, respectful, honest" prompt as Llama-3.1 runs (Llama-3 supports system role natively)
- Everything downstream (nb02 / nb08 / nb09 / nb10 / nb12 / nb13) runs unchanged after flipping `MODEL` at the top

### 4.2 Direction computation on the CB model
For each condition (full-context, no-context, compressed, single-turn):
- Compute `v_harmful(k)` and `v_harmful(ST)` at every layer, using within-pair subtraction on the CB model's hidden states.
- These are the CB-model-specific directions, computed identically to the Llama-3.1 runs.

### 4.3 Locating the rerouting direction

The CB model differs from the base at the rerouting layers. Two complementary ways to estimate the rerouting direction:

**Option A — weight-difference method.** Let `ΔW_l = W_l^CB − W_l^base` at the intervened layers. The top singular vector of `ΔW_l` approximates the direction along which the CB training modified the representation manifold. This is the "rerouting direction."

**Option B — activation-difference method.** On paired (harmful, benign) JBB goals, compute `h_l^CB − h_l^base` per example. The mean difference per goal, averaged across examples, gives the empirical rerouting direction at layer `l`.

Use both; they should agree up to sign and magnitude. Report as `v_rr_l` per intervened layer.

### 4.4 Key measurements

For each intervened layer `l`:

| Measurement | Interpretation under H-CB1 |
|---|---|
| `cos(v_rr_l, v_harmful_base(ST)_l)` | high — CB was trained on single-turn |
| `cos(v_rr_l, v_harmful_base(k=1)_l)` | high — k=1 is near single-turn |
| `cos(v_rr_l, v_harmful_base(k=3)_l)` | intermediate |
| `cos(v_rr_l, v_harmful_base(k=k_max)_l)` | low — multi-turn direction is orthogonal |

**H-CB1 prediction:** this cosine decays monotonically with k, paralleling the base model's own `cos(v_full(k), v_ST)` displacement curve.

For the residual-displacement analysis on the CB model (**H-CB2**), rerun nb08/nb12's displacement figures with `MODEL = "llama_3_RR"`. Expected pattern: the CB model has its own displacement curve that looks qualitatively similar to the base model's, even though its v_ST has shifted.

For **H-CB3**, correlate per-attempt jailbreak outcome (binary SR-jailbroken label) with per-attempt displacement magnitude (average `||h_k − h_1||` or similar proxy). Positive correlation → attempts that displace more are more likely to succeed.

For **H-CB4**, compute ASR on single-turn JBB goals (no attack framing) for both the base and the CB models, contrasted with Crescendo/ActorAttack/X-Teaming ASR. Expected: sharp drop in single-turn ASR on CB, modest or no drop in multi-turn.

## 5. Analysis notebooks

All new notebooks live at `notebooks/` (analysis is shared across models) or as a single dedicated notebook:

| Notebook | Purpose |
|---|---|
| `14_circuit_breakers_analysis.ipynb` | All CB-specific analyses: rerouting direction (A+B), cosine-with-v_harmful table, H-CB1/H-CB2/H-CB3/H-CB4 figures |
| `notebooks/llama_3_8B_RR_instruct/*.ipynb` | Data generation + extraction for CB model (same 2-notebook template as Gemma/Qwen) |
| `notebooks/llama_3_8B_instruct/*.ipynb` | Control, same template |

Existing shared notebooks (08, 09, 10, 12, 13) extend transparently when their `MODEL` constant is set to `llama_3_RR` or `llama_3_base`.

## 6. Figures for the paper

- **Fig CB-1 — Rerouting direction vs base model's v_harmful(k).** Per intervened layer, line plot of `cos(v_rr, v_harmful_base(k))` across k. One panel per framework. Prediction: descending line.
- **Fig CB-2 — CB model's own displacement.** Replicate nb08 Fig 1 with `MODEL=llama_3_RR`. Two lines per framework (`cos(v_full(k), v_ST)` and `cos(v_full(k), v_nc(k))`). Prediction: same shape as base Llama-3.
- **Fig CB-3 — Single-turn vs multi-turn ASR drop.** Bar chart: base ASR vs CB ASR for (single-turn JBB, Crescendo, ActorAttack, X-Teaming). Prediction: large gap only in single-turn column.
- **Fig CB-4 — Per-attempt displacement × jailbreak success.** Scatter or binned curve of displacement magnitude vs jailbreak probability, on CB model. Prediction: monotonic positive.

## 7. Expected contribution

> **Claim for the paper's CB section:** On Llama-3-8B-Instruct-RR, circuit breakers reroute representations along a direction that is close to `v_harmful(ST)` and far from `v_harmful(k)` at later turns (H-CB1). The displacement pattern persists under CB training (H-CB2). CB drops single-turn ASR substantially but multi-turn ASR minimally (H-CB4), and residual displacement magnitude correlates with attack success (H-CB3). The displacement phenomenon geometrically accounts for the known multi-turn failure of circuit breakers: they protect a direction that attacks have already moved past by the point of jailbreak.

This upgrades the paper's contribution from "displacement correlates with single-turn defenses degrading" to "displacement mechanistically explains the failure of the canonical representation-level defense."

## 8. Risks and caveats

1. **Model version mismatch.** CB release is Llama-3 (not 3.1). Direct comparisons with your existing Llama-3.1 runs are not apples-to-apples; must regenerate a base-Llama-3 dataset alongside the CB dataset. Cost: another ~$75 in attacker/judge API calls for the matched base.

2. **Intervention-layer uncertainty.** Zou et al.'s released model may not document the exact intervened layers. If unclear, identify them from `ΔW` analysis (Option A) — layers with the largest `||ΔW_l||_F` are the intervention targets. Include this as a methodology step.

3. **Rerouting direction is not necessarily rank-1.** The top singular vector of `ΔW_l` is an approximation; the true intervention may span a small subspace. Report cosine with top-k principal rerouting directions, not just the top-1.

4. **CB ASR might be nearly identical to base ASR.** If CB training barely moves ASR on multi-turn attacks (likely per Bullwinkel et al.), H-CB4 is easy to confirm but the "drop in single-turn" side still needs a clean measurement. Make sure the single-turn comparison is rigorous (same prompts, same decoding parameters).

5. **Attacker adaptation.** GPT-4o-as-attacker might behave differently against a CB model because the CB model's refusals look different. Sanity-check attempt-1 behavior on CB before full generation — confirm the attacker doesn't degenerate.

6. **Hidden-state extraction at intervened layers.** CB training may change the numerical range of activations at intervened layers. Check for NaN/Inf during extraction and ensure projection metrics are normalized consistently.

## 9. Timeline estimate

Approximately **one focused week** of researcher time plus GPU time for the regenerated data:

| Step | Days |
|---|---|
| Scaffold `llama_3_8B_RR_instruct/` + `llama_3_8B_instruct/` notebooks | 0.5 |
| Generate conversations on both models (3 frameworks × 2 splits each, vLLM runs) | 1.0 |
| Extract hidden states under all 4 conditions, both models | 1.0 |
| Compute rerouting direction (Options A and B, all intervened layers) | 0.5 |
| Run `14_circuit_breakers_analysis.ipynb` — all 4 hypotheses | 1.0 |
| Write up the CB section + produce paper figures | 1.5 |
| Slack for debugging template issues, sanity checks, figure polish | 0.5 |

GPU cost: approximately the same as generating one additional model's data (~$75 API + GPU time equivalent to a single-model pass for each of base-Llama-3 and Llama-3-RR).

## 10. What this plan intentionally does *not* cover

- **Training a new circuit-breakers variant.** Follow-up paper, option 3 in the earlier discussion.
- **Multi-turn-aware CB loss design.** Same.
- **Experimenting with other representation-level defenses** (e.g., RepControl, HarmBench-style defense training). Could extend the section but is out of scope for the initial contribution.
- **Interpreting the rerouting direction's alignment with arbitrary axes** (topic, refusal, etc.). Focus stays on the v_harmful family because that's what the paper's framework measures.

## 11. Next step if approved

Scaffold `notebooks/llama_3_8B_instruct/` (base control) and `notebooks/llama_3_8B_RR_instruct/` (CB model) from the Llama-3.1 template. Adjust `MODEL_ID` to the Llama-3 weights and to `GraySwanAI/Llama-3-8B-Instruct-RR` respectively; everything else stays the same since both use `<|eot_id|>` as the user-end token and native system role. Data folder convention: `data/llama_3_base/` and `data/llama_3_RR/`.

Start with a smoke-test generation (3 pairs, 2 attempts, Crescendo only) on both models to verify the vLLM + attacker + judge path works before committing to the full run.
