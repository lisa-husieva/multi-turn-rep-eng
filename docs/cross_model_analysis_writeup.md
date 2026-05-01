# Cross-Model Analysis — Thought Process, Findings, and Interpretations

## Overview

This document records the full thought process, analytical decisions, and interpretations from the cross-model analysis notebook (`notebooks/cross_model_comparison.ipynb`). The analysis spans Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-2-9B-it, across three attack frameworks (Crescendo, ActorAttack, xteaming), using JBB as training data and WJB as held-out test data.

The central question: **do representation engineering directions trained on single-turn or multi-turn data capture something meaningful about harmfulness, and how does the geometry of those directions change across a multi-turn attack trajectory?**

---

## Background: What We're Measuring

### Token positions

Following Zhao et al. (2024), hidden states are extracted at two positions:

- **t_inst**: the last *content* token of the user message (one before the end-of-turn marker)
- **t_post_inst**: the end-of-turn marker itself (e.g. `<|eot_id|>` for Llama, `<|im_end|>` for Qwen, `<end_of_turn>` for Gemma)

Zhao et al. found that t_inst carries *harmfulness* signal and t_post_inst carries *refusal* signal. We use t_inst for harmfulness directions and t_post_inst for outcome directions.

### Extraction conditions

For each conversation at each turn k:

- **full_context**: `[system, u1, a1, ..., uk]` — model sees full history
- **no_context**: `[system, uk]` — only the final turn, no history
- **compressed**: `[system, flattened_history_as_one_user_msg]` — history concatenated as plain text
- **single_turn**: `[system, raw_jbb_goal]` — no framing, no turns; one extraction per goal

### Directions

- **v_inst[L]**: unit vector from topic-paired mean difference (harmful − benign) on JBB single-turn at t_inst. Framework-independent. Built once per model.
- **v_outcome[L, fw]**: framework-specific direction from success − failure at t_post on JBB final-turn multi-turn conversations.
- **v_bin[b, L]**: per-bin direction built from JBB multi-turn data binned by relative position `rel = (k−1)/(K−1)`, bins early/mid/late.
- **v_ctrl[L]**: direction built from JBB goals wrapped in a single neutral filler turn (no attack content). Control condition.

### Relative position binning

Absolute turn number k conflates two structurally different situations: at k=3, a conversation might be at its *final* turn (short conversation) or an *early intermediate* turn (long conversation). These have different representational signatures.

To address this, we use relative position `rel = (k−1)/(K−1)` where K is the total conversation length. Bins: early ≤ 1/3, mid 1/3–2/3, late > 2/3. This ensures each bin contains turns at the same *stage* of the attack trajectory.

---

## §3–§4: Harmfulness Separability at t_inst

### §3: Final-turn AUROC

**Question**: Can v_inst separate harmful from benign WJB conversations at the final turn?

**Finding**: AUROC is generally high (~0.7–0.9) for most model/framework/condition combinations. full_context consistently outperforms no_context, meaning conversation history contributes to separability beyond just the final turn content.

**Interpretation**: v_inst generalises from JBB single-turn to WJB multi-turn data reasonably well. The harmfulness direction learned from simple single-turn prompts captures something that persists into the final turn of a multi-turn attack.

### §4: Per-turn AUROC trajectories (relative bins)

**Question**: How does separability evolve across the attack trajectory?

**Finding**: AUROC increases from early to late bins. The gap between full_context and no_context is larger at late turns — history becomes more informative as the attack progresses.

**Key design decision**: replaced absolute-turn plots with relative position bins after recognising the terminal/intermediate confound. The bin approach correctly isolates trajectory position from conversation length.

---

## §5–§6: Outcome Direction at t_post

**Motivation**: Zhao et al. showed t_post_inst carries refusal signal. We test whether it also carries jailbreak *success* signal.

**v_outcome**: built from success−failure mean differences at t_post on JBB harmful final turns.

**Finding**: v_outcome separates successful from failed WJB attacks above chance, though less cleanly than v_inst separates harmful/benign. The pattern holds across bins (§6), with late turns showing stronger separation.

**Note**: the Zhao et al. original intention was refusal detection; we are repurposing it for success prediction.

---

## §7: Direction Displacement

**Question**: How much has the per-turn harmfulness direction rotated away from the single-turn baseline?

**Finding**: `cos(v_bin, v_inst) ≈ 0.1` across all models, frameworks, and bins. This is nearly orthogonal — v_bin and v_inst share only ~1% of their variance.

**Key consideration**: does v_pt (per-turn direction at absolute k) have the same confound as the absolute-turn AUROC? Yes — v_pt(k=3) is built from a mixture of conversations where k=3 is the final turn and conversations where k=3 is an intermediate turn. These are structurally different. v_bin (relative binning) was used instead.

**Interpretation**: the representational geometry of multi-turn jailbreaks at t_inst is fundamentally different from single-turn. This is not a small rotation — it is an ~84° displacement. Multi-turn conversations shift the direction of harmfulness encoding into a nearly orthogonal subspace.

**Open question at this stage**: is this displacement meaningful, or is it noise? Does it translate to better predictive performance?

---

## §8: v_bin vs v_inst on Held-Out WJB Data

**Question**: Does the displaced direction (v_bin) actually perform better than v_inst for classifying harmful vs benign WJB conversations?

**Design**: v_bin built from JBB multi-turn data (training), evaluated on WJB multi-turn data (test). v_inst also evaluated on the same WJB samples. Fully out-of-sample comparison.

**Finding**: **v_bin consistently outperforms v_inst** across models, frameworks, and bins. The improvement is substantial — often 0.1–0.2 AUROC points.

**Interpretation at this stage**: multi-turn-native directions capture something that single-turn training misses. Single-turn RepE probes are suboptimal for detecting multi-turn attacks.

**Confound flagged**: v_bin was trained on multi-turn JBB data and tested on multi-turn WJB data (distribution match). v_inst was trained on single-turn JBB and tested on multi-turn WJB (distribution shift). The advantage might be format/distribution matching rather than genuine representational content.

---

## §8b: Do the Directions Predict Attack Success/Failure?

**Motivation**: a concerning alternative explanation — v_inst and v_bin might be detecting *surface harmfulness* (how explicitly harmful the conversation sounds right now) rather than *harmful intent*. If so, they would predict attack success/failure, since successful attacks produce more explicitly harmful final turns.

**Test**: among WJB *harmful* conversations only (same underlying goal, different outcome), compute AUROC of v_inst projections against success/failure label, per relative position bin.

**Finding**: almost everything is near chance (0.5). Small above-chance signals for Crescendo (~0.57) and Gemma+ActorAttack (~0.60), but no model×framework shows AUROC >> 0.5.

**Added v_bin to the test** after realising v_bin might predict success at early turns (hypothesis: it captures attack dynamics, not just content). Result: v_bin is also near chance across bins.

**Interpretation**: neither direction is primarily a success proxy. The signal appears to be about goal-level content, not attack outcome. This is reassuring — but "not a success proxy" is not the same as "encodes harmfulness." Other explanations (topic, style) remain possible.

**The ActorAttack early-turn observation** (AUROC ≈ 0.5 for early turns in §4) is consistent with this: early ActorAttack turns have a harmful underlying intent but benign surface text, and v_inst fires only weakly on them. This suggests the direction tracks something about how harmful the content *currently appears*, not purely latent intent.

---

## §9: Cross-Framework Transfer

**Question**: If v_bin captures harmfulness (a goal-level, model-internal property), it should generalise across frameworks — a direction trained on Crescendo should work on ActorAttack WJB conversations.

**Design**: for each (train_fw, test_fw) pair, build v_bin from JBB[train_fw] and evaluate AUROC on WJB[test_fw].

**Finding**: consistent three-tier ordering across essentially every panel:
```
same-framework (solid) > cross-framework (dashed) > v_inst (dotted triangle)
```

**Interpretation**:
- Cross-fw >> v_inst: v_bin captures something framework-agnostic that multi-turn training encodes better than single-turn. Not purely framework-specific style.
- Same-fw > cross-fw: there IS framework-specific signal in v_bin. The direction is a mixture of generalizable harmfulness signal and framework-specific stylistic/structural patterns.

**Assessment**: partial evidence for generalizable harmfulness encoding, partial evidence for framework-specific contamination. The single-turn baseline (v_inst) is consistently the worst, confirming §8.

---

## §10: PCA Diagnostic — Is v_inst Orthogonal to Topic Variance?

**Motivation**: the directions might be picking up on *topic* (e.g., harmful conversations are about weapons/drugs, benign about cooking/travel) rather than *harmfulness* per se.

**Test 1 — alignment**: compute `|cos(v_inst[L], PC_i)|` for top-20 PCs of the JBB hidden state distribution (harmful + benign combined). Compare to random baseline `1/√D`.

**Finding**: v_inst shows mild but consistent alignment with top PCs (cosines ~0.05–0.15, above the ~0.016 random baseline). Each top PC explains ~0.3–2% of v_inst's variance. Not dominated by topic PCs, but not perfectly orthogonal either.

**Test 2 — residual AUROC**: recompute WJB AUROC after projecting out top-k PCs from hidden states, as a function of k.

**Finding**: AUROC declines gradually (not a flat line), dropping from ~0.65–0.75 at k=0 to ~0.55–0.65 at k=20. Does not collapse to chance.

**Interpretation**: v_inst is a mixture — partly riding on high-variance topic subspace (captured by top PCs), partly in a low-variance subspace orthogonal to topic. The harmfulness signal is diffuse across many PCs rather than cleanly separable. Some topic contamination exists but it is not the dominant component.

**Note**: the gradual slope (no elbow) suggests the contamination is spread across many directions, not concentrated in one "topic PC" that could be cleanly removed.

---

## §11: Neutral Context Control — Format vs. Content

**The decisive test**: does the v_bin advantage over v_inst come from multi-turn *content* (attack dynamics, escalation, roleplay) or just from multi-turn *format* (having any prior context at all)?

**Control condition (v_ctrl)**: extract hidden states from JBB goals wrapped in a single neutral filler turn:
```
[system]
User: "Hi, could you help me with something?"
Asst: "Of course, I'd be happy to help. What do you need?"
User: [JBB goal]   ← t_inst extracted here
```

The assistant turn was hardcoded (not generated). This is correct — the model processes token sequences, not provenance. A fixed, generic filler is preferable: it is reproducible, interpretable, and avoids model-specific response variation across Llama/Qwen/Gemma.

**Design**: build v_ctrl from this neutral-context JBB data. Compare v_inst, v_ctrl, v_bin on WJB in §8.

**Finding**: **v_ctrl (dotted) tracks v_bin (solid) very closely in almost every panel. Both substantially outperform v_inst (dashed).**

**Interpretation**: the v_bin > v_inst advantage is **primarily a distribution-matching / format effect**. A direction trained on JBB goals with completely neutral prior context performs almost as well as one trained on actual attack conversations. The specific attack dynamics (Crescendo's gradual escalation, ActorAttack's roleplay, xteaming's strategy) add very little on top of simply having any multi-turn context in the training data.

---

## Summary of What Is Established

| Claim | Evidence | Confidence |
|-------|----------|------------|
| v_inst separates harmful/benign WJB at final turn (AUROC ~0.7–0.9) | §3 | High |
| Separability increases toward late turns | §4 | High |
| Multi-turn harmfulness direction is ~orthogonal to single-turn (cos ≈ 0.1) | §7 | High |
| v_bin > v_inst on WJB | §8 | High |
| The v_bin > v_inst gap is primarily a format effect, not attack content | §11 (v_ctrl ≈ v_bin) | High |
| Directions do not primarily predict attack success/failure | §8b | Moderate |
| Mild topic contamination in v_inst | §10 | Moderate |
| Cross-framework partial transfer (same-fw > cross-fw > v_inst) | §9 | High |

---

## What Remains Uncertain

1. **What specifically causes the format-induced shift?** Having any prior context shifts the representation at t_inst substantially. This is likely about how attention mechanisms process a user turn differently when conversation history is present — but we haven't analysed *which* layers or *what* components drive it.

2. **Is the mild topic contamination in v_inst practically important?** The residual AUROC after PC removal stays above chance, but the AUROC reduction (~0.1 over 20 PCs) suggests partial topic confounding. Better topic-paired data construction (e.g., JBB harmful goals explicitly matched to benign goals on identical topics) would address this.

3. **Why does v_ctrl not exactly equal v_bin?** There's a small but consistent gap. This might represent a genuine (small) contribution of attack-specific content, or it might reflect distributional differences between the JBB goal text phrasing in single-turn vs. multi-turn contexts.

4. **Does the format effect differ across layers?** All analyses aggregate across 8 layers via median ± IQR. Layer-specific patterns might reveal which parts of the network drive the format-induced shift.

---

## Implications for Probe Design

The main practical takeaway: **a detector trained on single-turn data systematically underperforms on multi-turn jailbreaks, but the fix is simpler than previously thought.** You don't need attack-specific multi-turn training data. Any multi-turn data (even completely benign conversations with a single neutral prior turn) shifts the training distribution to better match multi-turn inference.

This means:
- Building a robust multi-turn harmfulness detector requires multi-turn training examples
- The *content* of those training examples (attack vs. neutral) matters much less than their *format*
- v_inst's poor performance on multi-turn data is a training distribution mismatch problem, not a fundamental representational limitation

---

## Open Questions for Future Work

1. **What is the format shift actually encoding?** Attention pattern analysis or probing the position embeddings might reveal whether the shift is positional (attending to prior turns' tokens) or contextual (semantic meaning of prior context changing current representations).

2. **Does the neutral context need exactly one filler turn, or does turn count matter?** Running v_ctrl with k=1, 2, 3 filler turns and checking when the AUROC plateaus would reveal whether it's specifically multi-turn *format* (any k≥1) or there's a length dependence.

3. **MLP classifier comparison**: v_inst and v_bin are linear directions. An MLP trained on hidden states might capture nonlinear combinations. Given that v_ctrl ≈ v_bin and both are just mean-difference directions, the question becomes whether nonlinear classifiers trained on multi-turn data (any kind) do better than linear ones.

4. **Does steering along v_ctrl suppress harmful content as well as steering along v_bin?** If the format-induced shift is the relevant one, steering along v_ctrl might be a sufficient (and cheaper to construct) defense direction.
