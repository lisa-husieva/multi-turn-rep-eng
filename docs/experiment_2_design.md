# Experiment 2: Probe Design and Benign Data Generation

**Date:** 2026-03-30

---

## Context

Experiment 1 found that MLP probes and LAT reading vectors separate harmful from benign at k=1 — before any Crescendo history is in context — suggesting probes were picking up topic differences rather than multi-turn attack dynamics. Experiment 2 is designed to test whether jailbreak representations are attack-specific or universal, using JBB matched pairs (same topic, different intent) and three attack frameworks.

---

## Data

**Harmful conversations** (generated):
- `crescendo_harmful_v2`: 1000 conversations (100 objectives × 10 attempts)
- `actorattack_harmful_v2`: 1000 conversations (100 objectives × 10 attempts)
- `xteaming_harmful_v2`: 1000 conversations (100 objectives × 10 attempts)

**Benign conversations** (to be generated):
- Same three frameworks, same 100 JBB benign matched objectives, 10 attempts each
- Benign objectives are the JBB matched pairs — same topic as harmful, different intent

**Turn count distributions (harmful)**:
- Crescendo: mean 6.6, median 7 (max 10)
- ActorAttack: mean 4.0, median 6 (max 6)
- X-Teaming: mean 3.1, median 3 (max 7)

---

## Benign Conversation Generation

All three attack frameworks will be run on benign JBB objectives. This is valid — the frameworks take any objective string; benign objectives are harmless so the attacker LLM (gpt-4o) will comply readily.

**Why use the same frameworks for benign:**
Controls for conversational structure. When training a probe on Crescendo harmful vs. Crescendo benign, both have the same scaffolding — the probe must learn something about the direction of escalation rather than just "this looks like a multi-turn attack."

**Turn count matching:**
- ActorAttack and X-Teaming: turn count is determined by the attacker's plan generation, so benign conversations will naturally have similar lengths to harmful ones.
- Crescendo: `n_turns=7` (matching the harmful median), early stopping disabled for benign runs so every conversation runs to completion.

---

## Hidden State Extraction

**Model:** Llama-3.1-8B-Instruct (same model used as target during generation)

**Layer:** Last transformer layer (layer 32, `hidden_states[-1]`). Closest to the output; encodes the model's decision-relevant representation.

**Position:** Last token of each turn. After applying the chat template, the full conversation is one flat token sequence. The last token of each turn encodes everything the model has processed up to that point — the full context compressed into one vector.

**Which turns to extract:**
- All stored turns for each conversation
- For Crescendo: refusal turns are stored in the JSON but were rolled back from the target's context during generation. Only extract from non-refusal turns — i.e., turns where the target actually responded and the exchange was kept in context.
- For ActorAttack and X-Teaming: all stored turns are valid.

**Jailbreak turn:**
For all three frameworks, the jailbreak verdict is determined by scoring the last turn's response. The last stored turn is always the jailbreak turn (or the final failure turn). Turns 1 to N-1 are the escalation; turn N is the decision turn.

**Context for extraction:**
Use the stored turns as-is (filtered for non-refusals in Crescendo). This replicates the actual context the model conditioned on during generation.

---

## Probe Design

**Representation:** One hidden state vector per turn per conversation, shape `(4096,)` for Llama-3-8B. Label = 1 (harmful) or 0 (benign).

**Probe type:** Logistic regression (linear probe). Stronger test than MLP — if a linear probe works, the feature is genuinely linearly encoded in the residual stream.

**Training:**
- Train separate probes at each turn position k (turn 1, turn 2, ...)
- Coverage check: only train a probe at turn k if there are ≥50 examples of each class — sparsity at later turns (especially X-Teaming) may make late-turn probes unreliable
- Train/test split: **held-out pair_ids** — train on pairs 0–79, test on pairs 80–99. Forces generalization across topics rather than memorizing them.

**Topic control:**
JBB matched pairs ensure each topic appears in both harmful and benign classes equally. Holding out entire pair_ids for test means the probe must generalize to unseen topics.

---

## Research Questions and Build-Up

### Question 1: Does a linear probe separate harmful from benign at all? (Baseline)

- Train on Crescendo harmful vs. Crescendo benign, held-out topics
- Extract at each turn position; train one probe per turn
- Expected: probe accuracy increases across turns (early turns look similar, later turns diverge as escalation proceeds)
- This replicates and extends Bullwinkel et al. with topic control

### Question 2: Does the separation grow across turns? (Trajectory)

- Plot probe accuracy (or harmful probability score) at each turn position for harmful vs. benign conversations
- Visualization: probe score curves across turns, averaged by framework
- Expected: harmful trajectories diverge from benign over turns; benign stays flat

### Question 3: Are jailbreak representations universal across frameworks?

- Train probe on Crescendo harmful vs. Crescendo benign
- Test on ActorAttack and X-Teaming conversations (same held-out pair_ids)
- If probe transfers: representations are attack-framework-agnostic — the model ends up in a similar internal state regardless of how it was manipulated
- If probe doesn't transfer: representations are attack-specific

### Question 4: Does training on all frameworks improve generalization?

- Train on all three frameworks combined, test on held-out topics
- Comparison baseline: single-framework probe
- Expected: combined probe is more robust; also tests whether the three frameworks produce compatible representations

### Question 5: Where in the conversation does the harmful signal emerge?

- Compare probe accuracy at turn 1 vs. last turn across frameworks
- Turn 1 probes are the strongest topic-control test — both harmful and benign are asking something innocuous at turn 1, so any separation reflects something subtle about the setup or objective framing
- Late-turn probes capture the escalation signal

---

## Visualization Plan

**Trajectory plot (PCA/UMAP):**
Fit PCA on all extracted vectors. Plot each conversation as a path through 2D space — turn 1 → turn 2 → ... → turn N. Harmful and benign on the same topic start close (turn 1) and should diverge over turns. Compare paths across frameworks.

**Probe score curves:**
For each framework, plot mean harmful probability score at each turn position, separately for harmful and benign conversations. Shows when and how fast escalation is detectable in representation space.

**Cross-framework transfer heatmap:**
Train on framework A, test on framework B — report accuracy for all A→B combinations. 3×3 matrix showing universality.
