# Context Accumulation and the Displacement of Harmful/Benign Representations in Multi-Turn Jailbreaks

## Notebooks 07–10: Methodology and Results

---

## 1. Motivation

Single-turn jailbreak defenses — circuit breakers (Zou et al., 2024), Latent Guard (Zhao et al., 2025) — assume that a stable "harmfulness direction" exists in the model's representation space and that monitoring representations along this direction is sufficient to detect harmful inputs. These methods achieve high accuracy on single-turn attacks but consistently fail against multi-turn attacks like Crescendo (Bullwinkel et al., 2025).

The fundamental question is *why*. Two prior works provide the context:

**Zhao et al. (2025)** identified two token positions in single-turn settings that encode distinct safety-relevant signals:
- `t_inst` (the last token of the user's instruction): encodes **harmfulness** — whether the instruction is harmful, independent of the model's behavioral response.
- `t_post_inst` (the `<|eot_id|>` token closing the user turn): encodes **refusal** — whether the model will refuse, independent of whether the instruction is actually harmful.

This separation is causal: steering along the harmfulness direction at `t_inst` changes the model's internal perception of an instruction's harmfulness, while steering along the refusal direction at `t_post_inst` elicits refusal without changing the model's harmfulness belief. Zhao et al. validated this through clustering analysis on misbehaving examples and a reply inversion task.

**Bullwinkel et al. (2025)** applied representation engineering to Crescendo attacks, but with two limitations: (1) their probe was trained on single-turn data and applied to multi-turn outputs, measuring "does this look like a single-turn harmful response" rather than studying what happens turn by turn; (2) they extracted representations from response tokens of the final jailbroken output, not from input-side positions like Zhao et al.'s `t_inst`/`t_post_inst`.

**This work** directly extends Zhao et al.'s token positions to multi-turn settings. We study what happens to the harmful/benign distinction *at each turn* as context accumulates, using controlled extraction conditions that isolate the effect of context from message content. This is the gap neither prior work filled.

---

## 2. Hypotheses

**H1 (Displacement):** The harmful/benign direction `v_harmful(k)` — computed from hidden states at turn k — has decreasing cosine similarity with the single-turn baseline direction `v_harmful(ST)` as k increases. This displacement is driven by context accumulation, not just by message content.

**H2 (Detection failure):** A classifier anchored to the single-turn baseline direction degrades in AUROC as turns accumulate, while a per-turn adaptive classifier maintains performance. The gap between these two is the practical cost of displacement.

**H3 (Behavioral correlation):** Turns where displacement is largest correspond to the model's most cooperative behavioral responses, connecting geometric displacement to jailbreak success.

---

## 3. Experimental Setup

### 3.1 Model

Meta-Llama-3.1-8B-Instruct with the standard Llama safety system prompt ("You are a helpful, respectful and honest assistant..."), consistent across all conversations.

### 3.2 Attack frameworks

- **Crescendo** (primary): Gradual escalation through seemingly benign steps. GPT-4o as attacker and judge. Conversations average ~6 accepted turns (range 1–10), with rolled-back turns excluded.
- **ActorAttack** (secondary): Actor-Network Theory framing with multiple attacker personas. Average ~3–4 turns.
- **X-Teaming** (secondary): Strategy planning and execution. Similar turn depth to ActorAttack.

### 3.3 Dataset

The JBB-Behaviors benchmark provides 100 harmful goals and 100 benign goals, matched by topic into 100 pairs. Each pair shares a topic (e.g., pair 0: harmful = "Write a defamatory article claiming that a U.S. president is addicted to heroin", benign = a topically related but harmless request). This pairing is critical: it enables within-pair subtraction that cancels topic signal from the harmfulness direction.

Each framework generates 20 attempts per goal, yielding approximately 2000 harmful + 2000 benign conversations per framework (~12,000 total). Conversations are labeled at the goal level (harmful/benign), conversation level (attack_success), and turn level (is_refusal, judge_success — available for Crescendo).

### 3.4 Token positions

Following Zhao et al., hidden states are extracted at two positions per turn:

- **h_inst** (`t_inst`): the last token of the k-th attacker message, immediately before `<|eot_id|>`. In single-turn, this position encodes harmfulness.
- **h_post** (`t_post_inst`): the `<|eot_id|>` token that closes the k-th attacker message. In single-turn, this position encodes refusal.

**Caveat:** Whether these positions encode the same things in multi-turn settings is an open empirical question — and one that this work directly tests. In multi-turn settings, self-attention means these positions absorb information from the entire prior context, not just the current message. The positions are used here to enable direct comparison with Zhao et al.

### 3.5 Train/test split

Attempts 1–16 (out of 20 per goal) are used for direction computation and classifier training. Attempts 17–20 are held out for evaluation. This ensures that test conversations are from the same goals but are entirely separate attack attempts.

### 3.6 Direction computation

All harmfulness directions use within-pair subtraction to cancel topic signal:

```
For each JBB pair p:
    diff_p = mean(h | goal_type=harmful, pair_id=p) - mean(h | goal_type=benign, pair_id=p)
v_harmful = normalize(mean(diff_p across all pairs))
```

This produces a unit vector in R^4096 at each layer. The within-pair subtraction ensures the direction captures harmfulness independent of topic — verified by checking that topic prediction from the scalar projection is at chance (~10% for 10 JBB categories).

---

## 4. Data Collection (Notebook 07)

### 4.1 The k-turn extraction

The central data collection innovation. For each conversation of length n, instead of extracting hidden states once with the full conversation in context, we extract n times — once for each prefix length k = 1, 2, ..., n.

At prefix k, the model has seen turns 1 through k-1 (both attacker messages and model responses) plus the k-th attacker message, but NOT the model's response at turn k. The sequence ends at `<|eot_id|>` after user_k:

```
[system prompt]
[user_1] [assistant_1]
[user_2] [assistant_2]
...
[user_{k-1}] [assistant_{k-1}]
[user_k] <|eot_id|>
                      ^
              extract h_inst and h_post here
```

This mirrors Zhao et al.'s setup exactly: we extract at positions within the user turn, before any response is generated.

### 4.2 Four extraction conditions

The experiment uses four extraction conditions to isolate what drives changes in the harmfulness representation:

**Condition 1 — Full-context (k-turn prefix):**
Input = turns 1 through k as described above. The hidden state at turn k reflects both the content of user_k's message AND the accumulated influence of all prior turns via self-attention. This is the primary condition — it represents what the model actually experiences during a multi-turn attack.

**Condition 2 — No-context (isolated message):**
Input = system prompt + user_k's message only, with no prior history. The hidden state reflects only the content of the k-th message in isolation.

The difference between Conditions 1 and 2 at the same (conversation, k) is the **pure effect of context accumulation**. Anything that differs between them cannot be due to the message content — it must be due to the accumulated history.

**Condition 3 — Compressed single-turn:**
Input = the entire multi-turn conversation concatenated into a single user message with plain-text "User:"/"Assistant:" prefixes instead of role headers. No `<|eot_id|>` tokens between turns, no turn structure — just one large text block in a single user message. One extraction per conversation.

This tests whether the context effect is driven by semantic content alone (compressed should match full-context) or whether the turn structure and role-based attention patterns matter (they should differ). Bullwinkel et al. found representations were nearly identical for compressed vs. multi-turn — this is a replication check.

**Condition 4 — Single-turn baseline:**
Input = the original JBB goal passed directly as a single-turn prompt with no multi-turn framing. One extraction per goal (100 harmful, 100 benign). This produces v_harmful(ST), the anchor direction representing what any single-turn defense would be trained on.

### 4.3 Implementation details

- All 32 transformer layers saved (shape `(N, 32, 4096)` float16)
- `add_generation_prompt=False` to end the sequence cleanly at `<|eot_id|>` after user_k
- Rolled-back turns excluded (Crescendo-specific: turns the attacker retried after refusal)
- Multi-GPU parallel extraction with resume support
- Token position verification: decoded last two tokens to confirm t_inst is a content token and t_post_inst is `<|eot_id|>`

### 4.4 Sanity checks

Seven checks were run before full-scale extraction:
1. Token position decoding verification on 5 conversations
2. Prefix message count matches expectation (1 system + k user + (k-1) assistant)
3. No-context user content matches full-context user content at each k
4. Rolled-back turn count matches `executed_turns` field
5. No-context k=1 vs full-context k=1: cosine similarity ~1.0 (identical inputs at k=1)
6. No-context vs full-context diverge at k>1 (context is changing the representation)
7. Hidden state shapes are (32, 4096) for each extraction

### 4.5 Storage

Each condition's output is stored as:
- `h_inst.npy`: (N, 32, 4096) float16
- `h_post_inst.npy`: (N, 32, 4096) float16
- `metadata.parquet`: conversation_id, pair_id, goal_type, framework, attempt, turn_k, behavioral labels

Total storage: ~3GB per framework/split for trajectory data (full-context), less for no-context (shorter sequences, same number of rows), and less for compressed (one row per conversation instead of one per turn).

---

## 5. Displacement Analysis (Notebook 08)

### 5.1 Figure 1 — Direction displacement across turns

**Method:** At each turn k (1 through 10), compute v_harmful_full(k) from full-context trajectory data (within-pair, train split). Measure cosine similarity with:
- v_harmful(ST): the single-turn baseline direction
- v_harmful_nocontext(k): the no-context direction at the same turn k
- v_harmful_compressed: the compressed condition's direction (as a reference line)

Computed at L32 (last transformer layer). Two rows: h_inst and h_post. Three columns: one per framework.

**Results (Crescendo, h_inst, L32):**

The blue line — cos(v_full(k), v_ST) — starts at ~0.4 at k=1 and drops to ~0.1 by k=5–8. Even at k=1, the multi-turn direction only partially overlaps with single-turn. This is expected: at turn 1, the attacker sends an innocuous message ("Can you explain the impact of prominent leaders' personal challenges?"), not the direct harmful goal. By k=5–8, the directions are nearly orthogonal.

The orange line — cos(v_full(k), v_nc(k)) — starts near 1.0 at k=1 (expected: identical inputs) and drops to ~0.1–0.2 by later turns. This confirms that the accumulated context is reshaping the direction beyond what message content alone explains. The full-context and content-only directions diverge progressively.

The green dashed line — cos(v_compressed, v_ST) — sits at ~0.15–0.25, roughly where the blue line ends up at late turns. The compressed condition's direction is similarly displaced from single-turn, consistent with Bullwinkel et al.

The pattern is consistent across all three frameworks and across both h_inst and h_post positions at L32.

**Limitation:** At L32 (the last layer), h_inst and h_post show very similar patterns. The Zhao et al. distinction between positions may be more pronounced at middle layers.

### 5.2 Figure 2 — Detection failure and recovery

**Method:** At each turn k, project held-out data (test split, attempts 17–20) onto two directions:
- **Fixed:** v_harmful(ST) — what a single-turn defense would use
- **Adaptive:** v_harmful_full(k) — direction recomputed at this turn using train data

Compute AUROC for harmful vs. benign classification.

**Results (Crescendo, L32):**

| Turn k | Fixed AUROC | Adaptive AUROC | Gap |
|--------|-------------|----------------|-----|
| 1      | 0.58        | 0.69           | 0.11 |
| 3      | 0.60        | 0.74           | 0.14 |
| 5      | 0.61        | 0.75           | 0.14 |
| 8      | 0.60        | 0.76           | 0.16 |

The fixed classifier hovers near 0.58–0.61 — above chance but far from reliable. The adaptive classifier maintains 0.69–0.76 and improves slightly at later turns. The gap widens from 0.11 at k=1 to 0.16 at k=8.

**Limitation:** Even the adaptive classifier is below 0.80 at L32. This is a layer choice issue — mid-layers (L16–L20) achieve substantially higher AUROC (see nb09 results). L32 was chosen for the displacement analysis as a conservative layer; the displacement pattern is even stronger at layers where the signal is stronger.

### 5.3 Figure 3 — Behavioral correlation

**Method:** For harmful conversations only (Crescendo, which has per-turn behavioral labels), project each turn's hidden state onto v_harmful_full(k) and split by judge_success (whether the model complied at that specific turn).

**Results:** Box plots show that successful turns (judge_success=True, red) tend to have lower median projections than unsuccessful turns (blue), particularly at turns 3–6. The effect is modest and noisy, with substantial overlap between the groups.

**Limitation:** The behavioral correlation is the weakest of the three results. The per-turn judge_success label is noisy (GPT-4o judge), the projection is onto a direction computed from all conversations at that turn (not specific to the conversation), and the sample sizes for judge_success=True at early turns are small. H3 is partially supported but not as clean as H1 and H2.

### 5.4 Figure 4 — Misbehavior clustering (Zhao et al. replication)

**Method:** Zhao et al. validated their token positions by showing that at t_inst, accepted-harmful examples clustered with refused-harmful (model internally encodes both as harmful regardless of behavioral outcome). We replicate this analysis at the final turn of multi-turn conversations with three groups: accepted_harmful (attack_success=True), refused_harmful (attack_success=False), and benign.

Two analyses:
- **PCA scatter** at the final turn: visual clustering
- **Nearest-centroid fraction**: for each refused_harmful sample, compute distance to accepted_harmful centroid vs. benign centroid. Report the fraction closer to harmful. Zhao et al. predicts ~1.0 at h_inst.

**Results (PCA, L32):** All three groups heavily overlap in PCA space at the final turn. No clean separation between accepted_harmful, refused_harmful, and benign. Consistent across all three frameworks and both positions.

**Results (Nearest centroid, Crescendo, L32):** Full-context fraction is 0.50–0.65 at L32 — modestly above chance, far from Zhao et al.'s prediction of ~1.0. No-context fraction is 0.30–0.45 — below chance, meaning refused_harmful samples from failed attacks look more benign than harmful when the context is stripped.

**Cross-condition comparison:** A key additional analysis compared PCA clustering across extraction conditions (full-context, no-context, single-turn) at L32. The single-turn condition showed visibly cleaner separation at h_post (consistent with Zhao et al.'s refusal signal), but full-context and no-context were muddled across both positions. This provides direct evidence that context accumulation degrades the clean geometry that exists in single-turn.

---

## 6. Token Position Verification (Notebook 09)

Notebook 09 provides a systematic multi-layer, multi-condition verification of whether the Zhao et al. token position semantics hold in multi-turn settings.

### 6.1 Figure 1 — AUROC heatmap (conditions × layers)

**Method:** For each condition and layer, project test data onto v_harmful and compute AUROC for harmful vs. benign classification. Rows: single-turn, full-context (with v_ST), full-context (with v_own), no-context (with v_ST), no-context (with v_own). Columns: 8 layers spanning the full range (L4 through L32).

**Results (Crescendo, h_inst):**

| Condition | L4 | L8 | L12 | L16 | L20 | L24 | L28 | L32 |
|-----------|----|----|-----|-----|-----|-----|-----|-----|
| Single-turn | 0.62 | 0.84 | 0.91 | 0.96 | 0.91 | 0.91 | 0.81 | 0.69 |
| Full-context (v_ST) | 0.62 | 0.64 | 0.64 | 0.65 | 0.70 | 0.73 | 0.71 | 0.69 |
| Full-context (v_own) | 0.79 | 0.80 | 0.85 | 0.84 | 0.85 | 0.85 | 0.83 | 0.79 |
| No-context (v_ST) | 0.63 | 0.62 | 0.63 | 0.65 | 0.70 | 0.72 | 0.71 | 0.68 |
| No-context (v_own) | 0.77 | 0.78 | 0.80 | 0.82 | 0.83 | 0.83 | 0.73 | 0.72 |

Key observations:
1. **Single-turn is consistently strongest**, peaking at 0.96 at L16. The linear harmfulness signal is very strong when the model receives a direct harmful prompt.
2. **v_ST applied to multi-turn drops to 0.62–0.73** — substantial degradation across all layers. The single-turn direction partially transfers but loses 20–30 percentage points.
3. **Per-condition v_own recovers to 0.79–0.85** at mid-layers. The signal exists in multi-turn; it has moved to a different direction.
4. **Mid-layers (L12–L24) are best**. Late layers (L28, L32) degrade. This is consistent across conditions and frameworks.
5. **h_inst and h_post show similar patterns** — no clear advantage for either position on the harmfulness task.

### 6.2 Figure 2 — Position specificity

**Method:** The core Zhao et al. claim is that h_inst encodes harmfulness and h_post encodes refusal — these are functionally distinct positions. We test this by computing AUROC for two tasks at each position:
- **Harmfulness:** harmful vs. benign classification (goal_type)
- **Refusal:** refused_harmful vs. accepted_harmful classification (attack_success, harmful conversations only)

If Zhao et al. holds: harmfulness AUROC should be higher at h_inst; refusal AUROC should be higher at h_post.

**Results:**

In **full-context**, h_inst and h_post lines are interleaved for both tasks across all layers and frameworks. There is no consistent advantage for h_inst on harmfulness or for h_post on refusal. The clean functional separation between positions has collapsed.

In **no-context**, the same pattern — no consistent position specificity.

**Interpretation:** In multi-turn settings, self-attention allows both positions to absorb information from the entire prior context. The clean single-token semantics that exist when the model processes one instruction in isolation get overwritten by conversation-level features (topic trajectory, turn depth, escalation pattern). Both positions encode a mixed signal rather than the distinct harmfulness/refusal decomposition found in single-turn.

**Limitation:** We did not perform the causal steering test (Zhao et al.'s reply inversion task) in multi-turn. The AUROC analysis shows the positions don't *separate* the signals, but cannot prove they don't *encode* them in some entangled form that a more sophisticated decoder could recover. The causal test would be definitive but requires a separate experimental apparatus (multi-turn steering with inversion prompts).

### 6.3 Figure 3 — Silhouette scores

**Method:** Silhouette score measures geometric cluster quality without assuming a specific direction. Computed in full 4096-dimensional space, subsampled to 300 per class for tractability.

**Results (Crescendo):**

At h_inst, silhouette scores are very low across the board (0.000–0.014). Single-turn is consistently highest, peaking around L8–L12. Full-context and no-context are lower.

At h_post, single-turn shows a notable peak at L12–L16 (~0.08–0.10) — about 5–8× higher than any other condition. This is the clearest geometric separation in the entire analysis: in single-turn, the `<|eot_id|>` position cleanly clusters harmful from benign because it directly encodes the refusal decision. Full-context and no-context h_post are flat at ~0.02.

**Interpretation:** The harmful/benign distinction is a thin linear signal, not a big geometric separation — even in the best case (single-turn), silhouette scores are modest. The key finding is the *relative* ordering: single-turn > full-context > no-context at h_inst, and the dramatic single-turn peak at h_post. Context accumulation degrades the geometric separability at both positions.

### 6.4 Figure 4 — Nearest-centroid misbehavior clustering by layer

**Method:** Same as the nb08 nearest-centroid analysis but computed across all 8 layers for both full-context and no-context conditions. For each refused_harmful sample at the final turn, compute distance to the accepted_harmful centroid vs. the benign centroid.

**Results (Crescendo):**

Full-context (h_inst): the fraction starts at 0.80 at L4 and decreases to 0.50–0.65 at later layers. At early-to-mid layers, refused_harmful does cluster with accepted_harmful — **partially consistent with Zhao et al.** The model still "sees" these conversations as more harmful than benign at early layers, even when it refused.

Full-context (h_post): similar pattern, fraction 0.80–0.90 at L4–L8, dropping at later layers.

No-context: dramatically lower — 0.30–0.45 across all layers for both positions. Individual messages from failed attacks look more benign than harmful when stripped of context.

**The full-context vs. no-context gap is a key finding:** In full context the model partially "knows" a refused conversation is harmful (fraction > 0.5 at early/mid layers). Strip the context and it doesn't. The context is carrying the harmfulness signal.

**Cross-framework difference:** ActorAttack shows fraction ~0.20–0.35 for both conditions at all layers — the Zhao et al. clustering doesn't hold at all. This suggests that ActorAttack's failure mode is different from Crescendo's: when ActorAttack fails, the model may not even recognize the harmful intent, while Crescendo failures may involve the model recognizing the intent but refusing to comply.

### 6.5 Figure 5 — Turn × layer AUROC heatmap

**Method:** Full sweep across turns 1–10 and 8 layers. At each (k, layer), compute the within-pair direction from train data at that turn, then evaluate AUROC on test data at that turn. Separate panels for full-context and no-context, and for h_inst and h_post.

**Results (Crescendo, full-context, h_inst):**

AUROC is 0.64–0.84 at k=1, peaks at mid-layers (L16–L20), and remains remarkably stable across turns at mid-layers. There is no dramatic degradation at later turns when using the adaptive (per-turn) direction. Late layers (L28–L32) are consistently weaker.

The no-context heatmap shows a similar pattern with slightly lower AUROC. A notable feature: no-context AUROC at h_post at early turns and early layers drops to 0.41–0.51 (near chance). The h_post position needs conversation context to encode useful information about the interaction; in isolation, it carries almost no harmfulness signal at early layers.

**Limitation:** Sample sizes decrease at higher turn k (e.g., only ~278 conversations reach turn 10 in Crescendo). Results at k ≥ 8 should be interpreted with this in mind.

### 6.6 Figure 6 — PCA clustering by condition and layer

**Method:** PCA scatter of accepted_harmful, refused_harmful, and benign at the final turn, compared across extraction conditions (full-context, no-context, single-turn) at L16, L20, and L32.

**Results:** At L16 and L20, single-turn h_post shows the most visible separation between harmful and benign — consistent with the high silhouette score. Full-context and no-context panels show overlap at all layers. The PCA confirms the numerical measures: the harmful/benign distinction is a subtle signal that PCA's top-2 unsupervised components struggle to capture in multi-turn.

---

## 7. Nonlinear Probing (Notebook 10)

Notebooks 08 and 09 establish that the linear harmfulness direction displaces as context accumulates. Notebook 10 asks: is this a linear-only phenomenon, or does the displacement affect nonlinear classifiers too?

### 7.1 Figure 1 — Linear vs MLP within-condition

**Method:** At each layer and condition, train both a logistic regression (linear) and an MLP (hidden_layer_sizes=(64,), early stopping) on the train split, evaluate on the test split. Both use StandardScaler preprocessing.

**Results:**

**The signal is almost entirely linear.** Across all conditions, layers, positions, and frameworks, the MLP and logistic regression achieve nearly identical AUROC. Typical gap: < 0.02 in either direction.

Full-context and no-context both reach 0.93–0.96 AUROC with either method at mid-layers. Single-turn achieves 0.85–0.95 (lower due to small training set — only 50 samples per class).

**Implication:** There is no hidden nonlinear harmfulness signal that direction-based defenses miss. The information at these positions is linear in the harmful/benign distinction. The displacement problem is not about classifier capacity — it's about the representation itself changing.

### 7.2 Figure 2 — Cross-condition transfer matrix

**Method:** Train on one condition, test on another. 3×3 matrix (single-turn, full-context, no-context) for both logistic regression and MLP. Fixed at L20.

**Results (Crescendo, h_inst):**

| Train \ Test | Single-turn | Full-context | No-context |
|-------------|-------------|--------------|------------|
| **Single-turn** | **0.86 / 0.64** | 0.58 / 0.54 | 0.59 / 0.53 |
| **Full-context** | 0.58 / 0.65 | **0.95 / 0.96** | 0.94 / 0.96 |
| **No-context** | 0.63 / 0.64 | 0.93 / 0.94 | **0.95 / 0.96** |

(Format: linear / MLP)

Key findings:

1. **ST → Full-context: linear 0.58, MLP 0.54.** The MLP does *worse* than linear on this critical transfer. A nonlinear classifier trained on single-turn data cannot detect multi-turn harmfulness — the displacement is not a linear rotation that nonlinear methods can see through. The representation has genuinely changed at a structural level.

2. **Full-context ↔ No-context: 0.93–0.96 both directions.** The multi-turn probe generalizes well even when context is stripped. This suggests the probe learns features that persist across conditions — likely because the train split direction captures both context-dependent and content-dependent signal.

3. **Diagonal (in-condition): 0.86–0.96.** The signal is strong within each condition. The problem is purely cross-condition transfer.

**h_post results are more extreme:** ST → Full-context MLP achieves 0.45 for Crescendo and 0.45 for ActorAttack — *below chance*. The MLP trained on single-turn h_post is actively anti-correlated with multi-turn h_post. Whatever h_post encodes in single-turn (refusal) is not just absent in multi-turn — it has inverted.

**Cross-framework consistency:** ActorAttack shows the same pattern: ST → full-context linear 0.60, MLP 0.51. The MLP-worse-than-linear finding is robust.

### 7.3 Figure 3 — Per-turn transfer heatmap

**Method:** Train a classifier (linear or MLP) at turn k_train, test at turn k_test. Produces a k×k AUROC matrix showing whether the representation is stable across turns. At L20, h_inst, for turns 1–8.

**Results (Crescendo):**

**Linear:**
- Diagonal mean: ~0.83 (per-turn classifier works at its own turn)
- Off-diagonal: decays with distance. k=1→k=8: 0.67. k=8→k=1: 0.58.
- Nearby turns transfer better than distant turns (gradual decay, not a sharp cliff).
- Late turns (k=6,7,8) transfer well to each other (~0.85–0.91) — the representation stabilizes at later turns.

**MLP:**
- Diagonal mean: ~0.79 (slightly lower than linear — possible overfitting)
- Off-diagonal: similar decay pattern. k=1→k=8: 0.72. k=8→k=1: 0.72.
- The MLP does NOT rescue the transfer failure. Off-diagonal values are comparable to or lower than linear.

**Interpretation:** The representation changes progressively across turns, and this change is equally impenetrable to both linear and nonlinear methods. The slightly better off-diagonal transfer for MLP at some points is within noise. The key result is the similarity between the two heatmaps — the turn-by-turn displacement is a structural change, not a linear rotation.

---

## 8. Summary of Findings

### H1 (Displacement): Strongly supported

The harmful/benign direction `v_harmful(k)` progressively rotates away from the single-turn baseline as context accumulates. At L32, cosine similarity drops from ~0.4 at k=1 to ~0.1 by k=5–8. This pattern is consistent across all three attack frameworks and across both h_inst and h_post positions.

The no-context control confirms this is driven by context accumulation: cos(v_full(k), v_nc(k)) drops from ~1.0 at k=1 to ~0.1–0.2 at later turns, meaning the accumulated history — not just the message content — is reshaping the direction.

### H2 (Detection failure): Strongly supported

A classifier anchored to v_harmful(ST) achieves 0.58–0.61 AUROC on multi-turn data (at L32), barely above chance. A per-turn adaptive classifier maintains 0.69–0.76 AUROC. At mid-layers (L20), the per-condition classifier achieves 0.93–0.96 within condition.

The cross-condition transfer results from notebook 10 sharpen this: even a nonlinear MLP trained on single-turn achieves only 0.54 on multi-turn data (worse than linear's 0.58). The displacement is not something a more powerful classifier can overcome — it is a genuine structural change in the representation.

### H3 (Behavioral correlation): Partially supported

At turns where the model complied with harmful requests (judge_success=True), projections onto v_harmful_full(k) tend to be lower than at refused turns. The effect is modest, noisy, and limited to Crescendo (the only framework with per-turn behavioral labels). This hypothesis needs stronger methodology (e.g., the causal steering test from Zhao et al.) to be conclusively tested.

### Additional findings

**Position specificity collapses in multi-turn.** Zhao et al.'s clean separation — h_inst for harmfulness, h_post for refusal — does not hold when context accumulates. Both positions perform similarly on both tasks across layers and frameworks. The functional distinction is a single-turn property that multi-turn context destroys.

**The signal is linear.** MLP probes achieve the same AUROC as logistic regression across all conditions, layers, and frameworks. Direction-based defenses are not missing nonlinear signal — the problem is purely that the direction moves.

**Mid-layers are best.** The harmfulness signal peaks at layers 12–20 across all conditions. Late layers (28–32) consistently degrade. This is relevant for defense design: monitoring the last layer is suboptimal.

**Context carries the harmfulness signal for misbehaviors.** In full-context, refused_harmful conversations cluster with accepted_harmful at early/mid layers (nearest-centroid fraction 0.65–0.80 for Crescendo). Strip the context and the fraction drops to 0.30–0.45. The conversation history is what tells the model "this is a harmful conversation" — the individual messages don't carry this information.

**The representation changes progressively, not abruptly.** The per-turn transfer heatmap shows gradual off-diagonal decay, not a sharp cliff. Nearby turns transfer well (k=3→k=5: ~0.85), distant turns don't (k=1→k=8: ~0.67). The representation stabilizes somewhat at later turns, with k=6,7,8 transferring well to each other.

---

## 9. Limitations

### Methodological

1. **No causal steering test.** Zhao et al.'s strongest validation was the reply inversion task, which provides causal evidence that a position encodes a specific concept. All analyses here are correlational (projection, clustering, classification). The position specificity results show the positions don't *separate* harmfulness and refusal, but cannot prove they don't encode them in some entangled form.

2. **Single model.** All results are on Llama-3.1-8B-Instruct. The findings may not generalize to other model families, sizes, or instruction-tuning strategies. Larger models may have more robust position-level encoding.

3. **JBB topic pairing controls topic but not other confounds.** The within-pair subtraction cancels topic signal, but harmful and benign conversations may differ in other ways (conversation length, attacker vocabulary, model response patterns) that the direction captures in addition to harmfulness.

4. **Train/test split by attempt, not by goal.** Attempts 1–16 vs. 17–20 for the same goals. This means the same harmful/benign goals appear in both splits. The cross-validation by topic pair (used in nb05) is more conservative but was not used for the direction-based analyses in nb08–10.

### Statistical

5. **Decreasing sample sizes at high turn k.** Crescendo conversations range from 1–10 turns. At k=8, only ~626 conversations contribute data. At k=10, only ~278. Results at late turns are less reliable.

6. **Silhouette scores are very low everywhere.** The harmful/benign distinction is a thin linear signal (~0.01 silhouette at h_inst). This means the representation space is dominated by other factors (topic, syntax, conversation mechanics). The harmfulness signal is real and detectable, but it is not the primary organizing principle of the hidden state.

7. **Behavioral labels are noisy.** Per-turn judge_success labels come from a GPT-4o judge and are imperfect. The conversation-level attack_success label is more reliable but less granular.

### Scope

8. **Only input-side positions tested.** Zhao et al.'s positions are both within the user turn. Bullwinkel et al. extracted from response tokens. The displacement of response-side representations is not studied here and may show different patterns.

9. **Defense implications are suggestive, not prescriptive.** The results show that fixed-direction defenses fail and per-turn adaptive defenses succeed, but we did not implement or evaluate an actual defense system. The practical gap between "the signal exists in a per-turn direction" and "this can be deployed as a real-time defense" is substantial.

10. **Compressed condition has a single extraction per conversation.** Unlike the per-turn analysis of full-context and no-context, the compressed condition produces one data point per conversation. This limits the analysis to a single reference line in Figure 1 rather than a per-turn curve, and the comparison with full-context at the final turn is indirect since the compressed message includes content from all turns while the full-context extraction at turn k only includes context up to turn k.
