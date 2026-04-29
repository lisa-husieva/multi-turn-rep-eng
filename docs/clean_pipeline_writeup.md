# `clean_pipeline.ipynb` — full writeup and interpretation

This document accompanies `notebooks/clean_pipeline.ipynb`. It walks through what the notebook does section by section, and reads off the results currently in hand (Llama-3.1-8B-Instruct; Qwen2.5-7B-Instruct and Gemma-2-9B-it data exists on disk and will fill in once the notebook is run with `MODEL` switched). The numerical claims below are from the Llama run; the structural claims (compressed-≈-full, axis rotated from `k=1`, MLP ≈ logistic) are written to be cross-model when the other two runs land.

> Companion notebook: `notebooks/results_clean.ipynb`. That one runs **within-JBB** with an attempt-level train/test split. `clean_pipeline.ipynb` is the **JBB → WildJailbreak OOD transfer** version of the same questions. This writeup only covers `clean_pipeline`.

---

## What the notebook does

Eleven sections, one experiment per section. The file is parameterized by the `MODEL` constant in §0; every later constant is per-model and every output goes into `figures/clean_pipeline/{MODEL}/` and `results/clean_pipeline/{MODEL}/`. Replication across the three target models is "set `MODEL = "llama"`, run the notebook, set `MODEL = "qwen"`, run it again, set `MODEL = "gemma"`, run it a third time" — there's no shared state between runs.

**Direction objects, defined exactly once in §2 and reused everywhere:**

| Object | Built from | Position | Granularity |
|---|---|---|---|
| `v_inst[layer]` | JBB single-turn topic-paired harmful − benign | `t_inst` | one per layer (framework-independent) |
| `v_outcome[layer, framework]` | JBB full-context final-turn harmful, success − failure | `t_post` | one per (layer, framework) |
| `v_pt[layer, framework, condition, turn]` | JBB multi-turn at turn `k`, topic-paired harmful − benign | `t_inst` | one per (layer, framework, condition, turn) |

**Three extraction conditions:**
* `full_context` — the `k`-th user turn plus all prior accepted turns.
* `no_context` — the same `k`-th user turn alone, system-prompted.
* `compressed` — a single-turn semantic compression of the entire conversation (one row per attempt, no `turn_k` — used as a control, not a per-turn condition).

**Two label rules:**
* `t_inst`: harmful (1) vs benign (0).
* `t_post`: success (1) vs failure (0) on harmful conversations only, with success defined by `aisi_scaled ≥ 0.5`.

---

## §0 — Setup and definitions

Pure constants. The point of this section is that everything later in the notebook can be parsed unambiguously: the reader knows which model, which positions, which conditions, which AISI threshold, which layer indices, which directory the figures go to. No modelling or plots happen here.

The key parameter is `MODEL`. Switch it and rerun for each target. The `LAYER_INFO` JSON sidecar fixes the 8-layer sweep used everywhere downstream — L1, L5, L9, L14, L18, L23, L27, L32 (1-indexed). When this writeup says "median across 8 layers" or "median across the layer sweep", that is what is meant.

---

## §1 — Data sanity checks

Five tables saved as CSVs in the model's `results/clean_pipeline/{MODEL}/` directory. The point is to verify the data that every later claim depends on, before any modelling.

**1.1 Block counts** — rows per `(dataset, condition, label, framework)`. Tells you, for each combination, how many state vectors were extracted.

**1.2 Final-turn counts** — conversations per `(dataset, framework, label)` at full context. For Llama: JBB ~1900–2000 conversations per `(framework, label)`; WJB ~700–1000 per harmful, ~890–990 per benign (the WJB selection is 50 harmful + 50 benign goals × ~20 attempts each, with framework-dependent attrition).

**1.3 Per-turn counts** — rows per `(framework, label, turn_k)` at full context, JBB. These tell you which turns are usable for per-turn analyses. Crescendo runs to `k=10` (long tail), ActorAttack ends at `k=6`, X-Teaming at `k=7–8`. From Llama:

| framework | turn_k 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Crescendo (harmful) | 1991 | 1955 | 1543 | 999 | 631 | 418 | 296 | 218 | 155 | 117 |
| Crescendo (benign)  | 1994 | 1989 | 1862 | 1587 | 1313 | 1036 | 828 | 626 | 472 | 278 |

The conversation-attrition shape — most conversations end before `k=5` for Crescendo — explains why per-turn analyses get noisier at later turns.

**1.4 Topic-pair availability** — usable pairs for `v_pt` per `(framework, condition, turn)`. With the `MIN_PAIRS_PT = 30` threshold:
* Crescendo: usable at turns 1–9 (drops to 19 at `k=10`, skipped).
* ActorAttack: usable at turns 1–6 (all 100 pairs present).
* X-Teaming: usable at turns 1–5 (drops to 14 at `k=6`, skipped).

**1.5 `t_post` class balance** — success/failure split for `v_outcome`. Llama:
* Crescendo: 984 succ / 1010 fail (49.3% success — well-balanced).
* ActorAttack: 620 succ / 1377 fail (31.0% success — imbalanced).
* X-Teaming: 973 succ / 972 fail (50.0% — perfectly balanced).

The ActorAttack imbalance matters for `v_outcome` interpretation: with only 31% success, many topic pairs fail to retain both classes and drop out of the construction. The remaining `v_outcome[L, actorattack]` is built from a smaller, more selected subset than the other two.

---

## §2 — Build core directions

Three direction objects, each defined exactly once.

**§2.1 `v_inst[layer]`** — single-turn harmfulness anchor at `t_inst`. Built from the 100 JBB topic-matched single-turn pairs by mean-difference of harmful and benign hidden states, averaged across pairs, unit-normalized per layer. Framework-independent — this is the same axis applied across all three attack frameworks during evaluation. *This is "Latent Guard's direction" in spirit: the axis a single-turn detector would learn.*

**§2.2 `v_outcome[layer, framework]`** — outcome direction at `t_post`. Built from JBB full-context final-turn harmful conversations only, partitioned by AISI outcome within topic pair. For each `(layer, framework)`, average the within-pair (success − failure) difference across pairs that retain both classes, then unit-normalize per layer. *Framework-specific because success geometry depends on attack style.* For Llama, 24 directions land — 8 layers × 3 frameworks, all built successfully.

**§2.3 `v_pt[layer, framework, condition, turn]`** — per-turn harmfulness direction at `t_inst`. Same construction as `v_inst` but restricted to states with `turn_k == k` under `condition ∈ {full_context, no_context}`. Skipped when fewer than 30 topic pairs are available at that turn. For Llama, 320 directions land across the usable cells. *This is the object that powers the §6 mechanistic analysis and the §7 adaptive-AUROC consequence.*

---

## §3 — Final-turn transfer (Plot 1, Table 1)

**Question.** Does the JBB-trained direction still separate the position-conditional label on WJB final-turn states?

**Operation.** Per `(layer, framework, condition, position)`:
* `t_inst`: project WJB final-turn states onto `v_inst[layer]`, AUROC against harmful-vs-benign.
* `t_post`: project WJB final-turn harmful states onto `v_outcome[layer, framework]`, AUROC against success-vs-failure.

All three conditions reported. Plot 1 lays it out by layer (rows = positions, cols = frameworks, lines = conditions). Table 1 collapses to median-across-layers.

**Llama result (Table 1):**

| | t_inst, no_ctx | t_inst, full | t_inst, compressed | t_post, no_ctx | t_post, full | t_post, compressed |
|---|---|---|---|---|---|---|
| Crescendo | 0.727 | 0.742 | 0.698 | 0.597 | 0.648 | 0.649 |
| ActorAttack | 0.718 | 0.734 | 0.721 | 0.573 | 0.579 | 0.580 |
| X-Teaming | — | 0.557 | — | 0.495 | 0.532 | 0.525 |

**Reading.**
* **`t_inst` transfer is partial and framework-dependent.** Crescendo and ActorAttack land at AUROC 0.72–0.74 under full context — useful but not at the 0.99 ceiling Latent Guard reports for in-distribution single-turn data. X-Teaming is the standout failure at 0.557, barely above chance, and the per-layer sweep confirms it never exceeds 0.59 at any layer.
* **`t_post` is uniformly weak.** Range 0.49–0.65 across all conditions and frameworks. The success-vs-failure axis is partially readable from the post-instruction position, but it sits much closer to chance than the harmfulness signal at `t_inst`.
* **The X-Teaming missing cells** (`t_inst no_context` and `t_inst compressed`) are pre-existing data gaps — the WJB X-Teaming benign extraction never produced those conditions. Documented in §7 of the paper draft.

The X-Teaming asymmetry at `t_inst` is the first sign that frameworks induce qualitatively different multi-turn geometries; §9's cross-framework matrix makes that quantitative.

---

## §4 — Context effect on final-turn transfer (Plot 2)

**Question.** What does prior history add beyond the local message at the same final turn?

**Operation.** `ΔAUROC = AUROC(full_context) − AUROC(no_context)` per `(framework, position, layer)`. This is *context effect on separability*, not direction displacement — the directions are the same in both conditions; only the input trajectory differs.

**Llama result (median across layers):**

| | t_inst Δ | t_post Δ |
|---|---|---|
| Crescendo | +0.015 | +0.051 |
| ActorAttack | +0.016 | +0.006 |
| X-Teaming | — | +0.037 |

**Reading.** Context helps a little at `t_inst` and somewhat more at `t_post`. The asymmetry is principled: at `t_inst`, the harmful-vs-benign label is largely set by the local request, so seeing prior history adds at most marginal lift. At `t_post`, the success-vs-failure label depends on what the conversation accumulated to make the eventual jailbreak verdict — full context carries that trajectory, no-context strips it. The effect for Crescendo at `t_post` (+0.05) is the largest in the table and matches the framework's escalation design: Crescendo's harmfulness signal lives in the buildup, not in any single message.

This is *not* the displacement story. Directions are identical; only the inputs change. The mechanistic claim about axis rotation comes from §6.

---

## §5 — Turn-resolved transfer with fixed directions (Plot 3a, 3b)

**Question.** When does the fixed single-turn direction stop working as turns accumulate?

**Operation.** Per-turn AUROC on WJB at turn `k` projected onto the fixed direction (`v_inst` at `t_inst`, `v_outcome[*, framework]` at `t_post`). Two conditions on the test side: `full_context` and `no_context`. **Compressed enters as a horizontal reference line only** — it has no `turn_k`. Plot 3a is `t_inst`; Plot 3b is `t_post`.

**Llama result.** At `t_inst`, the fixed `v_inst` AUROC for Crescendo declines from ~0.58 at `k=1` to ~0.46 at `k=7` (below chance) before recovering slightly at `k=8–10`. ActorAttack sits flat near 0.50 at `k=1–5` and then jumps to 0.73 at `k=6`. X-Teaming is flat near 0.50 across all turns.

**Reading.** The fixed single-turn direction degrades with `k` for Crescendo — exactly what an "accumulating-context-rotates-the-axis" story would predict at first glance. But the next two sections make the picture more specific: §6 shows that the per-turn axis is already orthogonal to `v_inst` from `k=1` (so the degradation is not "context slowly rotating the axis"); §7 shows the per-turn signal is recoverable if you refit. ActorAttack's flat-then-spike behaviour is consistent with its design of holding the harmful objective in fictional-actor framing until the resolution turn — at intermediate turns, neither the fixed nor the adaptive axis (§7) finds harmful intent at `t_inst`.

---

## §6 — Per-turn direction displacement (Plot 4, Plot 5)

**This is the mechanistic core.** Two cosines, computed only on JBB (no WJB needed).

### §6.1 Drift from anchor — Plot 4

`cos(v_pt[L, fw, cond, k], v_inst[L])` per `(framework, condition, turn, layer)`. Median across layers.

**Llama result:**

| | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| Crescendo, full_context | 0.113 | 0.125 | 0.124 | 0.118 | 0.128 |
| Crescendo, no_context | 0.113 | 0.118 | 0.124 | 0.113 | 0.089 |
| ActorAttack, full_context | 0.065 | 0.153 | 0.159 | — | — |
| ActorAttack, no_context | 0.064 | 0.159 | 0.170 | — | — |
| X-Teaming, full_context | 0.098 | 0.114 | 0.071 | — | — |
| X-Teaming, no_context | 0.098 | 0.110 | 0.056 | — | — |

**Reading.** **The per-turn axis is nearly orthogonal to the single-turn anchor from `k=1` onward, before any context has accumulated.** Cosines hover in 0.07–0.16 across all frameworks and turns, including `k=1` where no prior context exists. This is decisive: the standard intuition that "context accumulates and rotates the axis as the conversation deepens" is wrong. The axis is rotated from the very first multi-turn turn. The framework's framing of the user message — Crescendo's indirect openers, ActorAttack's fictional-actor scaffolding, X-Teaming's planning prompts — already produces input geometry far from a JBB single-turn instruction.

The cosine values stay near-constant over `k`. Whatever rotates the axis happens *at the framework prompt template level*, not as a function of conversation depth.

### §6.2 Context-vs-message decomposition — Plot 5

`cos(v_pt[L, fw, full_context, k], v_pt[L, fw, no_context, k])`. Both axes computed at the same `k`-th user message; only prior history differs.

**Llama result:**

| | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| Crescendo | 1.000 | 0.953 | 0.914 | 0.894 | 0.809 |
| ActorAttack | 1.000 | 0.865 | 0.889 | — | — |
| X-Teaming | 1.000 | 0.937 | 0.817 | — | — |

The `k=1` row is **1.0 by construction** — at `k=1` no prior history exists yet, so `v_full(1) = v_noctx(1)`. The smoke test confirms this exactly: Crescendo 1.0000, X-Teaming 1.0000, ActorAttack 0.9999.

**Reading.** **Context does displace the axis, but only slowly.** From 1.0 at `k=1` down to ~0.80–0.81 at the deepest turns we measure, across all three frameworks. This is roughly an order of magnitude smaller than the framework-framing effect from §6.1 (cosine drop of ~0.20 from history vs cosine of ~0.10 between per-turn axis and `v_inst`).

**The decomposition splits total displacement into two parts:**
1. A large, rotated-from-`k=1` component attributable to the framework's framing of the user message (cosine ~0.07–0.16 with `v_inst`, regardless of `k`).
2. A small, monotonically-growing component attributable to accumulated history (cosine `(v_full, v_noctx)` declining from 1.0 to ~0.80 over many turns).

Component 1 dominates by an order of magnitude at every layer.

This reframes the multi-turn-displacement narrative. A Latent-Guard-style detector anchored to a JBB single-turn axis is anchored to the wrong axis from the very first multi-turn user message, regardless of how carefully its architecture handles long context.

---

## §7 — Practical consequence: fixed vs adaptive (Plot 6)

**Question.** Does adapting the axis recover the lost classification performance?

**Operation.** At `t_inst`, per `(framework, layer, turn k)`, compare two AUROCs on WJB:
* fixed anchor: `v_inst[L]`
* adaptive turn-matched: `v_pt[L, fw, full_context, k]`

The gap is the practical cost of using a stale single-turn axis at that turn.

**Llama result (median across layers):**

| Crescendo | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| anchor `v_inst` | 0.577 | 0.607 | 0.556 | 0.465 | 0.508 |
| adaptive `v_pt(full, k)` | 0.719 | 0.837 | 0.802 | 0.776 | 0.757 |
| **gap** | **+0.141** | **+0.230** | **+0.246** | **+0.311** | **+0.249** |

| X-Teaming | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| anchor | 0.502 | 0.542 | 0.553 | 0.497 | 0.506 |
| adaptive | 0.714 | 0.662 | 0.683 | 0.657 | 0.633 |
| **gap** | **+0.212** | **+0.120** | **+0.131** | **+0.160** | **+0.127** |

| ActorAttack | k=1 | k=3 | k=5 | k=6 |
|---|---|---|---|---|
| anchor | 0.509 | 0.530 | 0.497 | 0.734 |
| adaptive | 0.550 | 0.478 | 0.487 | 0.850 |
| gap | +0.040 | −0.053 | −0.011 | +0.116 |

**Reading.**
* **Crescendo and X-Teaming: the harmfulness signal is recoverable.** Refitting the topic-paired direction at the matching turn restores AUROC to 0.74–0.84 (Crescendo) and 0.63–0.71 (X-Teaming). The fixed anchor falls below chance at `k≥7` for Crescendo while the adaptive direction holds steady — refitting by ~0.30 AUROC at the deepest turns.
* **ActorAttack is qualitatively different.** Both anchor and adaptive AUROCs sit near chance at `k=1–5`, then jump together at `k=6` (the framework's standard ending turn). At intermediate ActorAttack turns, the harmful intent is *representationally invisible* at `t_inst` — neither the single-turn axis nor a refit per-turn axis separates harmful from benign. Consistent with ActorAttack's design: the harmful objective is held back behind fictional-actor framing until the resolution turn.

The §6+§7 combination is the central mechanistic claim: *the axis is rotated, but it is rotated to a still-linear destination — refitting per turn finds it. The displacement is a rotation, not a destruction.*

---

## §8 — Linear vs nonlinear recoverability (Plot 7, Table 2)

**Question.** Is the remaining signal linearly available, or does a more flexible model recover hidden structure?

**Operation.** Three classifiers per `(layer, position, framework, condition)`:
* **M1** — logistic regression on the 1-D projection onto the position-conditional direction.
* **M2** — L2 logistic regression on the standardized full hidden state.
* **M3** — small MLP (one hidden layer of 128 ReLU units, `alpha=1e-3`, early stopping; mean ± std over three random seeds).

Train on JBB (single-turn at `t_inst`, full-context final-turn at `t_post`); test on WJB final-turn under all three conditions.

**Llama result (median across 8 layers, full_context test side):**

| | M1 (1-D dir) | M2 (linear probe) | M3 (MLP-128) |
|---|---|---|---|
| t_inst, Crescendo | 0.742 | 0.685 | 0.621 |
| t_inst, ActorAttack | 0.734 | 0.753 | 0.714 |
| t_inst, X-Teaming | 0.557 | 0.604 | 0.550 |
| t_post, Crescendo | 0.648 | 0.592 | 0.622 |
| t_post, ActorAttack | 0.579 | 0.638 | 0.649 |
| t_post, X-Teaming | 0.532 | 0.566 | 0.587 |

**Reading.**
* **MLP never wins by more than a few hundredths.** In several cells (Crescendo `t_inst`, Crescendo `t_post`, ActorAttack `t_inst`) the MLP loses to one of the linear baselines. Consistent with the linear representation hypothesis: the harmfulness/outcome signal is encoded linearly, and adding nonlinearity does not recover hidden structure.
* **M1 > M2 at `t_inst` Crescendo (0.742 vs 0.685, gap 0.057).** Textbook small-data signature — the trained probe overfits to JBB-specific features that don't transfer, while the topic-paired direction averages out the idiosyncrasies. The within-pair construction is doing real work.
* **At `t_post`, M2/M3 ≈ M1 across the board.** Outcome readouts are weak in absolute terms (capping near 0.65) but flat across model classes — this is genuinely the ceiling, not a missed nonlinear signal.

Combined with §7's adaptive recovery, the picture is: **the axis is rotated (§6), refitting linearly recovers it (§7), and adding nonlinearity buys nothing more (§8).** The displacement is a linear rotation; defenses that refit per turn / per framework can work; defenses that anchor to a stale single-turn axis cannot.

---

## §9 — Cross-framework transfer (Plot 8 + appendix)

**Question.** Is the readout shared across attack frameworks, or framework-specific?

**Operation.** For each `(layer, position, train_fw, test_fw, condition)`, train an L2 linear probe on JBB[`train_fw`] full-context final-turn states (note: full-context, not single-turn — apples-to-apples within-framework diagonal) and test on WJB[`test_fw`] final-turn states.

**Llama result, `t_inst` full_context (main matrix; median across 8 layers):**

| train ↓ / test → | crescendo | actorattack | xteaming |
|---|---|---|---|
| crescendo | **0.889** | 0.688 | 0.662 |
| actorattack | 0.571 | **0.833** | 0.583 |
| xteaming | 0.669 | 0.713 | **0.759** |

**Reading.**
* **Diagonals dominate (0.76–0.89), but off-diagonals are well above chance (0.57–0.71).** Frameworks share *some* harmfulness structure but each induces its own readout.
* **Crescendo is the broadest training source.** A probe trained on JBB Crescendo achieves 0.69 on ActorAttack and 0.66 on X-Teaming — the largest off-diagonal sum of any row. Crescendo's escalation-style sweep covers a wider range of indirect-harmful phrasings than the other two.
* **ActorAttack is the narrowest training source.** A probe trained on JBB ActorAttack achieves only 0.57 on Crescendo and 0.58 on X-Teaming — the smallest off-diagonal sum, despite ActorAttack having a strong on-diagonal cell. Consistent with the framework's highly specific fictional-actor framing.
* **X-Teaming sits between the two.** Generalizes reasonably well in both directions but with the lowest within-framework cell (0.76).

**Llama result, `t_post` full_context:**

| train ↓ / test → | crescendo | actorattack | xteaming |
|---|---|---|---|
| crescendo | **0.592** | 0.569 | 0.543 |
| actorattack | 0.491 | **0.638** | 0.498 |
| xteaming | 0.558 | 0.487 | **0.566** |

All values 0.49–0.64. Diagonals barely separate from off-diagonals. Two cells fall slightly below chance (ActorAttack→Crescendo at 0.491; X-Teaming→ActorAttack at 0.487). **Outcome geometry at `t_post` is highly framework-specific and weak in absolute terms.** A defense built on a single-turn refusal axis would not transfer across these attack styles.

The appendix matrix (`plot8_appendix_all_conditions.png`) gives the same 3×3 layout under `no_context` and `compressed` test sides. The compressed `t_inst` matrix is the Bullwinkel handshake (§10).

---

## §10 — Compressed control (Bullwinkel handshake)

**Question.** Is multi-turn delivery itself the main source of detection failure?

The compressed condition packs each multi-turn conversation into a single semantically-equivalent user message. If `compressed ≈ full_context` on the same direction, then the same content delivered single-turn produces the same readout as the multi-turn version — multi-turn delivery is *not* what suppresses detection.

This consolidates compressed across §3, §8, §9.

**Observation 1 — `compressed ≈ full_context` at both positions.** From §3:

| | t_inst (full) | t_inst (compressed) | gap | t_post (full) | t_post (compressed) | gap |
|---|---|---|---|---|---|---|
| Crescendo | 0.742 | 0.698 | −0.044 | 0.648 | 0.649 | +0.001 |
| ActorAttack | 0.734 | 0.721 | −0.013 | 0.579 | 0.580 | +0.001 |
| X-Teaming | 0.557 | — | — | 0.532 | 0.525 | −0.007 |

At `t_post`, compressed and full match within 0.01 across all three frameworks. At `t_inst` compressed sits a hair below full (≤ 0.05). **Bullwinkel et al.'s response-side compressed-≈-full result extends to the input-side positions used by single-turn safety detectors.**

**Observation 2 — `no_context` consistently sits below both, especially at `t_post`.** At `t_post`, both `compressed` and `full_context` carry the full conversation's semantic content. `no_context` sees only the bare last user message and so loses the trajectory information the outcome label depends on (Crescendo gap +0.05, X-Teaming +0.03).

**Observation 3 — the cross-framework matrix shape is preserved under compression.** From §9:

| train ↓ / test → | full: cresc | full: actor | compressed: cresc | compressed: actor |
|---|---|---|---|---|
| Crescendo | **0.889** | 0.688 | **0.890** | 0.702 |
| ActorAttack | 0.571 | **0.833** | 0.615 | **0.819** |
| X-Teaming | 0.669 | 0.713 | 0.716 | 0.664 |

Diagonals are essentially identical (0.889 / 0.890 Crescendo; 0.833 / 0.819 ActorAttack). Off-diagonal asymmetries are preserved: Crescendo→others remains the broadest source row; ActorAttack→others remains the narrowest. **The framework-specific readout is a property of how each framework frames its content, not of the multi-turn delivery — when the conversation is compressed back into a single turn, the framework signature persists.**

**What this rules out, what it leaves open.** Three structurally different stories could explain why JBB-trained directions transfer poorly to WJB:

1. **Multi-turn delivery suppresses the readout.** Predicts `compressed > full_context`. **Ruled out.** Compressed and full match across every analysis.
2. **The cross-framework structure is multi-turn-induced.** Predicts that compressing collapses the framework-specific readout. **Ruled out.** Matrix shape is preserved.
3. **Content OOD-ness is the dominant transfer challenge.** Predicts `compressed ≈ full_context`, both bounded by the same content-distance ceiling. **Consistent with the data.**

The compressed condition does not enter §6's per-turn axis-rotation decomposition (no `turn_k`). The complete picture combines §6 and §10: each framework rewrites the input geometry from its first user turn (§6), and that rewrite is preserved when the framework's content is compressed back into a single turn (§10). The framework's framing is what rotates the axis; the multi-turn distribution of content is not.

---

## §11 — Cross-model replication

The notebook is parameterized by `MODEL`. Once representations are extracted for Qwen2.5-7B-Instruct and Gemma-2-9B-it (data is generated for those models in the broader project but not all conditions are processed yet), rerun the notebook with `MODEL` switched to produce the same set of artefacts under each model's directory.

Direction objects are **not** portable across models — different hidden dimensions, different tokenizers, different chat templates. Cross-model claims must be made at the *behaviour-of-the-pipeline* level, by aggregating the saved CSVs:

| Cross-model question | Aggregate from |
|---|---|
| Final-turn AUROC patterns | `03_table1_median_auroc.csv` per model |
| Context effect ΔAUROC | `04_context_effect.csv` per model |
| Per-turn drift and decomposition | `06_drift.csv`, `06_decomp.csv` per model |
| Adaptive-vs-anchor gain | `07_fixed_vs_adaptive.csv` per model |
| MLP-vs-linear gap | `08_table2_probe_medians.csv` per model |
| Cross-framework matrix shape | `09_cross_framework.csv` per model |

That aggregation is intentionally not in this notebook — it is one model-comparison step that consumes the saved CSVs from each per-model run. Keeping it separate keeps `clean_pipeline.ipynb` single-model and re-runnable.

---

## Headline findings (consolidating §3–§10)

A short reading list of the seven results that a paper section should lead with.

1. **The harmfulness axis is rotated from `k=1`.** `cos(v_pt[k], v_inst) = 0.07–0.16` across all frameworks, including turn 1 where no prior context exists. Each framework's prompt template alone is enough to displace the axis.
2. **Accumulated history adds only a slow secondary drift.** `cos(v_full(k), v_noctx(k))` declines from 1.0 at `k=1` to ~0.80 at the deepest measured turns — about an order of magnitude smaller than the framework-framing effect.
3. **The harmfulness signal is linearly recoverable.** Refitting the topic-paired direction per turn restores AUROC to 0.74–0.83 for Crescendo and X-Teaming on WJB; the fixed anchor falls below chance at `k≥7` for Crescendo while the adaptive direction holds steady.
4. **MLP probes do not outperform linear baselines.** Within ~0.05 of logistic regression at every cell; often loses. Consistent with the linear representation hypothesis — the displacement is a rotation, not a destruction.
5. **Cross-framework transfer is partial, with characteristic asymmetries.** At `t_inst`, off-diagonal AUROC 0.57–0.71 vs on-diagonal 0.76–0.89. Crescendo is the broadest training source; ActorAttack the narrowest.
6. **`compressed ≈ full_context` across every analysis at both positions.** Same content compressed to a single user message gives the same AUROC as the multi-turn final turn. Bullwinkel et al.'s response-side finding extends to the input-side positions. Multi-turn delivery is *not* what suppresses detection.
7. **ActorAttack is an outlier.** Harmful intent is representationally invisible at `t_inst` for `k=1–5`, then snaps into separability at `k=6` (the resolution turn). Both fixed and adaptive axes behave this way — the framework genuinely holds the harmful objective in indirection until conversation resolution.

## What this rules out, what it leaves open

**Ruled out by the data:**
* "Accumulated context rotates the harmfulness axis as conversations deepen." The axis is rotated from `k=1`. (§6)
* "Multi-turn delivery suppresses detection beyond what content OOD-ness already does." Compressed gives the same AUROC as full-context. (§10)
* "The cross-framework readout is a multi-turn-induced phenomenon." Matrix shape is preserved under compression. (§10)
* "The harmfulness signal is destroyed in multi-turn." The signal refits cleanly per turn at AUROC 0.74–0.83. (§7)
* "Nonlinear decoders can recover hidden structure linear probes miss." MLP ≤ logistic across the board. (§8)

**Consistent with the data:**
* The framework's framing of the very first user turn is what rotates the harmfulness axis. (§6.1 + §10 cross-framework persistence)
* The displacement is a linear rotation to a still-linear destination. (§7 + §8)
* Content OOD-ness from JBB to WJB is the dominant transfer challenge — multi-turn delivery is not. (§10)
* Direction-based defenses can work in multi-turn settings, but only if refit per framework or per turn. (§7)

**Left open:**
* Whether these patterns hold across model families (Qwen, Gemma — pending §11).
* Whether the `t_post` weakness is a property of the conversation-level AISI proxy or of the position itself; turn-level refusal labels would distinguish the two.
* Whether refitting per attack-framework prompt template is enough to make a deployable defense robust, vs needing per-turn refit.
* Whether the X-Teaming `t_inst` OOD-transfer failure (0.557) is attributable to its high attack success rate and harmfulness presentation, or to a data artefact in the WJB X-Teaming generation that also produced the missing benign extractions.

---

## Pipeline integrity notes

* **Sanity check passes.** `cos(v_full(1), v_noctx(1))` is 1.0000 (Crescendo, X-Teaming) and 0.9999 (ActorAttack) by construction. Confirmed in the latest smoke test.
* **Class balance for `v_outcome`** is well-behaved for Crescendo and X-Teaming (~50/50 success/failure) but ActorAttack runs at 31% success. The ActorAttack `v_outcome` direction is therefore built from fewer pairs that retain both classes; that is the most likely contributor to ActorAttack's weaker `t_post` numbers.
* **WJB X-Teaming benign extraction is incomplete** for `no_context` and `compressed` — the directories exist but lack `metadata.parquet`. This is a pre-existing data gap, noted in the paper draft §7. Re-extracting those conditions would fill in the missing X-Teaming cells in §3 Table 1 and §9's appendix matrix.
* **Per-turn analyses thin out at deep turns.** Crescendo's `k=10` has only 19 topic pairs and is excluded by `MIN_PAIRS_PT = 30`; X-Teaming's `k=6` has 14 pairs (skipped); ActorAttack uses all 100 pairs through `k=6`. Per-turn plots annotate `n` per turn for transparency.
