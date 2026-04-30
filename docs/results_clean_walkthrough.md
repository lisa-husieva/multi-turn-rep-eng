# `results_clean.ipynb` — pipeline walkthrough

This document walks through what `notebooks/results_clean.ipynb` does. The pipeline is **framework-agnostic and model-agnostic**: the same code runs identically for every (target model, attack framework) combination — the only thing that changes between runs is which directory of representations is loaded. Read this once, then trust that swapping `MODEL = 'llama'` for `'qwen'` or adding a new framework to `FRAMEWORKS` produces the same five figures with the new data fed in.

> Distinct from `clean_pipeline.ipynb`. `results_clean` runs **within-JBB**, with an attempt-level train/test split (attempts 1–16 fit directions, attempts 17–20 are evaluated). `clean_pipeline` runs the **JBB → WildJailbreak OOD transfer** version of essentially the same questions. Both notebooks live alongside each other; this walkthrough only covers `results_clean`.

## What the notebook produces

Five figures saved to `figures/clean/`, one per claim in §4 of the paper:

| Fig | § | Claim |
|---|---|---|
| 1 | 4.1 | The per-turn harmfulness direction rotates away from the single-turn baseline as `k` grows; most of that rotation is from accumulated context, not the local message. |
| 2 | 4.2a | A fixed single-turn direction degrades across turns; refitting per turn restores AUROC. |
| 3 | 4.2b | Within full-context, MLP ≈ logistic — the signal is linear. Across single-turn → full-context, both fail — the displacement is structural, not a rotation a more powerful model can undo. |
| 4 | 4.3 | Zhao et al.'s `t_inst` (harmfulness) / `t_post` (refusal) functional split holds in single-turn but collapses in multi-turn. |
| 5 | 4.4 | Among refused-harmful conversations at the final turn, the hidden state clusters with accepted-harmful (not benign) only when full context is present. The signal lives in the trajectory, not the bare local message. |

## §0 Setup

Configuration knobs that determine what gets loaded and how the split is drawn:

- **`MODEL = 'llama'`.** Source data root is `data/{MODEL}/representations/`. Switch to `'qwen'` or `'gemma'` to run the same pipeline against another target model — every cell downstream just reads from the new directory.
- **`FRAMEWORKS = ['crescendo', 'actorattack', 'xteaming']`.** One row per framework in the per-framework figures (Fig 1, 2, 5). Pooled figures (3, 4) iterate over whatever's in this list.
- **`TRAIN_MAX_ATTEMPT = 16`.** Each goal is attacked 20 times. Attempts 1–16 are TRAIN; 17–20 are TEST. Train and test share goals; the split is at the attempt level. This is the within-JBB version of generalization — different stochastic attempts on the same goal, not different goals.
- **`MAX_K = 10`.** Turns 1 through 10. Frameworks differ in natural depth (Crescendo 1–10, ActorAttack 1–6, X-Teaming 1–7); turns beyond a framework's max are NaN-filled in heatmaps.
- **`SUCCESS_COL = 'aisi_jailbroken'`.** Boolean derived from StrongReject's AISI rubric (`aisi_scaled > 0.5`). Used for Fig 4's refusal labels and Fig 5's refused/accepted partition.
- **8 layers** loaded from `layer_indices.json`. The saved tensors are `(rows, 8, hidden_dim)`; indexing is by sweep index, not absolute model-layer number.

## §1 Data loading

Four extraction conditions are loaded:

- **`traj`** — full-context trajectories. At each turn `k` the model saw the conversation through that turn. Source: `representations/trajectories/{fw}_{harmful|benign}/`.
- **`nc`** — no-context. The same `k`-th user message under the system prompt alone. Source: `representations/nocontext/{fw}_{harmful|benign}/`.
- **`comp`** — compressed. A single-turn semantic compression of the entire multi-turn conversation. One row per attempt; no `turn_k`. Loaded but not used in this notebook's per-turn analyses (the per-turn analyses are in `clean_pipeline.ipynb`); included in case downstream cells need it.
- **`st`** — single-turn baseline. The 100 JBB harmful + 100 JBB benign goals presented as one-shot prompts. Source: `representations/single_turn/{harmful|benign}/`.

Each load returns `{meta, h_inst, h_post}`:
- `meta` — per-row metadata DataFrame (`pair_id`, `turn_k`, `attempt`, `conversation_id`, framework, label, ...).
- `h_inst` — hidden state at the final attacker token (`(rows, 8, D)`).
- `h_post` — hidden state at the post-instruction end-of-turn token, same shape.

Two metadata-side enrichments happen here:
1. **`add_split`** stamps `data_split = 'train' if attempt <= TRAIN_MAX_ATTEMPT else 'test'` onto every row.
2. **`merge_labels`** joins SR judge labels (`sr_jailbroken`, `aisi_jailbroken`) onto harmful conversations via `conversation_id`. Without this, Fig 4's refusal task and Fig 5 cannot run.

## §2 Direction computation

The workhorse is `paired_direction(h_h, m_h, h_b, m_b)`. Given matched harmful/benign hidden-state arrays and metadata, it:

1. Finds `pair_id`s present in both classes.
2. For each pair, computes `mean(harmful_states) − mean(benign_states)` per layer.
3. Averages those differences across pairs.
4. Normalizes the result to unit length per layer.

The within-pair construction is what cancels topic-specific signal. Without it, the direction would partly encode "this pair is about money laundering" rather than "this is harmful" — the topic-paired pairs are the whole reason JBB is the right training set for this work.

`direction_at_k(data_dict, fw, h_key, k)` wraps `paired_direction` with two filters: `turn_k == k` AND `data_split == 'train'`. Per-turn directions are therefore fitted only on the train side; the test side stays unseen.

Three direction sets are constructed:

- **`v_st_inst[layer]`, `v_st_post[layer]`** — the single-turn baselines, one per layer at each token position. Built from the 100 JBB single-turn pairs (no train/test split there — the single-turn data itself is what trains the baseline). This is the "what a Latent-Guard-style detector would use" axis.
- **`v_full[fw][k][layer]`** — per-turn full-context harmfulness direction, fitted on the train side of `traj[fw, harmful/benign]` at turn `k`.
- **`v_nc[fw][k][layer]`** — per-turn no-context direction, fitted on the train side of `nc[fw, harmful/benign]` at turn `k`. Same `k`-th user message as full-context, but extracted in isolation.

Compressed enters as a *condition* (loaded into `comp`), not as a direction set — there's no `turn_k` to fit a per-turn axis on.

## Fig 1 — §4.1 Direction displacement

Two heatmaps per framework. Y-axis: turn `k` (1–10). X-axis: layer (L1, L5, L9, L14, L18, L23, L27, L32). Color: cosine similarity, RdBu_r colormap, range [−0.2, 1.0].

- **Left column — `cos(v_full[fw][k][L], v_st_inst[L])`.** How aligned is the per-turn axis with the single-turn anchor? Dark red = identical (~1.0). Going pale or blue downward in `k` = the axis is rotating away as the conversation deepens.

- **Right column — `cos(v_full[fw][k][L], v_nc[fw][k][L])`.** Both directions are computed at the same turn `k` from the same `k`-th user message; the only difference is whether prior history is visible. **Equal to 1.0 at `k=1` by construction** (no history exists yet — sanity check). Below 1.0 at later `k` is the *pure context contribution* to the rotation.

Read together: the left panel gives total displacement; the right panel attributes a chunk of it to context specifically. Whatever the right panel does *not* explain (i.e., the gap between the two cosines at `k=1`, where the right panel is 1.0 by construction) is attributable to the framework's framing of the user message itself.

## Fig 2 — §4.2a Detection AUROC (fixed vs adaptive)

Two heatmaps per framework. Y-axis: turn `k`. X-axis: layer. Color: AUROC, viridis, range [0.4, 1.0].

- **Left — fixed `v_ST`.** Project test-split (attempts > 16) hidden states at turn `k` onto `v_st_inst[L]`. AUROC computed against harmful (1) vs benign (0). This is what a single-turn-anchored detector scores on multi-turn data — the practical "would Latent-Guard work here" measurement.

- **Right — adaptive `v_full(k)`.** Project the same test-split rows onto `v_full[fw][k][L]` — the train-side per-turn direction. Per-turn refit, evaluated at the matching turn. This is the "best linear axis we can build for this turn" upper bound on what a single-direction detector could in principle achieve.

The gap between the two heatmaps is the cost of using a stale anchor instead of refitting per turn. `auroc_at_k_layer` requires ≥ 5 samples per class; otherwise NaN.

The train/test split is critical here: directions are fit on attempts 1–16, AUROC is scored on 17–20. Without the split, the adaptive AUROC would be train-set AUROC and would not generalize.

## Fig 3 — §4.2b Linearity

A single panel, frameworks **pooled**, four lines across the 8 layers.

Two training distributions × two classifiers:

- **Within-condition (blue):**
  - Train on TRAIN full-context final-turn rows (harmful + benign, pooled across all frameworks).
  - Test on TEST full-context final-turn rows.
  - Logistic regression (solid) and small MLP (dashed, 64-unit hidden, early stopping).

- **Cross-condition (red):**
  - Train on the 200 single-turn JBB pairs.
  - Test on the same TEST full-context final-turn rows.
  - Same two classifiers.

The combination answers two questions at once:

1. *Is the signal linear?* — within-condition MLP vs logistic. If they match, no nonlinear structure for a defender to exploit beyond a linear probe.
2. *Is the displacement recoverable by adding model power?* — cross-condition MLP vs logistic. If both fall to chance, the rotation between single-turn and full-context geometries is structural — adding a hidden layer does not "unrotate" an axis.

Both classifiers wrap a `StandardScaler` (the hidden states have very different scales across layers). Pooled across frameworks because the linearity question is about the representation, not framework-specific structure.

## Fig 4 — §4.3 Token-position collapse

Two panels side-by-side. X-axis: layer. Y-axis: AUROC. Up to four lines per panel — the cross product `{t_inst, t_post} × {harmfulness, refusal}`.

- **Left — single-turn baseline.** Harmfulness AUROC at both positions, exactly as Zhao et al. would compute it: difference-of-means direction from harmful vs benign at that position, projection score, AUROC against the goal label. Refusal is undefined here (no refusal labels on single-turn data — the model wasn't asked to refuse anything ambiguous).

- **Right — multi-turn full-context final turn, frameworks pooled.** Harmfulness uses the same construction on TRAIN final-turn pooled rows, evaluated on TEST. Refusal is defined within harmful conversations: refused (`aisi_jailbroken == False`) vs accepted (`True`). Same difference-of-means construction, same projection, AUROC against the refused/accepted label.

Zhao's prediction: in single-turn settings, `t_inst` is the harmfulness position and `t_post` is the refusal position — solid lines (harmfulness) should be higher at `t_inst`, dashed lines (refusal) higher at `t_post`. In multi-turn, that functional separation collapses: both positions carry both signals at similar strength. The figure puts the prediction next to its multi-turn counterpart so the collapse is visible directly.

## Fig 5 — §4.4 Signal location (context vs message)

One panel per framework. X-axis: layer. Y-axis: a fraction in [0, 1].

For each (framework, layer):
1. Take refused-harmful conversations at the final turn (where `aisi_jailbroken == False`).
2. Compute the centroid (mean) of accepted-harmful conversations at the same final turn (`aisi_jailbroken == True`).
3. Compute the centroid of all benign conversations at the final turn.
4. For each refused-harmful sample, measure Euclidean distance to both centroids and ask which one is closer.
5. Plot the fraction whose nearest centroid is the **harmful** one.

Two lines per panel:
- **Full-context (blue):** the model saw the whole trajectory through the final turn.
- **No-context (orange):** only the bare final user message, system-prompted.

`0.5` is chance. Above `0.5` means refused-harmful conversations live in the harmful neighborhood — the model declined to comply but its hidden state still says "this is harmful content." If that's true under full-context but at chance under no-context, the harmfulness signal lives in the *trajectory*, not the local message — the bare last user turn looks benign on its own, but the conversation that led to it doesn't.

Different frameworks may pattern differently here. A framework whose harmful intent is built up gradually (Crescendo's escalation) will tend to put the signal in the buildup, so the full-context vs no-context gap is large. A framework whose harmful intent rides on the final message's framing (a hypothetical role-play closer with the harmful instruction in plain sight) might keep the signal locally and show a smaller gap.

## What changes between runs

Nothing in the code changes. The same notebook runs identically for:

- **Different target models.** Change `MODEL = 'llama'` to `'qwen'` or `'gemma'`. Every downstream cell reads from `data/{MODEL}/representations/` and writes to `figures/clean/`. Either run them sequentially and rename the output dir between runs, or copy the notebook into `notebooks/{model}/results_clean.ipynb` per model.
- **Different attack frameworks.** `FRAMEWORKS` controls which directories get loaded and which framework rows appear in per-framework figures. Add or remove a framework name and the figure layouts adjust.
- **Different judge thresholds, layer sweeps, or attempt splits.** All driven by the constants in §0; no code changes needed for sensitivity analysis.

The pipeline is deliberately decoupled from any single (model, framework) combination so that the cross-model and cross-framework story in the paper can be read off three runs (or six, or nine) of the same notebook with different inputs — not from a separate analysis per model.
