# Experiment Plan — Multi-Turn Displacement Paper

Planning document for the research paper extending `paper_draft.md`. Defines the spine of the paper, refined research questions, experiment bundles, probe-quality certifications, external-generalization protocol, sample-size governance, and the circuit-breakers mechanism study.

Companion docs: `paper_draft.md`, `methods.md`, `experiment_design.md`, `circuit_breakers_plan.md`.

---

## 1. Spine of the paper

One sentence the paper earns:

> Multi-turn jailbreaks rotate the harmfulness direction out from under single-turn representation-level defenses; this rotation is structural (not a linear rotation a nonlinear probe can chase), it replicates across three open-weight model families, and it mechanistically explains why circuit breakers — the canonical representation-level defense — fail on multi-turn attacks.

Every experiment and figure either establishes this sentence, certifies it, or externally validates it. Anything that does none of the three is supplementary or cut.

## 2. Research questions, ranked by paper weight

| | RQ | Role | Paper tier |
|---|---|---|---|
| **RQ1** | Does the direction displace as context accumulates? | Phenomenon | Primary |
| **RQ2** | Does displacement break detection, and is it linear? | Consequence | Primary |
| **RQ5** | Does displacement explain circuit-breakers' multi-turn failure? | Mechanism | **Primary (new)** |
| RQ3 | Do `t_inst`/`t_post` roles persist across turns? | Tension with Zhao et al. | Secondary |
| RQ4 | Where does the harmfulness signal live? | Diagnostic | Secondary |

RQ3 and RQ4 remain in the paper but as shorter sections — they support the central claim without carrying it.

## 3. Hypotheses

**RQ1 — Displacement.**
- **H1a:** `cos(v_full(k), v_ST)` decreases monotonically with k on all three models.
- **H1b:** `cos(v_full(k), v_nc(k))` starts near 1.0 at k=1 and decreases with k — the gap is context, not message content.
- **H1c:** Displacement shape is consistent across frameworks; magnitude scales with escalation gradualness (Crescendo > X-Teaming > ActorAttack).

**RQ2 — Detection failure and linearity.**
- **H2a:** Fixed-ST linear AUROC is near chance at mid/late turns; per-turn adaptive linear recovers most of it.
- **H2b:** An MLP trained on single-turn data cannot recover the multi-turn signal (MLP ≲ fixed-ST linear). Displacement is structural, not a linear rotation.
- **H2c:** Within each condition, MLP ≈ LR — harmfulness is linear given the right anchor.

**RQ3 — Positional role persistence.**
- **H3:** The `t_inst` (harmfulness) vs `t_post` (refusal) separation from Zhao et al. collapses — both positions perform similarly on both tasks in multi-turn. Reported as a limit on the Zhao et al. result, not a refutation.

**RQ4 — Locus of the harmfulness signal.**
- **H4a:** In full-context, refused-harmful final-turn states cluster nearer to accepted-harmful than to benign at mid layers.
- **H4b:** In no-context, this collapses — the model only "knows" the conversation is harmful because of accumulated history.
- **H4c:** Effect is framework-specific (Crescendo yes, ActorAttack no) — the two fail for different representational reasons.

**RQ5 — Circuit-breakers mechanism.**
- **H5a:** The CB model's rerouting direction at its intervened layers is close to `v_harmful_base(ST)` but orthogonal to `v_harmful_base(k)` at late k.
- **H5b:** Displacement pattern persists on the CB model — `cos(v_full(k), v_ST)` drops with k with similar shape.
- **H5c:** CB training sharply reduces single-turn JBB ASR but leaves multi-turn ASR largely intact.
- **H5d:** *(Supplement-tier)* On the CB model, per-attempt displacement magnitude correlates with attack success.

## 4. Experiment bundles

Six bundles. Critical path is **E1 → E2 → E3 → E6** for main figures; **E5** runs in parallel after scaffolding.

### E1 — Probe/direction quality certification *(blocking — do first)*

Today `03_probe_quality.ipynb` has six descriptive checks. For the paper these must be promoted to *certifications* with pre-declared falsification thresholds.

**Additions:**
1. **Bootstrap CIs at the conversation level.** Resample whole conversations with replacement (not turn-rows — they're dependent within a conversation). Recompute direction and AUROC. Report 95% CI on every headline number.
2. **Topic-orthogonalization test.** Compute `v_topic` from JBB's 10 harm categories (one-vs-rest mean diffs → top PCs). Orthogonalize `v_harmful` against `v_topic`. Recompute AUROC. If AUROC drops <0.05, the direction is harmfulness, not topic. This is the cleanest answer to the "harmful vs benign confounds" limitation in the paper draft.
3. **Label-randomization null.** Shuffle harmful/benign labels within pairs, recompute direction, report AUROC distribution. Should be ≈0.5 — puts a floor on artifact.
4. **Leave-pair-out** *(already in nb03)* — promote to certification table with CIs.
5. **Layer-robustness band.** Report AUROC ± over the mid-depth layer block (e.g., proportional positions 3–5 of 8). Headline claim must survive layer choice.
6. **Direction-computation sensitivity** *(already in nb03)* — certify that paired subtraction beats unpaired / PCA / random baselines and that `v_harmful` is method-stable.

**Falsification thresholds (pre-declared in Methods):**
- Orthogonalized AUROC must stay ≥ 0.70 at mid layers → direction is "harmfulness," not "topic."
- Randomization null AUROC must fall in [0.48, 0.52].
- Bootstrap 95% CI on the displacement endpoint `cos(v_full(k_max), v_ST)` must exclude 0.5.

**Deliverables.**
- `notebooks/03_probe_quality.ipynb` extended with (1)–(3) and (5).
- One supplement table per model: AUROC, 95% CI, orthogonalized AUROC, LPOCV AUROC, null AUROC.

### E2 — Sample-size governance *(fast, one-shot)*

Per-turn sample sizes from current representations:

| | Crescendo | X-Teaming | ActorAttack |
|---|---|---|---|
| Llama  | n ≥ 278 through k=10 | falls off after k=5 (n≈295) | fixed 6 turns, n ≈ 2k |
| **Qwen**   | **n=4 at k=10** | **n=19 at k=6** | fixed 6 turns, n ≈ 2k |
| Gemma  | n ≥ 913 through k=10 | falls off after k=5 | fixed 6 turns, n ≈ 2k |

**Rules:**
- Pre-register: **n ≥ 100 per class per turn** for inclusion in primary figures.
- Report n(k) on every per-turn figure (secondary axis or sample-size strip beneath the plot).
- Qwen Crescendo and X-Teaming late turns: either truncate or pool across frameworks. Never draw a line through n=4.

**Deliverables.**
- Shared util that plots a per-turn n band alongside any per-turn AUROC/cosine line.
- Supplement table: `n(k)` for each `(model, framework, split)`.

### E3 — External generalization (JBB → WildJailbreak)

Addresses the paper draft's biggest limitation (attempt-based train/test split, same goals in both). Tiered by cost:

**Tier 1 — single-turn ST generalization (cheap).** JBB-trained `v_harmful(ST)` classifies WildJailbreak vanilla harmful vs benign. Already done for Llama (`04_wildjailbreak_single_turn.ipynb`). Replicate on Qwen and Gemma. Certifies that the ST anchor generalizes beyond JBB before any multi-turn claim anchored to it is credible.

**Tier 2 — adversarial single-turn (moderate).** Same as Tier 1 but include WJ `harmful_adversarial` / `benign_adversarial`. Tests whether single-turn adversarial prompts already induce partial displacement.

**Tier 3 — multi-turn external (expensive, stretch).** Run Crescendo on 50 WJ harmful + 50 WJ benign goals against each of the three models (~3,000 conversations per model, ~$30–50 attacker API per model). Extract full-context reps. Compute `v_full(k)` on JBB train, evaluate on WJ multi-turn test. If adaptive AUROC on WJ matches JBB-test, per-turn direction is a property of (model, k), not overfit to JBB goals.

**Recommendation:** ship with Tier 1 + Tier 2. Do Tier 3 for Crescendo only if time permits. Otherwise explicit "external multi-turn validation" future-work item.

### E4 — AISI-StrongREJECT as the one true label

**Policy.**
- **AISI is the paper-quality label.** Framework judges optimized against themselves; rubric is lenient; AISI is independent and strict. All RQ3/RQ4 figures default to AISI.
- Report ASR two ways in the main table: AISI and framework-judge (not rubric). Gap = standard overclaiming caveat, named.
- For CB work (RQ5), AISI-based ASR is the metric for H-CB3 and H-CB4.

**Per-turn labels: not required.** RQ3 and RQ4 use final-turn labels and the argument is clean ("for conversations that ultimately succeeded, what did the final-turn state look like?"). Per-turn AISI scoring is nice but expensive — defer unless a specific question demands it.

### E5 — Circuit breakers *(landing experiment)*

See `circuit_breakers_plan.md` for full detail. Two adjustments from that plan:

**Tightening.** H-CB3 (displacement ↔ per-attempt success on CB model) is weaker than the others and harder to power. Demote to supplement. Promote **H-CB1 (rerouting-direction mismatch)** as the headline of the CB section — clean, measurable geometric claim that directly ties the mechanism story.

**Addition.** Run the E1 probe-quality certifications on the CB model's directions. If `v_harmful_CB(k)` is noisier or more topic-entangled than base, that's itself a finding.

**Scope discipline.** Plan says 1 week; budget 2. Biggest hidden cost: verifying GPT-4o-as-attacker doesn't degenerate against CB refusals — smoke-test on 3 pairs before committing to the full run.

### E6 — Cross-model consolidation

Extend `08_cross_model_analysis.ipynb` so it:
- uses AISI consistently (already does via `SUCCESS_LABEL`)
- overlays per-turn n bands from E2
- displays bootstrap CI ribbons from E1
- handles `len(MD) < len(MODELS)` cleanly (drop Phi from `MODELS` unless extraction is run — see §6)

## 5. Suggested paper structure

1. Intro + prior work
2. Methods (§3.1–3.7 of current draft, + §3.8 Probe quality, + §3.9 CB methodology)
3. **Probe quality certification** (short, front-loaded — makes everything downstream credible)
4. **RQ1 Displacement** (3 models + WJ Tier 1/2)
5. **RQ2 Detection failure & linearity** (3 models + WJ)
6. **RQ3 Positional roles** (short)
7. **RQ4 Locus of signal** (short)
8. **RQ5 Circuit breakers** (mechanism + landing)
9. Discussion: generalization (WJ), limitations (per-turn n, model family, JBB-as-training), future work (multi-turn-aware CB loss)
10. Appendix: sample-size tables, per-framework breakdowns, full CIs

## 6. Scope decisions

**Phi-3.5-mini-instruct.** Conversations exist at `data/phi/conversations/`; representations don't. Extraction is ~4–8 GPU-hours. Two paths:

- **Drop Phi for the initial paper.** Title becomes "three open-weight models" (Llama/Qwen/Gemma). Clean story, existing figures. **Recommended.**
- **Keep Phi.** Fourth model adds robustness across scale (3.8B vs ~7–9B). Defensible. Requires re-score AISI, rerun nb03/nb08. Another 1–2 days.

**Default: drop Phi**, reintroduce only if E5 surfaces something that benefits from a smaller-model data point. Note as "robustness across scale" in limitations.

## 7. Execution order

Critical path (6-week plan):

```
Week 1   E1 (probe quality) · E2 (sample-size floor) · E4 (settle AISI policy)
Week 2   E3 Tier 1 + Tier 2 (WJ single-turn on Qwen & Gemma)
         E6 (nb08 overhaul: CIs, n bands, AISI defaults)
Week 3   E5 scaffold: notebooks/llama_3_8B_{instruct,RR_instruct}/ + smoke-test generation
Week 4   E5 full data gen + extraction, 14_circuit_breakers_analysis.ipynb
Week 5   E5 writeup + figures · E3 Tier 3 if budget allows
Week 6   Paper draft consolidation, revisions
```

**Parallelizable.** E1, E2, E4 run together in week 1 (no dependency). E3 Tier 1/2 is GPU-bound; E5 data generation is API-bound — they don't compete.

## 8. Connections to prior work

| Experiment | Contribution vs prior work |
|---|---|
| E1 | Certifies that the "harmfulness direction" deserves the name. Most RepE papers don't do this explicitly. |
| RQ1/RQ2 cross-model | Bullwinkel et al. studied one model; this generalizes to three open-weight families. |
| RQ2 linearity | Confirms linear-representation hypothesis *within* condition; shows it does not survive *across* conditions. |
| RQ3 position collapse | Bounds Zhao et al.'s Latent Guard finding — the `t_inst`/`t_post` separation is a single-turn property. |
| RQ4 locus of signal | Complements Arditi et al.'s refusal-direction work with a multi-turn recognition story. |
| E3 Tier 3 | External-dataset multi-turn replication Bullwinkel et al. did not do. |
| E5 CB | Mechanistic explanation of the Zou et al. defense's known multi-turn failure — the paper's contribution upgrade. |

## 9. Things explicitly cut

- **Per-turn SR scoring.** Final-turn labels support the paper's claims cleanly.
- **Phi-3.5-mini.** Drop unless E5 surfaces a need.
- **WJ Tier 3 for all frameworks.** Crescendo-only if at all.
- **H-CB3 as a headline.** Supplement.
- **Training a new multi-turn-aware CB variant.** Follow-up paper. Flagged in discussion.
- **Interpreting the rerouting direction vs arbitrary axes** (topic, refusal, etc.). Stays within the `v_harmful` family.

## 10. Open questions to confirm before kickoff

1. **Narrative fit.** Is CB the landing story, or is there more attachment to RQ3/RQ4 being primary?
2. **Compute / API budget.** Gates Tier 3 WJ and the CB study.
3. **Timeline.** 6-week plan is aggressive. Hard deadline would trim WJ Tier 3 and Phi first.
4. **Pre-registration depth.** The falsification thresholds in E1 are proposed; confirm the cutoffs before running certifications so they aren't retrofitted.
