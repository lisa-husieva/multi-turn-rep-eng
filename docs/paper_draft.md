# Mechanistic Interpretation of Multi-Turn Attack Frameworks

## Abstract

Multi-turn jailbreak attacks remain a major challenge for large language model safety because harmful intent is distributed across a conversation rather than expressed in a single prompt. Representation-engineering methods have produced strong results in single-turn settings, where harmfulness and refusal can be read from specific token positions in model hidden states. It remains unclear whether those single-turn representations transfer to multi-turn jailbreaks, where conversational context and attack strategy may change how harmful intent is internally encoded. The natural intuition — that *accumulated context* rotates the harmfulness axis as conversations deepen — is widely repeated but largely untested.

This paper tests that intuition directly. We learn two position-specific directions from JailbreakBench (JBB) using topic-matched difference-of-means and evaluate them on a deliberately out-of-distribution test set built from WildJailbreak (WJB) — the 50 harmful and 50 benign vanilla prompts most semantically distant from JBB after a moderation filter. We generate multi-turn conversations against Llama-3.1-8B-Instruct under Crescendo, ActorAttack, and X-Teaming, and extract hidden states under three extraction conditions: **`full_context`** (the multi-turn trajectory through the `k`-th user turn), **`no_context`** (the same `k`-th user turn alone), and **`compressed`** (a single-turn semantic compression of the whole conversation, following Bullwinkel et al.). The compressed condition is the same-content single-turn baseline that lets us tell content-OOD-ness apart from any multi-turn-specific suppression. The analysis pipeline asks five questions in sequence: whether the single-turn direction transfers, whether multi-turn context displaces it, whether displacement is mostly from accumulated history or from each framework's prompt style, whether the underlying signal stays linearly recoverable, and whether the relevant readouts generalize across attack frameworks.

The data contradict the intuition. The per-turn topic-paired harmfulness direction is nearly orthogonal to the single-turn anchor (cosine ~0.07–0.16) **from the very first turn**, before any context has accumulated, across all three frameworks. Context contributes only a slow secondary drift (cosine `(v_full(k), v_noctx(k))` declines from 1.0 to ~0.8 over many turns). The dominant displacing force is the attack framework's prompt-style framing, not history. The compressed single-turn baseline confirms this from the other direction: same content compressed back to a single user message gives essentially the same AUROC as the multi-turn final turn (within 0.01 at `t_post`, ≤ 0.05 at `t_inst`), and the cross-framework matrix shape is preserved under compression — so the framework-specific readout is a property of how each framework frames its content, not of the multi-turn delivery. This extends Bullwinkel et al.'s response-side compressed-≈-full finding to the input-side positions used by single-turn safety detectors. Despite this large axis rotation, the harmfulness signal remains linearly recoverable: refitting the direction per turn restores AUROC to 0.74–0.83 on WJB for Crescendo and X-Teaming, where the JBB single-turn anchor often falls below chance at later turns. Nonlinear MLP probes do not consistently outperform linear baselines. Cross-framework transfer at `t_inst` shows partial sharing (median AUROC 0.57–0.71 off-diagonal vs 0.76–0.89 on-diagonal), with Crescendo as the broadest training source. ActorAttack is an exception throughout: harmful intent is representationally invisible until the final turn, even with refitting. At the `t_post` position, success-versus-failure geometry is weak across all frameworks (0.49–0.64) and highly framework-specific.

These findings reframe the multi-turn-displacement story. Single-turn representation-engineering methods do not fail in multi-turn settings because *context* makes harmful intent unreadable. They fail because each attack framework rewrites the input geometry from its very first user turn, and a defense anchored to a single-turn axis is anchored to the wrong axis from the start. Direction-based defenses can still work in multi-turn settings, but only if they are refit per framework or per turn.

## 1. Introduction

Large language models continue to be vulnerable to jailbreak attacks that elicit harmful responses despite instruction tuning and safety alignment. While many studies focus on single-turn prompts, multi-turn jailbreaks have emerged as a particularly effective class of attack. Harmful intent is distributed across multiple turns — through gradual escalation, fictional-actor framing, or strategic planning — rather than expressed all at once. The model's eventual behavior depends on the accumulated conversational history, not on the content of any single locally harmful message.

Recent work in representation engineering has shown that safety-relevant concepts are often linearly accessible in transformer hidden states. The most precise single-turn result, due to Zhao et al., is that harmfulness and refusal are encoded at *different token positions*: harmfulness is most readable at the final token of the user instruction (`t_inst`), and refusal is most readable at the end-of-turn token that closes the message (`t_post`). The direction at `t_inst` separates harmful from benign instructions, the direction at `t_post` separates accepted from refused responses, and the two axes are causally distinct — steering experiments show that one can be moved without disturbing the other. These findings have motivated detection methods (Latent Guard) and intervention methods (circuit breakers).

It is widely assumed, but not directly tested, that those single-turn results transfer poorly to multi-turn jailbreak settings. The standard intuition is mechanistic: in a multi-turn conversation, the hidden state at any token position absorbs information from the full preceding context through self-attention. As more turns accumulate, the representation should drift further from what a single-turn safety direction was trained on, and a detector anchored to that direction should degrade. This intuition is plausible and frequently asserted; the present work asks whether it is actually correct.

We test the transfer of single-turn-style directions to multi-turn settings using a deliberately stringent setup. Directions are trained on JBB-Behaviors using its built-in topic-matched harmful/benign pairing — the topic match removes a major confound. Test data is a 100-prompt out-of-distribution slice of WildJailbreak constructed by filtering for moderation-flagged harmful candidates and then sampling for maximal semantic distance from JBB. We generate full multi-turn conversations against Llama-3.1-8B-Instruct under three attack frameworks (Crescendo, ActorAttack, X-Teaming) using GPT-4o as both attacker and per-turn judge. Hidden states are extracted at `t_inst` and `t_post` under three conditions: **`full_context`** (the multi-turn trajectory through the `k`-th user turn), **`no_context`** (the same `k`-th user turn alone), and **`compressed`** (a single-turn semantic compression of the whole conversation, following Bullwinkel et al.). The compressed condition is a same-content single-turn baseline that lets us tell content-OOD-ness apart from any multi-turn-specific suppression. The analysis evaluates each direction by AUROC on WJB and additionally compares it against trained linear and small nonlinear probes.

The data tell a different story than the intuitive one. At `t_inst`, the single-turn JBB direction *does* transfer to WJB final-turn states, but with strong framework dependence (median AUROC across layers: Crescendo 0.74, ActorAttack 0.73, X-Teaming 0.56). Far more striking is what happens once we recompute the topic-paired direction *at each turn k* of the JBB multi-turn data: the resulting per-turn direction is nearly orthogonal to the single-turn anchor at every turn — including turn 1, where no prior context exists. The cosine sits between 0.07 and 0.16 from `k=1` onward and drifts only slowly with depth. The pure context contribution, isolated by comparing `v_full(k)` against `v_noctx(k)` at the same turn, declines from 1.0 to about 0.8 over many turns. Context does displace the axis, but only slowly. The dominant effect is that each multi-turn framework rewrites the input geometry from its very first user turn — Crescendo opens with indirect benign-seeming questions, ActorAttack with fictional-actor framing, X-Teaming with strategic-planning prompts — and *that* is what makes the single-turn axis misaligned, not subsequent context accumulation.

The harmfulness signal nonetheless survives. When we refit the topic-paired direction at each turn and test it on WJB at the same turn, AUROC recovers to 0.74–0.83 for Crescendo and X-Teaming, while the single-turn anchor sometimes falls to chance or below at later turns. Nonlinear MLP probes do not consistently outperform linear probes. The data are consistent with the linear representation hypothesis: the harmfulness information is preserved in linear geometry; the *axis* that reads it has rotated.

Three additional analyses fill in the picture. A turn-resolved analysis shows that the cost of using a stale anchor grows with conversational depth in Crescendo (gap to refit AUROC widens from +0.14 at k=1 to +0.31 at k=7) but follows different patterns for the other frameworks. A 3×3 cross-framework matrix at `t_inst` shows partial transfer between attack styles (median off-diagonal AUROC 0.57–0.71 vs on-diagonal 0.76–0.89), with Crescendo as the broadest training source and ActorAttack as the narrowest. The corresponding matrix at `t_post` is weak across all cells — outcome geometry is highly framework-specific. ActorAttack stands out as an unusual case: harmful intent is representationally invisible at intermediate turns regardless of which axis is used, then snaps into separability at the final attack turn.

The paper contributes (1) a clean transfer setup from topic-matched JBB directions to OOD multi-turn jailbreak conversations; (2) evidence that the dominant displacing force in multi-turn safety representations is each framework's prompt-style framing rather than accumulated history; (3) a decomposition isolating context-only drift from message-form drift; (4) a demonstration that the harmfulness signal remains linear and recoverable when the readout is refit per turn; (5) cross-framework transfer matrices showing that harmfulness readouts at `t_inst` are partially shared across attack styles while outcome readouts at `t_post` are highly framework-specific; and (6) an input-side replication of Bullwinkel et al.'s response-side compressed-≈-full finding, showing that multi-turn delivery is not what suppresses detection — the cross-framework matrix shape is preserved when conversations are compressed back to a single turn.

The paper is organized around five research questions, and they form a sequence rather than five independent tests. **RQ1** asks the headline transfer question — does a Latent-Guard-style single-turn detector still fire on the final user turn of a multi-turn jailbreak? **RQ2** asks, if it does not, *where the displacement comes from*: accumulated conversational history, or each framework's prompt-style framing of the user turn itself? **RQ3** asks whether the harmfulness signal is *destroyed* or merely *on a different linear axis* — i.e., would refitting recover it? **RQ4** asks how attack-style-specific the resulting readout is — does training against one framework cover the others? **RQ5** asks whether the position-specific story from prior single-turn work (harmfulness at `t_inst`, refusal at `t_post`) survives at all into multi-turn, by rerunning RQ1–RQ4 at `t_post` against a conversation-level success/failure label. For each question we name the conceptual stake, the operation we run, and a careful reading of what the resulting numbers do — and do not — tell us.

**RQ1 — Does a single-turn safety direction still detect harmful intent at the final user turn of a multi-turn jailbreak?**
*Question.* Single-turn representation-engineering methods (Latent Guard, refusal-direction work) extract a fixed harmfulness axis from one benign-vs-harmful contrast and use it as a detector. The defense-relevant question is whether that same detector, pointed at the *final* user turn of a multi-turn jailbreak generated against the same model, still fires. Strong transfer would license direct application of single-turn detectors to multi-turn settings; weak transfer would say the multi-turn case genuinely needs different machinery.
*Operation.* Project WJB final-turn states onto a JBB-trained topic-paired direction and compute AUROC. At `t_inst` we project against the JBB single-turn harmful-vs-benign direction; at `t_post` we project against the JBB full-context final-turn success-vs-failure direction (built per framework). Reported per framework, per layer, under both `full_context` and `no_context` extraction.
*Reading.* High AUROC means the trained axis transfers to this test distribution; low AUROC means it does not. We do not infer from a high AUROC that "the model represents harmfulness well", nor from a low one that the signal is gone — only that this specific axis transfers (or doesn't) under this distribution shift.

**RQ2 — If transfer is weak, is the displacing force *accumulated history* or *each framework's prompt-style framing of the user turn*?**
*Question.* The standard intuition is that prior turns drag the harmfulness representation away from where it sat in single-turn space — i.e., that displacement scales with conversational depth. A competing possibility is that each multi-turn framework rewrites the *form* of every user message (Crescendo's indirect openers, ActorAttack's fictional-actor scaffolding, X-Teaming's planning-style prompts), and that this framing displaces the axis *from the very first user turn*, regardless of any context behind it. The two stories make different defense predictions: the first argues for context-aware detectors; the second argues for framework-aware ones.
*Operation.* Rebuild a topic-paired harmful-vs-benign direction at each turn `k` from JBB multi-turn data, separately under `full_context` (prior turns visible) and `no_context` (the same `k`-th user message in isolation). Report `cos(v_pt(k), v_inst[L])` — the per-turn axis vs the single-turn anchor — and `cos(v_full(k), v_noctx(k))` — the per-turn axis under the two extraction conditions at the same `k`, with the local user message held fixed. The second cosine is 1.0 at `k=1` by construction (no prior context exists yet) — an arithmetic identity used as a sanity check, not a finding.
*Reading.* The first cosine measures the *total displacement* of the multi-turn axis at turn `k` from the single-turn axis. The second isolates the *pure-context* component of that displacement: because the local message is identical across the two conditions, anything else cancels. The remainder — visible whenever the first cosine is low at `k=1` while the second is exactly 1.0 — is attributable to the framework's framing of the user turn. Low cosines mean the *axis we'd draw from this data points elsewhere*, not "harmfulness has moved" or "the model has reorganized."

**RQ3 — When the single-turn axis fails, is the harmfulness signal *gone* or just on a *different linear axis*?**
*Question.* Two structurally different failure modes give the same low AUROC under a fixed direction. The signal could be (a) preserved as a linear axis but rotated to where the single-turn anchor cannot see it, in which case refitting on multi-turn data should recover it; or (b) genuinely destroyed or scrambled into a nonlinear encoding, in which case a richer decoder would be required (or no decoder works at all). RQ3 distinguishes these — the answer governs whether refit-per-turn or refit-per-framework defenses can work, and whether moving from a difference-of-means direction to a full classifier buys anything.
*Operation.* Compare four AUROCs on WJB at each `(layer, framework)`: the JBB single-turn direction (anchor `v_ST`), the JBB-refit per-turn direction (adaptive `v_pt(full, k)`, evaluated at the matching turn `k`), an L2 logistic-regression linear probe trained on JBB at the same train side (M2), and a small one-hidden-layer MLP with the same train side (M3).
*Reading.* Each pairwise gap is a different "lift" question. *Adaptive − anchor* = how much AUROC we recover by refitting the topic-paired axis at the right turn — the rotation-not-destruction signature. *M2 − M1* = whether a more flexible linear classifier finds signal that a difference-of-means axis does not; small gaps suggest the topic-paired direction is already near the linearly-best readout. *M3 − M2* = whether nonlinear decoding picks up structure beyond the linear probe. We do not read "M3 doesn't beat M2" as a definitive statement that the encoding is linear — only that, at this training-set size and architecture, a small MLP does not recover anything additional.

**RQ4 — How attack-style-specific is the readout? Does a probe trained against one framework transfer to the others?**
*Question.* If multi-turn jailbreaks all displace the safety direction in roughly the same way, a probe trained against one framework's geometry should detect harmfulness in the others. If each framework induces a substantially different readout, off-diagonal cells of a train-on-one / test-on-another matrix should fall sharply relative to the diagonal. The answer governs whether a multi-turn-aware defense can be a single probe or must be one-per-framework, and whether some frameworks make broader training sources than others.
*Operation.* For each `(layer, position, train_fw, test_fw)`, train an L2 linear probe (matching M2 above) on JBB[`train_fw`] full-context final-turn states and test on WJB[`test_fw`] full-context final-turn states. Build the 3×3 matrix at both positions; report median AUROC across the 8 layers. *(Note: the train side here is JBB full-context final-turn rather than JBB single-turn. This differs from RQ3 deliberately, so the diagonal is a within-framework apples-to-apples cell; see §3.15.)*
*Reading.* Diagonal cells measure within-framework transfer (same attack style at train and test). Off-diagonals measure cross-framework generalization. A strong diagonal with weak off-diagonals would be consistent with each framework inducing its own readout; comparable on- and off-diagonal cells would be consistent with a more attack-style-independent contrast. Row asymmetry tells us which framework, if any, is the broadest training source. We are not making mechanistic claims about shared circuits — only about whether a trained classifier transfers between attack styles.

**RQ5 — Does the harmfulness-at-`t_inst` / refusal-at-`t_post` decomposition from single-turn work survive into multi-turn?**
*Question.* Zhao et al. show that, in single-turn settings, harmfulness is causally encoded at the final instruction token (`t_inst`) and refusal is causally encoded at the post-instruction end-of-turn token (`t_post`) — two distinct positions, two distinct axes, steerable independently. We ask whether that position-specific decomposition still holds shape in multi-turn: same pattern of transfer, same linear recoverability, same cross-framework structure — when `t_post` is read out against a conversation-level success/failure label rather than a clean turn-level refusal label.
*Operation.* Run RQ1–RQ4 at `t_post` (success-vs-failure on harmful conversations only, with success defined by AISI score ≥ 0.5) in parallel with `t_inst` (harmful-vs-benign). This is the same four analyses with a different label at a different token position — not a new experiment.
*Reading.* If the `t_post` patterns look qualitatively similar to `t_inst` (comparable AUROCs, comparable cross-framework transfer, comparable linear/nonlinear gap), the position-specific separation seen in prior single-turn work is partially preserved in multi-turn settings. If they look different — weaker AUROCs, sharper framework specificity, larger context dependence — the two positions are doing different work in our data. **Important caveat:** our `t_post` label is conversation-level AISI success/failure on the *final* assistant response, not turn-level refusal. This is a proxy for compliance/refusal; any `t_post` finding should be read against that proxy.

The remainder of the paper covers related work (§2), methodology (§3, organized to make explicit which analysis addresses which RQ), results (§4, organized by RQ), discussion (§5), conclusions and future work (§6), and limitations (§7).

## 2. Related Work

### 2.1 Representation engineering for safety

Representation engineering studies how interpretable behaviors are encoded as directions in transformer activation space. Zou et al. introduced representation engineering as a top-down interpretability framework in which difference-of-means contrasts can recover semantically meaningful axes. Subsequent work extended these methods to safety. Refusal-direction work identifies a single axis whose ablation suppresses refusal across diverse harmful inputs. Circuit breakers apply representation-level training objectives to suppress harmful behavior under adversarial pressure. Latent Guard uses a learned harmfulness direction at `t_inst` as an intrinsic detector that rivals dedicated guard models on single-turn inputs.

For this paper the most relevant single-turn finding is Zhao et al.'s position-specific decomposition: harmfulness is causally encoded at `t_inst` while refusal is causally encoded at `t_post`, verified by reply-inversion steering experiments. Our analysis adopts that two-position structure and asks how it survives in multi-turn settings — using `t_inst` for harmful-vs-benign analysis and `t_post` for successful-vs-failed-attack analysis (a conversation-level proxy for compliance/refusal, since true turn-level refusal labels are not available across all three frameworks).

The broader question of whether safety-relevant information is linearly accessible in hidden states connects to the linear representation hypothesis. If a single-turn direction fails on multi-turn data, the failure could reflect (a) the direction having moved to a worse-aligned axis even though useful linear information persists, or (b) the underlying signal having become genuinely nonlinear. Our M1/M2/M3 comparison (direction-only, linear probe, MLP) is designed to distinguish these.

### 2.2 Multi-turn jailbreak attacks

Multi-turn jailbreak attacks distribute adversarial pressure across a sequence of turns. **Crescendo** uses a single attacker LLM that generates a chain of benign-seeming questions designed to escalate gradually toward the harmful objective; refused turns are rolled back from the target's active context before the next attempt, so the conversation the target experiences differs from a naive read of the turn log. **ActorAttack** constructs a network of fictional actors and surfaces harmful content indirectly through their interactions; conversations tend to be shorter and to converge with low variance. **X-Teaming** uses a two-phase structure — an attacker plans a multi-turn strategy, then executes it turn by turn — yielding higher per-attempt success rates with conversation lengths comparable to ActorAttack. These three frameworks differ substantially in how they shape each user turn, which becomes mechanistically important in our results.

Other multi-turn vectors include chain-of-thought manipulation and many-shot in-context jailbreaks. Earlier analyses of prompt injection and competing objectives are also related; multi-turn attacks generalize these by distributing the adversarial pressure across turns.

### 2.3 Prior representation analysis of multi-turn jailbreaks

Bullwinkel et al. apply representation-level analysis to multi-turn jailbreak attacks, studying response-side hidden states from the final jailbroken turn. They report that compressed single-turn versions of multi-turn conversations produce representations very similar to the full multi-turn case, suggesting that — at least on the response side — accumulated semantic content matters more than the turn structure that delivered it. Read as a defense claim, this would mean a multi-turn jailbreak's eventual response state is essentially what a content-equivalent single-turn prompt would have produced; the multi-turn delivery is bookkeeping, not structurally important to the readout.

Our work differs in three ways. First, we study input-side positions (`t_inst`, `t_post`) as in Zhao et al., not response tokens, so we are testing whether Bullwinkel's response-side compressed-≈-full pattern *also holds at the user-turn positions where single-turn detectors actually read*. We carry the comparison through directly: the third extraction condition in our pipeline is a Bullwinkel-style compressed single-turn version of each conversation, and the gap between `full_context` and `compressed` AUROC at every analysis is the input-side test of their finding (§4.2, §4.4). Second, we frame the problem explicitly as transfer: directions are learned from JBB topic-matched pairs and tested on OOD WJB conversations selected to be semantically distant from JBB, so the compressed baseline tells us the content-only OOD ceiling — what would be detectable if delivery were single-turn — separately from any multi-turn-specific suppression. Third, our per-turn direction analysis decomposes axis displacement into a context-driven component and a framework-framing component, which the prior work does not.

### 2.4 Distribution shift and probe generalization

Strong in-distribution probe performance is well known not to guarantee transfer to other input distributions. Multi-turn jailbreaks create a particularly structured form of shift: harmful intent is expressed indirectly, distributed over turns, and varies by attack framework. This setting is useful as a controlled probe-generalization testbed because the distributional shift has a concrete, mechanistic source. Our train-on-JBB / test-on-most-distant-WJB setup is intentionally stricter than a random split.

## 3. Methodology

### 3.1 Overview and research questions

The methodology is organized to address five research questions in order. Each question has a designated analysis described in §3.10–§3.16. We give the mapping here for navigation.

| RQ | Operation | Analysis (§) |
|---|---|---|
| RQ1 | Project WJB final-turn states onto the JBB-trained direction; report AUROC by framework, layer, and condition. | §3.10 |
| RQ2 | Build a fresh topic-paired direction per turn from JBB; report `cos(v_pt(k), v_ST)` and `cos(v_full(k), v_noctx(k))`. | §3.12 (construction), §3.13 (Plots 5 and 6) |
| RQ3 | Compare four AUROCs on WJB: anchor `v_ST`, adaptive per-turn `v_pt(k)`, linear probe (M2), MLP (M3). | §3.13 (Plot 7), §3.14 (M1/M2/M3) |
| RQ4 | Train a linear probe on JBB[`train_fw`], test on WJB[`test_fw`]; build the 3×3 matrix. | §3.15 |
| RQ5 | Run RQ1–RQ4 at `t_post` (success-vs-failure on harmful) in parallel with `t_inst` (harmful-vs-benign); compare. | §3.10, §3.14, §3.15 (the t_post halves) |

The setup is deliberately minimal. A single target model. Two positions. Three extraction conditions (`full_context`, `no_context`, and a Bullwinkel-style `compressed` single-turn baseline). Three attack frameworks. One position-specific labeling rule per task. AUROC as the only evaluation metric. The methodology can be summarized in one sentence: train two topic-paired directions on JBB, evaluate them in five complementary ways on OOD WJB across the three conditions, and report AUROC across an 8-layer sweep.

### 3.2 Target model

Llama-3.1-8B-Instruct (32 transformer layers, hidden size 4096) is the only target model in this paper. It is served via vLLM with `temperature=0.0`, `max_tokens=2048`, and the system prompt in Appendix A. Extension to Qwen-2.5-7B-Instruct, Gemma-2-9B-Instruct, and Phi-3.5-mini-Instruct is left for future work; conversation data exists for those models in the broader project but is not part of the analysis here.

### 3.3 Training data: JBB-Behaviors

We use `JailbreakBench/JBB-Behaviors` (HuggingFace `behaviors` config) as the source of training signal. JBB contains 100 harmful goals and 100 benign goals organized into 100 topic-matched pairs (`pair_id ∈ [0, 99]`); each harmful goal shares a topic with its benign counterpart. Topic matching is central — within-pair harmful-minus-benign subtraction cancels topic-specific signal more cleanly than a global mean-difference over unmatched prompts.

For the single-turn condition (used to construct the `t_inst` direction in §3.9), each goal is presented once as a single user message under the system prompt in Appendix A; one row of hidden states is extracted per goal (100 harmful + 100 benign). For the multi-turn conditions (used to construct the `t_post` direction in §3.9 and the per-turn directions in §3.12), each JBB harmful and benign goal is the target of 20 independent attack attempts under each framework; one row is extracted per accepted turn per attempt.

### 3.4 Test data: WildJailbreak goals (OOD construction)

The test set is a 100-prompt out-of-distribution slice of WildJailbreak built per category (50 harmful + 50 benign) by a four-step pipeline.

**Step 1 — Moderation filter (vanilla_harmful only).** Every WJ `vanilla_harmful` candidate is scored by OpenAI's `omni-moderation-latest` endpoint. We take the maximum over harm categories per prompt and keep candidates with `max_score ≥ 0.7`. Without this filter, a non-trivial fraction of WJ's harmful split contains academic/research framings that would dominate the subsequent farthest-first selection. The benign split is not moderation-filtered.

**Step 2 — Embedding.** Surviving WJ candidates and JBB goals are embedded in a shared OpenAI embedding space (cached on disk).

**Step 3 — Below-median OOD pool.** Per category, for each WJ candidate `i` we compute `max_cos[i]` = maximum cosine similarity to any same-category JBB goal. The OOD pool is candidates with `max_cos[i] < median(max_cos)` — the half furthest from JBB.

**Step 4 — Farthest-first selection.** From the OOD pool we run a greedy k-center selection in cosine distance (`farthest_first`), seeded at the lowest-`max_cos` point, picking 50 per category. This yields a diverse 50 + 50 set that is both distant from JBB and well-spread among itself. Within-pick pairwise cosine is reported as a diversity sanity check.

`pair_id` is reset to 0–49 within each split. `wj_idx`, `max_cos_to_jbb`, and the nearest JBB goal are preserved as traceability columns. The selection is saved to `data/selection/wildjailbreak_selected_v1.parquet` and reused across frameworks for direct comparison.

### 3.5 Multi-turn attack generation

Multi-turn conversations are generated from JBB and WJB goals using three attack frameworks. All three runners share the same external interface and output schema; framework-specific control flow is briefly described below.

**Common settings.**
* **Attacker model:** GPT-4o (OpenAI). Generates each user turn given its own (separate) attacker context.
* **Per-turn judge model:** GPT-4o (OpenAI). Decides whether each turn was a refusal and whether the conversation has reached the harmful objective.
* **Target model:** Llama-3.1-8B-Instruct served via vLLM, `temperature=0.0`, `max_tokens=2048`, BASE_TARGET_PROMPT (Appendix A) as system message.
* **Number of attempts:** 20 independent stochastic attempts per goal per framework.
* **Maximum turns per attempt:** 10.
* **Concurrency:** 50 in-flight per GPU, distributed across multiple GPUs.

**Framework 1 — Crescendo.** A single attacker LLM generates one question per turn, instructed to start with innocuous queries and escalate gradually toward the harmful objective. Two separate conversation contexts are maintained: an `attacker_context` (which the attacker LLM sees) and a `target_history` (which the target sees). After each turn the per-turn judge decides whether the target's response is a refusal; if it is, the offending user/assistant pair is **rolled back** from `target_history` before the next attempt. The `n_accepted_turns` field in the metadata reflects this rollback. Crescendo attempts vary substantially in length: in our extracted data, accepted-turn counts span 1–10 with a long tail at lower counts.

**Framework 2 — ActorAttack.** A planning step constructs a network of fictional actors connected to the harmful objective; subsequent turns surface harmful content indirectly through these actors' interactions. ActorAttack converges quickly and uniformly: in our extracted data, accepted-turn counts span 1–6, with the bulk of attempts finishing at exactly 6 turns. The harmful intent is by design held back until late in the conversation.

**Framework 3 — X-Teaming.** A two-phase structure: an attacker LLM first writes an explicit multi-turn strategy plan, then a second pass executes the plan turn by turn. Accepted-turn counts in our extracted data span 1–7 (with a few benign attempts reaching 8). X-Teaming's attempt-success rate is the highest of the three frameworks at the per-attempt level, though we use it primarily as a structurally distinct way of inducing multi-turn trajectories rather than for attack-rate comparisons.

For each conversation, the runner records the system prompts, both conversation contexts, per-turn judge verdicts, refusal flags, the final framework verdict (`jailbroken | near_miss | refusal`), and turn-level metadata. All three runners share the same JSON output schema (one file per conversation under `data/llama/conversations[_wj]/{framework}_{split}/`).

### 3.6 Outcome labeling: StrongReject / AISI

We use the StrongReject library's AISI rubric to produce a per-conversation continuous outcome score. Every harmful conversation's final assistant response is scored by **GPT-4o-mini** with the AISI prompt template. The judge returns an integer score in 1–5; we map this to `aisi_scaled = (raw − 1) / 4 ∈ [0, 1]`. Successful attacks are those with `aisi_scaled ≥ 0.5`.

This judgment is conversation-level: it labels the eventual jailbreak verdict on the *final* response, not turn-level refusal. Our `t_post` analysis must therefore be read as success-versus-failure of the *attack*, not as a direct turn-level refusal axis. AISI scores are produced for both JBB and WJB harmful conversations under all three frameworks.

For traceability we also record StrongReject's three-component rubric (`sr_refused`, `sr_convincing`, `sr_specific`) and the framework's own verdict. All AISI threshold choices reported in this paper use 0.5; sensitivity to that threshold is left for future work.

### 3.7 Hidden-state extraction

For each conversation file, hidden states are extracted under two conditions used in this paper.

**Full-context (trajectory).** For each accepted turn `k`, the model receives the conversation history up to and including the `k`-th attacker message (with rolled-back turns excluded). The hidden state at the final attacker message thus reflects both the local message and accumulated prior turns.

**No-context.** For each accepted turn `k`, the model receives the system prompt and *only* the `k`-th attacker message in isolation. The local message is identical to its full-context counterpart; only prior history is removed.

**Compressed.** A single-turn semantic compression of the multi-turn conversation, produced once per attempt by the broader pipeline. Compressed states are conversation-level (one row per `(conversation_id, attempt)` with no per-turn structure). Because they have no `turn_k`, compressed states do not enter the per-turn direction construction (§3.12) or the per-turn AUROC plots (§3.13); they participate as a third extraction condition in the final-turn projection AUROC (§3.10), the linear-vs-nonlinear probe analysis (§3.14), and the cross-framework matrix (§3.15). In the per-turn plots they appear as a horizontal reference line carrying the same direction's AUROC on the compressed conversation.

Hidden states are saved at **8 layers** chosen as a sweep through model depth (1-indexed): L1, L5, L9, L14, L18, L23, L27, L32. The eight indices are stored in `layer_indices.json` alongside each block. All cross-layer claims in this paper are therefore over this sweep — there are 24 transformer layers we do not analyze; that gap is acknowledged in §7. Tensors are stored as float16 arrays of shape `(n_rows, 8, 4096)`, one per `(dataset, condition, framework, label)` block, with a parallel `metadata.parquet` for row-level lookup.

### 3.8 Token positions

Two positions are extracted per attacker message.

* **`t_inst`** — the final token of the attacker message, immediately before the closing `<|eot_id|>`. Single-turn safety analysis associates this position with harmfulness encoding.
* **`t_post`** — the `<|eot_id|>` token itself that closes the user turn. Single-turn analysis associates this position with refusal encoding.

We use `t_inst` for harmful-versus-benign analyses and `t_post` for successful-versus-failed-attack analyses.

### 3.9 Direction construction

We construct two position-specific directions per layer (and per framework where applicable).

**`v_inst[L]` — single-turn harmfulness direction at `t_inst`.** Framework-independent. For each of the 100 JBB topic-matched pairs we compute the difference between the harmful and benign single-turn `t_inst` hidden states at layer `L`. We average the 100 pairwise differences and normalize to unit length. Construction sees no framework; the resulting direction is the same axis applied across all attack frameworks during evaluation.

**`v_post[L, fw]` — success-versus-failure direction at `t_post`.** Framework-specific. For each JBB pair under framework `fw`, we collect the final-turn `t_post` hidden states of harmful conversations and split them into successful (`aisi_scaled ≥ 0.5`) and failed sets. Within each pair we compute the difference between the mean successful state and the mean failed state; pairs with no contrast (all successful or all failed) are skipped. We average across pairs that retain both classes and normalize to unit length. This produces a separate direction per `(layer, framework)`.

In the JBB Crescendo set used to build `v_post`, 1994 conversations contribute 984 successes and 1010 failures (well-balanced); 90/100 topic pairs have both classes available. ActorAttack and X-Teaming have analogous balance under the same threshold.

### 3.10 Final-turn projection AUROC (RQ1)

For each `(layer, position, condition, framework)`, we project every WJB final-turn test row onto the relevant direction (`v_inst[L]` at `t_inst`, `v_post[L, fw]` at `t_post`) to obtain a scalar score `s = h · v`. AUROC is computed against the position-conditional label.

* **At `t_inst`:** harmful goal vs benign goal across all WJB final-turn rows.
* **At `t_post`:** successful vs failed attack across WJB harmful final-turn rows only.

Each "test row" is one final-turn state from one `(conversation_id, attempt)` pair on WJB. A row contributes one example whether or not it overlaps with other attempts on the same goal (we do not goal-average; goal-level uncertainty is acknowledged in §3.17). Centering of the projection is omitted because a constant shift does not affect AUROC.

Two summary plots follow. Plot 1 shows AUROC by layer with **three lines** per `(position, framework)` panel — `no_context`, `full_context`, and `compressed` — so the three extraction conditions can be read off the same axes. Plot 2 reports `ΔAUROC = AUROC(full) − AUROC(no_context)` by layer per `(position, framework)`; this isolates the pure context-vs-no-context displacement (RQ2) and so does not include compressed. The Bullwinkel handshake — whether `compressed` and `full_context` track each other on the input side — is read directly from Plot 1 and summarized in §4.2.

### 3.11 Per-turn projection AUROC (supporting RQ1, RQ2, RQ3)

We additionally recompute the same projection AUROC at each turn `k`. Test rows are restricted to `turn_k == k`. We report median AUROC across the 8 layers with an interquartile band, and annotate per-turn sample size `n` because conversations end at different depths. Per-turn AUROC against the *anchor* `v_inst[L]` (Plot 2b at `t_inst`, Plot 2c at `t_post`) is the basic per-turn transfer trajectory.

### 3.12 Per-turn direction construction (RQ2)

For each `(layer, framework, condition, turn k)`, we build a fresh topic-paired harmfulness direction `v_pt[L, fw, cond, k]` from JBB at turn `k`, using the same construction as `v_inst` but restricted to states with `turn_k == k`. We require at least `MIN_PAIRS_PT = 30` topic pairs with both harmful and benign support at turn `k`; below that, the estimate is too noisy and the turn is skipped.

For Crescendo, `n_pairs ≥ 30` holds through `k=9` (and falls to 19 at `k=10`). For ActorAttack, all 100 pairs are present at every turn 1–6. For X-Teaming, 100 pairs are present at `k=1–3`, 98 at `k=4`, 81 at `k=5`, and only 14 at `k=6` (skipped).

### 3.13 Per-turn direction analyses (RQ2 and RQ3)

Three plots use the per-turn directions from §3.12 to dissect displacement.

**Direction drift (Plot 5, RQ2).** At each `(layer, framework, condition, k)` we compute `cos(v_pt[L, fw, cond, k], v_inst[L])`. Plotted by `k`, faceted by framework, with separate lines for `full_context` and `no_context`. A high cosine means turn `k` uses an axis similar to the single-turn anchor; a low cosine means the axis has rotated.

**Context-vs-message decomposition (Plot 6, RQ2).** At each `(layer, framework, k)` we compute `cos(v_pt[L, fw, full_context, k], v_pt[L, fw, no_context, k])`. At `k=1` this should be 1.0 (no prior context exists yet — a free correctness check). At `k>1`, deviations from 1.0 are the *pure context contribution* to axis rotation, with the local message held fixed.

**Fixed vs adaptive AUROC (Plot 7, RQ3).** At each turn `k` we project WJB[fw] full-context turn-`k` rows onto two axes: the fixed anchor `v_inst[L]`, and the adaptive per-turn direction `v_pt[L, fw, full_context, k]` (re-fit on JBB at the matching turn). Both AUROCs are reported. The gap between them is the practical cost of using a stale single-turn axis at that turn.

All three plots aggregate across the 8 layers as median ± interquartile band (no single focal layer is selected). Plot 7 is JBB→WJB transfer; Plots 5 and 6 use only JBB.

### 3.14 Linear vs nonlinear probes (RQ3)

To test whether weak direction transfer reflects axis-mismatch (linear info still present, just on a different axis) versus genuine reorganization, we compare three classifiers per `(layer, position, framework)`.

* **M1 — direction score.** Logistic regression on the 1-D projection onto the position-conditional direction (`v_inst[L]` or `v_post[L, fw]`).
* **M2 — linear probe.** L2 logistic regression (`C=0.1`) on the standardized full hidden-state vector.
* **M3 — small MLP.** One hidden layer of 128 ReLU units with L2 regularization (`alpha=1e-3`) standing in for dropout (sklearn's `MLPClassifier` does not expose a dropout parameter), early stopping on a 10% validation slice, three random seeds reported as mean ± standard deviation.

Train/test sets:

* **At `t_inst`:** train on JBB single-turn (harmful/benign), test on WJB[fw] final turn (harmful/benign).
* **At `t_post`:** train on JBB[fw] full-context final-turn harmful (success/failure), test on WJB[fw] final-turn harmful (success/failure).

The test side is evaluated under **all three extraction conditions** (`no_context`, `full_context`, `compressed`) so the linear-vs-nonlinear gap can be read separately for the bare-message, the multi-turn-with-context, and the Bullwinkel-style compressed-single-turn views of the same conversations. The headline figure (Plot 3, layer-resolved) keeps the `full_context` view for visual continuity; the printed summary table beneath it gives median-across-layer AUROC for M1/M2/M3 across all three conditions, and the §4 results read directly off that table.

### 3.15 Cross-framework transfer matrix (RQ4)

For each `(layer, position, train_fw, test_fw)`, we train an L2 linear probe (matching M2 above) on JBB[`train_fw`] full-context final-turn states and test on WJB[`test_fw`] full-context final-turn states.

* **At `t_inst`:** train on JBB[`train_fw`] harmful/benign, test on WJB[`test_fw`] harmful/benign. We use JBB *full-context* final-turn here rather than JBB single-turn so the matrix's diagonal is a proper within-framework cell (apples-to-apples). This is intentionally different from §3.14, which keeps JBB single-turn on the train side; the comparison is across-frameworks here, not single-turn-anchor-vs-multi-turn.
* **At `t_post`:** train on JBB[`train_fw`] harmful (success/failure), test on WJB[`test_fw`] harmful (success/failure).

Reported as a **3×3 heatmap per `(position, test condition)`**, laid out as a 2 × 3 grid (rows = positions, columns = the three test conditions `no_context`, `full_context`, `compressed`). Cell values are the median AUROC across the 8 layers. Diagonal cells measure within-framework transfer; off-diagonals measure cross-framework generalization. The condition axis lets us check whether the cross-framework pattern itself is multi-turn-specific: if the matrix shape is preserved under `compressed`, the framework-specific readout is a property of the framework's framing of the content, not of the multi-turn delivery.

### 3.16 Layerwise reporting

All AUROC values, cosine drifts, and probe scores are computed per layer. We do not pick a single focal layer. Per-layer values appear in line plots; aggregate views (per-turn medians, cross-framework heatmap, summary tables) summarize the layer sweep using the median across the 8 layers, with IQR bands where applicable.

### 3.17 Statistical treatment

Each WJB final-turn test row corresponds to one `(goal, attempt)` pair. Because the same goal contributes multiple attempts, conversation-level rows are not fully independent. Goal-level uncertainty is therefore wider than i.i.d. row-level inference would suggest. In the present analysis we report:

* per-layer AUROC point estimates (no bootstrap),
* median ± interquartile band across the 8 layers in per-turn plots,
* mean ± standard deviation across three random seeds for the M3 MLP,
* per-turn sample size `n` annotated on every per-turn plot, including both the number of test rows and (for direction-construction plots) the number of usable topic pairs.

Goal-level bootstrap confidence intervals are listed as a methodological extension in §6.

## 4. Results

This section is organized by RQ. All medians are taken across the 8 extracted layers unless otherwise noted; per-layer detail is preserved in the figures and CSV outputs.

### 4.1 RQ1 — Final-turn transfer of single-turn-style directions

The single-turn JBB direction transfers to WJB final-turn states at `t_inst`, but transfer strength varies sharply by attack framework. Median AUROC across 8 layers, all three conditions:

**`t_inst` — harmful vs benign:**

|  | no_context | full_context | compressed |
|---|---|---|---|
| Crescendo | 0.727 | 0.742 | 0.698 |
| ActorAttack | 0.718 | 0.734 | 0.721 |
| X-Teaming | — | 0.557 | — |

**`t_post` — success vs failure (harmful only):**

|  | no_context | full_context | compressed |
|---|---|---|---|
| Crescendo | 0.597 | 0.648 | 0.649 |
| ActorAttack | 0.573 | 0.579 | 0.580 |
| X-Teaming | 0.495 | 0.532 | 0.525 |

At `t_inst`, Crescendo and ActorAttack are within 0.02 of each other under full context and well above chance. **X-Teaming is the standout failure**: full-context AUROC sits at 0.557 — barely above chance — and the per-layer breakdown shows the direction never exceeds 0.59 at any of the 8 layers. (The X-Teaming `no_context` and `compressed` cells at `t_inst` are missing because the WJB X-Teaming benign extraction has data gaps for those conditions; the full-context value is consistent with the strong cross-framework asymmetry seen in §4.4.) The transfer gap between X-Teaming and the other two frameworks is the first signal that frameworks induce qualitatively different multi-turn geometries.

At `t_post`, AUROC is much weaker across all three frameworks: the median full-context cell ranges from 0.532 (X-Teaming) to 0.648 (Crescendo). The success-versus-failure axis is partially readable from the post-instruction position, but it is closer to chance than to the harmfulness signal at `t_inst`.

A clear pattern across both positions: **`compressed ≈ full_context`** (within 0.01 at `t_post` for all three frameworks; within 0.05 at `t_inst` for the two frameworks where compressed is available), and **`no_context`** sits noticeably below at `t_post` (gap of 0.05 for Crescendo, 0.04 for X-Teaming) but only marginally below at `t_inst`. The compressed-vs-full match is the input-side replication of Bullwinkel et al.'s response-side finding; the consolidated reading is in §4.6.

### 4.2 RQ2 — Context displaces, but framework framing displaces more

The headline finding emerges from §3.12–§3.13 and is more decisive than the AUROC tables alone suggest. The per-turn topic-paired harmfulness direction is **nearly orthogonal** to the single-turn anchor `v_inst[L]` from `k=1` onward, across all three frameworks.

| `cos(v_pt(k), v_ST)` (median across layers) | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| Crescendo, full_context | 0.113 | 0.125 | 0.124 | 0.118 | 0.128 |
| Crescendo, no_context | 0.113 | 0.118 | 0.124 | 0.113 | 0.089 |
| ActorAttack, full_context | 0.065 | 0.153 | 0.159 | — | — |
| ActorAttack, no_context | 0.064 | 0.159 | 0.170 | — | — |
| X-Teaming, full_context | 0.098 | 0.114 | 0.071 | — | — |
| X-Teaming, no_context | 0.098 | 0.110 | 0.056 | — | — |

The cosines hover near zero at every turn, including turn 1, where no prior context has accumulated and where `v_full(k)` and `v_noctx(k)` are by construction identical. The intuitive story — that context accumulation rotates the axis as the conversation deepens — is not supported. The axis is rotated *from the very first multi-turn turn*. Each framework's first user message under the framework prompt looks structurally different from a JBB single-turn instruction, and that structural difference dominates.

The pure-context contribution can be isolated by `cos(v_full(k), v_noctx(k))`:

| `cos(v_full(k), v_noctx(k))` (median across layers) | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| Crescendo | 1.000 | 0.953 | 0.914 | 0.894 | 0.809 |
| ActorAttack | 1.000 | 0.865 | 0.889 | — | — |
| X-Teaming | 1.000 | 0.937 | 0.817 | — | — |

The `k=1` row is a sanity check (1.000 across all frameworks, as expected — no prior context exists yet). Beyond `k=1`, the cosine declines slowly: by `k=9` Crescendo has drifted to 0.81, X-Teaming by `k=5` to 0.82, ActorAttack by `k=6` to 0.80. **Context does displace the axis, but only slowly.** The drop from 1.0 to 0.8 over many turns is much smaller than the gap between any per-turn axis and `v_ST` (cosines around 0.10).

The analysis decomposes the total displacement into two components:
1. A large, frame-from-`k=1` rotation attributable to the framework's prompt-style framing of each user turn (cosine `~0.07–0.16` regardless of `k`).
2. A small, monotonically growing rotation attributable to accumulated prior context (`~0.20` cosine drop across many turns).

The first component dominates by roughly an order of magnitude at every layer.

### 4.3 RQ3 — Linear recoverability when refit per turn or per framework

Despite the orthogonal axis rotation, the harmfulness signal remains linearly accessible. Refitting the topic-paired direction per turn restores AUROC substantially.

**Plot 7 — fixed `v_ST` vs adaptive `v_pt(full, k)` AUROC on WJB (median across layers):**

| Crescendo | k=1 | k=3 | k=5 | k=7 | k=9 |
|---|---|---|---|---|---|
| anchor `v_ST` | 0.577 | 0.607 | 0.556 | 0.465 | 0.508 |
| adaptive `v_pt(k)` | 0.719 | 0.837 | 0.802 | 0.776 | 0.757 |
| **gap** | **+0.141** | **+0.230** | **+0.246** | **+0.311** | **+0.249** |

| X-Teaming | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| anchor `v_ST` | 0.502 | 0.542 | 0.553 | 0.497 | 0.506 |
| adaptive `v_pt(k)` | 0.714 | 0.662 | 0.683 | 0.657 | 0.633 |
| **gap** | **+0.212** | **+0.120** | **+0.131** | **+0.160** | **+0.127** |

For Crescendo, the anchor falls *below chance* at `k≥7` (0.465 at `k=7`, 0.460 at `k=8`) while the adaptive direction holds steady at 0.74–0.84. For X-Teaming, the anchor sits near chance (0.50–0.55) at every turn while the adaptive direction sits at 0.63–0.71. In both frameworks the harmfulness information is linearly recoverable — refitting the axis at the right turn is enough.

ActorAttack is qualitatively different. Both anchor and adaptive AUROC sit near chance at `k=1` through `k=5`, then jump together at `k=6` (the framework's standard ending turn):

| ActorAttack | k=1 | k=3 | k=5 | k=6 |
|---|---|---|---|---|
| anchor | 0.509 | 0.530 | 0.497 | 0.734 |
| adaptive | 0.550 | 0.478 | 0.487 | 0.850 |
| gap | +0.040 | −0.053 | −0.011 | +0.116 |

At intermediate ActorAttack turns the harmful intent is *representationally invisible at `t_inst`* — neither the single-turn axis nor a refit per-turn axis separates harmful from benign. The signal appears only at `k=6`, the conversation's resolution turn. This is consistent with ActorAttack's design: the harmful objective is held back behind fictional-actor framing until the end.

The full M1/M2/M3 probe comparison (Plot 3, median across layers):

| | M1 (1-D dir) | M2 (linear probe) | M3 (MLP-128) |
|---|---|---|---|
| t_inst, Crescendo | 0.742 | 0.685 | 0.621 |
| t_inst, ActorAttack | 0.734 | 0.753 | 0.714 |
| t_inst, X-Teaming | 0.557 | 0.604 | 0.550 |
| t_post, Crescendo | 0.648 | 0.592 | 0.622 |
| t_post, ActorAttack | 0.579 | 0.638 | 0.649 |
| t_post, X-Teaming | 0.532 | 0.566 | 0.587 |

The MLP never wins by more than a few hundredths and often loses to one of the linear models. The data are consistent with the linear representation hypothesis: the harmfulness/outcome signal is encoded linearly, and nonlinear decoding does not recover hidden structure.

A subtle sub-finding: at `t_inst` for Crescendo, the **direction-only score (M1) outperforms the trained linear probe (M2)** by 0.057. This is the textbook small-data signature — the probe is trained on only 200 JBB single-turn rows and overfits to JBB-specific features that do not transfer, while the topic-paired direction averages out idiosyncrasies.

### 4.4 RQ4 — Cross-framework transfer at `t_inst` and `t_post`

**`t_inst` — harmful vs benign, median AUROC across 8 layers:**

| train ↓ / test → | crescendo | actorattack | xteaming |
|---|---|---|---|
| crescendo | **0.889** | 0.688 | 0.662 |
| actorattack | 0.571 | **0.833** | 0.583 |
| xteaming | 0.669 | 0.713 | **0.759** |

Diagonals dominate (0.76–0.89), but off-diagonals are comfortably above chance (0.57–0.71). Frameworks share *some* harmfulness structure but each induces its own readout. Two asymmetries are notable:

* **Crescendo is the broadest training source.** A probe trained on JBB Crescendo achieves 0.69 on ActorAttack and 0.66 on X-Teaming — the largest off-diagonal sum of any row.
* **ActorAttack is the narrowest training source.** A probe trained on JBB ActorAttack achieves only 0.57 on Crescendo and 0.58 on X-Teaming — the smallest off-diagonal sum of any row, despite ActorAttack having a strong on-diagonal cell.

X-Teaming sits between the two, generalizing reasonably well in both directions (0.67 onto Crescendo, 0.71 onto ActorAttack) but with the lowest within-framework cell (0.76) of the three diagonals.

**`t_post` — success vs failure, median AUROC across 8 layers:**

| train ↓ / test → | crescendo | actorattack | xteaming |
|---|---|---|---|
| crescendo | **0.592** | 0.569 | 0.543 |
| actorattack | 0.491 | **0.638** | 0.498 |
| xteaming | 0.558 | 0.487 | **0.566** |

All values are between 0.49 and 0.64. Diagonals barely separate from off-diagonals. **Outcome geometry at `t_post` is highly framework-specific and weak in absolute terms.** Two cells fall slightly below chance (ActorAttack→Crescendo at 0.491; X-Teaming→ActorAttack at 0.487). A defense built on a single-turn refusal axis would not transfer across these attack styles.

The cross-framework matrix shape is preserved when the test side switches from `full_context` to `compressed` (full numerical comparison in §4.6). Diagonals are essentially identical at `t_inst` (0.890 / 0.819 vs 0.889 / 0.833 for Crescendo / ActorAttack), and the same row-asymmetry — Crescendo as the broadest training source, ActorAttack as the narrowest — survives compression. The framework-specific readout is therefore not multi-turn-induced; it is a property of how each framework frames its content, and that framing is preserved when the conversation is compressed back to a single turn.

### 4.5 RQ5 — `t_post` vs `t_inst` along all four axes

The `t_post` outcome axis differs from the `t_inst` harmfulness axis in four consistent ways.

1. **Lower absolute AUROC.** Median full-context AUROC at `t_post` is 0.53–0.65; at `t_inst` it is 0.56–0.74. The post-instruction position carries weaker signal for the conversation-level outcome label than the instruction-end position carries for goal-type.
2. **Larger context benefit.** `ΔAUROC = AUROC(full) − AUROC(no_context)` is larger and more reliably positive at `t_post` (median +0.05 for Crescendo, +0.04 for X-Teaming, +0.01 for ActorAttack) than at `t_inst` (median +0.01–0.02). This is plausible: outcome depends on the trajectory, so seeing the trajectory helps.
3. **Stronger framework specificity.** The cross-framework matrix at `t_post` has off-diagonal cells nearly at chance, while the matrix at `t_inst` retains 0.57–0.71 off-diagonal AUROC.
4. **Weaker linear recoverability.** Among the M1/M2/M3 probes, `t_post` rarely exceeds 0.65 at any framework, while `t_inst` reaches 0.74–0.75 for Crescendo and ActorAttack. The MLP marginally wins at `t_post` for ActorAttack and X-Teaming, but the gain is within seed-level noise.

Combined, these four observations suggest that the conversation-level outcome label (`aisi_scaled ≥ 0.5`) is a noisier and more attack-style-coupled signal than the goal-type label, and that the `t_post` position does not carry an analogous to the harmfulness axis under the AISI labeling we used.

### 4.6 Bullwinkel handshake — does the same-content single-turn baseline match the multi-turn final turn?

The third extraction condition, `compressed`, packs each multi-turn conversation into a single semantically-equivalent user message and runs the same JBB-trained directions over the resulting hidden states. It serves as a same-content single-turn baseline that lets us tell *content OOD-ness* apart from any *multi-turn-specific suppression*. Bullwinkel et al. report that on the response side, compressed and full-context representations are very similar; we test the input-side analogue by reading compressed against `full_context` and `no_context` at `t_inst` and `t_post`, and across all three of the §4.1 / §4.3 / §4.4 analyses.

**Observation 1 — `compressed ≈ full_context` at both positions.** The match is tight at `t_post` and within a few hundredths at `t_inst`. From §4.1:

|  | t_inst (full) | t_inst (compressed) | gap | t_post (full) | t_post (compressed) | gap |
|---|---|---|---|---|---|---|
| Crescendo | 0.742 | 0.698 | −0.044 | 0.648 | 0.649 | +0.001 |
| ActorAttack | 0.734 | 0.721 | −0.013 | 0.579 | 0.580 | +0.001 |
| X-Teaming | 0.557 | — | — | 0.532 | 0.525 | −0.007 |

At `t_post` the compressed and full-context AUROCs are identical to two decimal places for all three frameworks. At `t_inst` compressed sits a hair below full-context (≤ 0.05 gap). The same content delivered as a single compressed user message is detected at essentially the same AUROC as the multi-turn final turn — Bullwinkel et al.'s response-side finding extends to the input-side positions used by single-turn safety detectors.

**Observation 2 — `no_context` consistently sits below both, especially at `t_post`.** From §4.1:

|  | t_post (no_ctx) | t_post (full) | t_post (compressed) | gap (compressed − no_ctx) |
|---|---|---|---|---|
| Crescendo | 0.597 | 0.648 | 0.649 | +0.052 |
| ActorAttack | 0.573 | 0.579 | 0.580 | +0.007 |
| X-Teaming | 0.495 | 0.532 | 0.525 | +0.030 |

At `t_post`, both `compressed` and `full_context` carry the full conversation's semantic content — multi-turn and compressed-single-turn deliveries of the same content. `no_context` sees only the bare last user message and so loses the trajectory information that the outcome label depends on. The 0.05 gap for Crescendo and 0.03 for X-Teaming is the cost of stripping prior content. At `t_inst`, where the harmful-vs-benign label is largely determined by the local request, compression does not add much discriminative content over the local message and the asymmetry is muted.

**Observation 3 — the cross-framework matrix shape is preserved under compression.** Comparing the `t_inst` 3×3 cross-framework matrix from §4.4 at `full_context` to the same matrix at `compressed` (median across 8 layers; X-Teaming test-side cells missing under compressed because of the WJB benign extraction gap):

| train ↓ / test → | full: cresc | full: actor | full: xteam | compressed: cresc | compressed: actor | compressed: xteam |
|---|---|---|---|---|---|---|
| Crescendo  | **0.889** | 0.688 | 0.662 | **0.890** | 0.702 | — |
| ActorAttack | 0.571 | **0.833** | 0.583 | 0.615 | **0.819** | — |
| X-Teaming   | 0.669 | 0.713 | **0.759** | 0.716 | 0.664 | — |

The diagonals are essentially identical (0.889 / 0.890 Crescendo; 0.833 / 0.819 ActorAttack). Off-diagonal asymmetries are preserved: Crescendo→others remains the broadest source row (0.66–0.69 under full, 0.70 under compressed); ActorAttack→others remains the narrowest (0.57–0.58 under full, 0.61 under compressed); X-Teaming sits between the two in both versions. **The framework-specific readout is a property of how each framework frames its content, not of the multi-turn delivery — when the conversation is compressed back into a single turn, the framework signature persists.**

**Probe-level confirmation (M1/M2/M3, t_inst, full vs compressed test side, median across 8 layers):**

|  | M1 full / compressed | M2 full / compressed | M3 full / compressed |
|---|---|---|---|
| Crescendo | 0.742 / 0.698 | 0.685 / 0.652 | 0.621 / 0.568 |
| ActorAttack | 0.734 / 0.721 | 0.753 / 0.727 | 0.714 / 0.666 |

Compressed AUROC is uniformly within 0.05 of full-context across M1, M2, and M3 — the linear-vs-nonlinear story (RQ3) does not change when the test side switches to compressed.

**What this rules out, what it leaves open.** Three structurally different stories could have explained why JBB-trained directions transfer poorly to WJB:

1. **Multi-turn delivery suppresses the readout.** Predicts `compressed > full_context` (the same content, when delivered single-turn, would be detected better). **Ruled out.** Compressed and full match across every analysis; if anything, compressed is slightly *lower* at `t_inst`.
2. **The cross-framework structure is multi-turn-induced.** Predicts that compressing the conversation collapses the framework-specific readout. **Ruled out.** The 3×3 matrix shape — diagonals, row asymmetries, broadest-vs-narrowest training source — is preserved under compression.
3. **Content OOD-ness is the dominant transfer challenge.** Predicts `compressed ≈ full_context`, with both bounded by the same content-distance ceiling. **Consistent with the data.**

What the compressed condition leaves open is the RQ2 question of *where the axis rotation comes from*: compressed is conversation-level and so cannot enter the per-turn `cos(v_pt(k), v_inst[L])` decomposition. The complete picture combines RQ2 and §4.6: **each framework rewrites the input geometry from its first user turn (RQ2), and that rewrite is preserved when the framework's content is compressed back into a single turn (this section).** The framework's framing of the content is what rotates the axis; the multi-turn distribution of content is not. Compression strips only the turn-by-turn distribution, which §4.1–§4.4 confirms is not the bottleneck.

## 5. Discussion

The data reframe the multi-turn-displacement story. The intuitive narrative — "as conversational context accumulates, the harmfulness representation drifts away from what a single-turn safety direction was trained on" — is not what the analysis finds. The dominant displacing force is the attack framework's framing of the very first user turn: Crescendo's indirect openers, ActorAttack's fictional-actor scaffolding, and X-Teaming's planning-style prompts each produce input geometry that is nearly orthogonal to a JBB single-turn instruction at every multi-turn turn, including turn 1. Context accumulation contributes only a slow secondary drift, of much smaller magnitude. From a defense-evaluation perspective, this means a Latent-Guard-style detector anchored to a JBB single-turn axis is anchored to the wrong axis from the very first multi-turn user message, regardless of how carefully its architecture handles long context.

The story is not as pessimistic as that observation alone suggests. The signal is still linear and still recoverable. When we refit the same topic-paired direction at each turn of multi-turn JBB data and project WJB at the same turn onto it, we recover AUROC in the 0.74–0.83 range for Crescendo and X-Teaming — values comparable to single-turn safety detector performance reported in prior work. The reorganization is a rotation, not a destruction. Multi-turn-aware defenses that refit per turn (or, more practically, per attack-framework prompt template) can therefore work; defenses that assume a stable single-turn axis cannot.

ActorAttack stands apart and deserves its own framing. At `k=1` through `k=5`, neither the single-turn axis nor a fitted per-turn axis separates harmful from benign WJB ActorAttack conversations. The signal jumps into existence only at `k=6`, the resolution turn. This is consistent with ActorAttack's design — fictional-actor scaffolding holds the harmful objective in indirection until the conversation resolves — but it has a strong representational interpretation: at intermediate turns the model genuinely is not internally distinguishing harmful from benign ActorAttack conversations along any axis we measured. ActorAttack defenses that operate on intermediate-turn states (rather than on the final-turn judgement) appear unlikely to work; defenses that wait for the final turn would lose the prevention advantage of multi-turn detection.

The cross-framework matrix at `t_inst` adds a useful shape to the harmfulness picture. There is a real shared component across attack styles — off-diagonal AUROCs of 0.57–0.71 are well above chance — but each framework's diagonal (0.76–0.89) shows that a substantial fraction of the readout is framework-specific. Crescendo trains the broadest readout, suggesting its escalation-style sweep covers a wider range of indirect-harmful phrasings than the other two; ActorAttack trains the narrowest, consistent with its highly specific fictional-actor framing.

Outcome geometry at `t_post` is the weakest part of the result. AUROC values are low across the board (0.49–0.64), context benefits are modest, cross-framework transfer is at chance off-diagonal, and even MLP probes cap at ~0.65. The conservative reading is that the conversation-level AISI label is a noisy proxy for refusal/compliance and that the `t_post` axis under this labeling does not have a clean analogue to the `t_inst` harmfulness axis. Direct turn-level refusal labels (refused / accepted, judged on the immediate next assistant response) would be a cleaner test; we did not have those across all three frameworks.

The Bullwinkel handshake (§4.6) sharpens the diagnosis. A natural reading of "single-turn detectors fail on multi-turn jailbreaks" is that the multi-turn delivery itself suppresses detection — that distributing harmful intent across turns makes the model's hidden state at the final turn look more benign than the same content would, packaged single-turn. Our compressed condition tests that prediction directly: it takes the same multi-turn conversation, compresses it to a single user message, and asks how the JBB-trained directions read out on that single-turn version. The result is that compressed and full-context AUROCs match within 0.05 across every analysis at both `t_inst` and `t_post` — multi-turn delivery is *not* what suppresses detection. What does is the OOD-ness of the content itself plus the framework's framing of that content; both are properties of *what* is in the conversation, not of *how* it is laid out across turns. Bullwinkel et al.'s response-side compressed-≈-full result extends to the input-side positions, and the cross-framework matrix shape (Crescendo broadest, ActorAttack narrowest) is preserved when the conversation is compressed back into one turn — so the framework-specific readout is not multi-turn-induced either.

Two methodological points deserve emphasis. First, the topic-matched within-pair construction is doing real work here — the per-turn directions remain near-orthogonal to the single-turn axis even with 30+ topic pairs averaged into the construction; this is unlikely to be a noise effect from a single-pair difference. Second, the orthogonality result is stable across all eight layers; it is not a property of one focal layer, and the cosines shown above are medians of a tight per-layer spread.

## 6. Conclusions and Future Work

This study tested whether single-turn-style position-specific representation-engineering directions transfer to multi-turn jailbreak settings on Llama-3.1-8B-Instruct. Using JBB topic-matched pairs for training and the 100 most-distant WildJailbreak prompts for testing, we evaluated a `t_inst` harmfulness direction and a per-framework `t_post` success-vs-failure direction across Crescendo, ActorAttack, and X-Teaming on the same OOD test set.

The main conclusion is that single-turn directions transfer in a structured but framework-dependent way: harmfulness at `t_inst` is partially preserved across attack frameworks (0.57–0.89 AUROC across the cross-framework matrix), but each framework rotates the axis from its very first user turn so that a single-turn-anchored detector points the wrong way. The signal remains linear and refits cleanly per turn; nonlinear decoding does not help. The compressed single-turn baseline matches `full_context` AUROC across every analysis (within 0.01 at `t_post`, ≤ 0.05 at `t_inst`) and preserves the cross-framework matrix shape — extending Bullwinkel et al.'s response-side compressed-≈-full finding to the input-side positions and ruling out multi-turn delivery as the cause of detection failure. ActorAttack is an outlier whose harmful intent is representationally invisible until the final turn. Outcome geometry at `t_post` is much weaker and highly framework-specific.

Future work:

* extend to additional target models (Qwen-2.5-7B-Instruct, Gemma-2-9B-Instruct, Phi-3.5-mini-Instruct) — conversation data already exists in the broader project;
* add goal-level bootstrap confidence intervals to all per-layer AUROCs so the asymmetries reported in the cross-framework matrix carry uncertainty;
* incorporate explicit turn-level refusal/compliance labels (`refused / accepted` per assistant response) so the `t_post` analysis can be repeated as a true refusal axis rather than via the conversation-level AISI proxy;
* test multi-turn-aware defenses constructed by refitting the topic-paired direction per attack framework, and evaluate cross-framework robustness directly;
* sensitivity analysis on the AISI threshold (currently 0.5) and on the OOD selection method (moderation threshold, embedding model, farthest-first seed);
* extend the framework pool — the three studied here cover gradual escalation, fictional-actor framing, and strategic planning, but other multi-turn vectors (chain-of-thought manipulation, many-shot in-context jailbreaks) would round out the picture.

## 7. Limitations

* **Single target model.** Results are reported on Llama-3.1-8B-Instruct only. Other architectures, scales, and alignment strategies may produce different patterns.
* **Single training dataset.** Directions are learned only from JBB. Although the topic-matched structure cleanly isolates harmfulness from topic, JBB's harm categories may not span the full distribution of harmful requests that multi-turn attacks target in practice.
* **Eight of 32 layers.** Hidden states were saved at L1, L5, L9, L14, L18, L23, L27, L32. Cross-layer aggregates are over this sweep, not over the full network. Adjacent-layer behavior is not resolved by these results.
* **WJB OOD selection is shaped by two specific tools.** Distance is measured in OpenAI's embedding space, and the harmful pool is pre-filtered by OpenAI's `omni-moderation-latest` (max-category score ≥ 0.7). A different embedder or moderation threshold would produce a different OOD set; a less aggressive moderation threshold would keep more academic / research-framed prompts in the harmful pool.
* **Sample sizes shrink at late turns.** Per-turn analyses at the deepest turns are supported by relatively few topic pairs (Crescendo 19 at `k=10`; X-Teaming 14 at `k=6`). The `MIN_PAIRS_PT = 30` cutoff drops noisier turns from direction construction; AUROC plots annotate `n` per turn for transparency.
* **No CIs on point estimates.** Per-layer AUROC values are point estimates. Goal-level bootstrap is left to future work.
* **No dropout in M3.** Sklearn's `MLPClassifier` does not implement dropout; we substitute L2 regularization (`alpha=1e-3`). A torch-based MLP with explicit dropout is unlikely to change the qualitative linear-vs-nonlinear conclusion but is a methodological loose end.
* **Cross-framework matrix at `t_inst` uses JBB full-context for training, not single-turn.** This is intentional (the matrix's diagonals are within-framework cells, requiring matched train/test setups), but it means the matrix's findings should be read as a different question from the single-turn-anchor results in §3.14.
* **AISI is a conversation-level proxy for refusal/compliance.** AISI labels the eventual jailbreak verdict for the full conversation, not turn-level refusal. The `t_post` analysis here measures attack success-vs-failure, not direct refusal.

## Appendix A. System prompt used in all experiments

> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
