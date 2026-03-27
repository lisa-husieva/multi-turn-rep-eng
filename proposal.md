# Research Proposal: Representational Dynamics of Multi-Turn Jailbreaks Across Attack Frameworks and Models

## Motivation

This study addresses the topic confound present in prior probing work — specifically, whether harmful/benign separation in hidden states reflects genuine attack dynamics or simply topic-level differences between harmful and benign prompts. It implements a more natural trajectory operationalization, and expands the analysis to three attack frameworks and multiple open-source models to ask whether representational dynamics are attack-specific or universal properties of successful multi-turn jailbreaks.

---

## Research Questions

1. Does harmful/benign separation at k=1 collapse when topic is held constant within JBB matched pairs, confirming topic as the dominant signal in prior probing work?
2. Does extracting hidden states from each turn's own response r_t at full context reveal genuine turn-by-turn representational drift that the r_n-at-varying-k design obscures?
3. Do different attack frameworks — Crescendo, ActorAttack, X-Teaming — produce different representational trajectories, or do all successful jailbreaks converge on the same region of representation space regardless of how the model was steered there?
4. Does a probe trained on one attack framework's conversations transfer to another framework's conversations, and what does that tell us about the generality of the harmful representation direction?
5. Is the harmful/benign signal localized to specific layers, and does this vary by attack framework or model?
6. Do representational dynamics generalize across model families, or are they architecture- and scale-specific?

---

## Dataset Generation

**Behaviors:** JBB-Behaviors — 100 harmful objectives with matched benign counterparts constructed to differ only in harmful intent. Topic is held constant within each pair by design.

**Attack frameworks:** Crescendo, ActorAttack, and X-Teaming, all run via AJAR's shared interface. Using AJAR for all three ensures consistent planning, prompt generation, evaluation, and verdict labeling across frameworks — the only thing varying between conditions is attack strategy.

**Models:** Multiple open-source models evaluated across families and scales (e.g., Llama, Mistral, Qwen). Each model is attacked independently to enable cross-model comparison of representational dynamics.

**Harmful conversations:** 10 attempts per objective per framework per model, max 10 turns. All verdicts saved — Jailbroken, Refusal, Near-Miss — filter at analysis time.

**Benign conversations:** One set of matched benign conversations per objective per model, generated once using Crescendo-style escalating multi-turn structure targeting the JBB benign counterpart objective. Same turn depth distribution as harmful conversations, same model. Benign conversations do not need to be triplicated across frameworks since the benign side does not vary by attack strategy.

**Verdict labeling:** AJAR's shared judge applied uniformly across all three frameworks, ensuring consistent labeling criteria.

---

## Hidden State Extraction

Two extraction designs implemented in the same forward pass, so conversations are only processed once:

**Design A — r_n at varying k:** For each conversation, extract hidden states at final response tokens r_n with k=1 to k=k_max turns of context. Mean-pool over response tokens at each k. Replicates the Bullwinkel et al. design, enabling direct comparison.

**Design B — r_t at full context:** For each conversation, extract hidden states from each turn's own response r_t at full context up to that turn. r_1 with context p_1; r_2 with context p_1, r_1, p_2; and so on. Mean-pool over response tokens at each turn. This is the more natural operationalization of trajectory — tracking how the model represents its own responses as the conversation evolves rather than how it represents the same final response under varying context.

**Layers:** Mean-pooled vectors extracted at all layers for both designs. Per-token hidden states stored at a subset of layers only for storage efficiency.

**Output:** One `representations.jsonl` record per (conversation, model, extraction design, k or t, layer), with fields for attack framework, verdict, harm category, objective pair ID, and paths to stored numpy arrays.

---

## Analysis

### Part 1: Topic Confound Test

Directly addresses the central limitation of prior work. Train the MLP probe within JBB matched pairs — for each objective pair, train on jailbroken vs. benign representations from that pair only, evaluate separability at k=1. Compare within-pair AUC to across-pair AUC. If within-pair AUC collapses relative to across-pair, topic is confirmed as the dominant signal. If it remains high, something beyond topic is driving separation.

Repeat for LAT reading vector: compute difference vectors within each matched pair, fit PCA, project k=1 representations onto the resulting reading vector. Does the reading vector still separate jailbroken from benign at k=1 when computed within pairs?

Run this analysis separately per attack framework and per model to establish clean within-framework, within-model baselines before cross-framework and cross-model comparison.

### Part 2: r_t Trajectory Analysis

Using Design B extractions, plot probe score and LAT projection for each turn's own response r_t across turns 1 to n, separately for jailbroken and benign conversations, per attack framework, and per model. This is the most important methodological innovation relative to prior work.

Key questions: Does the jailbroken trajectory drift differently from the benign trajectory under this design? At which turn does divergence between jailbroken and benign begin? Does drift rate differ across attack frameworks — does X-Teaming or ActorAttack show faster or more abrupt representational shift than Crescendo's gradual escalation? Does higher ASR correlate with faster drift or with a different trajectory shape entirely? Are these patterns consistent across model families?

### Part 3: Cross-Framework Probe Transfer

The central novel contribution. Train the MLP probe on jailbroken vs. benign conversations from one attack framework and evaluate it on conversations from the other two. Six transfer directions total: Crescendo→ActorAttack, Crescendo→X-Teaming, ActorAttack→Crescendo, ActorAttack→X-Teaming, X-Teaming→Crescendo, X-Teaming→ActorAttack.

If probes transfer well across frameworks, all three attacks push representations into the same harmful region and the probe has learned something general about harmful generation. If transfer is poor, different attacks produce representationally distinct jailbreaks despite similar surface-level harmful text — which would mean single-turn defenses trained on one attack type would fail not just because of the single-turn vs. multi-turn generalization gap, but because of cross-framework representational divergence.

Repeat with LAT reading vector: compute the reading vector from one framework's matched pairs and project the other frameworks' conversations onto it. Do successful jailbreaks from all three frameworks cluster in the same region of each reading vector space?

### Part 4: Convergence Analysis

For JBB objectives where all three frameworks produced at least one successful jailbreak, extract the full-context (k=k_max) representations of those jailbroken final responses and compare them directly. Do they converge on the same point in representation space — suggesting a universal jailbroken region that all paths lead to — or do they cluster separately by attack framework?

Visualize using 2D PCA fitted on all jailbroken representations regardless of framework, with points colored by framework. If clusters are framework-specific rather than objective-specific, attack strategy shapes the final representation. If clusters are objective-specific regardless of framework, the destination in representation space is determined by content, not path.

Run this analysis per model. If convergence structure is consistent across models, the finding generalizes; if it varies, model architecture or scale moderates how attack strategy shapes internal representations.

### Part 5: Layer-by-Layer Analysis

For each layer, train a probe on full-context representations and compute AUC, separately per attack framework and per model, and separately for across-pair and within-pair settings. Plot AUC as a function of layer for each condition.

This reveals where in the network the harmful/benign signal is encoded and whether Crescendo's effect is localized to different layers than ActorAttack or X-Teaming. If all three frameworks show the same layer profile, the network processes multi-turn harmful generation consistently regardless of how it was elicited. If layer profiles diverge by framework or by model, different attack strategies (or different architectures) engage different parts of the network — with direct implications for where defenses should intervene.

### Part 6: MLP vs. LAT Consistency Under Topic Control

Repeat the drift correlation analysis from prior work — Pearson correlation between MLP drift and LAT drift per conversation — but now within matched pairs and under both extraction designs, and across all models.

Does controlling for topic increase the MLP–LAT correlation? Does Design B produce higher correlation than Design A? Does the relationship between the two probing methods vary by model family? If topic control increases the correlation, the two methods were previously detecting different topic-correlated signals. If correlation remains low under topic control, MLP probe and LAT reading vector genuinely capture different dimensions of the representation space and both are needed for a complete picture.
