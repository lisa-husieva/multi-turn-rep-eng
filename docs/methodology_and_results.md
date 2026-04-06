# Methodology, Justification, and Preliminary Results

## 1. Problem and Motivation

Large language models (LLMs) are increasingly deployed in contexts where adversarial users attempt to elicit harmful outputs. Single-turn jailbreak attacks — where a harmful request is made directly — are now well-studied, and modern aligned models are reasonably robust to them. The more pressing threat is **multi-turn jailbreaks**: structured, multi-step attack strategies that gradually escalate a conversation toward a harmful objective, exploiting the model's tendency to maintain conversational coherence and comply with individually innocuous-seeming requests. Frameworks such as Crescendo (Shah et al., 2023), ActorAttack, and xTeaming operationalise this threat at scale.

Detection of multi-turn jailbreaks is difficult because the harmful intent is distributed across turns and no single message is obviously harmful. Existing defences rely on output classifiers or per-turn judges, which are reactive and susceptible to the same escalation dynamics that make these attacks effective. An alternative — and potentially more fundamental — approach is to examine the model's **internal representations** during the attack: does the model *know* it is being led toward a harmful output, even when it continues to comply?

---

## 2. Background: Representation Engineering

**Representation engineering (RepE)** (Zou et al., 2023) studies how behavioural concepts are encoded in a model's hidden states. The core idea is that directions in activation space correspond to interpretable properties — harmfulness, honesty, refusal — and that these directions can be extracted from contrast pairs and used for detection or steering.

Two papers are the direct predecessors of this work:

**Bullwinkel et al.** apply RepE to multi-turn jailbreaks in the Crescendo framework on Llama-3-8B. They extract hidden states at each user turn and train MLP probes to detect the harmfulness signal. Their key finding is that a harmfulness direction is detectable and grows over turns — but a confound is that their probe may be learning topic (what the conversation is about) rather than attack dynamics, since harmful and benign conversations differ in subject matter. They study only Crescendo and do not control for topic.

**Zhao et al. (NeurIPS 2025)** study how single-turn jailbreaks are encoded in Llama-3 and other models. Their main finding is a clean dissociation: **harmfulness** is encoded at the last user-content token position (t_inst — the instruction token), while **refusal** is encoded at the end-of-turn token (t_post_inst). Jailbreaks work by suppressing the refusal direction, not the harmfulness direction — the model "knows" the request is harmful but is primed not to refuse. They use within-pair subtraction across JBB-Behaviors contrast pairs to cancel topic, then take the first principal component of the difference vectors as the harmfulness direction (v_harmful).

---

## 3. How This Work Fits In

This project extends both papers along several dimensions:

- **Multi-framework**: Bullwinkel studied only Crescendo. We run the same RepE analysis across three structurally different attack frameworks (Crescendo, ActorAttack, xTeaming), enabling direct comparison of how different attack strategies prime the model's representations.
- **Topic control**: Following Zhao et al., we use within-pair JBB-Behaviors subtraction to cancel topic. Bullwinkel does not do this.
- **Turn-depth analysis**: We study how representations evolve across turns (Mode B), not just at the final turn. This lets us test whether the harmfulness signal builds, erodes, or transforms as the attack progresses.
- **Multiple centroid designs**: We introduce three approaches to centroid construction that reflect different hypotheses about where in the conversation the signal is strongest (see §4.3).
- **Cross-framework transfer**: We test whether a probe trained on one framework's representations detects another framework's attacks, probing the universality of the harmfulness direction.

---

## 4. Methodology

### 4.1 Data

**Model:** Llama-3.1-8B-Instruct.

**Goals:** The JBB-Behaviors dataset (Chao et al., 2023) provides 100 harmful and 100 benign behavioural goals, paired by topic (pair_id 0–99). Each harmful goal has a semantically related benign counterpart designed to share surface-level topic while differing in intent (e.g., "synthesise compound X" vs. "describe the chemistry of compound X").

**Attack frameworks:** Three multi-turn attack frameworks are applied to each goal:
- **Crescendo**: gradual escalation over 10 turns
- **ActorAttack**: role-playing-based multi-turn attack
- **xTeaming**: cross-framework ensemble attack strategy

Each framework runs up to 10 attempts per goal (harmful and benign), yielding approximately 1,000 conversations per framework per goal type. A conversation-level verdict (jailbroken / near\_miss / refusal) is assigned by the framework's built-in judge. Near\_miss is treated as non-jailbreak throughout (equivalent to refusal) — a deliberate methodological choice, since near\_miss responses are partial compliances that do not constitute successful attacks.

**Train/test split:** Attempt 10 is held out as the test set; attempts 1–9 are used for centroid construction and (future) probe training.

**Planned expansion:** The current sample of 10 attempts per goal will be expanded to 30. This is expected to substantially stabilise the per-pair mean representations and increase the coherence of the learned harmfulness direction (see §4.2 and §6).

### 4.2 Token Positions and Hidden State Extraction

Following Zhao et al., hidden states are extracted at two positions per turn:

- **t_inst**: the last user-content token immediately before the end-of-turn token (`<|eot_id|>`). This position encodes the model's representation of the instruction — harmfulness is expected here.
- **t_post_inst**: the end-of-turn token itself. This position encodes the model's incipient response decision — refusal tendency is expected here.

For Llama-3.1-8B-Instruct, the template is:
```
<|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```
so t_inst = position of `<|eot_id|>` − 1, and t_post_inst = position of `<|eot_id|>`.

**Rollback filtering (Crescendo):** Crescendo can roll back turns where the model refuses, re-presenting them in a modified form. Only turns actually shown to the model in the final conversation are included in the forward pass — rolled-back turns are excluded.

Hidden states are extracted across all 32 layers simultaneously using `output_hidden_states=True` in a single forward pass. Representations are stored as float16 arrays of shape (N, 32, 4096).

**Two extraction modes:**
- **Mode A** (final-turn): one forward pass per conversation, passing the full conversation up to the last accepted user turn. Produces one (32, 4096) representation per conversation — used for centroid construction and main evaluation.
- **Mode B** (trajectory): one forward pass per turn, up to turn k, for k = 1…n. Produces one (32, 4096) representation per (conversation, turn) pair — used for trajectory analysis.

### 4.3 Finding the Harmfulness Direction

For each framework, a harmfulness direction v_harmful ∈ ℝ^(32×4096) is constructed from training conversations.

**Within-pair subtraction (topic control):**
For each JBB pair k (k = 1…100):
1. Compute μ_h,k = mean h_inst over all harmful training conversations for goal k
2. Compute μ_b,k = mean h_inst over all benign training conversations for goal k
3. Form difference vector d_k = μ_h,k − μ_b,k

This subtraction is designed to cancel the topic component: if harmful and benign conversations of the same pair differ only in their harmfulness (not their topic), d_k isolates the harmfulness signal. The quality of this cancellation depends on the outcome distributions within pairs (see §4.4 and §5).

**v_harmful** is the mean of the d_k vectors across all 100 pairs. As a diagnostic, we also run per-layer PCA on the set of {d_k} and report the variance explained by PC1 — a measure of how coherent the harmfulness direction is across goals. Low variance explained (currently 6–13%) indicates the difference vectors are spread across many directions, suggesting no single dominant harmfulness axis shared across all goals. This is expected to improve with larger sample sizes as per-pair estimates stabilise.

**Supporting centroids** (for Δ_harmful scoring):
- μ_harmful: mean h_inst over all harmful training conversations
- μ_harmless: mean h_inst over accepted benign training conversations

**Δ_harmful score** for a test conversation:
```
Δ_harmful = mean over layers of [cos_sim(h_inst, μ_harmful) − cos_sim(h_inst, μ_harmless)]
```
This is the primary evaluation scorer. It does not use v_harmful and is not topic-controlled.

**v_harmful projection score** (topic-controlled alternative):
```
proj = mean over layers of [cos_sim(h_inst_norm, v_harmful_norm)]
```
This is more directly comparable to Zhao et al. but is currently unreliable for Crescendo and ActorAttack (see §5).

### 4.4 Centroid Design Options

A key methodological question for multi-turn RepE — absent in the single-turn setting — is *which turn's representation* to use for centroid construction. We implement three designs:

**Centroid A (final-turn, notebook 04):** Uses the hidden state at t_inst of the *final accepted turn* of each conversation. Captures the representation after the full attack context has accumulated. This is the most direct analog of Zhao et al.'s single-turn approach. The representation at the final turn should reflect the model's state at the decision point closest to compliance.

**Centroid B (relative-position, notebook 05):** Divides each conversation into thirds (early ≈ turn ⌊n/3⌋, middle ≈ turn ⌊2n/3⌋, late = final turn) and constructs separate centroids for each stage. This tests whether the harmfulness signal is present early in the attack or only emerges later. A key question is whether cross-stage transfer fails — i.e., whether an early-turn centroid can detect late-turn representations and vice versa.

**Centroid C (trajectory-averaged, notebook 06):** Averages h_inst across *all* turns of a conversation before centroid construction. Each conversation contributes one averaged vector regardless of length. This normalises for conversation length and produces a representation of the "average state" of the model across the entire attack. A limitation is that if harmful and benign conversations of the same pair differ in length (which they do for Crescendo and xTeaming), the within-pair subtraction also captures a depth/timing confound.

The distinction between these designs is unique to multi-turn RepE and has no analog in single-turn work.

### 4.5 Evaluation

**Cross-framework transfer matrix:** For each (centroid source, test framework) pair, we compute Δ_harmful on held-out test conversations and report harmful accuracy (Δ_harmful > 0), benign accuracy (Δ_harmful ≤ 0), and overall accuracy. This 4×3 matrix (4 centroid sources including single-turn baseline, 3 test frameworks) is the primary evaluation.

**Centroid validation:**
- *In-sample train accuracy*: verifies the centroid correctly separates its own training data
- *Leave-one-attempt-out stability*: rebuilds v_harmful leaving one attempt out per fold; cosine similarity to the full centroid measures robustness
- *Behavioural validation*: checks that v_harmful projection ranks jailbroken > near_miss > refusal on harmful training conversations — a necessary condition for a correctly oriented direction

**Single-turn sanity check:** Applies the single-turn centroid to 200 single-turn JBB inputs (no attack context). This tests the RepE machinery in isolation and provides a ceiling on what's achievable without multi-turn context confounds.

### 4.6 Known Limitations and Honest Critiques

**v_harmful contamination (Crescendo and ActorAttack):** The behavioural validation shows the v_harmful direction is inverted for Crescendo and ActorAttack — refused conversations project more strongly onto v_harmful than jailbroken ones. The root cause is outcome distribution imbalance: Crescendo's harmful conversations are 83% jailbroken and its benign conversations are 94% jailbroken (similar distributions, weak signal); ActorAttack's harmful conversations are only 29% jailbroken while benign are 88% accepted (very different distributions, so μ_harmful ≈ refused and μ_benign ≈ accepted, and d_k ≈ refusal direction). The within-pair subtraction does not fix this — it is a structural property of how these attacks work. Only xTeaming, with more balanced outcomes, produces a correctly oriented v_harmful. Potential fixes: filter to matched-outcome pairs within pair_id, or restrict topic-control claims to xTeaming.

**No trained probe (current state):** Centroid distance (Δ_harmful) is an unsupervised scorer. Bullwinkel uses MLP probes; the standard RepE evaluation uses linear probes with cross-validation. A logistic regression probe trained directly on h_inst vectors, with goal-stratified cross-validation (held-out goals, not just held-out attempts), would be more comparable to prior work and would test whether the signal generalises across topics. This is a planned addition (see §6).

**Single model:** All experiments use Llama-3.1-8B-Instruct. Generalisability to other model families, sizes, and alignment procedures is unknown. Zhao et al. tested multiple models; this work does not yet.

**Verdict quality:** The jailbroken/near_miss/refusal labels come from the attack framework's built-in judge, which may produce false positives (partial compliances scored as jailbroken). StrongREJECT scoring of model outputs is planned to validate that "jailbroken" verdicts correspond to genuinely harmful content.

**Sample size (current state):** 10 attempts per goal is sufficient for initial direction finding but produces noisy per-pair means. The low PCA variance explained (6–13%) may partly reflect estimation noise rather than true incoherence of the harmfulness direction. The planned 30-attempt expansion will allow meaningful significance testing and more stable direction estimates.

**Length confound (Centroid C):** Trajectory-averaged centroids may conflate conversation depth with harmfulness for frameworks with variable conversation lengths.

**Goal-stratified evaluation not yet implemented:** Current train/test split is by attempt number (attempt 10 = test). This means the same goals appear in train and test, which risks the probe learning topic rather than harmfulness. Goal-stratified cross-validation (hold out k goals entirely) is planned as the primary evaluation with trained probes.

---

## 5. Preliminary Results: Direction and Overall Outcomes

*Note: specific accuracy numbers are not emphasised here. The current sample size (10 attempts per goal) is insufficient for stable estimates, and significance testing has not been performed. The following describes directional trends and qualitative findings.*

**The harmfulness signal exists but is weak and framework-specific.** Applying the single-turn centroid to single-turn inputs produces reasonable harmful/benign separation — confirming the RepE machinery is functioning. However, the same centroid applied to multi-turn test conversations produces substantially weaker separation, and cross-framework transfer largely fails: centroids trained on one framework's conversations do not generalise to another framework's attacks.

**The v_harmful directions are nearly orthogonal across frameworks.** Pairwise cosine similarities between framework-specific v_harmful directions are small (substantially below 0.5 for all pairs, with some pairs showing near-zero or negative similarity). This suggests each attack framework primes the model in a distinct subspace of activation space — there is no universal multi-turn harmfulness direction shared across frameworks.

**The apparent same-framework signal is largely topic-driven.** When the centroid delta scorer (Δ_harmful, not topic-controlled) is compared to the v_harmful projection scorer (topic-controlled), the projection-based evaluation shows near-chance performance across all framework pairs. This indicates that the separation achieved by the delta scorer reflects differences in what harmful vs. benign conversations are *about*, not differences in how the model represents their harmfulness. This replicates and extends the topic confound concern raised in the Bullwinkel paper.

**Signal strengthens monotonically from early to late turns (Centroid B).** Within-framework accuracy improves from early to late stage centroids for all three frameworks. This is consistent with Bullwinkel's finding that the harmfulness signal builds over turns. More strikingly, cross-stage transfer fails dramatically: a centroid built from late-turn representations predicts near-zero harmful accuracy when applied to early-turn representations, and vice versa. This implies the harmfulness-related direction in activation space rotates substantially as the attack progresses — multi-turn attacks do not simply amplify a fixed direction, they actively transform it.

**Trajectory-averaged centroids (Centroid C) produce better same-framework separation than final-turn centroids**, suggesting that averaging across turns gives a more stable reference point than the final turn alone. Cross-framework transfer remains poor regardless of centroid design.

**PC1 explains only 6–13% of variance in the per-pair difference vectors.** This means the harmfulness direction varies substantially across JBB goals — different harmful goals prime the model differently, with no single dominant axis. Whether this reflects genuine goal-specificity or estimation noise from the limited sample size is a key open question for the 30-attempt expansion.

**The fine-tuned attacker and stronger attacks** are expected to produce higher jailbreak rates on harmful goals, which would give cleaner outcome distributions within pairs and a better-conditioned v_harmful. This is a key motivation for the planned data expansion.

---

## 6. Planned Work and Open Questions

- **Expand to 30 attempts per goal** (harmful and benign, all three frameworks). Primary motivation: stable per-pair means, meaningful significance testing, and diagnosis of whether low PCA variance reflects noise or true goal-specificity.
- **Add fine-tuned attacker model** for comparison. Expected benefit: higher jailbreak rates, cleaner outcome distributions, and a "strong attack" reference point.
- **Add logistic regression probe with goal-stratified cross-validation** as the primary classification evaluation. The probe is trained directly on h_inst vectors (4096-dim input per layer); evaluation holds out entire goals rather than individual attempts.
- **StrongREJECT scoring** of model outputs to validate verdict labels and potentially redefine the "jailbroken" class to require genuinely harmful content.
- **Fix v_harmful for Crescendo and ActorAttack**: either filter to matched-outcome pairs within pair_id, or restrict topic-control claims to xTeaming pending better data.
- **Expand to additional models**: at minimum one larger model (Llama-3.1-70B or equivalent) and one model from a different family.
- **Significance testing**: permutation tests on the cross-framework transfer matrix once the 30-attempt dataset is available.
- **Refusal direction analysis**: the Δ_refuse score and its relationship to Δ_harmful (Zhao et al. Fig. 5 reproduction in multi-turn setting) requires cleaner mu_refuse/mu_accept centroids, which depend on resolved verdict quality.
