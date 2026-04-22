# Mechanistic Interpretation of Multi-Turn Attack Frameworks

## Abstract

Large language models remain vulnerable to jailbreak attacks that elicit harmful outputs by circumventing safety alignment. Multi-turn jailbreak attacks pose a particular challenge, as harmful intent is distributed across conversational turns that each appear individually benign or safe. Existing representation engineering approaches have been developed primarily for single-turn settings, leaving open the question of whether and how they generalize to multi-turn attacks. This work examines how multi-turn context affects harmfulness and refusal representations in transformer hidden states. We do [] and find [].

## 1. Introduction

Multi-turn jailbreak attacks represent a harder problem than single-turn attacks. Rather than creating a single harmful prompt, an adversary distributes harmful intent across multiple conversational turns, each of which may appear individually benign. The model's compliance emerges gradually from accumulated context, which means that any defense anchored to the content of an individual message may not succeed in detecting this kind of attack. This project studies multi-turn jailbreaks through the lens of representation engineering, where instead of focusing on the model's outputs, we study what internal hidden states reveal about how it is processing a conversation.

### 1.1 Representation engineering

Representation engineering finds directions in a model's activation space that correspond to interpretable behaviors. For safety-relevant applications, the key method is: take contrastive pairs (for example, harmful vs. benign instructions), extract hidden states at specific token positions, take the mean difference, and normalize. The resulting direction vector can be used to classify inputs or steer model behavior. In single-turn settings, this approach has produced important findings. Prior work shows that harmfulness and refusal are encoded separately and at different token positions: the last token of the user's instruction encodes whether the input is harmful, while the end-of-turn special token encodes whether the model will refuse, independently of whether the instruction is actually harmful. This separation is causal, verified by steering experiments and a reply inversion task. Detection methods like Latent Guard and circuit breakers build on these findings to create defenses against single-turn jailbreaks.

### 1.2 Multi-turn gap

One goal of this work is to extend representation engineering methods previously applied to single-turn attacks to multi-turn ones. In a multi-turn conversation, the same token positions are present at each turn, but they sit in a context that includes the entire prior conversation history. Self-attention means these positions absorb information from everything the model has seen, not just the current message. Whether the harmfulness and refusal signals remain stable or shift as context accumulates, and whether any such shift is general or strategy-specific, is a question that prior work has not answered.

The closest related work applied representation engineering to multi-turn attacks with a few limitations. First, they extract representations from the response tokens of the final jailbroken turn, not from the input-side positions identified in single-turn work. Their method included training an MLP probe on single-turn harmful vs. benign data, so their setup measures how similar the model's output is to a single-turn harmful response, and not whether the model's internal representation of the input shifts across turns. They also test a compressed condition, where they concatenate the multi-turn conversation into a single prompt, and find that the representations are nearly identical to the full multi-turn condition. They interpret it as semantic context dominating over turn structure. They only study output-side representations, and not whether it holds for input-side positions at each turn.

### 1.3 This work

We extend the single-turn positional-semantics approach to multi-turn settings, studying how the harmful/benign direction changes at each turn as context accumulates, across three attack frameworks. At each turn `k`, we extract representations under full conversational context, under no prior context (the `k`-th message in isolation), and under a compressed single-turn baseline.

## 2. Literature Review

### 2.1 Representation engineering for safety

The study of how safety-relevant concepts are encoded in transformer hidden states draws on a broader body of mechanistic interpretability work showing that interpretable features are often linearly recoverable from activation space. Early work introduced representation engineering as a top-down approach to AI transparency, applying contrastive difference-in-means to recover behavioral directions. Follow-up work extended this to circuit breakers, using representation-level constraints to suppress harmful outputs robustly under adversarial pressure. Other work has identified a single refusal direction in instruction-tuned models and shown that ablating it suppresses refusal across diverse inputs.

The most precise single-turn analysis relevant to this work shows that harmfulness is encoded at `t_inst` and refusal at `t_post` as causally distinct concepts, verified by a reply inversion task in which steering along the harmfulness direction reverses the model's internal judgment of an instruction's harmfulness while steering along the refusal direction does not. Their Latent Guard method, using the `t_inst` direction as an intrinsic safeguard, matches or exceeds Llama Guard 3 8B on single-turn inputs. These findings are only about single-turn settings; their extension to multi-turn is the central question of this work.

### 2.2 Multi-turn attacks

Multi-turn jailbreak attacks exploit conversational context to elicit harmful outputs that direct single-turn requests would fail to produce. Crescendo uses gradual escalation through seemingly benign steps, with refused turns rolled back from the target model's context before the next attempt — meaning the context the model experiences is not the same as a naive reading of the turn log. ActorAttack constructs a network of fictional actors whose interaction progressively surfaces harmful content, concentrating around 3–4 turns with low variance. X-Teaming uses a two-phase approach — strategic planning followed by execution — achieving higher per-attempt success rates with similar conversation lengths to ActorAttack.

Other work studies chain-of-thought manipulation as a multi-turn vector, showing that models can be led to reason themselves into harmful outputs across a sequence of ostensibly reasonable steps. Earlier analyses of prompt injection and competing objectives as jailbreak mechanisms are also relevant; multi-turn attacks generalize these by distributing the adversarial pressure across turns. Many-shot jailbreaking, in which a large number of in-context examples normalize harmful behavior, is related in mechanism to conversational priming but distinct in that it relies on in-context learning within a single context rather than turn-by-turn escalation.

### 2.3 Probing and linear representation

Prior research supports linear encoding of semantic features in transformer activations. The linear representation hypothesis implies that direction-based classifiers are not missing nonlinear signal, and that transfer failures reflect the direction moving rather than a capacity limitation. We test this directly. The broader question of probe generalization under distributional shift connects to our cross-condition transfer analysis: multi-turn attacks provide a controlled setting in which the shift has a specific mechanistic cause (context accumulation via self-attention), making them a useful testbed for this question beyond safety.

## 3. Methods

### 3.1 Dataset and attack data

Primary dataset is the JBB-Behaviors benchmark, which provides 100 harmful goals and 100 benign goals matched by topic into 100 pairs. Because each harmful goal shares a topic with its benign counterpart, computing directions via within-pair subtraction should cancel topic-specific signal and isolate harmfulness better.

Three attack frameworks generate multi-turn conversations against the model: Crescendo (gradual escalation through benign-seeming steps; primary framework), ActorAttack (Actor-Network Theory framing with multiple attacker personas), and X-Teaming (strategy planning and execution). Each framework runs 20 independent attack attempts per goal using GPT-4o as both attacker and judge, yielding approximately 2,000 harmful and 2,000 benign conversations per framework (~12,000 total). Crescendo conversations average approximately 6 accepted turns (range 1–10); ActorAttack and X-Teaming are shorter (3–4 turns on average).

### 3.2 Target models

Target model is Llama-3.1-8B-Instruct with a standard helpful assistant system prompt. GPT-4o is the attacker and judge for all three frameworks.

We also include Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct as additional target models. These span different architectures, training data, and safety alignment strategies. Phi-3.5-mini is notably smaller (3.8B) than Llama and Qwen (~7–8B), which lets us check whether the displacement phenomenon is robust to target-model scale.

### 3.3 Token positions

Following prior work, hidden states are extracted at two positions within each user turn. The first position, `t_inst`, is the last token of the attacker's message — the token immediately before the closing `<|eot_id|>`. In the single-turn analysis, this position encodes harmfulness: whether the instruction is harmful, independent of the model's behavioral response. The second position, `t_post_inst`, is the `<|eot_id|>` token itself that closes the user turn. In single-turn, this position encodes refusal: whether the model will refuse, independent of actual harmfulness. One of our questions is whether these positions encode the same signals in multi-turn settings.

### 3.4 Direction computation

All harmfulness directions are computed via within-pair subtraction. For each of the 100 JBB pairs, we compute the difference between the mean hidden state of harmful-goal conversations and the mean hidden state of benign-goal conversations at that pair. The harmfulness direction is the mean of these 100 pair-level differences, normalized to unit length.

For per-turn directions, this computation is repeated separately at each turn `k`, using only conversations that reach at least turn `k`. For single-turn, the 100 harmful and 100 benign goals are sorted by pair ID and subtracted directly. Directions are computed separately for `h_inst` and `h_post` positions and for full-context and no-context conditions.

Attempts 1–16 are used for direction computation and classifier training; attempts 17–20 are held out for evaluation.

### 3.5 Displacement analysis

To measure direction displacement, we compute the cosine similarity between the per-turn full-context direction `v_harmful_full(k)` and the single-turn (ST) baseline `v_harmful(ST)` at each turn `k`. A decreasing cosine indicates that the multi-turn harmfulness direction is rotating away from what a single-turn defense would monitor. To attribute this displacement to context accumulation, we also compute the cosine similarity between the full-context direction and the no-context direction at the same turn `k`. At `k=1`, both conditions receive identical inputs (no prior context exists), so this cosine should be near 1.0; divergence at later turns quantifies the contribution of accumulated history. The compressed direction's cosine with `v_harmful(ST)` provides a reference for comparison with prior work.

To measure the practical cost of displacement, we compare two classifiers at each turn `k` on held-out data: a fixed-direction projection onto `v_harmful(ST)` and an adaptive projection onto the per-turn direction `v_harmful_full(k)`. AUROC is computed for harmful-vs-benign classification at each turn. The gap between fixed and adaptive AUROC quantifies how much detection performance is lost by relying on a single-turn direction.

### 3.6 Token position verification

To test whether the position-specific semantics from single-turn work (`t_inst` encodes harmfulness, `t_post` encodes refusal) hold in multi-turn settings, we compute AUROC for two tasks at each position across layers: harmfulness classification (harmful vs. benign goal type) and refusal classification (refused-harmful vs. accepted-harmful within harmful conversations only). If the single-turn finding holds, `t_inst` should outperform `t_post` on harmfulness and `t_post` should outperform `t_inst` on refusal.

All position verification analyses are computed at 8 layers spanning the model (L4, L8, L12, L16, L20, L24, L28, L32) and across all three attack frameworks.

### 3.7 Non-linear probing

To determine whether the displacement is a linear rotation (recoverable by a nonlinear classifier) or a deeper structural change, we compare logistic regression with an MLP classifier (single hidden layer of 64 units, early stopping, StandardScaler preprocessing).

Within-condition comparison trains and evaluates both classifiers on the same condition (single-turn, full-context, or no-context) to measure whether nonlinear signal exists beyond the linear direction. Cross-condition transfer trains on one condition and evaluates on another, producing a 3×3 matrix (single-turn, full-context, no-context) for both classifier types at a fixed mid-layer (L20). The critical cell is single-turn to full-context: if MLP succeeds where linear fails, the displacement is a linear rotation; if both fail equally, the representation has genuinely reorganized.

Per-turn transfer trains a classifier at each turn `k` and evaluates at every other turn `k'`, producing a `k`-by-`k` AUROC matrix for both linear and MLP. This reveals whether the nonlinear representation is stable across turns even though the linear direction rotates — and whether nearby turns share more representational structure than distant turns.

## 4. Results (for now)

### 4.1 Does the harmful/benign direction change as context accumulates, and is this change driven by context rather than message content?

The harmfulness direction at `t_inst` progressively rotates away from the single-turn baseline as context accumulates. Full-context and no-context directions are nearly identical at the first turn but become nearly orthogonal by later turns, isolating accumulated history as the cause. The compressed single-turn condition, which preserves all semantic content but removes turn structure, produces a direction similarly displaced from the single-turn baseline as the late-turn full-context direction, roughly consistent with prior findings that turn structure matters less than accumulated content. The representation changes progressively across turns rather than abruptly: per-turn classifiers transfer well to nearby turns but degrade for distant ones, and the representation stabilizes somewhat at later turns with late-turn classifiers transferring well to each other.

### 4.2 Does this displacement cause detection methods to fail, and can more powerful classifiers recover the lost signal?

A fixed-direction linear projection anchored to the single-turn baseline performs poorly across all turn depths, while a per-turn adaptive projection that recomputes the direction from multi-turn data maintains substantially better performance, strongest at middle layers, with late layers consistently degrading across all conditions. A nonlinear MLP trained on single-turn data performs no better than logistic regression on multi-turn inputs — and in fact performs slightly worse — indicating the displacement is not a linear rotation that a more powerful classifier can see through but a genuine reorganization of the representation. Within each condition, MLP and logistic regression achieve nearly identical performance, confirming the harmfulness signal is almost entirely linear and that direction-based defenses are not missing hidden nonlinear structure.

### 4.3 Do the token positions identified by prior work retain their functional roles in multi-turn settings?

The clean functional separation between `t_inst` (harmfulness) and `t_post` (refusal) identified in single-turn work collapses when context accumulates: in multi-turn settings, both positions perform similarly on both harmfulness and refusal classification tasks across all layers and frameworks. At `t_post` the single-turn geometry is not merely absent but actively anti-predictive, with cross-condition MLP transfer falling below chance.

### 4.4 Where does the harmfulness signal reside — in the messages or in the conversation history?

When the model has access to the full conversation, refused-harmful conversations cluster closer to accepted-harmful than to benign at early and middle layers — the model partially recognizes these conversations as harmful even though it refused. Strip the conversation history and present only the isolated attacker message, and this signal disappears: refused-harmful messages look indistinguishable from or even more similar to benign than to harmful. The gap between the two lines is the pure contribution of accumulated context to the model's internal representation of harmfulness.

This pattern is framework-dependent. Crescendo, which builds harmful intent gradually through escalation, produces enough accumulated context that even failed attacks register as partially harmful. ActorAttack, which distributes harmful intent through fictional actor framing, does not — failed ActorAttack conversations look benign to the model regardless of whether context is present. This suggests that different attack strategies fail differently at the representation level: Crescendo failures are cases where the model recognized the harmful trajectory but refused to comply, while ActorAttack failures may be cases where the harmful intent was never internally recognized at all.

## 5. Limitations and problems

Many:

- **Three models, all dense-attention decoder-only transformers.** Models with different attention mechanisms (e.g., sliding-window or sparse attention) and different safety alignment strategies may show different displacement patterns than the ones we measure here.
- **Only one dataset.** The 10 harm categories included probably do not represent the full range of harmful requests that multi-turn attacks target in practice. Although to be fair, the 10 harm categories and individual behaviors included are quite diverse and validated, and the topic-matched structure is important. What needs to be done is leave training and direction computation on JBB only, and test on an external dataset of harmful/benign goals outside of JBB. Good candidates would be AdvBench, HarmBench, and OR-Bench.
- **Harmful vs. benign confounds beyond topic.** Harmful and benign conversations may differ in ways other than topic (attacker vocabulary patterns, model response styles, and turn-level dynamics may all differ between harmful and benign conditions in ways that the direction captures alongside harmfulness). We have not verified the magnitude of this confound, though the fact that no-context directions (where conversation dynamics are absent) achieve similar AUROC to full-context directions with their own `v_own` suggests that at least some of the captured signal is content-based rather than purely structural. Honestly, at least trying to estimate the magnitude of those confounders would be valuable on its own — no prior work I've seen talks or cares much about that.
- **Train/test split by attempt, not by goal.** For now, the 80:20 train:test split divides by attempt number (1–16 for training, 17–20 for evaluation) rather than by goal. This means the same harmful and benign goals appear in both splits. Again, training should stay on JBB but testing should use a totally different dataset of goals. Might need to include how similar the test dataset goals are to JBB ones.
- **Small sample sizes, especially at higher turns.**
- **What's the impact of all this?**

## 6. Discussion

*(To be written.)*

## 7. Conclusions & Future Work

*(To be written.)*

## Appendix A. System prompt used in all experiments

> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
