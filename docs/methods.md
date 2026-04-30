# Methods

## Model and Dataset

All experiments use Meta-Llama-3.1-8B-Instruct with the standard Llama safety system prompt, consistent across all conversations and extraction conditions. The dataset is drawn from the JBB-Behaviors benchmark, which provides 100 harmful goals and 100 benign goals matched by topic into 100 pairs. The topic pairing is central to the analysis: because each harmful goal shares a topic with its benign counterpart, computing directions via within-pair subtraction cancels topic-specific signal and isolates harmfulness.

Three attack frameworks generate multi-turn conversations against the model: Crescendo (gradual escalation through benign-seeming steps; primary framework), ActorAttack (Actor-Network Theory framing with multiple attacker personas), and X-Teaming (strategy planning and execution). Each framework runs 20 independent attack attempts per goal using GPT-4o as both attacker and judge, yielding approximately 2,000 harmful and 2,000 benign conversations per framework (~12,000 total). Crescendo conversations average approximately 6 accepted turns (range 1–10); ActorAttack and X-Teaming are shorter (3–4 turns on average). Conversations are labeled at the goal level (harmful or benign), conversation level (attack success), and — for Crescendo — at the turn level (is_refusal, judge_success).

## Token Positions

Following Zhao et al. (2025), hidden states are extracted at two positions within each user turn. The first position, t_inst, is the last token of the attacker's message — the token immediately before the closing `<|eot_id|>`. In Zhao et al.'s single-turn analysis, this position encodes harmfulness: whether the instruction is harmful, independent of the model's behavioral response. The second position, t_post_inst, is the `<|eot_id|>` token itself that closes the user turn. In single-turn, this position encodes refusal: whether the model will refuse, independent of actual harmfulness.

Whether these positions encode the same signals in multi-turn settings is an open empirical question that the analysis directly tests. In multi-turn settings, causal self-attention means both positions absorb information from the entire prior conversation context, not just the current message. The positions are adopted here to enable direct comparison with Zhao et al. while explicitly studying whether their single-turn semantics survive context accumulation.

## Extraction Conditions

The analysis relies on four extraction conditions that together isolate what drives changes in the harmfulness representation.

**Full-context extraction.** For each conversation of length n, we run n separate forward passes at increasing context depths. At prefix k, the model receives the system prompt, turns 1 through k−1 with both attacker messages and model responses, and the k-th attacker message without the model's response at turn k. The sequence ends at the `<|eot_id|>` token after the k-th attacker message. The hidden state at this point reflects both the content of the k-th message and the accumulated influence of all prior turns via self-attention. Rolled-back turns (Crescendo-specific: turns the attacker retried after the model refused) are excluded from all prefix constructions.

**No-context extraction.** For each conversation and each turn k, we run a forward pass containing only the system prompt and the k-th attacker message in isolation — no prior conversation history. The hidden state reflects only the content of that specific message. The difference between full-context and no-context at the same (conversation, k) is the pure effect of context accumulation: anything that differs between them cannot be due to message content.

**Compressed single-turn extraction.** The entire multi-turn conversation is concatenated into a single user message with plain-text "User:" and "Assistant:" prefixes replacing the chat template's role headers and `<|eot_id|>` delimiters between turns. The conversation content is preserved but the turn structure is removed. This produces one extraction per conversation and tests whether context effects are driven by semantic content alone or whether the turn structure and role-based attention patterns contribute independently. Bullwinkel et al. found compressed and multi-turn representations to be nearly identical for Crescendo.

**Single-turn baseline extraction.** Each of the 200 raw JBB goals (100 harmful, 100 benign) is passed directly as a single user message with the same system prompt and no multi-turn framing. This produces the baseline direction v_harmful(ST) — the direction that any single-turn defense would learn from.

All conditions extract hidden states at all 32 transformer layers, producing arrays of shape (N, 32, 4096) in float16. The full-context extraction (Condition 1) and single-turn baseline (Condition 4) were extracted prior to this study; the no-context (Condition 2) and compressed (Condition 3) extractions are new.

## Direction Computation

All harmfulness directions are computed via within-pair subtraction. For each of the 100 JBB pairs, we compute the difference between the mean hidden state of harmful-goal conversations and the mean hidden state of benign-goal conversations at that pair. The harmfulness direction is the mean of these 100 pair-level differences, normalized to unit length at each layer independently. This produces a unit vector in R^4096 per layer that captures the harmful/benign distinction while cancelling topic-specific signal.

For per-turn directions, this computation is repeated separately at each turn k, using only conversations that reach at least turn k. For single-turn, the 100 harmful and 100 benign goals are sorted by pair ID and subtracted directly. Directions are computed separately for h_inst and h_post positions and for full-context and no-context conditions.

Attempts 1–16 are used for direction computation and classifier training; attempts 17–20 are held out for evaluation. This ensures that the test set contains entirely separate attack attempts on the same goals.

## Displacement Analysis

To measure direction displacement, we compute the cosine similarity between the per-turn full-context direction v_harmful_full(k) and the single-turn baseline v_harmful(ST) at each turn k. A decreasing cosine indicates that the multi-turn harmfulness direction is rotating away from what a single-turn defense would monitor. To attribute this displacement to context accumulation, we also compute the cosine similarity between the full-context direction and the no-context direction at the same turn k. At k=1, both conditions receive identical inputs (no prior context exists), so this cosine should be near 1.0; divergence at later turns quantifies the contribution of accumulated history. The compressed direction's cosine with v_harmful(ST) provides a reference for comparison with Bullwinkel et al.

To measure the practical cost of displacement, we compare two classifiers at each turn k on held-out data: a fixed-direction projection onto v_harmful(ST) and an adaptive projection onto the per-turn direction v_harmful_full(k). AUROC is computed for harmful-vs-benign classification at each turn. The gap between fixed and adaptive AUROC quantifies how much detection performance is lost by relying on a single-turn direction.

For behavioral correlation, we project each turn's hidden state (harmful conversations only, Crescendo) onto v_harmful_full(k) and split by the per-turn judge_success label. We compare the distribution of projections between turns where the model complied and turns where it did not.

## Token Position Verification

To test whether the Zhao et al. position-specific semantics (t_inst encodes harmfulness, t_post encodes refusal) hold in multi-turn settings, we compute AUROC for two tasks at each position across layers: harmfulness classification (harmful vs. benign goal type) and refusal classification (refused-harmful vs. accepted-harmful within harmful conversations only). If the Zhao et al. finding holds, t_inst should outperform t_post on harmfulness and t_post should outperform t_inst on refusal.

Silhouette scores are computed in the full 4096-dimensional space (subsampled to 300 per class for tractability) to measure geometric cluster quality without assuming a specific direction. This captures whether harmful and benign representations form distinct clusters or overlap heavily.

To replicate Zhao et al.'s misbehavior clustering analysis, we compute the nearest-centroid fraction for refused-harmful conversations: for each sample, we measure distance to the accepted-harmful centroid and to the benign centroid, and report the fraction closer to the harmful centroid. In Zhao et al.'s single-turn setting, this fraction is near 1.0 at t_inst (the model internally encodes refused-harmful instructions as harmful regardless of the refusal). We compute this across all layers and for both full-context and no-context conditions to test whether the pattern holds in multi-turn.

The PCA scatter comparison visualizes the three groups (accepted-harmful, refused-harmful, benign) at the final turn across extraction conditions and layers, providing a qualitative complement to the quantitative measures.

All position verification analyses are computed at 8 layers spanning the model (L4, L8, L12, L16, L20, L24, L28, L32) and across all three attack frameworks.

## Nonlinear Probing

To determine whether the displacement is a linear rotation (recoverable by a nonlinear classifier) or a deeper structural change, we compare logistic regression with an MLP classifier (single hidden layer of 64 units, early stopping, StandardScaler preprocessing).

Within-condition comparison trains and evaluates both classifiers on the same condition (single-turn, full-context, or no-context) to measure whether nonlinear signal exists beyond the linear direction. Cross-condition transfer trains on one condition and evaluates on another, producing a 3x3 matrix (single-turn, full-context, no-context) for both classifier types at a fixed mid-layer (L20). The critical cell is single-turn to full-context: if MLP succeeds where linear fails, the displacement is a linear rotation; if both fail equally, the representation has genuinely reorganized.

Per-turn transfer trains a classifier at each turn k and evaluates at every other turn k', producing a k-by-k AUROC matrix for both linear and MLP. This reveals whether the nonlinear representation is stable across turns even though the linear direction rotates — and whether nearby turns share more representational structure than distant turns.
d