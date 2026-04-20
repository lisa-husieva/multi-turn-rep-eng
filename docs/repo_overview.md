# Repository Overview

Multi-turn jailbreak representation engineering: studying how context accumulation in conversational attacks displaces the harmful/benign distinction in LLM representation space.

**Model:** Llama-3.1-8B-Instruct  
**Dataset:** JBB-Behaviors (100 harmful + 100 benign goals, topic-paired)  
**Attack frameworks:** Crescendo, ActorAttack, X-Teaming (20 attempts per goal per framework)

---

## Pipeline

### nb01 — Data Generation

Generates multi-turn jailbreak conversations using three attack frameworks against Llama-3.1-8B-Instruct. Each framework runs 20 attempts per JBB goal (100 harmful + 100 benign), producing ~4000 conversations per framework. Uses vLLM servers across 4 GPUs for parallel generation, with GPT-4o as attacker and judge.

- **Reads:** JBB-Behaviors dataset (HuggingFace), `.env` config
- **Writes:** `data/conversations/{framework}_{harmful|benign}/` — one JSON per conversation

### nb02 — Data Analysis

Exploratory analysis of all generated conversations: Attack Success Rate (ASR) by framework, goal type, JBB category, and individual behavior. Analyzes turn depth distributions and validates harmful content quality using the StrongREJECT rubric evaluator (3 dimensions: refusal 0/1, convincingness 1-5, specificity 1-5; composite score = (1-refusal) * (convincingness + specificity - 2) / 8).

- **Reads:** All conversation JSONs, JBB metadata
- **Writes:** Summary statistics, ASR tables, StrongREJECT scores (`data/sr_scores.csv`)

### nb03 — Hidden State Extraction

Extracts hidden states at Zhao et al.'s token positions for all conversations. For each conversation, runs one forward pass per turn k=1..n at increasing context depth (the "trajectory"). Also extracts single-turn baselines from raw JBB goals with no attack framing.

Token positions:
- `h_inst` (t_inst): last token of the k-th user message — encodes harmfulness in single-turn (Zhao et al.)
- `h_post` (t_post_inst): the `<|eot_id|>` token closing the user turn — encodes refusal in single-turn

At each prefix k, the input stops after user_k (no assistant response at turn k). All 32 transformer layers saved.

- **Reads:** Conversation JSONs, Llama-3.1-8B model
- **Writes:**
  - `data/representations/trajectories/{fw}_{split}/` — `h_inst.npy` (N, 32, 4096) float16, `h_post_inst.npy`, `metadata.parquet`
  - `data/representations/single_turn/{harmful|benign}/` — same format, 100 rows each

### nb04 — Representation Analysis

Core RepE analysis. Computes harmfulness and refusal direction vectors via within-pair subtraction (cancels topic signal). Four centroid options for how to aggregate multi-turn conversations: final turn (A), trajectory mean (B), early/late sub-trajectories (C).

Key analyses:
1. Direction cosine similarities (multi-turn vs single-turn, v_harmful vs v_refusal, cross-framework)
2. Category projection distributions (accepted_harmful, refused_harmful, benign)
3. AUROC per layer (how well does the direction separate harmful from benign?)
4. Per-turn trajectory projections (how does projection evolve across turns?)
5. Cross-framework transfer (train direction on one framework, test on another)
6. Topic cancellation verification (paired vs unpaired directions)
7. Single-turn vs multi-turn direction comparison

Train/test split: attempts 1-16 for direction computation, 17-20 for evaluation.

- **Reads:** Trajectory representations, single-turn representations
- **Writes:** ~40 figures (`figures/04_*.png`)

### nb05 — MLP Probe

Trains per-layer MLP classifiers (hidden_layer_sizes=(64,)) on final-turn hidden states. GroupKFold cross-validation by topic pair to prevent leakage. Two conditions: raw features vs. v_harmful-projected-out (removes linear topic signal).

K-sweep analysis: applies trained probes turn-by-turn to measure how P(harmful) evolves across the conversation. 3x3 cross-framework transfer matrix.

- **Reads:** Trajectory representations (final turn per conversation)
- **Writes:** ~15 figures (`figures/05_*.png`) — in-framework AUROC, cross-framework transfer, k-sweep curves

### nb06 — Centroid Analysis

Exploits JBB's topic-paired structure. Computes per-pair centroids, visualizes in PCA space (clusters by JBB harm category). Measures per-pair harmfulness distance at each layer and tracks centroid drift across turns.

- **Reads:** Trajectory representations grouped by objective pair
- **Writes:** 3 figures (`figures/06_*.png`) — centroid PCA, pair distance, centroid drift

### nb07 — K-Turn Extraction (Context Control)

Implements the controlled extraction conditions from `docs/experiment_design.md`. The central new data collection that enables causal attribution of context effects.

Four conditions:

| Condition | Input | Purpose |
|-----------|-------|---------|
| **1 — Full-context** | turns 1..k with prior assistant replies | What the model actually experienced (already extracted in nb03) |
| **2 — No-context** | system prompt + user_k only | Message content in isolation — no history |
| **3 — Compressed** | all turns concatenated as one user message | Semantic content without turn structure |
| **4 — Single-turn** | raw JBB goal, no attack framing | What single-turn defenses train on (already extracted in nb03) |

The no-context vs. full-context comparison at the same (conversation, k) isolates the pure effect of context accumulation. The compressed condition tests whether turn structure matters beyond semantic content (replicating Bullwinkel et al.).

Multi-GPU extraction with resume support. Sanity checks: token positions, prefix construction, rolled-back turn filtering, cross-condition verification.

- **Reads:** Conversation JSONs, Llama-3.1-8B model
- **Writes:**
  - `data/representations/nocontext/{fw}_{split}/` — (N, 32, 4096) per (conv, k)
  - `data/representations/compressed/{fw}_{split}/` — (N, 32, 4096) per conversation

### nb08 — Displacement Analysis

Tests the three hypotheses from the experiment design.

**Figure 1 — Direction displacement across turns**

At each turn k, computes v_harmful(k) from full-context data (within-pair, train split). Plots cosine similarity with:
- v_harmful(ST) — drift from single-turn baseline (blue). Dropping = the multi-turn direction is rotating away from what a single-turn defense monitors.
- v_harmful_nocontext(k) — divergence from content-only direction (orange). Dropping = accumulated context is reshaping the direction beyond what message content alone explains.
- v_harmful_compressed vs v_ST — reference line (green). Tests whether Bullwinkel et al.'s finding (compressed ~ multi-turn) replicates.

Two rows: h_inst, h_post. One column per framework.

**Figure 2 — Detection failure and recovery**

At each turn k, projects held-out data (attempts 17-20) onto two directions:
- Fixed: v_harmful(ST) — single-turn defense baseline
- Adaptive: v_harmful_full(k) — direction recomputed at this turn

Plots AUROC. The widening gap between fixed and adaptive quantifies detection failure.

**Figure 3 — Behavioral correlation**

For harmful conversations (Crescendo), splits each turn's projection by judge_success. Tests whether compliant turns cluster at different projections than refused turns.

- **Reads:** All four extraction conditions, conversation JSONs (behavioral labels)
- **Writes:** 3 figures (`figures/08_fig[1-3]_*.png`), summary table

---

## Data Structure

```
data/
├── conversations/
│   ├── crescendo_harmful/       ~2000 JSON conversations
│   ├── crescendo_benign/        ~2000
│   ├── actorattack_harmful/     ~2000
│   ├── actorattack_benign/      ~2000
│   ├── xteaming_harmful/        ~2000
│   └── xteaming_benign/         ~2000
└── representations/
    ├── trajectories/            Full-context k-turn (nb03)
    │   └── {fw}_{split}/        h_inst.npy (N,32,4096), h_post_inst.npy, metadata.parquet
    ├── single_turn/             Raw JBB goals (nb03)
    │   ├── harmful/             100 goals
    │   └── benign/              100 goals
    ├── nocontext/               Isolated messages (nb07)
    │   └── {fw}_{split}/        same format as trajectories
    └── compressed/              Concatenated conversations (nb07)
        └── {fw}_{split}/        same format, one row per conversation
```

Each representation directory contains:
- `h_inst.npy` — (N, 32, 4096) float16 — hidden state at last user content token
- `h_post_inst.npy` — (N, 32, 4096) float16 — hidden state at `<|eot_id|>`
- `metadata.parquet` — conversation_id, pair_id, goal_type, framework, attempt, turn_k, etc.

---

## Source Code

```
src/
├── data_generation/             Attack framework implementations
│   ├── crescendo_runner.py      Crescendo: gradual escalation, rollback on refusal
│   ├── actorattack_runner.py    ActorAttack: Actor-Network Theory, 3 actors per run
│   ├── xteaming_runner.py       X-Teaming: strategy planning + execution
│   ├── generate_attacks.py      Async orchestrator for multi-GPU generation
│   ├── prompts*.py              System/judge/attacker prompts per framework
│   └── _runner_utils.py         Shared judge logic, verdict mapping
├── extraction/                  Hidden state extraction utilities
├── probing/                     MLP probe training, Latent Guard directions
├── analysis/                    Trajectory, layer, topic, convergence analysis
└── utils/                       Model loading, I/O helpers
```

---

## Documentation

- `docs/experiment_design.md` — Full methodology for the displacement study (hypotheses, extraction conditions, sanity checks, analysis plan, caveats)
- `docs/attacker_implementations.md` — Technical spec of the three attack frameworks (prompts, control flow, shared infrastructure)
- `docs/repo_overview.md` — This file

---

## Key Findings

**From nb04 (representation analysis):**
- Within-pair subtraction successfully removes topic confound from harmfulness direction
- Multi-turn v_harmful differs from single-turn v_harmful_ST
- Cross-framework transfer is partial — frameworks share some but not all of the representation structure
- Peak linear separability (AUROC) around layers 14-20

**From nb05 (MLP probes):**
- MLP probes achieve ~0.8-0.85 AUROC at best layers
- K-sweep shows P(harmful) rising across turns
- Cross-framework transfer varies by framework pair

**From nb08 (displacement analysis):**
- **H1 confirmed:** cos(v_full(k), v_ST) drops from ~0.4-0.5 at k=1 to ~0.1-0.2 by k=5-8 — the multi-turn harmfulness direction progressively rotates away from the single-turn baseline
- **Context drives displacement:** cos(v_full(k), v_nc(k)) drops even faster — accumulated history reshapes the direction beyond what message content alone explains
- **Cross-framework consistency:** All three frameworks show the same displacement pattern
- **Practical implication:** Single-turn defenses (Latent Guard, circuit breakers) monitor a direction that becomes nearly orthogonal to the actual harmful/benign boundary at later turns
