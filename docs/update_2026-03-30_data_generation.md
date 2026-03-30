# Data Generation Update
**Date:** 2026-03-30

---

Experiment 1 found that MLP probes and LAT reading vectors separate harmful from benign at k=1 — before any Crescendo history is in context — suggesting the probes were picking up topic differences rather than multi-turn attack dynamics. The current study is designed to test that directly, using JBB-Behaviors matched pairs (same topic, different intent) and three attack frameworks to ask whether jailbreak representations are attack-specific or universal. This update covers the data generation infrastructure built to support that.

## What was built

Three async attack runners (Crescendo, ActorAttack, X-Teaming) were implemented from scratch, faithfully reimplementing each framework with verbatim prompts from their original repos. All use gpt-4o as attacker, gpt-4o as judge, and local vLLM as target with no system prompt. A shared orchestration function handles all three with concurrent execution, per-conversation progress bars, and resume-safe file output.

Crescendo generates questions one at a time using a binary judge (refusal check → success check). ActorAttack routes toward the harmful objective indirectly by first discussing semantically related "actors" from the target's network, then synthesizing via a final formatting turn. X-Teaming generates multiple persona-driven strategies upfront via gpt-4o, executes each turn by turn, and revises the plan if initial turns fail.

## Data generated

Two Crescendo runs of 1000 conversations each (100 objectives × 10 attempts). The first was archived after discovering bugs and using the weaker gpt-4o-mini attacker. The second (v2) is clean. One ActorAttack run was also archived — 83% of files have empty turns (see below). All data is gitignored.

## Bugs found and fixed

In Crescendo, refused exchanges were staying in target history (priming further refusals) and the attacker context wasn't getting the target response appended on refusal (causing consecutive user messages that the OpenAI API rejects). Both fixed.

In X-Teaming, the conversation extraction was using paired regex matching on `<conversation>` tags — when the LLM omitted the closing tag, the full raw output including chain-of-thought leaked to the target. Also, when gpt-4o-mini refused to generate an attack turn, the refusal text was being sent to the target verbatim. Both fixed; attacker model also switched to gpt-4o.

Across all runners, `max_tokens=512` was cutting off ~73% of target responses mid-sentence. Increased to 2048.

## Known issues

The internal judges produce false positive jailbreak verdicts at a meaningful rate — academic responses about harmful topics get scored as successful. This is a weakness of the original judge prompts, not an implementation error. StrongREJECT scoring of Crescendo v2 is currently running to quantify this. The ActorAttack v1 empty turns were caused by gpt-4o-mini refusing to generate queries for sensitive objectives; fixed by switching to gpt-4o.

## Next steps

Review StrongREJECT results, then rerun ActorAttack and X-Teaming with all fixes. Generate benign conversations. Begin hidden state extraction.
