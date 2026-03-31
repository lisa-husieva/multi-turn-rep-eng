"""
Generate harmful multi-turn attack conversations.

Orchestrates multi-turn attack generation across all three frameworks:
Crescendo, ActorAttack, and X-Teaming. All runners share the same interface.

Each saved conversation file is a JSON object matching the schema:
{
    "conversation_id": str,          # UUID
    "objective_pair_id": int,        # Links to JBB matched pair index
    "objective": str,                # The harmful behavior
    "attack_framework": str,         # "crescendo" | "actorattack" | "xteaming"
    "model": str,                    # Model shortname, e.g. "llama3_8b"
    "verdict": str,                  # "jailbroken" | "near_miss" | "refusal"
    "n_turns": int,                  # Number of turns attempted
    "executed_turns": int,           # Turns that got a non-refusal response
    "turns": [
        {"role": "user"|"assistant", "content": str, "turn_idx": int},
        ...
    ]
}
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from tqdm.auto import tqdm

from src.data_generation.actorattack_runner import ActorAttackRunner
from src.data_generation.crescendo_runner import CrescendoRunner
from src.data_generation.xteaming_runner import XTeamingRunner

logger = logging.getLogger(__name__)

FRAMEWORK_RUNNERS = {
    "crescendo":   CrescendoRunner,
    "actorattack": ActorAttackRunner,
    "xteaming":    XTeamingRunner,
}


async def generate_all(
    objectives: list[dict],
    framework: str,
    model_shortname: str,
    target_model_id: str,
    n_turns: int = 3,
    n_attempts: int = 1,
    output_dir: str | Path = "data/conversations/harmful",
    concurrency: int = 5,
    openai_api_key: str | None = None,
    vllm_base_url: str = "http://localhost:8000/v1",
    attacker_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    resume: bool = True,
    desc: str = "Conversations",
    stop_on_success: bool = True,
) -> list[dict]:
    """
    Run attacks for all objectives and save one JSON file per conversation.

    Args:
        objectives: List of dicts with keys "objective" and "pair_id".
        framework: Attack framework name ("crescendo", "actorattack", "xteaming").
        model_shortname: Target model shortname (e.g. "llama3_8b").
        target_model_id: Full HuggingFace model ID as served by vLLM.
        n_turns: Number of attack turns per conversation.
        n_attempts: Number of independent attack attempts per objective.
        output_dir: Directory to save conversation JSON files.
        concurrency: Max simultaneous attack conversations.
        openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        vllm_base_url: Local vLLM server URL.
        attacker_model: OpenAI model for question generation.
        judge_model: OpenAI model for verdict evaluation.
        resume: If True, skip objectives that already have saved files.

    Returns:
        List of all conversation dicts.
    """
    if framework not in FRAMEWORK_RUNNERS:
        raise ValueError(f"Unknown framework '{framework}'. Choose from: {list(FRAMEWORK_RUNNERS)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)

    runner = FRAMEWORK_RUNNERS[framework](
        attacker_model=attacker_model,
        judge_model=judge_model,
        target_model_id=target_model_id,
        openai_api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        vllm_base_url=vllm_base_url,
        vllm_api_key=os.environ.get("VLLM_API_KEY", "token-abc123"),
        attacker_semaphore=asyncio.Semaphore(concurrency * 2),
        judge_semaphore=asyncio.Semaphore(concurrency * 2),
        target_semaphore=asyncio.Semaphore(concurrency),
    )

    async def run_one(objective: str, pair_id: int, attempt: int) -> dict | None:
        save_path = (
            output_dir
            / f"{framework}_{model_shortname}_pair{pair_id:03d}_attempt{attempt}.json"
        )
        if resume and save_path.exists():
            logger.info(f"Skipping pair {pair_id} attempt {attempt} (already exists)")
            with open(save_path) as f:
                return json.load(f)

        async with semaphore:
            logger.info(
                f"[{framework}] pair_id={pair_id} attempt={attempt} "
                f"objective={objective[:60]}..."
            )
            try:
                kwargs = dict(
                    objective=objective,
                    objective_pair_id=pair_id,
                    model_shortname=model_shortname,
                    n_turns=n_turns,
                )
                if framework in ("crescendo", "xteaming"):
                    kwargs["stop_on_success"] = stop_on_success
                conv = await runner.run_attack(**kwargs)
                conv["attempt"] = attempt
                with open(save_path, "w") as f:
                    json.dump(conv, f, indent=2)
                logger.info(
                    f"[{framework}] pair_id={pair_id} attempt={attempt} -> "
                    f"{conv['verdict']} ({conv['executed_turns']}/{n_turns} turns)"
                )
                return conv
            except Exception as e:
                logger.error(
                    f"[{framework}] pair_id={pair_id} attempt={attempt} FAILED: {e}",
                    exc_info=True,
                )
                return None

    tasks = [
        run_one(obj["objective"], obj["pair_id"], attempt)
        for obj in objectives
        for attempt in range(1, n_attempts + 1)
    ]

    results = []
    pbar = tqdm(total=len(tasks), desc=desc, unit="conv")
    for fut in asyncio.as_completed(tasks):
        result = await fut
        results.append(result)
        pbar.update(1)
    pbar.close()

    return [r for r in results if r is not None]
