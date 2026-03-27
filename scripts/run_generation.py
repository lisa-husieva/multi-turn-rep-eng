"""
Load JBB-Behaviors and run multi-turn attack generation.

Before running:
  1. Copy .env.example to .env and fill in OPENAI_API_KEY
  2. Start the vLLM server: bash scripts/launch_vllm.sh
  3. Install dependencies: pip install -r requirements.txt

Usage:
  python scripts/run_generation.py [options]

Examples:
  # Smoke test: 3 objectives, 3 turns, 1 attempt
  python scripts/run_generation.py --max-objectives 3 --n-turns 3

  # Full run: all 100 JBB objectives, 3 turns, 1 attempt, Crescendo
  python scripts/run_generation.py --n-turns 3 --framework crescendo

  # Multiple attempts per objective
  python scripts/run_generation.py --n-turns 3 --n-attempts 3
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import dotenv
from datasets import load_dataset

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generation.generate_attacks import generate_all

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jbb_objectives(max_objectives: int | None = None) -> list[dict]:
    """
    Load harmful objectives from JBB-Behaviors.

    Returns list of dicts: [{"objective": str, "pair_id": int, "category": str}, ...]
    """
    logger.info("Loading JBB-Behaviors dataset...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", split="harmful")

    objectives = []
    for i, row in enumerate(ds):
        if max_objectives is not None and i >= max_objectives:
            break
        objectives.append({
            "objective": row["Goal"],
            "pair_id": i,
            "category": row.get("Category", "unknown"),
            "behavior_id": row.get("BehaviorID", str(i)),
        })

    logger.info(f"Loaded {len(objectives)} objectives.")
    return objectives


def save_manifest(objectives: list[dict], output_dir: Path) -> None:
    """Save a manifest of all objectives to output_dir/objectives_manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "objectives_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(objectives, f, indent=2)
    logger.info(f"Objectives manifest saved to {manifest_path}")


def print_summary(conversations: list[dict]) -> None:
    verdicts = [c["verdict"] for c in conversations]
    total = len(verdicts)
    jailbroken = verdicts.count("jailbroken")
    near_miss = verdicts.count("near_miss")
    refusal = verdicts.count("refusal")

    print("\n" + "=" * 50)
    print(f"Generation complete. {total} conversations saved.")
    print(f"  Jailbroken : {jailbroken:3d} ({100*jailbroken/total:.1f}%)")
    print(f"  Near-miss  : {near_miss:3d} ({100*near_miss/total:.1f}%)")
    print(f"  Refusal    : {refusal:3d} ({100*refusal/total:.1f}%)")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-turn attack generation.")

    parser.add_argument(
        "--framework",
        default="crescendo",
        choices=["crescendo"],  # extend as more runners are implemented
        help="Attack framework to use (default: crescendo).",
    )
    parser.add_argument(
        "--model",
        default="llama3_8b",
        help="Target model shortname from configs/models.yaml (default: llama3_8b).",
    )
    parser.add_argument(
        "--target-model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Full HuggingFace model ID as served by vLLM.",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=3,
        help="Number of attack turns per conversation (default: 3).",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=1,
        help="Independent attack attempts per objective (default: 1).",
    )
    parser.add_argument(
        "--max-objectives",
        type=int,
        default=None,
        help="Limit number of JBB objectives (default: all 100).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max simultaneous conversations (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/conversations/harmful",
        help="Directory for saved conversation files.",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM server URL (default: $VLLM_BASE_URL or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--attacker-model",
        default=os.environ.get("ATTACKER_MODEL", "gpt-4o-mini"),
        help="OpenAI model for question generation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", "gpt-4o"),
        help="OpenAI model for verdict evaluation (default: gpt-4o).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run even if output files already exist.",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    objectives = load_jbb_objectives(max_objectives=args.max_objectives)
    output_dir = Path(args.output_dir)
    save_manifest(objectives, output_dir)

    logger.info(
        f"Starting generation: framework={args.framework}, model={args.model}, "
        f"n_turns={args.n_turns}, n_attempts={args.n_attempts}, "
        f"concurrency={args.concurrency}, objectives={len(objectives)}"
    )

    conversations = await generate_all(
        objectives=objectives,
        framework=args.framework,
        model_shortname=args.model,
        target_model_id=args.target_model_id,
        n_turns=args.n_turns,
        n_attempts=args.n_attempts,
        output_dir=output_dir,
        concurrency=args.concurrency,
        openai_api_key=openai_api_key,
        vllm_base_url=args.vllm_url,
        attacker_model=args.attacker_model,
        judge_model=args.judge_model,
        resume=not args.no_resume,
    )

    print_summary(conversations)


if __name__ == "__main__":
    asyncio.run(main())
