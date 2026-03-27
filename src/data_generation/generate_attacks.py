"""
Generate harmful multi-turn attack conversations using Crescendo, ActorAttack,
and X-Teaming via AJAR's shared interface.

For each (objective, framework, model) combination, runs n_attempts conversations
and saves all turns plus the AJAR verdict to data/conversations/.

Output schema per conversation (JSON):
{
    "conversation_id": str,
    "objective_pair_id": int,
    "objective": str,
    "attack_framework": str,
    "model": str,
    "verdict": str,          # jailbroken | refusal | near_miss
    "turns": [
        {"role": "user"|"assistant", "content": str},
        ...
    ]
}
"""


def run_attack(objective: str, framework: str, model_shortname: str, max_turns: int = 10) -> dict:
    """Run a single attack conversation via AJAR. Returns conversation dict."""
    raise NotImplementedError


def generate_all(
    objectives_path: str,
    frameworks: list[str],
    model_shortnames: list[str],
    n_attempts: int = 10,
    output_dir: str = "data/conversations/harmful",
) -> None:
    """Run all (objective, framework, model) combinations and save conversations."""
    raise NotImplementedError
