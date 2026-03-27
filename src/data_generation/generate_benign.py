"""
Generate matched benign multi-turn conversations.

One benign conversation per (objective_pair_id, model) — not triplicated across
frameworks since the benign side does not vary by attack strategy. Uses a
Crescendo-style escalating multi-turn structure targeting the JBB benign
counterpart objective.

Output schema matches generate_attacks.py with attack_framework="benign" and
verdict="benign".
"""


def generate_benign_conversation(
    benign_objective: str,
    objective_pair_id: int,
    model_shortname: str,
    max_turns: int = 10,
) -> dict:
    """Generate one benign multi-turn conversation. Returns conversation dict."""
    raise NotImplementedError


def generate_all(
    objective_pairs_path: str,
    model_shortnames: list[str],
    output_dir: str = "data/conversations/benign",
) -> None:
    """Generate benign conversations for all (objective_pair, model) combinations."""
    raise NotImplementedError
