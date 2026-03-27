"""
Part 3: Cross-framework probe transfer.

Trains the MLP probe on jailbroken vs. benign conversations from one attack
framework and evaluates on conversations from the other two.

Six transfer directions:
  Crescendo -> ActorAttack, Crescendo -> X-Teaming,
  ActorAttack -> Crescendo, ActorAttack -> X-Teaming,
  X-Teaming -> Crescendo, X-Teaming -> ActorAttack

Repeated with LAT reading vector.
"""

FRAMEWORKS = ["crescendo", "actorattack", "xteaming"]
TRANSFER_DIRECTIONS = [(s, t) for s in FRAMEWORKS for t in FRAMEWORKS if s != t]


def run_all_transfer_experiments(
    representations_jsonl: str,
    model: str,
    layer: int,
    aggregation: str = "mean_pool",
) -> dict:
    """
    Returns {(source, target): auc} for all six transfer directions,
    for both MLP probe and LAT reading vector.
    """
    raise NotImplementedError
