"""
Part 1: Topic confound test.

Trains the MLP probe within JBB matched pairs — for each objective pair,
trains on jailbroken vs. benign representations from that pair only, evaluates
separability at k=1. Compares within-pair AUC to across-pair AUC.

Repeated for LAT reading vector: computes difference vectors within each matched
pair and evaluates whether reading vector separates at k=1 when computed within pairs.

Run separately per attack framework and per model.
"""


def run_topic_confound_analysis(
    representations_jsonl: str,
    model: str,
    framework: str,
    layer: int,
    aggregation: str = "mean_pool",
) -> dict:
    """
    Returns dict with keys:
      within_pair_aucs: {pair_id: auc}
      across_pair_auc: float
      lat_within_pair_scores: {pair_id: {"jailbroken": float, "benign": float}}
    """
    raise NotImplementedError
