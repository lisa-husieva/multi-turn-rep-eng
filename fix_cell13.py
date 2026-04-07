import json

with open('notebooks/04_rep_analysis.ipynb') as f:
    nb = json.load(f)

new_src = r"""# ── Per-category direction cosine similarity (Zhao et al. Fig 14a / Fig 9a) ──
#
# For each JBB harm category, compute:
#   v_harmful_c[l]  = normalize( mean(h_inst, harmful, cat c) - mean(h_inst, benign, cat c) )
#   v_refusal_c[l]  = normalize( mean(h_post, refused_harmful, cat c) - mean(h_post, accepted_harmful, cat c) )
#
# Then for each category c report:
#   avg_cos_harmful = mean cosine similarity of v_harmful_c with all other categories' v_harmful
#   avg_cos_refusal = mean cosine similarity of v_refusal_c with all other categories' v_refusal
#
# Average taken over middle layers (10-25) where harmfulness signal is clearest.

# ── Build pair_id → JBB harm category mapping ─────────────────────────────────
# Use single-turn metadata which already has the JBB category labels.
# Note: traj meta has a 'category' column but that's the behavioral label
# (accepted_harmful etc.) set in cell 3 — not the JBB category.
pair_to_cat = dict(zip(
    st["harmful"]["meta"]["pair_id"],
    st["harmful"]["meta"]["category"],
))
categories  = sorted(set(pair_to_cat.values()))
MIDDLE_LAYERS = list(range(9, 25))   # 0-indexed layers 10-25

FW = "crescendo"  # use crescendo; could pool frameworks for robustness


def compute_per_category_directions(framework):
    """
    Returns:
        v_harm_cat: dict {jbb_cat: (32, 4096) unit vector}
        v_ref_cat:  dict {jbb_cat: (32, 4096) unit vector}
    """
    v_harm_cat = {}
    v_ref_cat  = {}

    h_harm_all, m_harm_all = option_a(
        traj[(framework, "harmful")]["h_inst"],
        traj[(framework, "harmful")]["meta"],
        data_split="train",
    )
    h_beni_all, m_beni_all = option_a(
        traj[(framework, "benign")]["h_inst"],
        traj[(framework, "benign")]["meta"],
        data_split="train",
    )
    hp_harm_all, _ = option_a(
        traj[(framework, "harmful")]["h_post"],
        traj[(framework, "harmful")]["meta"],
        data_split="train",
    )

    # Map pair_id → JBB category for the returned subsets
    harm_jbb = m_harm_all["pair_id"].map(pair_to_cat)
    beni_jbb = m_beni_all["pair_id"].map(pair_to_cat)

    for cat in categories:
        # ── v_harmful_c ──────────────────────────────────────────────────────
        mask_h = (harm_jbb == cat).values
        mask_b = (beni_jbb == cat).values
        if mask_h.sum() == 0 or mask_b.sum() == 0:
            continue
        mu_h = h_harm_all[mask_h].mean(axis=0)   # (32, 4096)
        mu_b = h_beni_all[mask_b].mean(axis=0)
        diff = mu_h - mu_b
        norms = np.linalg.norm(diff, axis=-1, keepdims=True)
        v_harm_cat[cat] = diff / (norms + 1e-12)

        # ── v_refusal_c ───────────────────────────────────────────────────────
        mask_ref = ((harm_jbb == cat) & (m_harm_all["attack_success"] == False)).values
        mask_acc = ((harm_jbb == cat) & (m_harm_all["attack_success"] == True)).values
        if mask_ref.sum() == 0 or mask_acc.sum() == 0:
            v_ref_cat[cat] = v_harm_cat[cat] * 0   # fallback zeros
            continue
        mu_ref = hp_harm_all[mask_ref].mean(axis=0)
        mu_acc = hp_harm_all[mask_acc].mean(axis=0)
        diff_r = mu_ref - mu_acc
        norms_r = np.linalg.norm(diff_r, axis=-1, keepdims=True)
        v_ref_cat[cat] = diff_r / (norms_r + 1e-12)

    return v_harm_cat, v_ref_cat


v_harm_cat, v_ref_cat = compute_per_category_directions(FW)
print(f"Computed per-category directions for {len(v_harm_cat)} categories")

# ── Compute average cross-category cosine similarities ────────────────────────
def avg_cross_cat_cosine(direction_dict, layers):
    """
    For each category c, average cosine similarity with all other categories,
    averaged over the given layer indices.
    Returns {cat: mean_cosine}
    """
    cats = list(direction_dict.keys())
    result = {}
    for c in cats:
        sims = []
        for c2 in cats:
            if c2 == c:
                continue
            cos_layers = []
            for l in layers:
                v1 = direction_dict[c][l]
                v2 = direction_dict[c2][l]
                cos_layers.append(float(np.dot(v1, v2)))
            sims.append(np.mean(cos_layers))
        result[c] = float(np.mean(sims))
    return result


cos_harmful = avg_cross_cat_cosine(v_harm_cat, MIDDLE_LAYERS)
cos_refusal = avg_cross_cat_cosine(v_ref_cat,  MIDDLE_LAYERS)

# ── Plot ──────────────────────────────────────────────────────────────────────
short = {
    "Disinformation":               "Disinform.",
    "Economic harm":                "Economic",
    "Expert advice":                "Expert\nAdvice",
    "Fraud/Deception":              "Fraud/\nDeception",
    "Government decision-making":   "Government",
    "Harassment/Discrimination":    "Harassment",
    "Malware/Hacking":              "Malware",
    "Physical harm":                "Physical\nHarm",
    "Privacy":                      "Privacy",
    "Sexual/Adult content":         "Sexual/\nAdult",
}

cats_sorted = sorted(categories, key=lambda c: cos_harmful[c])
labels = [short.get(c, c) for c in cats_sorted]
x = np.arange(len(cats_sorted))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars_h = ax.bar(x - width/2,
                [cos_harmful[c] for c in cats_sorted],
                width, label="Harmfulness direction", color="#d62728", alpha=0.85)
bars_r = ax.bar(x + width/2,
                [cos_refusal[c] for c in cats_sorted],
                width, label="Refusal direction", color="#1f77b4", alpha=0.85)

for bar in bars_h:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
for bar in bars_r:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Average cosine similarity with other categories")
ax.set_title(
    f"Per-category direction similarity — {FW} train set (layers {MIDDLE_LAYERS[0]+1}–{MIDDLE_LAYERS[-1]+1})\n"
    "Harmfulness directions are more category-specific; refusal directions are more universal"
)
ax.set_ylim(0.0, 1.1)
mean_h = np.mean(list(cos_harmful.values()))
mean_r = np.mean(list(cos_refusal.values()))
ax.axhline(mean_h, color="#d62728", lw=1, ls="--", alpha=0.6)
ax.axhline(mean_r, color="#1f77b4", lw=1, ls="--", alpha=0.6)
ax.legend([bars_h, bars_r,
           plt.Line2D([0],[0],color="#d62728",lw=1,ls="--"),
           plt.Line2D([0],[0],color="#1f77b4",lw=1,ls="--")],
          ["Harmfulness direction", "Refusal direction",
           f"mean harmful = {mean_h:.2f}",
           f"mean refusal = {mean_r:.2f}"],
          fontsize=9)

plt.tight_layout()
plt.savefig("../figures/04_category_direction_cosines.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Mean harmful cosine: {mean_h:.3f}")
print(f"Mean refusal cosine: {mean_r:.3f}")
print("Saved figures/04_category_direction_cosines.png")
"""

nb['cells'][13]['source'] = new_src

with open('notebooks/04_rep_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Done — cell 13 updated")
