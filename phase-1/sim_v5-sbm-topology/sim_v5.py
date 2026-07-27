"""
sim_v5.py — Decentralised Consensus Learning Network (Phase 1 PoC, Run 5)

Model: scalar beliefs, competence-weighted trust, bounded confidence (D-001),
       Stochastic block model topology. No idea objects (Phase 2).

Parameters (changes from Run 4 in brackets):
  N=120, M=3, T=500, alpha=0.3, eta=0.15, lambda=0.1, epsilon=0.15,
  noise=Uniform(-0.3, 0.3), seed=42
  Graph: StochasticBlockModel(4 communities x 30 nodes) [was Watts-Strogatz]
    p_in=0.1  (intra-community connection probability)
    p_out=0.01 (inter-community connection probability)
    Expected avg degree ≈ 29*0.1 + 90*0.01 = 3.8 (cf. WS k=4, avg degree=4.0)

Comparison questions vs Run 4:
  - Does block structure slow convergence? (tests Run 4 t=17 seed-artefact hypothesis)
  - Does inter-community barrier reduce SME participation below Run 4's 100%?
  - Do competence distributions show community clustering not seen in Run 4?
  Nodes are ordered by community: 0-29=C1, 30-59=C2, 60-89=C3, 90-119=C4.
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_HERE = os.path.dirname(os.path.abspath(__file__))
# ── Parameters ───────────────────────────────────────────────────────────────

N = 120
M = 3
T = 500
ALPHA = 0.3             # social learning rate
ETA = 0.15              # innovation rate
LAMBDA = 0.1            # competence plasticity
EPSILON = 0.15          # bounded confidence threshold (D-001)
SME_PCT = 90            # top-10% threshold for SME status
ENTROPY_THRESHOLD = 0.1 # entropy below this on all topics = converged

BLOCK_SIZES = [30, 30, 30, 30]   # 4 communities of 30 nodes
P_IN  = 0.10                      # intra-community edge probability
P_OUT = 0.01                      # inter-community edge probability
COMMUNITY_BOUNDARIES = [30, 60, 90]  # node indices where communities divide

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

# ── Graph (stochastic block model) ────────────────────────────────────────────

n_blocks = len(BLOCK_SIZES)
probs = [
    [P_IN if r == c else P_OUT for c in range(n_blocks)]
    for r in range(n_blocks)
]
G = nx.stochastic_block_model(BLOCK_SIZES, probs, seed=42)
A = nx.to_numpy_array(G)   # shape (N, N), symmetric, 0-diagonal

degrees = np.array([d for _, d in G.degree()])
print("-- Graph: Stochastic Block Model --")
print(f"  Communities: {len(BLOCK_SIZES)} x {BLOCK_SIZES[0]} nodes | "
      f"p_in={P_IN}  p_out={P_OUT}")
print(f"  Edges: {G.number_of_edges()} | "
      f"Avg degree: {degrees.mean():.2f} (cf. WS k=4 avg degree=4.0) | "
      f"Min: {degrees.min()}  Max: {degrees.max()}")

# ── Initial state ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
B = rng.uniform(0.0, 1.0, (N, M))    # beliefs[node, topic]
C = rng.uniform(0.1, 0.9, (N, M))    # competence[node, topic]

# ── Diagnostic: filter activity at t=0 ──────────────────────────────────────

print("\n-- Diagnostic: bounded confidence filter activity at t=0 --")
for k in range(M):
    diffs = np.abs(B[:, k][:, np.newaxis] - B[:, k][np.newaxis, :])
    within_eps_v1 = (diffs <= 0.3).sum() - N   # exclude self-pairs
    within_eps_v5 = (diffs <= EPSILON).sum() - N
    total_pairs = N * (N - 1)
    print(f"{TOPIC_LABELS[k]}: pairs within eps=0.3: "
          f"{within_eps_v1}/{total_pairs} "
          f"({100*within_eps_v1/total_pairs:.1f}%) | "
          f"within eps=0.15: "
          f"{within_eps_v5}/{total_pairs} "
          f"({100*within_eps_v5/total_pairs:.1f}%)")

# ── History ───────────────────────────────────────────────────────────────────

B_hist = np.empty((T + 1, N, M))
C_hist = np.empty((T + 1, N, M))
SME_hist = np.zeros((T + 1, N, M), dtype=bool)

B_hist[0] = B.copy()
C_hist[0] = C.copy()


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_trust(A, c_k):
    """
    Competence-weighted, normalised trust matrix for one topic.
    w_tilde_ij = a_ij * c_j; then row-normalise.
    Falls back to uniform-over-adjacency for isolated nodes.
    """
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    fallback = A / np.where(A.sum(axis=1, keepdims=True) == 0, 1,
                            A.sum(axis=1, keepdims=True))
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence(W, b_k, epsilon):
    """
    Zero out trust to neighbours whose belief differs by more than epsilon (D-001).
    Re-normalise rows; nodes with no in-epsilon neighbours keep their own belief.
    """
    mask = (np.abs(b_k[:, np.newaxis] - b_k[np.newaxis, :]) <= epsilon).astype(float)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit(x):
    """Map array to [0, 1] via min-max; return 0.5 if constant."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


def belief_entropy(b_k, bins=10):
    """Shannon entropy of belief distribution for one topic."""
    counts, _ = np.histogram(b_k, bins=bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ── Simulation loop ───────────────────────────────────────────────────────────

for t in range(T):
    B_new = np.empty_like(B)
    C_new = np.empty_like(C)

    for k in range(M):
        W_k = compute_trust(A, C[:, k])
        W_k = apply_bounded_confidence(W_k, B[:, k], EPSILON)

        social = W_k @ B[:, k]
        u = rng.uniform(-0.3, 0.3, N)
        B_new[:, k] = np.clip(
            (1 - ALPHA - ETA) * B[:, k] + ALPHA * social + ETA * u,
            0.0, 1.0
        )

        diff_sq = (B[:, k][:, np.newaxis] - B[:, k][np.newaxis, :]) ** 2
        delta = -(W_k * diff_sq).sum(axis=1)
        C_new[:, k] = np.clip(
            (1 - LAMBDA) * C[:, k] + LAMBDA * minmax_unit(delta),
            0.0, 1.0
        )

    B, C = B_new, C_new
    B_hist[t + 1] = B
    C_hist[t + 1] = C

    for k in range(M):
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[t + 1, :, k] = C[:, k] >= threshold


# ── Derived metrics ───────────────────────────────────────────────────────────

entropy_hist = np.array(
    [[belief_entropy(B_hist[t, :, k]) for k in range(M)] for t in range(T + 1)]
)   # shape (T+1, M)

sme_fraction = SME_hist.mean(axis=0)   # (N, M)


def cumulative_unique_sme(k):
    seen = set()
    result = []
    for t in range(T + 1):
        seen.update(np.where(SME_hist[t, :, k])[0].tolist())
        result.append(len(seen))
    return result


# ── Post-convergence rotation analysis ───────────────────────────────────────

converged_steps = np.where(
    np.all(entropy_hist < ENTROPY_THRESHOLD, axis=1)
)[0]
t_conv = int(converged_steps[0]) if len(converged_steps) > 0 else None

post_conv_new_sme = {}
if t_conv is not None:
    for k in range(M):
        pre_ever = set(np.where(np.any(SME_hist[:t_conv, :, k], axis=0))[0])
        post_ever = set(np.where(np.any(SME_hist[t_conv:, :, k], axis=0))[0])
        post_conv_new_sme[k] = len(post_ever - pre_ever)

# ── Per-community SME participation ──────────────────────────────────────────

community_edges = [0] + COMMUNITY_BOUNDARIES + [N]
community_labels = [f"C{i+1}({community_edges[i]}-{community_edges[i+1]-1})"
                    for i in range(len(BLOCK_SIZES))]

community_sme = {}   # (community_idx, topic) -> ever-SME count in that community
for i in range(len(BLOCK_SIZES)):
    lo, hi = community_edges[i], community_edges[i + 1]
    for k in range(M):
        ever = int(np.any(SME_hist[:, lo:hi, k], axis=0).sum())
        community_sme[(i, k)] = ever


# ── Console summary ───────────────────────────────────────────────────────────

print(f"\n-- sim_v5.py complete --")
print(f"N={N}  M={M}  T={T}  alpha={ALPHA}  eta={ETA}  "
      f"lambda={LAMBDA}  epsilon={EPSILON}  graph=StochasticBlockModel  "
      f"noise=Uniform(-0.3,0.3)")

print("\nBelief entropy (t=0 -> t=500):")
for k in range(M):
    print(f"  {TOPIC_LABELS[k]}: {entropy_hist[0, k]:.3f} -> {entropy_hist[-1, k]:.3f}")

if t_conv is not None:
    print(f"\nConvergence (all topics entropy < {ENTROPY_THRESHOLD}): t={t_conv} "
          f"(Run 4 WS: t=17)")
else:
    print(f"\nConvergence: NOT reached by t={T} (Run 4 WS: t=17)")

print(f"\nSME participation — overall (target: >{int(0.6*N)}/{N} = >60%):")
for k in range(M):
    ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
    pct = 100 * ever / N
    top = np.argsort(sme_fraction[:, k])[-3:][::-1]
    target_met = "PASS" if ever > 0.6 * N else "FAIL"
    print(f"  {TOPIC_LABELS[k]}: {ever}/{N} ({pct:.1f}%) [{target_met}] | "
          f"top holders: nodes {top.tolist()} "
          f"({sme_fraction[top, k].round(2).tolist()})")

print("\nSME participation — per community (30 nodes each):")
for i in range(len(BLOCK_SIZES)):
    row = f"  {community_labels[i]}: "
    parts = []
    for k in range(M):
        ever = community_sme[(i, k)]
        parts.append(f"{TOPIC_LABELS[k]}: {ever}/30 ({100*ever/30:.0f}%)")
    print(row + " | ".join(parts))

print("\nPost-convergence rotation (new SME holders appearing after t_conv):")
if t_conv is not None:
    for k in range(M):
        pct = 100 * post_conv_new_sme[k] / N
        print(f"  {TOPIC_LABELS[k]}: {post_conv_new_sme[k]} new nodes entered SME "
              f"after t={t_conv} ({pct:.1f}%) (Run 4: 88-94 nodes, 73-78%)")
else:
    print("  N/A — convergence not reached")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle(
    "sim_v5.py — Decentralised Consensus Learning Network\n"
    "N=120, M=3, T=500 | SBM(4x30, p_in=0.10, p_out=0.01) | "
    "eps=0.15  eta=0.15  noise=U(-0.3,0.3)",
    fontsize=10
)

# Row 0: belief distributions at t=0, 250, 500
for col, snap in enumerate([0, 250, 500]):
    ax = fig.add_subplot(gs[0, col])
    for k in range(M):
        ax.hist(B_hist[snap, :, k], bins=10, range=(0, 1),
                alpha=0.6, color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
    ax.set_title(f"Belief distribution  t={snap}", fontsize=9)
    ax.set_xlabel("Belief value", fontsize=8)
    ax.set_ylabel("Nodes", fontsize=8)
    ax.legend(fontsize=7)

# Row 1 left+centre: belief entropy over time; mark convergence point
ax_ent = fig.add_subplot(gs[1, :2])
for k in range(M):
    ax_ent.plot(entropy_hist[:, k], color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
if t_conv is not None:
    ax_ent.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                   label=f"converged t={t_conv}")
ax_ent.axhline(ENTROPY_THRESHOLD, color="grey", linestyle=":", linewidth=0.8)
ax_ent.set_title("Belief entropy over time", fontsize=9)
ax_ent.set_xlabel("Timestep", fontsize=8)
ax_ent.set_ylabel("Shannon entropy", fontsize=8)
ax_ent.legend(fontsize=7)

# Row 1 right: final competence heatmap
ax_comp = fig.add_subplot(gs[1, 2])
im = ax_comp.imshow(C_hist[-1], aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
for b in COMMUNITY_BOUNDARIES:
    ax_comp.axhline(b - 0.5, color="white", linewidth=0.8, linestyle="--")
ax_comp.set_title("Competence at t=500", fontsize=9)
ax_comp.set_xlabel("Topic", fontsize=8)
ax_comp.set_ylabel("Node", fontsize=8)
ax_comp.set_xticks(range(M))
ax_comp.set_xticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im, ax=ax_comp, fraction=0.046, pad=0.04)

# Row 2 left+centre: SME fraction heatmap with community boundary markers
ax_sme = fig.add_subplot(gs[2, :2])
im2 = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
for b in COMMUNITY_BOUNDARIES:
    ax_sme.axvline(b - 0.5, color="white", linewidth=1.0, linestyle="--")
ax_sme.set_title("SME fraction (proportion of timesteps as SME) | dashes = community bounds",
                 fontsize=9)
ax_sme.set_xlabel("Node", fontsize=8)
ax_sme.set_ylabel("Topic", fontsize=8)
ax_sme.set_yticks(range(M))
ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im2, ax=ax_sme, fraction=0.046, pad=0.04)

# Row 2 right: cumulative unique SME holders; mark convergence and 60% target
ax_turn = fig.add_subplot(gs[2, 2])
for k in range(M):
    ax_turn.plot(cumulative_unique_sme(k), color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
if t_conv is not None:
    ax_turn.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                    label=f"converged t={t_conv}")
ax_turn.axhline(0.6 * N, color="grey", linestyle=":", linewidth=0.8,
                label="60% target")
ax_turn.set_title("Cumulative unique SME holders", fontsize=9)
ax_turn.set_xlabel("Timestep", fontsize=8)
ax_turn.set_ylabel("Unique nodes ever SME", fontsize=8)
ax_turn.legend(fontsize=7)

plt.savefig(os.path.join(_HERE, "sim_v5_plots.png"), dpi=150, bbox_inches="tight")
print(f"Plots saved -> {os.path.join(_HERE, 'sim_v5_plots.png')}")

# ── CSV export (Session 52 backfill — same values as the console printout above) ──

import csv

csv_path = os.path.join(_HERE, "sim_v5_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "scope", "community", "topic", "entropy_t0", "entropy_t500", "t_conv",
        "ever_sme", "ever_sme_pct", "target_met",
        "top1_node", "top1_frac", "top2_node", "top2_frac", "top3_node", "top3_frac",
        "post_conv_new_sme", "post_conv_pct",
    ])
    # Overall (per-topic) rows — same values as the "SME participation -- overall" block
    for k in range(M):
        ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
        pct = 100 * ever / N
        top = np.argsort(sme_fraction[:, k])[-3:][::-1]
        target_met = "PASS" if ever > 0.6 * N else "FAIL"
        row = [
            "overall", "", TOPIC_LABELS[k],
            f"{entropy_hist[0, k]:.4f}", f"{entropy_hist[-1, k]:.4f}",
            t_conv if t_conv is not None else "NA",
            ever, f"{pct:.2f}", target_met,
        ]
        for node, frac in zip(top.tolist(), sme_fraction[top, k].tolist()):
            row += [node, f"{frac:.4f}"]
        if t_conv is not None:
            post_pct = 100 * post_conv_new_sme[k] / N
            row += [post_conv_new_sme[k], f"{post_pct:.2f}"]
        else:
            row += ["NA", "NA"]
        writer.writerow(row)
    # Per-community rows — same values as the "SME participation -- per community" block
    # (only ever-SME count is computed per community; other columns are NA, not invented)
    for i in range(len(BLOCK_SIZES)):
        for k in range(M):
            ever = community_sme[(i, k)]
            writer.writerow([
                "community", community_labels[i], TOPIC_LABELS[k],
                "NA", "NA", t_conv if t_conv is not None else "NA",
                ever, f"{100*ever/30:.2f}", "NA",
                "NA", "NA", "NA", "NA", "NA", "NA",
                "NA", "NA",
            ])
print(f"CSV saved -> {csv_path}")

plt.show()
