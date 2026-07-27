"""
sim_v6.py — Decentralised Consensus Learning Network (Phase 1 PoC, Run 6)

Model: scalar beliefs, competence-weighted trust, bounded confidence (D-001),
       Stochastic block model topology. No idea objects (Phase 2).
       Sparse matrix implementation — scales to N=10,000.

Parameters (changes from Run 5 in brackets):
  N=10000 [was 120], M=3, T=500, alpha=0.3, eta=0.15, lambda=0.1, epsilon=0.15,
  noise=Uniform(-0.3, 0.3), seed=42
  Graph: StochasticBlockModel(500 communities x 20 nodes)
    p_in=0.30  (intra-community; 20 nodes x 0.30 ≈ 6 intra neighbours)
    p_out=0.001 (inter-community; 9980 nodes x 0.001 ≈ 10 inter neighbours)
    Expected avg degree ≈ 6 + 10 = 16

Memory budget (history arrays):
  B_hist  (501 x 10000 x 3 float64) ≈ 115 MB
  C_hist  (501 x 10000 x 3 float64) ≈ 115 MB
  SME_hist (501 x 10000 x 3 bool)   ≈  15 MB
  Total                              ≈ 245 MB

Sparse design decisions:
  - A stored as scipy.sparse CSR — never materialised as dense N×N
  - compute_trust: sparse column-scale + row-normalise
  - apply_bounded_confidence: CSR row iteration — no dense N×N mask
  - peer_consistency: sparse matvec identity (avoids O(N²) diff_sq)
"""

import os
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


_HERE = os.path.dirname(os.path.abspath(__file__))
# ── Parameters ───────────────────────────────────────────────────────────────

N = 10_000
M = 3
T = 500
ALPHA = 0.3             # social learning rate
ETA = 0.15              # innovation rate
LAMBDA = 0.1            # competence plasticity
EPSILON = 0.15          # bounded confidence threshold (D-001)
SME_PCT = 90            # top-10% threshold for SME status
ENTROPY_THRESHOLD = 0.1 # entropy below this on all topics = converged

BLOCK_SIZES = [20] * 500   # 500 communities of 20 nodes
P_IN  = 0.30               # intra-community edge probability
P_OUT = 0.001              # inter-community edge probability

# Plot community boundaries: every 100 communities = every 2000 nodes
PLOT_BOUNDARIES = list(range(2000, N, 2000))   # [2000, 4000, 6000, 8000]

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

# ── Graph (stochastic block model) ────────────────────────────────────────────

n_blocks = len(BLOCK_SIZES)
probs = [
    [P_IN if r == c else P_OUT for c in range(n_blocks)]
    for r in range(n_blocks)
]
print("Building graph (500 communities x 20 nodes) ...")
G = nx.stochastic_block_model(BLOCK_SIZES, probs, seed=42)

# Convert directly to CSR sparse — never materialise dense N×N adjacency
A = nx.to_scipy_sparse_array(G, format='csr', dtype=np.float64)

degrees = np.array([d for _, d in G.degree()])
print("-- Graph: Stochastic Block Model --")
print(f"  Communities: {n_blocks} x {BLOCK_SIZES[0]} nodes | "
      f"p_in={P_IN}  p_out={P_OUT}")
print(f"  Edges: {G.number_of_edges()} | "
      f"Avg degree: {degrees.mean():.2f} | "
      f"Min: {degrees.min()}  Max: {degrees.max()}")
del G   # free NetworkX graph object — A is all we need

# ── Initial state ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
B = rng.uniform(0.0, 1.0, (N, M))    # beliefs[node, topic]
C = rng.uniform(0.1, 0.9, (N, M))    # competence[node, topic]

# ── Diagnostic: filter activity at t=0 ──────────────────────────────────────
# At N=10,000 a full N×N pairwise matrix is ~800MB — not materialised.
# Diagnostic is computed only over actual graph edges (the pairs that matter
# for the bounded confidence filter during simulation).

print("\n-- Diagnostic: bounded confidence filter activity at t=0 --")
print("  (computed over graph edges only — N×N pairwise matrix not materialised)")
rows, cols = A.nonzero()
directed_edge_count = len(rows)   # number of directed edges (both directions)
for k in range(M):
    diffs = np.abs(B[rows, k] - B[cols, k])
    within_v1 = int((diffs <= 0.3).sum())
    within_v6 = int((diffs <= EPSILON).sum())
    print(f"  {TOPIC_LABELS[k]}: edge pairs within eps=0.3: "
          f"{within_v1}/{directed_edge_count} "
          f"({100*within_v1/directed_edge_count:.1f}%) | "
          f"within eps=0.15: "
          f"{within_v6}/{directed_edge_count} "
          f"({100*within_v6/directed_edge_count:.1f}%)")

# ── History ───────────────────────────────────────────────────────────────────

B_hist = np.empty((T + 1, N, M), dtype=np.float64)
C_hist = np.empty((T + 1, N, M), dtype=np.float64)
SME_hist = np.zeros((T + 1, N, M), dtype=bool)

B_hist[0] = B.copy()
C_hist[0] = C.copy()


# ── Helper functions ──────────────────────────────────────────────────────────

def compute_trust(A, c_k):
    """
    Competence-weighted, normalised trust matrix for one topic.
    Returns sparse CSR matrix. No dense N×N materialised.
    """
    # Scale each column j of A by c_j: W_ij = a_ij * c_j
    W = A.multiply(c_k[np.newaxis, :])   # sparse elementwise broadcast
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv_sums = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    W = W.multiply(inv_sums[:, np.newaxis])
    # Fallback for zero-sum rows: uniform over adjacency
    # (rare in a well-connected SBM; leave as all-zero rows — peer_consistency
    # handles them explicitly)
    return W.tocsr()


def apply_bounded_confidence(W, b_k, epsilon):
    """
    Zero out trust to neighbours whose belief differs by more than epsilon.
    Operates on CSR data arrays row by row — no dense N×N mask materialised.
    """
    W = W.tocsr().copy()
    for i in range(W.shape[0]):
        start, end = W.indptr[i], W.indptr[i + 1]
        cols = W.indices[start:end]
        diffs = np.abs(b_k[i] - b_k[cols])
        W.data[start:end] *= (diffs <= epsilon).astype(np.float64)
    # Re-normalise rows
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv_sums = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    W = W.multiply(inv_sums[:, np.newaxis])
    return W.tocsr()


def peer_consistency(W, b_k):
    """
    Compute delta_i = -sum_j w_ij (b_i - b_j)^2 without materialising N×N.

    Expanding: sum_j w_ij(b_i - b_j)^2
             = b_i^2 * row_sum_i - 2*b_i*(W@b)_i + (W@b^2)_i
    For row-normalised W, row_sum_i = 1 for connected nodes.
    Zero-sum rows (no in-epsilon neighbours) are forced to delta=0.
    """
    Wb  = np.asarray(W @ b_k).ravel()
    Wb2 = np.asarray(W @ (b_k ** 2)).ravel()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    delta = -(b_k ** 2 - 2.0 * b_k * Wb + Wb2)
    delta = np.where(row_sums > 0, delta, 0.0)
    return delta


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

print(f"\nRunning simulation: N={N}, M={M}, T={T} ...")
for t in range(T):
    if t % 50 == 0:
        print(f"  t={t}/{T} ...", flush=True)

    B_new = np.empty_like(B)
    C_new = np.empty_like(C)

    for k in range(M):
        W_k = compute_trust(A, C[:, k])
        W_k = apply_bounded_confidence(W_k, B[:, k], EPSILON)

        # Belief update: decay + social consensus + innovation
        social = np.asarray(W_k @ B[:, k]).ravel()
        u = rng.uniform(-0.3, 0.3, N)
        B_new[:, k] = np.clip(
            (1 - ALPHA - ETA) * B[:, k] + ALPHA * social + ETA * u,
            0.0, 1.0
        )

        # Competence update: peer-consistency via sparse matvec (no N×N matrix)
        delta = peer_consistency(W_k, B[:, k])
        C_new[:, k] = np.clip(
            (1 - LAMBDA) * C[:, k] + LAMBDA * minmax_unit(delta),
            0.0, 1.0
        )

    B, C = B_new, C_new
    B_hist[t + 1] = B
    C_hist[t + 1] = C

    # SME status: top-10% competence per topic
    for k in range(M):
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[t + 1, :, k] = C[:, k] >= threshold

print(f"  t={T}/{T} ... done")

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

# ── Per-community SME participation (summary stats over 500 communities) ─────

community_size = BLOCK_SIZES[0]   # 20
community_ever_sme = np.zeros((n_blocks, M), dtype=int)
for i in range(n_blocks):
    lo = i * community_size
    hi = lo + community_size
    for k in range(M):
        community_ever_sme[i, k] = int(
            np.any(SME_hist[:, lo:hi, k], axis=0).sum()
        )


# ── Console summary ───────────────────────────────────────────────────────────

print(f"\n-- sim_v6.py complete --")
print(f"N={N}  M={M}  T={T}  alpha={ALPHA}  eta={ETA}  "
      f"lambda={LAMBDA}  epsilon={EPSILON}  "
      f"graph=SBM(500x20)  noise=Uniform(-0.3,0.3)")

print("\nBelief entropy (t=0 -> t=500):")
for k in range(M):
    print(f"  {TOPIC_LABELS[k]}: {entropy_hist[0, k]:.3f} -> {entropy_hist[-1, k]:.3f}")

if t_conv is not None:
    print(f"\nConvergence (all topics entropy < {ENTROPY_THRESHOLD}): t={t_conv} "
          f"(Run 5 SBM N=120: t=18)")
else:
    print(f"\nConvergence: NOT reached by t={T} (Run 5 SBM N=120: t=18)")

print(f"\nSME participation — overall (target: >{int(0.6*N)}/{N} = >60%):")
for k in range(M):
    ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
    pct = 100 * ever / N
    top = np.argsort(sme_fraction[:, k])[-3:][::-1]
    target_met = "PASS" if ever > 0.6 * N else "FAIL"
    print(f"  {TOPIC_LABELS[k]}: {ever}/{N} ({pct:.1f}%) [{target_met}] | "
          f"top holders: nodes {top.tolist()} "
          f"({sme_fraction[top, k].round(3).tolist()})")

print(f"\nSME participation — per community summary ({n_blocks} communities of "
      f"{community_size} nodes each):")
for k in range(M):
    col = community_ever_sme[:, k]
    mean_pct = 100 * col.mean() / community_size
    min_pct  = 100 * col.min()  / community_size
    max_pct  = 100 * col.max()  / community_size
    n_full   = int((col == community_size).sum())
    print(f"  {TOPIC_LABELS[k]}: mean {col.mean():.1f}/{community_size} "
          f"({mean_pct:.1f}%) | "
          f"min {col.min()}/{community_size} ({min_pct:.0f}%) | "
          f"max {col.max()}/{community_size} ({max_pct:.0f}%) | "
          f"full-participation communities: {n_full}/{n_blocks}")

print("\nPost-convergence rotation (new SME holders appearing after t_conv):")
if t_conv is not None:
    for k in range(M):
        pct = 100 * post_conv_new_sme[k] / N
        print(f"  {TOPIC_LABELS[k]}: {post_conv_new_sme[k]} new nodes entered SME "
              f"after t={t_conv} ({pct:.1f}%) (Run 5: 74-78%)")
else:
    print("  N/A — convergence not reached")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle(
    "sim_v6.py — Decentralised Consensus Learning Network\n"
    "N=10,000, M=3, T=500 | SBM(500x20, p_in=0.30, p_out=0.001) | "
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
# Community boundary lines every 2000 nodes (every 100 communities)
ax_comp = fig.add_subplot(gs[1, 2])
im = ax_comp.imshow(C_hist[-1], aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
for b in PLOT_BOUNDARIES:
    ax_comp.axhline(b - 0.5, color="white", linewidth=0.8, linestyle="--")
ax_comp.set_title("Competence at t=500", fontsize=9)
ax_comp.set_xlabel("Topic", fontsize=8)
ax_comp.set_ylabel("Node", fontsize=8)
ax_comp.set_xticks(range(M))
ax_comp.set_xticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im, ax=ax_comp, fraction=0.046, pad=0.04)

# Row 2 left+centre: SME fraction heatmap; boundaries every 2000 nodes
ax_sme = fig.add_subplot(gs[2, :2])
im2 = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
for b in PLOT_BOUNDARIES:
    ax_sme.axvline(b - 0.5, color="white", linewidth=1.0, linestyle="--")
ax_sme.set_title(
    "SME fraction (proportion of timesteps as SME) | dashes every 2000 nodes",
    fontsize=9
)
ax_sme.set_xlabel("Node", fontsize=8)
ax_sme.set_ylabel("Topic", fontsize=8)
ax_sme.set_yticks(range(M))
ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im2, ax=ax_sme, fraction=0.046, pad=0.04)

# Row 2 right: cumulative unique SME holders; convergence marker and 60% line
ax_turn = fig.add_subplot(gs[2, 2])
for k in range(M):
    ax_turn.plot(cumulative_unique_sme(k), color=TOPIC_COLORS[k],
                 label=TOPIC_LABELS[k])
if t_conv is not None:
    ax_turn.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                    label=f"converged t={t_conv}")
ax_turn.axhline(0.6 * N, color="grey", linestyle=":", linewidth=0.8,
                label="60% target")
ax_turn.set_title("Cumulative unique SME holders", fontsize=9)
ax_turn.set_xlabel("Timestep", fontsize=8)
ax_turn.set_ylabel("Unique nodes ever SME", fontsize=8)
ax_turn.legend(fontsize=7)

plt.savefig(os.path.join(_HERE, "sim_v6_plots.png"), dpi=150, bbox_inches="tight")
print(f"Plots saved -> {os.path.join(_HERE, 'sim_v6_plots.png')}")

# ── CSV export (Session 52 backfill — same values as the console printout above) ──

import csv

csv_path = os.path.join(_HERE, "sim_v6_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "scope", "topic", "entropy_t0", "entropy_t500", "t_conv",
        "ever_sme", "ever_sme_pct", "target_met",
        "top1_node", "top1_frac", "top2_node", "top2_frac", "top3_node", "top3_frac",
        "post_conv_new_sme", "post_conv_pct",
        "community_mean", "community_mean_pct", "community_min", "community_min_pct",
        "community_max", "community_max_pct", "community_n_full",
    ])
    # Overall (per-topic) rows -- same values as "SME participation -- overall"
    for k in range(M):
        ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
        pct = 100 * ever / N
        top = np.argsort(sme_fraction[:, k])[-3:][::-1]
        target_met = "PASS" if ever > 0.6 * N else "FAIL"
        row = [
            "overall", TOPIC_LABELS[k],
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
        row += ["NA", "NA", "NA", "NA", "NA", "NA", "NA"]
        writer.writerow(row)
    # Per-community summary rows -- same values as "SME participation -- per community summary"
    for k in range(M):
        col = community_ever_sme[:, k]
        mean_pct = 100 * col.mean() / community_size
        min_pct = 100 * col.min() / community_size
        max_pct = 100 * col.max() / community_size
        n_full = int((col == community_size).sum())
        writer.writerow([
            "community_summary", TOPIC_LABELS[k],
            "NA", "NA", t_conv if t_conv is not None else "NA",
            "NA", "NA", "NA",
            "NA", "NA", "NA", "NA", "NA", "NA",
            "NA", "NA",
            f"{col.mean():.2f}", f"{mean_pct:.2f}",
            int(col.min()), f"{min_pct:.2f}",
            int(col.max()), f"{max_pct:.2f}",
            n_full,
        ])
print(f"CSV saved -> {csv_path}")

plt.show()
