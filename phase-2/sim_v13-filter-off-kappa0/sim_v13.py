"""
sim_v13.py — Decentralised Consensus Learning Network (Phase 2, Run 13)

Vector beliefs, Gaussian init, KAPPA=0.0 (no bounded confidence filter).

Run 12 showed that KAPPA=0.5 freezes social learning under Gaussian init:
with cosim ~0 at t=0, only ~15-20% of neighbour pairs pass the filter,
giving each node fewer than 1 effective interacting neighbour. Beliefs
barely moved (cosim ~0.0 -> 0.035 over T=200). No convergence. Oligarchy.

Run 13 removes the BC filter (KAPPA=0.0 passes all neighbours) to isolate
the effect of initialisation from the effect of the threshold. If convergence
and rotation return with KAPPA=0.0, the BC threshold is the root cause.

Direct comparison:
  Run 11 (uniform init, KAPPA=0.5): cosim ~0.75 at t=0, conv t=14
  Run 12 (Gaussian init, KAPPA=0.5): cosim ~0.00 at t=0, no conv, oligarchy
  Run 13 (Gaussian init, KAPPA=0.0): cosim ~0.00 at t=0, BC filter OFF

Parameters: identical to Run 12 except KAPPA=0.0.
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- Parameters ---------------------------------------------------------------

N = 30
M = 3
T = 200
D = 5               # belief vector dimension
ALPHA = 0.3         # social learning rate
ETA = 0.15          # innovation rate
LAMBDA = 0.1        # competence plasticity
# Two separate thresholds — do not confuse:
KAPPA = 0.0         # bounded confidence filter: 0.0 = all neighbours pass (filter OFF)
CONV_THRESHOLD = 0.99  # convergence detection only — never used inside the sim loop
SME_PCT = 90        # top-10% threshold for SME status

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

# -- Graph (Watts-Strogatz small-world) ----------------------------------------

G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=42)
A = nx.to_numpy_array(G)

degrees = np.array([d for _, d in G.degree()])
isolated = degrees == 0   # expected: none for WS k=4

# -- Initial state -------------------------------------------------------------

rng = np.random.default_rng(42)
B_raw = rng.standard_normal((N, M, D))
norms_init = np.linalg.norm(B_raw, axis=2, keepdims=True)
B = B_raw / np.where(norms_init > 0, norms_init, 1.0)   # unit vectors, shape (N, M, D)
C = rng.uniform(0.1, 0.9, (N, M))                        # competence stays scalar

# -- History -------------------------------------------------------------------

B_hist   = np.empty((T + 1, N, M, D))           # note extra D dimension
C_hist   = np.empty((T + 1, N, M))
SME_hist = np.zeros((T + 1, N, M), dtype=bool)

B_hist[0] = B.copy()
C_hist[0] = C.copy()


# -- Helper functions ----------------------------------------------------------

def compute_trust(A, c_k):
    """
    Competence-weighted, row-normalised trust matrix for one topic.
    Fallback to uniform-over-adjacency for isolated nodes (degree=0).
    """
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence_cosine(W, B_k, kappa):
    """
    Zero out trust to neighbours whose cosine similarity < kappa.
    B_k: (N, D) unit vectors for one topic.
    With kappa=0.0, all neighbours with cosim >= 0 pass — effectively
    the filter passes all neighbours on an undirected graph (cosim can be
    negative, but adjacency matrix already zeros out non-neighbours).
    Re-normalise rows; nodes with no in-kappa neighbours receive zero social term.
    """
    cosim = B_k @ B_k.T                         # (N, N) pairwise cosine similarities
    mask = (cosim >= kappa).astype(float)
    np.fill_diagonal(mask, 0.0)                  # no self-influence
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit_nan_safe(x):
    """Min-max scaling ignoring NaN values. NaN entries remain NaN."""
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.where(np.isnan(x), np.nan, 0.5)


def mean_pairwise_cosim(B_k):
    """
    Mean pairwise cosine similarity for one topic snapshot.
    B_k: (N, D) unit vectors.
    Returns scalar in [-1, 1]. 1.0 = full consensus; ~0 = maximum diversity.
    """
    cosim = B_k @ B_k.T                         # (N, N)
    idx = np.triu_indices(N, k=1)               # upper triangle, no diagonal
    return float(cosim[idx].mean())


def cumulative_unique_sme(k):
    seen = set()
    result = []
    for t in range(T + 1):
        seen.update(np.where(SME_hist[t, :, k])[0].tolist())
        result.append(len(seen))
    return result


# -- Diagnostic: confirm initialisation before running ------------------------

print(f"Initial mean pairwise cosim at t=0:")
for k in range(M):
    cosim_t0 = mean_pairwise_cosim(B[:, k, :])
    print(f"  {TOPIC_LABELS[k]}: {cosim_t0:.3f}  "
          f"(Run 12 reference: ~0.00 | KAPPA={KAPPA} — filter {'OFF' if KAPPA == 0.0 else 'ON'})")

# -- Simulation loop -----------------------------------------------------------

for t in range(T):
    B_new = np.empty_like(B)   # (N, M, D)
    C_new = np.empty_like(C)   # (N, M)

    for k in range(M):
        W_k = compute_trust(A, C[:, k])                          # (N, N)
        W_k = apply_bounded_confidence_cosine(W_k, B[:, k, :], KAPPA)

        # Belief update: trust-weighted sum of neighbour vectors + Gaussian noise
        social    = W_k @ B[:, k, :]                             # (N, D)
        innovation = rng.normal(0.0, 0.1, (N, D))               # (N, D)
        b_raw = ((1 - ALPHA - ETA) * B[:, k, :]
                 + ALPHA * social
                 + ETA * innovation)                             # (N, D)
        norms_new = np.linalg.norm(b_raw, axis=1, keepdims=True)
        B_new[:, k, :] = b_raw / np.where(norms_new > 0, norms_new, 1.0)

        # Competence update: peer-consistency via cosine similarity
        # delta_i = sum_j w_ij * dot(b_i, b_j) — positive, higher = more aligned
        cosim_matrix = B[:, k, :] @ B[:, k, :].T                # (N, N)
        delta = (W_k * cosim_matrix).sum(axis=1)                 # (N,)

        # Isolated-node guard (carry from sim_v10.py; sign is positive but guard is identical)
        delta[isolated] = np.nan           # exclude from min/max range
        delta_scaled = minmax_unit_nan_safe(delta)
        delta_scaled[isolated] = 0.0       # no update; competence decays naturally

        C_new[:, k] = np.clip(
            (1 - LAMBDA) * C[:, k] + LAMBDA * delta_scaled,
            0.0, 1.0
        )

    B, C = B_new, C_new
    B_hist[t + 1] = B
    C_hist[t + 1] = C

    for k in range(M):
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[t + 1, :, k] = C[:, k] >= threshold


# -- Derived metrics -----------------------------------------------------------

mean_cosim_hist = np.array(
    [[mean_pairwise_cosim(B_hist[t, :, k, :]) for k in range(M)]
     for t in range(T + 1)]
)   # shape (T+1, M)

sme_fraction = SME_hist.mean(axis=0)   # (N, M)

# Convergence: first t where mean pairwise cosim > CONV_THRESHOLD on all topics
# CONV_THRESHOLD is detection-only — this block never feeds back into the sim loop
converged  = np.all(mean_cosim_hist > CONV_THRESHOLD, axis=1)
conv_steps = np.where(converged)[0]
t_conv     = int(conv_steps[0]) if len(conv_steps) > 0 else None

# Post-convergence rotation: nodes that first appeared as SME after t_conv
post_conv_new_sme = {}
if t_conv is not None:
    for k in range(M):
        pre_ever  = set(np.where(np.any(SME_hist[:t_conv, :, k], axis=0))[0])
        post_ever = set(np.where(np.any(SME_hist[t_conv:, :, k], axis=0))[0])
        post_conv_new_sme[k] = len(post_ever - pre_ever)


# -- Console summary -----------------------------------------------------------

print("\n-- sim_v13.py complete --")
print(f"N={N}  M={M}  T={T}  D={D}  alpha={ALPHA}  eta={ETA}  lambda={LAMBDA}  "
      f"KAPPA={KAPPA}  CONV_THRESHOLD={CONV_THRESHOLD}  "
      f"graph=WattsStrogatz(k=4,p=0.1)  seed=42  init=Gaussian")

isolated_ids = np.where(isolated)[0].tolist()
print(f"Isolated nodes (degree=0): {isolated_ids}  "
      f"({'none — expected for WS k=4' if not isolated_ids else 'WARNING: unexpected'})")

print("\nMean pairwise cosim (t=0 -> t=200):")
for k in range(M):
    print(f"  {TOPIC_LABELS[k]}: {mean_cosim_hist[0, k]:.3f} -> {mean_cosim_hist[-1, k]:.3f}")

print("\nComparison:")
print("  Run 11 (uniform init, KAPPA=0.5): cosim ~0.75 -> 0.997  conv t=14")
print("  Run 12 (Gaussian init, KAPPA=0.5): cosim ~0.00 -> 0.035  NO convergence")
vec_str = " | ".join(
    f"{TOPIC_LABELS[k]}: {mean_cosim_hist[0, k]:.3f} -> {mean_cosim_hist[-1, k]:.3f}"
    for k in range(M)
)
print(f"  Run 13 (Gaussian init, KAPPA=0.0): cosim  {vec_str}")

if t_conv is not None:
    print(f"\nConvergence (all topics cosim > {CONV_THRESHOLD}): t={t_conv}")
else:
    print(f"\nConvergence: NOT reached by t={T}")

print("\nUnique nodes ever SME / top-3 holders:")
for k in range(M):
    ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
    top  = np.argsort(sme_fraction[:, k])[-3:][::-1]
    print(f"  {TOPIC_LABELS[k]}: {ever}/{N} nodes ever SME | "
          f"top holders: nodes {top.tolist()} "
          f"({sme_fraction[top, k].round(2).tolist()})")

print("\nPost-convergence rotation:")
if t_conv is not None:
    for k in range(M):
        n_new = post_conv_new_sme.get(k, 0)
        print(f"  {TOPIC_LABELS[k]}: {n_new} new nodes entered SME after t={t_conv} "
              f"({100 * n_new / N:.1f}%)")
else:
    print("  N/A — convergence not reached")


# -- Plots (3x3 grid) ---------------------------------------------------------

fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)
fig.suptitle(
    "sim_v13.py — Vector Beliefs (D=5, Gaussian init, KAPPA=0.0)  |  Run 13\n"
    "N=30, M=3, T=200, D=5 | Watts-Strogatz(k=4,p=0.1) | "
    "KAPPA=0.0 (BC filter OFF)  Gaussian init",
    fontsize=10
)

# Row 0: belief vector heatmaps at t=0, 100, 200 (Topic 1)
for col, snap in enumerate([0, 100, 200]):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(B_hist[snap, :, 0, :], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_title(f"Belief vectors Topic 1  t={snap}", fontsize=9)
    ax.set_xlabel("Dimension", fontsize=8)
    ax.set_ylabel("Node", fontsize=8)
    ax.set_xticks(range(D))
    ax.set_xticklabels([f"d{d + 1}" for d in range(D)], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 1 left+centre: mean pairwise cosim over time (all 3 topics)
ax_cos = fig.add_subplot(gs[1, :2])
for k in range(M):
    ax_cos.plot(mean_cosim_hist[:, k], color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
ax_cos.axhline(CONV_THRESHOLD, color="black", linestyle="--", linewidth=0.9,
               label=f"conv threshold = {CONV_THRESHOLD}")
if t_conv is not None:
    ax_cos.axvline(t_conv, color="grey", linestyle=":", linewidth=0.8,
                   label=f"t_conv={t_conv}")
ax_cos.set_title("Mean pairwise cosine similarity over time", fontsize=9)
ax_cos.set_xlabel("Timestep", fontsize=8)
ax_cos.set_ylabel("Mean pairwise cosim", fontsize=8)
ax_cos.set_ylim(-0.1, 1.05)
ax_cos.legend(fontsize=8)

# Row 1 right: competence heatmap at t=200
ax_comp = fig.add_subplot(gs[1, 2])
im_c = ax_comp.imshow(C_hist[-1], aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax_comp.set_title("Competence at t=200", fontsize=9)
ax_comp.set_xlabel("Topic", fontsize=8)
ax_comp.set_ylabel("Node", fontsize=8)
ax_comp.set_xticks(range(M))
ax_comp.set_xticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im_c, ax=ax_comp, fraction=0.046, pad=0.04)

# Row 2 left+centre: SME fraction heatmap
ax_sme = fig.add_subplot(gs[2, :2])
im_s = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax_sme.set_title("SME fraction (proportion of timesteps as SME)", fontsize=9)
ax_sme.set_xlabel("Node", fontsize=8)
ax_sme.set_ylabel("Topic", fontsize=8)
ax_sme.set_yticks(range(M))
ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im_s, ax=ax_sme, fraction=0.046, pad=0.04)

# Row 2 right: cumulative unique SME holders with 60% target and t_conv marker
ax_turn = fig.add_subplot(gs[2, 2])
for k in range(M):
    ax_turn.plot(cumulative_unique_sme(k), color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
ax_turn.axhline(0.6 * N, color="grey", linestyle=":", linewidth=0.8,
                label="60% target")
if t_conv is not None:
    ax_turn.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                    label=f"t_conv={t_conv}")
ax_turn.set_title("Cumulative unique SME holders", fontsize=9)
ax_turn.set_xlabel("Timestep", fontsize=8)
ax_turn.set_ylabel("Unique nodes ever SME", fontsize=8)
ax_turn.legend(fontsize=7)

plt.savefig(os.path.join(_HERE, "sim_v13_plots.png"),
            dpi=150, bbox_inches="tight")
print(f"\nPlots saved -> {os.path.join(_HERE, 'sim_v13_plots.png')}")

# -- CSV export (Session 52 backfill -- same values as the console printout above) --

import csv

csv_path = os.path.join(_HERE, "sim_v13_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "topic", "cosim_t0", "cosim_t200", "t_conv",
        "ever_sme", "top1_node", "top1_frac", "top2_node", "top2_frac",
        "top3_node", "top3_frac", "post_conv_new_sme", "post_conv_pct",
    ])
    for k in range(M):
        ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
        top = np.argsort(sme_fraction[:, k])[-3:][::-1]
        row = [
            TOPIC_LABELS[k],
            f"{mean_cosim_hist[0, k]:.4f}", f"{mean_cosim_hist[-1, k]:.4f}",
            t_conv if t_conv is not None else "NA",
            ever,
        ]
        for node, frac in zip(top.tolist(), sme_fraction[top, k].tolist()):
            row += [node, f"{frac:.4f}"]
        if t_conv is not None:
            n_new = post_conv_new_sme.get(k, 0)
            row += [n_new, f"{100 * n_new / N:.2f}"]
        else:
            row += ["NA", "NA"]
        writer.writerow(row)
print(f"CSV saved -> {csv_path}")

plt.show()
