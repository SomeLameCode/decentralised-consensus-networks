"""
sim_v4.py — Decentralised Consensus Learning Network (Phase 1 PoC, Run 4)

Model: scalar beliefs, competence-weighted trust, bounded confidence (D-001),
       Watts-Strogatz small-world topology (D-002). No idea objects (Phase 2).

Parameters (changes from Run 3 in brackets):
  N=120 [was 30], M=3, T=500 [was 200], alpha=0.3, eta=0.15, lambda=0.1,
  epsilon=0.15, noise=Uniform(-0.3, 0.3)
  Graph: Watts-Strogatz(N=120, k=4, p=0.1)

Success metrics for this run:
  - % of nodes ever holding SME status (target: >60% = >72/120 nodes)
  - Whether SME rotation continues after belief convergence (post-convergence
    rotation) — identified as the key open question from the stocktake.
  Convergence is defined as entropy < ENTROPY_THRESHOLD on all topics.
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

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

# ── Graph (D-002: Watts-Strogatz small-world) ─────────────────────────────────

G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=42)
A = nx.to_numpy_array(G)   # shape (N, N), symmetric, 0-diagonal

# ── Initial state ─────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
B = rng.uniform(0.0, 1.0, (N, M))    # beliefs[node, topic]
C = rng.uniform(0.1, 0.9, (N, M))    # competence[node, topic]

# ── Diagnostic: filter activity at t=0 ──────────────────────────────────────

print("-- Diagnostic: bounded confidence filter activity at t=0 --")
for k in range(M):
    diffs = np.abs(B[:, k][:, np.newaxis] - B[:, k][np.newaxis, :])
    within_eps_v1 = (diffs <= 0.3).sum() - N   # exclude self-pairs
    within_eps_v4 = (diffs <= EPSILON).sum() - N
    total_pairs = N * (N - 1)
    print(f"{TOPIC_LABELS[k]}: pairs within eps=0.3: "
          f"{within_eps_v1}/{total_pairs} "
          f"({100*within_eps_v1/total_pairs:.1f}%) | "
          f"within eps=0.15: "
          f"{within_eps_v4}/{total_pairs} "
          f"({100*within_eps_v4/total_pairs:.1f}%)")

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

# Find first timestep where entropy < threshold on all topics simultaneously
converged_steps = np.where(
    np.all(entropy_hist < ENTROPY_THRESHOLD, axis=1)
)[0]
t_conv = int(converged_steps[0]) if len(converged_steps) > 0 else None

post_conv_new_sme = {}   # topic -> count of nodes that first became SME post-conv
if t_conv is not None:
    for k in range(M):
        pre_ever = set(np.where(np.any(SME_hist[:t_conv, :, k], axis=0))[0])
        post_ever = set(np.where(np.any(SME_hist[t_conv:, :, k], axis=0))[0])
        post_conv_new_sme[k] = len(post_ever - pre_ever)


# ── Console summary ───────────────────────────────────────────────────────────

print("\n-- sim_v4.py complete --")
print(f"N={N}  M={M}  T={T}  alpha={ALPHA}  eta={ETA}  "
      f"lambda={LAMBDA}  epsilon={EPSILON}  graph=WattsStrogatz(k=4,p=0.1)  "
      f"noise=Uniform(-0.3,0.3)")

print("\nBelief entropy (t=0 -> t=500):")
for k in range(M):
    print(f"  {TOPIC_LABELS[k]}: {entropy_hist[0, k]:.3f} -> {entropy_hist[-1, k]:.3f}")

if t_conv is not None:
    print(f"\nConvergence (all topics entropy < {ENTROPY_THRESHOLD}): t={t_conv}")
else:
    print(f"\nConvergence: NOT reached by t={T}")

print(f"\nSME participation (target: >{int(0.6*N)}/{N} nodes = >60%):")
for k in range(M):
    ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
    pct = 100 * ever / N
    top = np.argsort(sme_fraction[:, k])[-3:][::-1]
    target_met = "PASS" if ever > 0.6 * N else "FAIL"
    print(f"  {TOPIC_LABELS[k]}: {ever}/{N} ({pct:.1f}%) [{target_met}] | "
          f"top holders: nodes {top.tolist()} "
          f"({sme_fraction[top, k].round(2).tolist()})")

print("\nPost-convergence rotation (new SME holders appearing after t_conv):")
if t_conv is not None:
    for k in range(M):
        print(f"  {TOPIC_LABELS[k]}: {post_conv_new_sme[k]} new nodes entered SME "
              f"after t={t_conv}")
else:
    print("  N/A — convergence not reached")


# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle(
    "sim_v4.py — Decentralised Consensus Learning Network\n"
    "N=120, M=3, T=500 | Watts-Strogatz(k=4,p=0.1) | eps=0.15  eta=0.15  noise=U(-0.3,0.3)",
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
ax_comp.set_title("Competence at t=500", fontsize=9)
ax_comp.set_xlabel("Topic", fontsize=8)
ax_comp.set_ylabel("Node", fontsize=8)
ax_comp.set_xticks(range(M))
ax_comp.set_xticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im, ax=ax_comp, fraction=0.046, pad=0.04)

# Row 2 left+centre: SME fraction heatmap
ax_sme = fig.add_subplot(gs[2, :2])
im2 = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax_sme.set_title("SME fraction (proportion of timesteps as SME)", fontsize=9)
ax_sme.set_xlabel("Node", fontsize=8)
ax_sme.set_ylabel("Topic", fontsize=8)
ax_sme.set_yticks(range(M))
ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
plt.colorbar(im2, ax=ax_sme, fraction=0.046, pad=0.04)

# Row 2 right: cumulative unique SME holders; mark convergence point
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

plt.savefig(os.path.join(_HERE, "sim_v4_plots.png"), dpi=150, bbox_inches="tight")
print(f"Plots saved -> {os.path.join(_HERE, 'sim_v4_plots.png')}")

# ── CSV export (Session 52 backfill — same values as the console printout above) ──

import csv

csv_path = os.path.join(_HERE, "sim_v4_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "topic", "entropy_t0", "entropy_t500", "t_conv",
        "ever_sme", "ever_sme_pct", "target_met",
        "top1_node", "top1_frac", "top2_node", "top2_frac", "top3_node", "top3_frac",
        "post_conv_new_sme",
    ])
    for k in range(M):
        ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
        pct = 100 * ever / N
        top = np.argsort(sme_fraction[:, k])[-3:][::-1]
        target_met = "PASS" if ever > 0.6 * N else "FAIL"
        row = [
            TOPIC_LABELS[k],
            f"{entropy_hist[0, k]:.4f}", f"{entropy_hist[-1, k]:.4f}",
            t_conv if t_conv is not None else "NA",
            ever, f"{pct:.2f}", target_met,
        ]
        for node, frac in zip(top.tolist(), sme_fraction[top, k].tolist()):
            row += [node, f"{frac:.4f}"]
        row.append(post_conv_new_sme[k] if t_conv is not None else "NA")
        writer.writerow(row)
print(f"CSV saved -> {csv_path}")

plt.show()
