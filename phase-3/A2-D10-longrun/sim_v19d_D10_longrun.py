"""
sim_v19d_D10_longrun.py -- Phase 3 Series A2: D=10 run (Run 58)

Context: D=2 slower than D=5 for this seed. D=5 reference confirmed (Run 57).
D=10 tests whether timescales continue to grow with D or whether
the relationship is non-monotone.

Fixed: N=30, M=3, D=10, T=50000, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, KAPPA=0.0
       Gaussian init, WS k=4 p=0.1, seed=42

Outputs (workspace/simulation/phase-3/):
  sim_v19d_D10_longrun_plots.png
  D10-longrun-summary.csv

Run from project root:
  python workspace/simulation/phase-3/sim_v19d_D10_longrun.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Parameters ---------------------------------------------------------------

N       = 30
M       = 3
D       = 10
T       = 50000
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
SEED    = 42

CONV_THRESHOLD = 0.99
SME_PCT = 90
MILESTONE_STEP = 5000

SNAP_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions ---------------------------------------------------------

def compute_trust(A, c_k):
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence_cosine(W, B_k, kappa):
    cosim = B_k @ B_k.T
    mask = (cosim >= kappa).astype(float)
    np.fill_diagonal(mask, 0.0)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit_nan_safe(x):
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.where(np.isnan(x), np.nan, 0.5)


def mean_pairwise_cosim(B_k):
    cosim = B_k @ B_k.T
    idx = np.triu_indices(N, k=1)
    return float(cosim[idx].mean())


# -- Simulation ---------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print(f"sim_v19d_D10_longrun.py -- D=10 (Run 58) | seed={SEED} | T={T}")
print(f"N={N}  M={M}  D={D}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}")
print(f"Milestone every {MILESTONE_STEP} steps")
print("=" * 72)

rng = np.random.default_rng(SEED)
G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=SEED)
A = nx.to_numpy_array(G)
degrees = np.array([d for _, d in G.degree()])
isolated = degrees == 0

B_raw = rng.standard_normal((N, M, D))
norms = np.linalg.norm(B_raw, axis=2, keepdims=True)
B = B_raw / np.where(norms > 0, norms, 1.0)
C = rng.uniform(0.1, 0.9, (N, M))

mean_cosim_hist = np.zeros((T + 1, M))
SME_hist = np.zeros((T + 1, N, M), dtype=bool)

for k in range(M):
    mean_cosim_hist[0, k] = mean_pairwise_cosim(B[:, k, :])
    threshold = np.percentile(C[:, k], SME_PCT)
    SME_hist[0, :, k] = C[:, k] >= threshold

print(f"Initial cosim: " +
      "  ".join(f"T{k+1}:{mean_cosim_hist[0, k]:.3f}" for k in range(M)))
print()

for t in range(T):
    B_new = np.empty_like(B)
    C_new = np.empty_like(C)

    for k in range(M):
        W_k = compute_trust(A, C[:, k])
        W_k = apply_bounded_confidence_cosine(W_k, B[:, k, :], KAPPA)

        social = W_k @ B[:, k, :]
        innovation = rng.normal(0.0, 0.1, (N, D))
        b_raw = (1 - ALPHA - ETA) * B[:, k, :] + ALPHA * social + ETA * innovation
        norms_new = np.linalg.norm(b_raw, axis=1, keepdims=True)
        B_new[:, k, :] = b_raw / np.where(norms_new > 0, norms_new, 1.0)

        cosim_matrix = B[:, k, :] @ B[:, k, :].T
        delta = (W_k * cosim_matrix).sum(axis=1)
        delta[isolated] = np.nan
        delta_scaled = minmax_unit_nan_safe(delta)
        delta_scaled[isolated] = 0.0

        C_new[:, k] = np.clip(
            (1 - LAMBDA) * C[:, k] + LAMBDA * delta_scaled,
            0.0, 1.0
        )

    B, C = B_new, C_new

    for k in range(M):
        mean_cosim_hist[t + 1, k] = mean_pairwise_cosim(B[:, k, :])
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[t + 1, :, k] = C[:, k] >= threshold

    if (t + 1) % MILESTONE_STEP == 0:
        cosims_str = "  ".join(
            f"T{k+1}:{mean_cosim_hist[t+1, k]:.3f}" for k in range(M)
        )
        conv_flags = "  ".join(
            f"T{k+1}:{'CONV' if mean_cosim_hist[t+1, k] > CONV_THRESHOLD else '----'}"
            for k in range(M)
        )
        print(f"  t={t+1:>6}  |  {cosims_str}  |  {conv_flags}")

# -- Derived metrics ----------------------------------------------------------

sme_fraction = SME_hist.mean(axis=0)
ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]
final_cosims = [float(mean_cosim_hist[-1, k]) for k in range(M)]

per_topic_conv = {}
for k in range(M):
    steps = np.where(mean_cosim_hist[:, k] > CONV_THRESHOLD)[0]
    per_topic_conv[k] = int(steps[0]) if len(steps) > 0 else None

snap_times = [max(1, int(f * T)) for f in SNAP_FRACS]
snap_times[-1] = T
snapshots = {snap_t: [float(mean_cosim_hist[snap_t, k]) for k in range(M)]
             for snap_t in snap_times}

# -- Console summary ----------------------------------------------------------

print(f"\n{'=' * 72}")
print(f"SUMMARY -- D={D}  seed={SEED}  T={T}  ETA={ETA}")
print(f"\nFinal cosim at T={T}:")
for k in range(M):
    conv_t = per_topic_conv[k]
    conv_str = f"converged at t={conv_t}" if conv_t is not None else "NOT converged"
    print(f"  {TOPIC_LABELS[k]}: {final_cosims[k]:.4f}  ({conv_str})")

print(f"\nCosim snapshots (fractions of T={T}):")
frac_labels = "  ".join(f"{f:.2f}xT(t={snap_times[i]})" for i, f in enumerate(SNAP_FRACS))
print(f"  {'':12}  {frac_labels}")
for k in range(M):
    vals = "  ".join(f"{snapshots[snap_t][k]:>14.3f}" for snap_t in snap_times)
    print(f"  {TOPIC_LABELS[k]:<12}  {vals}")

print(f"\never-SME: {ever_sme}  (min={min(ever_sme)}/{N})")
print(f"top_holder: {[round(x, 3) for x in top_holder]}")

# -- Write CSV ----------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "D10-longrun-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["D", "seed", "T"]
    for frac, snap_t in zip(SNAP_FRACS, snap_times):
        for k in range(M):
            header.append(f"snap_{frac:.2f}_T{k+1}")
    header += ["t_conv_T1", "t_conv_T2", "t_conv_T3",
               "ever_sme_T1", "ever_sme_T2", "ever_sme_T3",
               "min_ever_sme", "max_top_holder"]
    writer.writerow(header)

    row = [D, SEED, T]
    for snap_t in snap_times:
        for k in range(M):
            row.append(f"{snapshots[snap_t][k]:.4f}")
    row += [
        per_topic_conv[0] if per_topic_conv[0] is not None else "NA",
        per_topic_conv[1] if per_topic_conv[1] is not None else "NA",
        per_topic_conv[2] if per_topic_conv[2] is not None else "NA",
        ever_sme[0], ever_sme[1], ever_sme[2],
        min(ever_sme),
        f"{max(top_holder):.4f}",
    ]
    writer.writerow(row)
print(f"\nCSV saved -> {csv_path}")

# -- Plot: single panel -------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle(
    f"sim_v19d_D10_longrun.py -- D=10 (Run 58) | seed={SEED} | T={T}\n"
    f"N={N}, M={M}, D={D}, ALPHA={ALPHA}, ETA={ETA}, KAPPA={KAPPA}, Gaussian init",
    fontsize=10
)

for k in range(M):
    conv_t = per_topic_conv[k]
    label = (TOPIC_LABELS[k] +
             (f" (conv t={conv_t})" if conv_t is not None else " (no conv)"))
    ax.plot(mean_cosim_hist[:, k], color=TOPIC_COLORS[k],
            linewidth=0.7, label=label)

ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
           linewidth=0.8, alpha=0.8, label=f"threshold ({CONV_THRESHOLD})")
ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7,
           label="0.5 boundary")

for snap_t in snap_times[:-1]:
    ax.axvline(snap_t, color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.5)

ax.set_xlabel("Timestep", fontsize=9)
ax.set_ylabel("Mean pairwise cosim", fontsize=9)
ax.set_ylim(-0.15, 1.05)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v19d_D10_longrun_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print(f"\n-- sim_v19d_D10_longrun.py complete --")
