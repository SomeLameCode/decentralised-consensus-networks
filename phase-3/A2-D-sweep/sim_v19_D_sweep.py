"""
sim_v19_D_sweep.py -- Phase 3 Series A2: D sweep

Context: A1 extended and Run 54 established convergence timescales in D=5
range from t=1,626 to t=13,574 per topic, driven by initial angular geometry.
A2 tests how dimensionality D shifts this timescale distribution. Prediction:
higher D -> more orthogonal noise directions -> longer, more heterogeneous
timescales per topic.

Fixed: ALPHA=0.3, ETA=0.15, KAPPA=0.0, Gaussian init, WS k=4 p=0.1
Fixed: N=30, M=3, LAMBDA=0.1, seed=42
Variable: D in {1, 2, 5, 10, 20}

T per run:
  D=1:  T=500    (expect fast convergence -- 1D unit vector: values +1 or -1)
  D=2:  T=2000   (expect moderate timescale)
  D=5:  T=20000  (Run 54 reference -- should reproduce key checkpoints)
  D=10: T=50000  (expect very long timescale)
  D=20: T=100000 (expect extremely long or no convergence in window)

Note: D=1 is NOT scalar beliefs -- it is a 1D unit vector (+1 or -1 after
normalisation). Distinct from Phase 1 scalar beliefs but useful as the
lowest-dimensional vector case.

Run numbering: Series A2 (Runs 55-59)
  D=1:  Run 55
  D=2:  Run 56
  D=5:  Run 57
  D=10: Run 58
  D=20: Run 59

Outputs (workspace/simulation/phase-3/):
  sim_v19_D_sweep_plots.png  -- one panel per D run (1xN layout)
  D-sweep-summary.csv        -- one row per D, per-topic t_conv

Run from project root:
  python workspace/simulation/phase-3/sim_v19_D_sweep.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Parameters ---------------------------------------------------------------

N       = 30
M       = 3
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
SEED    = 42

CONV_THRESHOLD = 0.99
SME_PCT = 90

SNAP_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00]

# D values to run -- set to [1, 2] for initial run; expand to full list later:
# D_VALUES = [1, 2, 5, 10, 20]
D_VALUES = [1, 2]

T_PER_D = {
    1:  500,
    2:  2000,
    5:  20000,
    10: 50000,
    20: 100000,
}

MILESTONE_EVERY = {d: (2000 if d <= 5 else 5000) for d in [1, 2, 5, 10, 20]}

RUN_START = 55  # D=1 -> Run 55, D=2 -> Run 56, D=5 -> Run 57, etc.
D_TO_RUN  = {1: 55, 2: 56, 5: 57, 10: 58, 20: 59}

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions (unchanged from sim_v14-v18) ----------------------------

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


# -- Single-D run -------------------------------------------------------------

def run_one_D(D):
    run_num = D_TO_RUN[D]
    T = T_PER_D[D]
    milestone_every = MILESTONE_EVERY[D]

    # Snapshots at fixed fractions of T
    snap_times = [max(1, int(f * T)) for f in SNAP_FRACS]
    snap_times[-1] = T  # ensure final snapshot is exactly at T

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

    print(f"  Initial cosim: " +
          "  ".join(f"T{k+1}:{mean_cosim_hist[0, k]:.3f}" for k in range(M)))

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

        if (t + 1) % milestone_every == 0:
            cosims_str = "  ".join(
                f"T{k+1}:{mean_cosim_hist[t+1, k]:.3f}" for k in range(M)
            )
            conv_flags = "  ".join(
                f"T{k+1}:{'CONV' if mean_cosim_hist[t+1, k] > CONV_THRESHOLD else '----'}"
                for k in range(M)
            )
            print(f"    t={t+1:>7}  |  {cosims_str}  |  {conv_flags}")

    # -- Derived metrics
    sme_fraction = SME_hist.mean(axis=0)
    ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]
    final_cosims = [float(mean_cosim_hist[-1, k]) for k in range(M)]

    per_topic_conv = {}
    for k in range(M):
        steps = np.where(mean_cosim_hist[:, k] > CONV_THRESHOLD)[0]
        per_topic_conv[k] = int(steps[0]) if len(steps) > 0 else None

    snapshots = {}
    for snap_t in snap_times:
        snapshots[snap_t] = [float(mean_cosim_hist[snap_t, k]) for k in range(M)]

    return {
        "run_num":         run_num,
        "D":               D,
        "T":               T,
        "mean_cosim_hist": mean_cosim_hist,
        "final_cosims":    final_cosims,
        "per_topic_conv":  per_topic_conv,
        "ever_sme":        ever_sme,
        "top_holder":      top_holder,
        "snap_times":      snap_times,
        "snapshots":       snapshots,
    }


# -- Main ---------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v19_D_sweep.py -- Phase 3 Series A2 | D sweep")
print(f"N={N}  M={M}  ALPHA={ALPHA}  ETA={ETA}  KAPPA={KAPPA}  seed={SEED}")
print(f"D values this run: {D_VALUES}")
print(f"T per D: " + "  ".join(f"D={d}:{T_PER_D[d]}" for d in D_VALUES))
print("=" * 72)

results = []
for D in D_VALUES:
    run_num = D_TO_RUN[D]
    T = T_PER_D[D]
    print(f"\nRun {run_num}  D={D}  T={T}")
    r = run_one_D(D)
    results.append(r)

    print(f"\n  SUMMARY -- Run {run_num}  D={D}  T={T}")
    for k in range(M):
        conv_t = r["per_topic_conv"][k]
        conv_str = f"converged at t={conv_t}" if conv_t is not None else "NOT converged"
        print(f"    {TOPIC_LABELS[k]}: final cosim={r['final_cosims'][k]:.4f}  ({conv_str})")
    print(f"  ever-SME: {r['ever_sme']}  top_holder: {[round(x, 3) for x in r['top_holder']]}")
    print(f"  Cosim snapshots at fractions of T:")
    frac_labels = "  ".join(f"{f:.2f}xT" for f in SNAP_FRACS)
    print(f"    {'':12}  {frac_labels}")
    for k in range(M):
        snap_vals = "  ".join(
            f"{r['snapshots'][snap_t][k]:.3f}" for snap_t in r["snap_times"]
        )
        print(f"    {TOPIC_LABELS[k]:<12}  {snap_vals}")
    print()


# -- Final summary table -------------------------------------------------------

print("=" * 72)
print(f"SUMMARY TABLE -- Series A2  D sweep  seed={SEED}  ALPHA={ALPHA}  ETA={ETA}")
print(f"{'Run':>4}  {'D':>3}  {'T':>7}  {'T1 t_conv':>10}  "
      f"{'T2 t_conv':>10}  {'T3 t_conv':>10}  {'minSME':>7}")
print("-" * 72)
for r in results:
    def fmt_conv(v):
        return str(v) if v is not None else "NA"
    print(f"{r['run_num']:>4}  {r['D']:>3}  {r['T']:>7}  "
          f"{fmt_conv(r['per_topic_conv'][0]):>10}  "
          f"{fmt_conv(r['per_topic_conv'][1]):>10}  "
          f"{fmt_conv(r['per_topic_conv'][2]):>10}  "
          f"{min(r['ever_sme']):>7d}")


# -- Write CSV ----------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "D-sweep-summary.csv")
header = ["run_num", "D", "T"]
for frac in SNAP_FRACS:
    for k in range(M):
        header.append(f"snap_{frac:.2f}_T{k+1}")
header += ["t_conv_T1", "t_conv_T2", "t_conv_T3",
           "ever_sme_T1", "ever_sme_T2", "ever_sme_T3",
           "min_ever_sme", "max_top_holder"]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in results:
        row = [r["run_num"], r["D"], r["T"]]
        for snap_t in r["snap_times"]:
            for k in range(M):
                row.append(f"{r['snapshots'][snap_t][k]:.4f}")
        row += [
            r["per_topic_conv"][0] if r["per_topic_conv"][0] is not None else "NA",
            r["per_topic_conv"][1] if r["per_topic_conv"][1] is not None else "NA",
            r["per_topic_conv"][2] if r["per_topic_conv"][2] is not None else "NA",
            r["ever_sme"][0], r["ever_sme"][1], r["ever_sme"][2],
            min(r["ever_sme"]),
            f"{max(r['top_holder']):.4f}",
        ]
        writer.writerow(row)
print(f"CSV saved -> {csv_path}")


# -- Plot: one panel per D run (1 x n_panels layout) -------------------------

n_panels = len(results)
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
if n_panels == 1:
    axes = [axes]

fig.suptitle(
    f"sim_v19_D_sweep.py -- Phase 3 Series A2 | D sweep\n"
    f"N={N}, M={M}, ALPHA={ALPHA}, ETA={ETA}, KAPPA={KAPPA}, Gaussian init, seed={SEED}\n"
    f"D values: {[r['D'] for r in results]}",
    fontsize=9
)

for j, r in enumerate(results):
    ax = axes[j]
    for k in range(M):
        conv_t = r["per_topic_conv"][k]
        label = (TOPIC_LABELS[k] +
                 (f" (conv t={conv_t})" if conv_t is not None else " (no conv)"))
        ax.plot(r["mean_cosim_hist"][:, k],
                color=TOPIC_COLORS[k], linewidth=0.8, label=label)

    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
               linewidth=0.8, alpha=0.8, label=f"threshold ({CONV_THRESHOLD})")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7,
               label="0.5 boundary")

    for snap_t in r["snap_times"][:-1]:
        ax.axvline(snap_t, color="lightgrey", linestyle="-",
                   linewidth=0.5, alpha=0.5)

    ax.set_title(
        f"Run {r['run_num']}  D={r['D']}  T={r['T']}",
        fontsize=9
    )
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Mean pairwise cosim", fontsize=8)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v19_D_sweep_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print("\n-- sim_v19_D_sweep.py complete --")
