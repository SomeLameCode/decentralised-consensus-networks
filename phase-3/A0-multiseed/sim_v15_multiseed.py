"""
sim_v15_multiseed.py — Phase 3 Series A0: Multi-Seed Robustness Study (D=5)

Series A0 runs (Runs 15-23): D=5 vector beliefs, 9 seeds.
Tests whether Run 14 findings (diversity regime, SME rotation) are robust
across different random seeds.

Parameters fixed at Run 14 baseline:
  N=30, M=3, D=5, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, KAPPA=0.0
  Gaussian init, WS k=4 p=0.1, T=500

Seed 42 is the Run 14 reference point — should reproduce Run 14 behaviour
(note: T=500 here vs T=1000 in Run 14, so final cosims will differ slightly).

Outputs (written to workspace/simulation/phase-3/):
  sim_v15_vector_plots.png  — 3x3 grid, one panel per seed
  multiseed-summary.csv     — one row per seed with key metrics

Run from project root:
  python workspace/simulation/phase-3/sim_v15_multiseed.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Parameters ---------------------------------------------------------------

N       = 30
M       = 3
D       = 5
T       = 500
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
KAPPA   = 0.0
CONV_THRESHOLD = 0.99
SME_PCT = 90    # top-10% SME threshold

SEEDS = [1, 7, 13, 21, 42, 77, 99, 123, 256]

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions (unchanged from sim_v14.py) ----------------------------

def compute_trust(A, c_k):
    """Competence-weighted, row-normalised trust matrix for one topic."""
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence_cosine(W, B_k, kappa):
    """Zero out trust to neighbours with cosine similarity < kappa; re-normalise."""
    cosim = B_k @ B_k.T
    mask = (cosim >= kappa).astype(float)
    np.fill_diagonal(mask, 0.0)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit_nan_safe(x):
    """Min-max scaling ignoring NaN values; NaN entries stay NaN."""
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    return np.where(np.isnan(x), np.nan, 0.5)


def mean_pairwise_cosim(B_k):
    """Mean pairwise cosine similarity for one topic snapshot. B_k: (N, D)."""
    cosim = B_k @ B_k.T
    idx = np.triu_indices(N, k=1)
    return float(cosim[idx].mean())


# -- Single-seed simulation --------------------------------------------------

def run_one_seed(run_num, seed):
    """
    Run Series A0 simulation for one seed.
    Returns a result dict containing full cosim history and summary metrics.
    """
    rng = np.random.default_rng(seed)

    # Graph: both topology and initial state seeded independently
    G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=seed)
    A = nx.to_numpy_array(G)
    degrees = np.array([d for _, d in G.degree()])
    isolated = degrees == 0

    # Gaussian init: neutral starting point (~0 initial cosim)
    B_raw = rng.standard_normal((N, M, D))
    norms = np.linalg.norm(B_raw, axis=2, keepdims=True)
    B = B_raw / np.where(norms > 0, norms, 1.0)
    C = rng.uniform(0.1, 0.9, (N, M))

    # History arrays (only cosim and SME needed for output)
    mean_cosim_hist = np.zeros((T + 1, M))
    SME_hist = np.zeros((T + 1, N, M), dtype=bool)

    # t = 0
    for k in range(M):
        mean_cosim_hist[0, k] = mean_pairwise_cosim(B[:, k, :])
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[0, :, k] = C[:, k] >= threshold

    # Simulation loop
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

    # -- Derived metrics ------------------------------------------------------

    sme_fraction = SME_hist.mean(axis=0)                        # (N, M)
    ever_sme = [
        int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)
    ]
    top_holder = [float(sme_fraction[:, k].max()) for k in range(M)]
    final_cosims = [float(mean_cosim_hist[-1, k]) for k in range(M)]

    diversity_regime = all(c < 0.5 for c in final_cosims)
    converged_flag   = any(c > CONV_THRESHOLD for c in final_cosims)

    conv_all  = np.all(mean_cosim_hist > CONV_THRESHOLD, axis=1)
    conv_steps = np.where(conv_all)[0]
    t_conv = int(conv_steps[0]) if len(conv_steps) > 0 else None

    return {
        "run_num":          run_num,
        "seed":             seed,
        "mean_cosim_hist":  mean_cosim_hist,
        "final_cosims":     final_cosims,
        "ever_sme":         ever_sme,
        "top_holder":       top_holder,
        "diversity_regime": diversity_regime,
        "converged":        converged_flag,
        "t_conv":           t_conv,
    }


# -- Run all seeds -----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v15_multiseed.py -- Phase 3 Series A0 | D=5 vector | Runs 15-23")
print(f"N={N}  M={M}  D={D}  T={T}  ALPHA={ALPHA}  ETA={ETA}  "
      f"LAMBDA={LAMBDA}  KAPPA={KAPPA}")
print(f"Seeds: {SEEDS}")
print("=" * 72)

results = []
for i, seed in enumerate(SEEDS):
    run_num = 15 + i
    print(f"  Run {run_num:02d}  seed={seed:>3} ...", end=" ", flush=True)
    r = run_one_seed(run_num, seed)
    results.append(r)
    cosims_str = "  ".join(f"T{k+1}:{r['final_cosims'][k]:.3f}" for k in range(M))
    print(f"done |  {cosims_str}  |  "
          f"minSME={min(r['ever_sme'])}/{N}  "
          f"div={'Y' if r['diversity_regime'] else 'N'}  "
          f"conv={'Y' if r['converged'] else 'N'}")


# -- Console summary table ---------------------------------------------------

print(f"\n{'=' * 72}")
print(f"SUMMARY TABLE -- Series A0  D={D}  T={T}  N={N}  ALPHA={ALPHA}  ETA={ETA}")
print(f"{'Run':>4}  {'Seed':>5}  {'T1 cosim':>9}  {'T2 cosim':>9}  {'T3 cosim':>9}  "
      f"{'minSME':>7}  {'maxTop':>7}  {'div':>4}  {'conv':>5}")
print("-" * 72)
for r in results:
    print(f"{r['run_num']:>4}  {r['seed']:>5}  "
          f"{r['final_cosims'][0]:>9.3f}  "
          f"{r['final_cosims'][1]:>9.3f}  "
          f"{r['final_cosims'][2]:>9.3f}  "
          f"{min(r['ever_sme']):>7d}  "
          f"{max(r['top_holder']):>7.3f}  "
          f"{'Y' if r['diversity_regime'] else 'N':>4}  "
          f"{'Y' if r['converged'] else 'N':>5}")

print()
for k in range(M):
    vals = [r['final_cosims'][k] for r in results]
    print(f"  {TOPIC_LABELS[k]} cosim at T={T}:  "
          f"mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
          f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")


# -- Success criteria check --------------------------------------------------

print(f"\n{'=' * 72}")
print("SUCCESS CRITERIA  (from phase3-plan-A0-amendment.md)")
print()

n_div      = sum(1 for r in results if r['diversity_regime'])
n_sme_pass = sum(1 for r in results if min(r['ever_sme']) >= int(0.9 * N))
cosim_stds = [np.std([r['final_cosims'][k] for r in results]) for k in range(M)]
std_pass   = all(s < 0.15 for s in cosim_stds)

print(f"  SC1  Diversity regime in >=7/9 seeds (all topics cosim < 0.5 at T={T}):  "
      f"{n_div}/9  {'PASS' if n_div >= 7 else 'FAIL'}")
print(f"  SC2  [Scalar convergence -- A0-scalar, not this script]")
print(f"  SC3  ever-SME >=90% (>={int(0.9*N)}/{N}) in >=8/9 seeds:  "
      f"{n_sme_pass}/9  {'PASS' if n_sme_pass >= 8 else 'FAIL'}")
print(f"  SC4  cosim std < 0.15 across seeds (structural, not noise -- target: stable regime):")
for k in range(M):
    print(f"         {TOPIC_LABELS[k]}: std={cosim_stds[k]:.3f}  "
          f"{'PASS' if cosim_stds[k] < 0.15 else 'FAIL'}")
print(f"       SC4 overall: {'PASS' if std_pass else 'FAIL'}")

sc_passed = (n_div >= 7) and (n_sme_pass >= 8) and std_pass
print(f"\n  A0 overall: {'PASS (arXiv gate cleared for this workstream)' if sc_passed else 'FAIL -- review before proceeding'}")


# -- Write CSV ---------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "multiseed-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_num", "seed",
        "topic1_cosim_T500", "topic2_cosim_T500", "topic3_cosim_T500",
        "ever_sme_t1", "ever_sme_t2", "ever_sme_t3",
        "min_ever_sme", "max_top_holder",
        "diversity_regime", "converged", "t_conv",
    ])
    for r in results:
        writer.writerow([
            r["run_num"],
            r["seed"],
            f"{r['final_cosims'][0]:.4f}",
            f"{r['final_cosims'][1]:.4f}",
            f"{r['final_cosims'][2]:.4f}",
            r["ever_sme"][0],
            r["ever_sme"][1],
            r["ever_sme"][2],
            min(r["ever_sme"]),
            f"{max(r['top_holder']):.4f}",
            int(r["diversity_regime"]),
            int(r["converged"]),
            r["t_conv"] if r["t_conv"] is not None else "NA",
        ])
print(f"\nCSV saved -> {csv_path}")


# -- Plots: 3x3 grid, one panel per seed -------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes_flat = axes.flatten()

fig.suptitle(
    f"sim_v15_multiseed.py — Phase 3 Series A0 | D={D} Vector Beliefs | 9 Seeds\n"
    f"N={N}, M={M}, T={T}, ALPHA={ALPHA}, ETA={ETA}, KAPPA={KAPPA}, Gaussian init  |  "
    f"Dashed: conv threshold ({CONV_THRESHOLD}) | Dotted: 0.5 diversity boundary",
    fontsize=10
)

for idx, r in enumerate(results):
    ax = axes_flat[idx]
    for k in range(M):
        ax.plot(r["mean_cosim_hist"][:, k],
                color=TOPIC_COLORS[k], linewidth=0.9, label=TOPIC_LABELS[k])
    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
               linewidth=0.7, alpha=0.8)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_title(
        f"Run {r['run_num']}  seed={r['seed']}  "
        f"div={'Y' if r['diversity_regime'] else 'N'}  "
        f"conv={'Y' if r['converged'] else 'N'}",
        fontsize=8
    )
    ax.set_xlabel("Timestep", fontsize=7)
    ax.set_ylabel("Mean pairwise cosim", fontsize=7)
    ax.set_ylim(-0.3, 1.05)
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v15_vector_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
plt.show()
print("\n-- sim_v15_multiseed.py complete --")
