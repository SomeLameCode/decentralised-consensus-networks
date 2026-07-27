"""
sim_v21_A3b.py -- Phase 3 Series A3b: Fine D sweep + ETA sensitivity at Pareto target

Sub-run 1 (fine D sweep):
  Fix ETA=0.15, ALPHA=0.3, KAPPA=0.0, Gaussian init, WS k=4 p=0.1, seed=42
  N=30, M=3, T=5000. Vary D: {125, 150, 175}
  Goal: locate D where eq_cosim closest to 0.80

Sub-run 2 (ETA sensitivity at D=150):
  Fix D=150, ALPHA=0.3, KAPPA=0.0, Gaussian init, WS k=4 p=0.1, seed=42
  N=30, M=3, T=5000. Vary ETA: {0.05, 0.10, 0.15}
  Goal: does lower ETA recover rotation health (maxHolder < 0.50) without
        collapsing eq_cosim toward 1.0?

D=150, ETA=0.15 is shared (Run 66): run once in sub-run 1, reused in sub-run 2.

Run numbering: R65 (D=125), R66 (D=150), R67 (D=175), R68 (D=150 ETA=0.05),
               R69 (D=150 ETA=0.10)

Outputs (workspace/simulation/phase-3/):
  sim_v21_A3b_subrun1_plots.png  (1x3 grid: D=125, 150, 175 at ETA=0.15)
  sim_v21_A3b_subrun2_plots.png  (1x3 grid: D=150 at ETA=0.05, 0.10, 0.15)
  A3b-summary.csv

Run from project root:
  python workspace/simulation/phase-3/sim_v21_A3b.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Fixed parameters ---------------------------------------------------------

N       = 30
M       = 3
ALPHA   = 0.3
LAMBDA  = 0.1
KAPPA   = 0.0
SEED    = 42
T       = 5000

CONV_THRESHOLD = 0.99
PARETO_TARGET  = 0.80
SME_PCT        = 90
MILESTONE_STEP = 500
EQ_START       = 1000

SNAP_TIMES     = [100, 500, 1000, 5000]

TOPIC_COLORS   = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS   = [f"T{k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Sub-run definitions ------------------------------------------------------

SUBRUN1_D_VALUES = [125, 150, 175]
SUBRUN1_ETA      = 0.15
SUBRUN1_RUNS     = {125: 65, 150: 66, 175: 67}

SUBRUN2_ETA_VALUES = [0.05, 0.10, 0.15]
SUBRUN2_D          = 150
SUBRUN2_RUNS       = {0.05: 68, 0.10: 69, 0.15: 66}

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


# -- Single run ---------------------------------------------------------------

def run_one(D, ETA, run_num, label):
    print()
    print("=" * 72)
    print(f"Run {run_num} -- {label}")
    print(f"N={N}  M={M}  D={D}  ETA={ETA}  ALPHA={ALPHA}  KAPPA={KAPPA}  "
          f"seed={SEED}  T={T}")
    print(f"Milestone every {MILESTONE_STEP} | eq window t={EQ_START}-{T}")
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
            mean_all = float(np.mean(mean_cosim_hist[t + 1, :]))
            print(f"  t={t+1:>5}  |  {cosims_str}  |  mean={mean_all:.3f}")

    # -- Derived metrics ------------------------------------------------------

    sme_fraction = SME_hist.mean(axis=0)
    ever_sme     = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder   = [float(sme_fraction[:, k].max()) for k in range(M)]

    eq_cosim = float(np.mean(mean_cosim_hist[EQ_START:, :]))

    mean_over_topics = mean_cosim_hist.mean(axis=1)
    t_to_eq_arr = np.where(mean_over_topics >= 0.95 * eq_cosim)[0]
    t_to_eq = int(t_to_eq_arr[0]) if len(t_to_eq_arr) > 0 else T

    snapshots = {snap_t: [float(mean_cosim_hist[snap_t, k]) for k in range(M)]
                 for snap_t in SNAP_TIMES if snap_t <= T}

    # -- Per-run summary ------------------------------------------------------
    print()
    print(f"--- {label} summary ---")
    print(f"eq_cosim (t={EQ_START}-{T}): {eq_cosim:.4f}  "
          f"(Pareto target={PARETO_TARGET}, delta={abs(eq_cosim - PARETO_TARGET):.4f})")
    print(f"t_to_eq (95% of eq): t={t_to_eq}")
    print(f"Snapshots:")
    for snap_t in SNAP_TIMES:
        vals = "  ".join(f"T{k+1}:{snapshots[snap_t][k]:.3f}" for k in range(M))
        mean_s = float(np.mean(snapshots[snap_t]))
        print(f"  t={snap_t:>5}: {vals}  mean={mean_s:.3f}")
    print(f"ever-SME: {ever_sme}  (min={min(ever_sme)}/{N})")
    print(f"top_holder: {[round(x, 3) for x in top_holder]}  "
          f"(max={max(top_holder):.3f})")

    return {
        "D": D, "ETA": ETA, "run": run_num, "label": label,
        "eq_cosim": eq_cosim, "t_to_eq": t_to_eq,
        "ever_sme": ever_sme, "top_holder": top_holder,
        "snapshots": snapshots, "mean_cosim_hist": mean_cosim_hist,
    }


# -- Main ---------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v21_A3b.py -- Phase 3 Series A3b")
print("Sub-run 1: fine D sweep (D={125,150,175}, ETA=0.15)")
print("Sub-run 2: ETA sensitivity (D=150, ETA={0.05,0.10,0.15})")
print(f"Fixed: N={N}  M={M}  ALPHA={ALPHA}  KAPPA={KAPPA}  seed={SEED}  T={T}")
print(f"Pareto target: eq_cosim ~ {PARETO_TARGET}")
print("=" * 72)

# -- Sub-run 1 ----------------------------------------------------------------

print()
print("##" * 36)
print("## SUB-RUN 1 -- Fine D sweep (ETA=0.15, D={125, 150, 175})")
print("##" * 36)

subrun1_results = []
for D in SUBRUN1_D_VALUES:
    label = f"Sub-run1 D={D} ETA={SUBRUN1_ETA}"
    res = run_one(D, SUBRUN1_ETA, SUBRUN1_RUNS[D], label)
    subrun1_results.append(res)

# Identify closest to Pareto in sub-run 1
closest_d = min(subrun1_results, key=lambda r: abs(r["eq_cosim"] - PARETO_TARGET))

print()
print("Sub-run 1 summary:")
print(f"{'D':>5}  {'ETA':>5}  {'Run':>4}  {'eq_cosim':>9}  "
      f"{'t_to_eq':>8}  {'minSME':>7}  {'maxHolder':>10}  {'delta_0.80':>10}")
print("-" * 75)
for res in subrun1_results:
    marker = " <--" if res["D"] == closest_d["D"] else ""
    print(f"{res['D']:>5}  {res['ETA']:>5}  {res['run']:>4}  "
          f"{res['eq_cosim']:>9.4f}  {res['t_to_eq']:>8}  "
          f"{min(res['ever_sme']):>7}  {max(res['top_holder']):>10.3f}  "
          f"{abs(res['eq_cosim']-PARETO_TARGET):>10.4f}{marker}")
print(f"\nClosest to Pareto (cosim={PARETO_TARGET}): D={closest_d['D']}  "
      f"eq_cosim={closest_d['eq_cosim']:.4f}")

# -- Sub-run 2 ----------------------------------------------------------------

print()
print("##" * 36)
print("## SUB-RUN 2 -- ETA sensitivity (D=150, ETA={0.05, 0.10, 0.15})")
print("##" * 36)

subrun2_results = []
d150_result = next(r for r in subrun1_results if r["D"] == 150)

for ETA in SUBRUN2_ETA_VALUES:
    if ETA == 0.15:
        # Reuse Run 66 from sub-run 1
        res = dict(d150_result)
        res["label"] = f"Sub-run2 D={SUBRUN2_D} ETA={ETA} (reused Run 66)"
        print(f"\nETA={ETA}: reusing D=150 result from sub-run 1 (Run 66)")
        print(f"  eq_cosim={res['eq_cosim']:.4f}  t_to_eq={res['t_to_eq']}  "
              f"maxHolder={max(res['top_holder']):.3f}")
    else:
        label = f"Sub-run2 D={SUBRUN2_D} ETA={ETA}"
        res = run_one(SUBRUN2_D, ETA, SUBRUN2_RUNS[ETA], label)
    subrun2_results.append(res)

print()
print("Sub-run 2 summary:")
print(f"{'D':>5}  {'ETA':>5}  {'Run':>4}  {'eq_cosim':>9}  "
      f"{'t_to_eq':>8}  {'minSME':>7}  {'maxHolder':>10}  {'delta_0.80':>10}")
print("-" * 75)
for res in subrun2_results:
    print(f"{res['D']:>5}  {res['ETA']:>5}  {res['run']:>4}  "
          f"{res['eq_cosim']:>9.4f}  {res['t_to_eq']:>8}  "
          f"{min(res['ever_sme']):>7}  {max(res['top_holder']):>10.3f}  "
          f"{abs(res['eq_cosim']-PARETO_TARGET):>10.4f}")

# -- Combined summary table ---------------------------------------------------

print()
print("=" * 72)
print("COMBINED SUMMARY TABLE -- Series A3b")
print(f"{'Sub-run':>8}  {'D':>5}  {'ETA':>5}  {'Run':>4}  {'eq_cosim':>9}  "
      f"{'t_to_eq':>8}  {'minSME':>7}  {'maxHolder':>10}")
print("-" * 75)
for res in subrun1_results:
    print(f"{'D-sweep':>8}  {res['D']:>5}  {res['ETA']:>5}  {res['run']:>4}  "
          f"{res['eq_cosim']:>9.4f}  {res['t_to_eq']:>8}  "
          f"{min(res['ever_sme']):>7}  {max(res['top_holder']):>10.3f}")
print("-" * 75)
for res in subrun2_results:
    print(f"{'ETA-sens':>8}  {res['D']:>5}  {res['ETA']:>5}  {res['run']:>4}  "
          f"{res['eq_cosim']:>9.4f}  {res['t_to_eq']:>8}  "
          f"{min(res['ever_sme']):>7}  {max(res['top_holder']):>10.3f}")

# -- Write CSV ----------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "A3b-summary.csv")
all_results = []
for res in subrun1_results:
    all_results.append(("D-sweep", res))
for res in subrun2_results:
    if not (res["D"] == 150 and res["ETA"] == 0.15 and
            any(r["D"] == 150 and r["ETA"] == 0.15 for _, r in all_results)):
        all_results.append(("ETA-sens", res))

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["subrun", "D", "ETA", "run", "eq_cosim", "t_to_eq", "delta_pareto"]
    for snap_t in SNAP_TIMES:
        for k in range(M):
            header.append(f"snap_t{snap_t}_T{k+1}")
    header += ["ever_sme_T1", "ever_sme_T2", "ever_sme_T3",
               "min_ever_sme", "max_top_holder"]
    writer.writerow(header)

    for subrun_label, res in all_results:
        row = [
            subrun_label, res["D"], res["ETA"], res["run"],
            f"{res['eq_cosim']:.4f}", res["t_to_eq"],
            f"{abs(res['eq_cosim'] - PARETO_TARGET):.4f}",
        ]
        for snap_t in SNAP_TIMES:
            for k in range(M):
                row.append(f"{res['snapshots'][snap_t][k]:.4f}")
        row += [
            res["ever_sme"][0], res["ever_sme"][1], res["ever_sme"][2],
            min(res["ever_sme"]),
            f"{max(res['top_holder']):.4f}",
        ]
        writer.writerow(row)

print(f"\nCSV saved -> {csv_path}")

# -- Plot sub-run 1: 1x3 grid -------------------------------------------------

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
fig1.suptitle(
    "sim_v21_A3b.py Sub-run 1 -- Fine D sweep | ETA=0.15 | "
    f"seed={SEED} T={T} N={N} ALPHA={ALPHA}\n"
    "Orange = eq_cosim | Purple dashed = Pareto target (0.80) | "
    "Black dashed = conv threshold (0.99)",
    fontsize=8
)

for i, res in enumerate(subrun1_results):
    ax = axes1[i]
    hist = res["mean_cosim_hist"]

    for k in range(M):
        ax.plot(hist[:, k], color=TOPIC_COLORS[k], linewidth=0.8,
                label=TOPIC_LABELS[k], alpha=0.9)

    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(PARETO_TARGET, color="purple", linestyle="--",
               linewidth=0.9, alpha=0.8, label=f"target({PARETO_TARGET})")
    ax.axhline(res["eq_cosim"], color="orange", linestyle="-",
               linewidth=1.0, alpha=0.9, label=f"eq={res['eq_cosim']:.3f}")
    ax.axvline(EQ_START, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

    marker = " [*]" if res["D"] == closest_d["D"] else ""
    ax.set_title(
        f"D={res['D']}  Run {res['run']}  ETA=0.15\n"
        f"eq={res['eq_cosim']:.3f}  maxH={max(res['top_holder']):.3f}{marker}",
        fontsize=8
    )
    ax.set_xlabel("Timestep", fontsize=7)
    if i == 0:
        ax.set_ylabel("Mean pairwise cosim", fontsize=7)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=6)

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "sim_v21_A3b_subrun1_plots.png")
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
print(f"Sub-run 1 plot saved -> {plot1_path}")

# -- Plot sub-run 2: 1x3 grid -------------------------------------------------

ETA_COLORS_SR2 = ["#E91E63", "#FF9800", "#9C27B0"]

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4), sharey=False)
fig2.suptitle(
    "sim_v21_A3b.py Sub-run 2 -- ETA sensitivity | D=150 | "
    f"seed={SEED} T={T} N={N} ALPHA={ALPHA}\n"
    "Orange = eq_cosim | Purple dashed = Pareto target (0.80) | "
    "Black dashed = conv threshold (0.99)",
    fontsize=8
)

for i, res in enumerate(subrun2_results):
    ax = axes2[i]
    hist = res["mean_cosim_hist"]

    for k in range(M):
        ax.plot(hist[:, k], color=TOPIC_COLORS[k], linewidth=0.8,
                label=TOPIC_LABELS[k], alpha=0.9)

    ax.axhline(CONV_THRESHOLD, color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(PARETO_TARGET, color="purple", linestyle="--",
               linewidth=0.9, alpha=0.8, label=f"target({PARETO_TARGET})")
    ax.axhline(res["eq_cosim"], color="orange", linestyle="-",
               linewidth=1.0, alpha=0.9, label=f"eq={res['eq_cosim']:.3f}")
    ax.axvline(EQ_START, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_title(
        f"D=150  Run {res['run']}  ETA={res['ETA']}\n"
        f"eq={res['eq_cosim']:.3f}  maxH={max(res['top_holder']):.3f}",
        fontsize=8
    )
    ax.set_xlabel("Timestep", fontsize=7)
    if i == 0:
        ax.set_ylabel("Mean pairwise cosim", fontsize=7)
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=6, loc="lower right")
    ax.tick_params(labelsize=6)

plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, "sim_v21_A3b_subrun2_plots.png")
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
print(f"Sub-run 2 plot saved -> {plot2_path}")

print()
print("-- sim_v21_A3b.py complete --")
