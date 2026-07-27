"""
sim_v10.py — Decentralised Consensus Learning Network (Phase 2, Run 10)

Isolated-node peer-consistency fix.

Runs 7-9 showed that nodes 16, 45, 85 hold SME status 97-100% of timesteps
across all parameter variations (mu, degree normalisation, uniform C0).
Run 9 revealed the cause: those three nodes have degree=0 (isolated in the
SBM draw). Their peer-consistency delta is identically zero because their
W_k row is all zeros. All connected nodes have delta<0 (they have neighbours
to disagree with). minmax_unit() maps the entire array to [0,1], so delta=0
becomes 1.0 — the isolated node is labelled maximally competent every step
and its competence converges to 1.0 via (1-lambda)*c + lambda*1.0.
Non-participation is rewarded as perfect expertise.

Fix — two guards, not one:
  isolated = (degrees == 0)
  delta[isolated] = np.nan            # exclude from min/max scaling
  delta_scaled = minmax_unit_nan_safe(delta)  # nanmin/nanmax skip NaN
  delta_scaled[isolated] = 0.0        # no update; competence decays naturally

Setting isolated nodes to the array minimum is wrong: it would give them the
same update as the worst connected node. Setting to NaN and then to 0.0 gives
them no update at all — competence decays via (1-lambda) each step, converging
toward whatever it started at, driven lower over time without any positive signal.

Base: sim_v9.py / sim_v5.py — N=120, SBM(4x30, p_in=0.10, p_out=0.01),
M=3, T=500, alpha=0.3, eta=0.15, lambda=0.1, epsilon=0.15,
noise=Uniform(-0.3, 0.3), seed=42. Random initial C0 (same as Runs 5-9 baseline).

Two-run structure:
  Run A (baseline) — bug present; should reproduce Runs 5-9 (nodes 16,45,85 at 97-100%)
  Run B (fix)      — isolated nodes excluded from minmax scaling; zero competence update
"""

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -- Parameters ---------------------------------------------------------------

N = 120
M = 3
T = 500
ALPHA = 0.3
ETA = 0.15
LAMBDA = 0.1
EPSILON = 0.15
SME_PCT = 90
ENTROPY_THRESHOLD = 0.1

BLOCK_SIZES = [30, 30, 30, 30]
P_IN  = 0.10
P_OUT = 0.01
COMMUNITY_BOUNDARIES = [30, 60, 90]

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

RUN_LABELS = ["baseline (bug present)", "isolated-node fix"]

# -- Graph --------------------------------------------------------------------

n_blocks = len(BLOCK_SIZES)
probs = [
    [P_IN if r == c else P_OUT for c in range(n_blocks)]
    for r in range(n_blocks)
]
G = nx.stochastic_block_model(BLOCK_SIZES, probs, seed=42)
A = nx.to_numpy_array(G)

degrees = np.array([d for _, d in G.degree()])
isolated = degrees == 0

print("-- Graph: Stochastic Block Model --")
print(f"  Communities: {len(BLOCK_SIZES)} x {BLOCK_SIZES[0]} nodes | "
      f"p_in={P_IN}  p_out={P_OUT}")
print(f"  Edges: {G.number_of_edges()} | "
      f"Avg degree: {degrees.mean():.2f} | "
      f"Min: {degrees.min()}  Max: {degrees.max()}")
isolated_ids = np.where(isolated)[0].tolist()
print(f"  Isolated nodes (degree=0): {isolated_ids}  ({len(isolated_ids)} total)")

# -- Initial state ------------------------------------------------------------

_init_rng = np.random.default_rng(42)
B0 = _init_rng.uniform(0.0, 1.0, (N, M))
C0 = _init_rng.uniform(0.1, 0.9, (N, M))

print(f"\n-- Initial C0 of isolated nodes --")
for node in isolated_ids:
    vals = C0[node, :]
    print(f"  Node {node}: T1={vals[0]:.3f}  T2={vals[1]:.3f}  T3={vals[2]:.3f}")

# -- Diagnostic ---------------------------------------------------------------

print("\n-- Diagnostic: bounded confidence filter activity at t=0 --")
for k in range(M):
    diffs = np.abs(B0[:, k][:, np.newaxis] - B0[:, k][np.newaxis, :])
    within_v1 = (diffs <= 0.3).sum() - N
    within_v10 = (diffs <= EPSILON).sum() - N
    total_pairs = N * (N - 1)
    print(f"  {TOPIC_LABELS[k]}: pairs within eps=0.3: "
          f"{within_v1}/{total_pairs} "
          f"({100 * within_v1 / total_pairs:.1f}%) | "
          f"within eps=0.15: "
          f"{within_v10}/{total_pairs} "
          f"({100 * within_v10 / total_pairs:.1f}%)")


# -- Helper functions ---------------------------------------------------------

def compute_trust(A, c_k):
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence(W, b_k, epsilon):
    mask = (np.abs(b_k[:, np.newaxis] - b_k[np.newaxis, :]) <= epsilon).astype(float)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit(x):
    """Standard min-max scaling. Isolated nodes' delta=0 maps to 1.0 (the bug)."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


def minmax_unit_nan_safe(x):
    """Min-max scaling that ignores NaN values. NaN entries remain NaN."""
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    result = np.where(np.isnan(x), np.nan, 0.5)
    return result


def belief_entropy(b_k, bins=10):
    counts, _ = np.histogram(b_k, bins=bins, range=(0.0, 1.0))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def cumulative_unique_sme(SME_hist, k):
    seen = set()
    result = []
    for t in range(SME_hist.shape[0]):
        seen.update(np.where(SME_hist[t, :, k])[0].tolist())
        result.append(len(seen))
    return result


# -- Simulation function ------------------------------------------------------

def run_simulation(fix_isolated):
    """
    Run T=500 steps from identical initial conditions (B0, C0).
    fix_isolated=False: original minmax_unit — isolated nodes map to 1.0 (bug).
    fix_isolated=True:  NaN-safe scaling; isolated nodes get zero competence update.
    """
    B = B0.copy()
    C = C0.copy()

    rng = np.random.default_rng(42)
    rng.uniform(0.0, 1.0, (N, M))   # advance past B0
    rng.uniform(0.1, 0.9, (N, M))   # advance past C0

    B_hist   = np.empty((T + 1, N, M))
    C_hist   = np.empty((T + 1, N, M))
    SME_hist = np.zeros((T + 1, N, M), dtype=bool)

    B_hist[0] = B
    C_hist[0] = C

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

            if fix_isolated:
                # Guard 1: exclude isolated nodes from min/max scaling
                delta[isolated] = np.nan
                # Guard 2: scale over connected nodes only; set isolated to 0.0
                delta_scaled = minmax_unit_nan_safe(delta)
                delta_scaled[isolated] = 0.0
            else:
                delta_scaled = minmax_unit(delta)

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

    entropy_hist = np.array(
        [[belief_entropy(B_hist[t, :, k]) for k in range(M)]
         for t in range(T + 1)]
    )

    sme_fraction = SME_hist.mean(axis=0)

    converged_steps = np.where(np.all(entropy_hist < ENTROPY_THRESHOLD, axis=1))[0]
    t_conv = int(converged_steps[0]) if len(converged_steps) > 0 else None

    post_conv_new_sme = {}
    if t_conv is not None:
        for k in range(M):
            pre_ever  = set(np.where(np.any(SME_hist[:t_conv, :, k], axis=0))[0])
            post_ever = set(np.where(np.any(SME_hist[t_conv:, :, k], axis=0))[0])
            post_conv_new_sme[k] = len(post_ever - pre_ever)

    return {
        "entropy_hist":      entropy_hist,
        "sme_fraction":      sme_fraction,
        "SME_hist":          SME_hist,
        "t_conv":            t_conv,
        "post_conv_new_sme": post_conv_new_sme,
    }


# -- Run both -----------------------------------------------------------------

flags = [False, True]
results = {}
for fix, label in zip(flags, RUN_LABELS):
    print(f"\n{'=' * 60}")
    print(f"Running simulation: {label}  N={N}  M={M}  T={T} ...")
    results[label] = run_simulation(fix_isolated=fix)
    print("  Done.")


# -- Per-run console output ---------------------------------------------------

community_edges = [0] + COMMUNITY_BOUNDARIES + [N]
community_labels = [
    f"C{i+1}({community_edges[i]}-{community_edges[i+1]-1})"
    for i in range(len(BLOCK_SIZES))
]

for label in RUN_LABELS:
    r            = results[label]
    t_conv       = r["t_conv"]
    sme_fraction = r["sme_fraction"]
    SME_hist     = r["SME_hist"]
    pc           = r["post_conv_new_sme"]
    entropy_hist = r["entropy_hist"]

    print(f"\n{'=' * 60}")
    print(f"-- {label} --")
    print(f"N={N}  M={M}  T={T}  alpha={ALPHA}  eta={ETA}  "
          f"lambda={LAMBDA}  eps={EPSILON}  graph=SBM(4x30)  noise=U(-0.3,0.3)")

    print("\nBelief entropy (t=0 -> t=500):")
    for k in range(M):
        print(f"  {TOPIC_LABELS[k]}: {entropy_hist[0, k]:.3f} -> "
              f"{entropy_hist[-1, k]:.3f}")

    if t_conv is not None:
        print(f"\nConvergence (all topics entropy < {ENTROPY_THRESHOLD}): t={t_conv}")
    else:
        print(f"\nConvergence: NOT reached by t={T}")

    print(f"\nSME participation -- overall (target: >{int(0.6*N)}/{N} = >60%):")
    for k in range(M):
        ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
        pct  = 100 * ever / N
        top  = np.argsort(sme_fraction[:, k])[-3:][::-1]
        top_degs = [int(degrees[n]) for n in top]
        flag = "PASS" if ever > 0.6 * N else "FAIL"
        print(f"  {TOPIC_LABELS[k]}: {ever}/{N} ({pct:.1f}%) [{flag}] | "
              f"top holders: nodes {top.tolist()} "
              f"(frac={sme_fraction[top, k].round(2).tolist()}, "
              f"deg={top_degs})")

    print("\nSME participation -- per community (30 nodes each):")
    for i in range(len(BLOCK_SIZES)):
        lo, hi = community_edges[i], community_edges[i + 1]
        parts = []
        for k in range(M):
            ever = int(np.any(SME_hist[:, lo:hi, k], axis=0).sum())
            parts.append(f"{TOPIC_LABELS[k]}: {ever}/30 ({100*ever/30:.0f}%)")
        print(f"  {community_labels[i]}: " + " | ".join(parts))

    print("\nPost-convergence rotation (new SME holders after t_conv):")
    if t_conv is not None:
        for k in range(M):
            pct = 100 * pc.get(k, 0) / N
            print(f"  {TOPIC_LABELS[k]}: {pc.get(k, 0)} new nodes entered SME "
                  f"after t={t_conv} ({pct:.1f}%)")
    else:
        print("  N/A -- convergence not reached")


# -- Comparison table ---------------------------------------------------------

BL = RUN_LABELS[0]
FX = RUN_LABELS[1]

print(f"\n{'=' * 60}")
print("COMPARISON TABLE  (baseline bug vs isolated-node fix)")
sep = "+" + "-" * 36 + "+" + "-" * 22 + "+" + "-" * 22 + "+"
hdr = (f"| {'Metric':<34} | {'baseline (bug)':^20} | "
       f"{'isolated-node fix':^20} |")
print(sep)
print(hdr)
print(sep)

def _row(label, v_bl, v_fx):
    return f"| {label:<34} | {v_bl:^20} | {v_fx:^20} |"

for k in range(M):
    top_bl = results[BL]["sme_fraction"][:, k].max()
    top_fx = results[FX]["sme_fraction"][:, k].max()
    print(_row(f"Top holder {TOPIC_LABELS[k]} (% steps)",
               f"{top_bl*100:.1f}%", f"{top_fx*100:.1f}%"))

print(sep)

for k in range(M):
    ever_bl = int(np.any(results[BL]["SME_hist"][:, :, k], axis=0).sum())
    ever_fx = int(np.any(results[FX]["SME_hist"][:, :, k], axis=0).sum())
    print(_row(f"Ever-SME {TOPIC_LABELS[k]}", f"{ever_bl}/120", f"{ever_fx}/120"))

print(sep)

for k in range(M):
    r_bl = results[BL]
    r_fx = results[FX]
    v_bl = (f"{100 * r_bl['post_conv_new_sme'].get(k, 0) / N:.1f}%"
            if r_bl["t_conv"] is not None else "N/A")
    v_fx = (f"{100 * r_fx['post_conv_new_sme'].get(k, 0) / N:.1f}%"
            if r_fx["t_conv"] is not None else "N/A")
    print(_row(f"Post-conv rotation {TOPIC_LABELS[k]}", v_bl, v_fx))

print(sep)

tc_bl = results[BL]["t_conv"]
tc_fx = results[FX]["t_conv"]
print(_row("Convergence timestep",
           str(tc_bl) if tc_bl is not None else "N/A",
           str(tc_fx) if tc_fx is not None else "N/A"))

print(sep)

# Report top holders with their degrees in the fix run
print("\nTop-3 SME holders in fix run (with degrees):")
for k in range(M):
    top = np.argsort(results[FX]["sme_fraction"][:, k])[-3:][::-1]
    info = [(int(n), float(results[FX]["sme_fraction"][n, k]), int(degrees[n]))
            for n in top]
    print(f"  {TOPIC_LABELS[k]}: " +
          ", ".join(f"node {n} ({frac*100:.1f}%, deg={d})"
                    for n, frac, d in info))

# Validation
top_bl_mean = results[BL]["sme_fraction"].max(axis=0).mean()
top_fx_mean = results[FX]["sme_fraction"].max(axis=0).mean()
print(f"\nValidation: baseline top-holder mean = {top_bl_mean*100:.1f}% "
      f"(expected ~99%). ", end="")
print("OK." if top_bl_mean >= 0.90 else "WARNING: does not reproduce prior runs.")

isolated_still_top = any(
    len(set(np.argsort(results[FX]["sme_fraction"][:, k])[-3:]) &
        set(isolated_ids)) > 0
    for k in range(M)
)
if not isolated_still_top:
    print("Fix confirmed: no isolated node appears in top-3 holders. "
          f"Fix run top-holder mean = {top_fx_mean*100:.1f}%.")
else:
    print("WARNING: isolated nodes still appear in top-3 holders after fix. "
          "Check implementation.")


# -- Plots (2 columns x 2 rows) -----------------------------------------------

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    "sim_v10.py -- Isolated-Node Peer-Consistency Fix\n"
    "N=120, M=3, T=500 | SBM(4x30, p_in=0.10, p_out=0.01) | "
    "eps=0.15  eta=0.15  noise=U(-0.3,0.3)",
    fontsize=10
)

for col, label in enumerate(RUN_LABELS):
    r            = results[label]
    sme_fraction = r["sme_fraction"]
    SME_hist     = r["SME_hist"]
    t_conv       = r["t_conv"]

    ax_sme = fig.add_subplot(gs[0, col])
    im = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for b in COMMUNITY_BOUNDARIES:
        ax_sme.axvline(b - 0.5, color="white", linewidth=1.0, linestyle="--")
    # Mark isolated nodes with a red vertical line
    for node in isolated_ids:
        ax_sme.axvline(node, color="red", linewidth=0.8, linestyle=":",
                       alpha=0.7)
    ax_sme.set_title(f"SME fraction  [{label}]\n(red dots = isolated nodes)",
                     fontsize=8)
    ax_sme.set_xlabel("Node", fontsize=8)
    ax_sme.set_ylabel("Topic", fontsize=8)
    ax_sme.set_yticks(range(M))
    ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
    plt.colorbar(im, ax=ax_sme, fraction=0.046, pad=0.04)

    ax_turn = fig.add_subplot(gs[1, col])
    for k in range(M):
        ax_turn.plot(cumulative_unique_sme(SME_hist, k),
                     color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
    if t_conv is not None:
        ax_turn.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                        label=f"conv t={t_conv}")
    ax_turn.axhline(0.6 * N, color="grey", linestyle=":", linewidth=0.8,
                    label="60% target")
    ax_turn.set_title(f"Cumul. unique SME holders  [{label}]", fontsize=9)
    ax_turn.set_xlabel("Timestep", fontsize=8)
    ax_turn.set_ylabel("Unique nodes ever SME", fontsize=8)
    ax_turn.legend(fontsize=7)

plt.savefig(os.path.join(_HERE, "sim_v10_plots.png"),
            dpi=150, bbox_inches="tight")
print(f"\nPlots saved -> {os.path.join(_HERE, 'sim_v10_plots.png')}")

# -- CSV export (Session 52 backfill -- same values as the console printout above) --

import csv

csv_path = os.path.join(_HERE, "sim_v10_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_label", "scope", "community", "topic", "entropy_t0", "entropy_t500", "t_conv",
        "ever_sme", "ever_sme_pct", "target_met",
        "top1_node", "top1_frac", "top1_deg",
        "top2_node", "top2_frac", "top2_deg",
        "top3_node", "top3_frac", "top3_deg",
        "post_conv_new_sme", "post_conv_pct",
    ])
    for label in RUN_LABELS:
        r = results[label]
        t_conv = r["t_conv"]
        sme_fraction = r["sme_fraction"]
        SME_hist = r["SME_hist"]
        pc = r["post_conv_new_sme"]
        entropy_hist = r["entropy_hist"]

        # Overall (per-topic) rows -- same as "SME participation -- overall"
        for k in range(M):
            ever = int(np.any(SME_hist[:, :, k], axis=0).sum())
            pct = 100 * ever / N
            top = np.argsort(sme_fraction[:, k])[-3:][::-1]
            top_degs = [int(degrees[n]) for n in top]
            target_met = "PASS" if ever > 0.6 * N else "FAIL"
            row = [
                label, "overall", "", TOPIC_LABELS[k],
                f"{entropy_hist[0, k]:.4f}", f"{entropy_hist[-1, k]:.4f}",
                t_conv if t_conv is not None else "NA",
                ever, f"{pct:.2f}", target_met,
            ]
            for node, frac, deg in zip(top.tolist(), sme_fraction[top, k].tolist(), top_degs):
                row += [node, f"{frac:.4f}", deg]
            if t_conv is not None:
                post_pct = 100 * pc.get(k, 0) / N
                row += [pc.get(k, 0), f"{post_pct:.2f}"]
            else:
                row += ["NA", "NA"]
            writer.writerow(row)

        # Per-community rows -- same as "SME participation -- per community"
        for i in range(len(BLOCK_SIZES)):
            lo, hi = community_edges[i], community_edges[i + 1]
            for k in range(M):
                ever = int(np.any(SME_hist[:, lo:hi, k], axis=0).sum())
                writer.writerow([
                    label, "community", community_labels[i], TOPIC_LABELS[k],
                    "NA", "NA", t_conv if t_conv is not None else "NA",
                    ever, f"{100*ever/30:.2f}", "NA",
                    "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA",
                    "NA", "NA",
                ])
print(f"CSV saved -> {csv_path}")

plt.show()
