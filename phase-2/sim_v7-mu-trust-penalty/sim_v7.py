"""
sim_v7.py — Decentralised Consensus Learning Network (Phase 2, Run 7)

Anti-oligarchy test: three runs with mu=0, mu=1.0, mu=5.0 from identical
initial conditions. All other parameters held constant at Run 5 values.

Base: sim_v5.py — N=120, SBM(4x30, p_in=0.10, p_out=0.01),
M=3, T=500, alpha=0.3, eta=0.15, lambda=0.1, epsilon=0.15,
noise=Uniform(-0.3, 0.3), seed=42.

New parameter: mu — anti-oligarchy trust penalty (model-spec.md Section 5).
  w_ij *= exp(-mu * w_ij)  applied BEFORE final renormalisation in compute_trust().
  mu=0 path is mathematically identical to sim_v5.py.

The mu=0 run should reproduce Run 5's hub oligarchy (top holders 97-100%).
If it does not, a WARNING is printed before the mu>0 results are interpreted.
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
ALPHA = 0.3             # social learning rate
ETA = 0.15              # innovation rate
LAMBDA = 0.1            # competence plasticity
EPSILON = 0.15          # bounded confidence threshold (D-001)
SME_PCT = 90            # top-10% threshold for SME status
ENTROPY_THRESHOLD = 0.1 # entropy below this on all topics = converged

MU_VALUES = [0.0, 1.0, 5.0]   # anti-oligarchy penalty values to test

BLOCK_SIZES = [30, 30, 30, 30]
P_IN  = 0.10
P_OUT = 0.01
COMMUNITY_BOUNDARIES = [30, 60, 90]

TOPIC_LABELS = [f"Topic {k+1}" for k in range(M)]
TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]

# -- Graph (built once; shared across all three runs) -------------------------

n_blocks = len(BLOCK_SIZES)
probs = [
    [P_IN if r == c else P_OUT for c in range(n_blocks)]
    for r in range(n_blocks)
]
G = nx.stochastic_block_model(BLOCK_SIZES, probs, seed=42)
A = nx.to_numpy_array(G)

degrees = np.array([d for _, d in G.degree()])
print("-- Graph: Stochastic Block Model --")
print(f"  Communities: {len(BLOCK_SIZES)} x {BLOCK_SIZES[0]} nodes | "
      f"p_in={P_IN}  p_out={P_OUT}")
print(f"  Edges: {G.number_of_edges()} | "
      f"Avg degree: {degrees.mean():.2f} | "
      f"Min: {degrees.min()}  Max: {degrees.max()}")

# -- Initial state (shared across all three runs) -----------------------------
# B0, C0 drawn with seed=42 to match sim_v5.py exactly.
# Each run_simulation() re-seeds the noise rng and fast-forwards past these
# draws so that the noise sequence is also identical to sim_v5.py for mu=0.

_init_rng = np.random.default_rng(42)
B0 = _init_rng.uniform(0.0, 1.0, (N, M))
C0 = _init_rng.uniform(0.1, 0.9, (N, M))

# -- Diagnostic (once -- same initial beliefs for all runs) -------------------

print("\n-- Diagnostic: bounded confidence filter activity at t=0 --")
for k in range(M):
    diffs = np.abs(B0[:, k][:, np.newaxis] - B0[:, k][np.newaxis, :])
    within_v1 = (diffs <= 0.3).sum() - N
    within_v7 = (diffs <= EPSILON).sum() - N
    total_pairs = N * (N - 1)
    print(f"  {TOPIC_LABELS[k]}: pairs within eps=0.3: "
          f"{within_v1}/{total_pairs} "
          f"({100 * within_v1 / total_pairs:.1f}%) | "
          f"within eps=0.15: "
          f"{within_v7}/{total_pairs} "
          f"({100 * within_v7 / total_pairs:.1f}%)")


# -- Helper functions ---------------------------------------------------------

def compute_trust(A, c_k, mu=0.0):
    """
    Competence-weighted, row-normalised trust matrix for one topic.
    Falls back to uniform-over-adjacency for isolated nodes.

    If mu > 0, applies the self-damping penalty from model-spec.md Section 5
    before final renormalisation: W *= exp(-mu * W).
    High existing weights are penalised more strongly, limiting trust
    concentration on hub nodes.
    """
    W = A * c_k[np.newaxis, :]
    row_sums = W.sum(axis=1, keepdims=True)
    A_row_sums = A.sum(axis=1, keepdims=True)
    fallback = A / np.where(A_row_sums == 0, 1, A_row_sums)
    W = np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)
    # Anti-oligarchy penalty: apply BEFORE final renormalisation
    if mu > 0:
        W = W * np.exp(-mu * W)
        row_sums2 = W.sum(axis=1, keepdims=True)
        W = np.where(row_sums2 > 0,
                     W / np.where(row_sums2 > 0, row_sums2, 1),
                     fallback)
    return W


def apply_bounded_confidence(W, b_k, epsilon):
    """
    Zero out trust to neighbours whose belief differs by more than epsilon (D-001).
    Re-normalise rows; nodes with no in-epsilon neighbours contribute nothing.
    """
    mask = (np.abs(b_k[:, np.newaxis] - b_k[np.newaxis, :]) <= epsilon).astype(float)
    W_bc = W * mask
    row_sums = W_bc.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, W_bc / safe_sums, 0.0)


def minmax_unit(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


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

def run_simulation(mu):
    """
    Run T=500 steps from identical initial conditions (B0, C0) with the given mu.
    The noise rng is re-seeded to 42 and fast-forwarded past the B0/C0 draws
    so the noise sequence is identical to sim_v5.py (mu=0 exactly reproduces Run 5).
    """
    B = B0.copy()
    C = C0.copy()

    # Match sim_v5.py rng state: seed=42, advance past B0 and C0 draws
    rng = np.random.default_rng(42)
    rng.uniform(0.0, 1.0, (N, M))   # advance past B0 draws
    rng.uniform(0.1, 0.9, (N, M))   # advance past C0 draws

    B_hist   = np.empty((T + 1, N, M))
    C_hist   = np.empty((T + 1, N, M))
    SME_hist = np.zeros((T + 1, N, M), dtype=bool)

    B_hist[0] = B
    C_hist[0] = C

    for t in range(T):
        B_new = np.empty_like(B)
        C_new = np.empty_like(C)

        for k in range(M):
            W_k = compute_trust(A, C[:, k], mu=mu)
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


# -- Run all three ------------------------------------------------------------

results = {}
for mu in MU_VALUES:
    print(f"\n{'=' * 60}")
    print(f"Running simulation: mu={mu}  N={N}  M={M}  T={T} ...")
    results[mu] = run_simulation(mu)
    print("  Done.")


# -- Per-mu console output ----------------------------------------------------

community_edges = [0] + COMMUNITY_BOUNDARIES + [N]
community_labels = [
    f"C{i+1}({community_edges[i]}-{community_edges[i+1]-1})"
    for i in range(len(BLOCK_SIZES))
]

for mu in MU_VALUES:
    r            = results[mu]
    t_conv       = r["t_conv"]
    sme_fraction = r["sme_fraction"]
    SME_hist     = r["SME_hist"]
    pc           = r["post_conv_new_sme"]
    entropy_hist = r["entropy_hist"]

    print(f"\n{'=' * 60}")
    print(f"-- mu={mu} results --")
    print(f"N={N}  M={M}  T={T}  alpha={ALPHA}  eta={ETA}  "
          f"lambda={LAMBDA}  eps={EPSILON}  mu={mu}  "
          f"graph=SBM(4x30)  noise=U(-0.3,0.3)")

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
        flag = "PASS" if ever > 0.6 * N else "FAIL"
        print(f"  {TOPIC_LABELS[k]}: {ever}/{N} ({pct:.1f}%) [{flag}] | "
              f"top holders: nodes {top.tolist()} "
              f"({sme_fraction[top, k].round(2).tolist()})")

    print("\nSME participation -- per community (30 nodes each):")
    for i in range(len(BLOCK_SIZES)):
        lo, hi = community_edges[i], community_edges[i + 1]
        parts = []
        for k in range(M):
            ever = int(np.any(SME_hist[:, lo:hi, k], axis=0).sum())
            parts.append(f"{TOPIC_LABELS[k]}: {ever}/30 ({100*ever/30:.0f}%)")
        print(f"  {community_labels[i]}: " + " | ".join(parts))

    print("\nPost-convergence rotation (new SME holders appearing after t_conv):")
    if t_conv is not None:
        for k in range(M):
            pct = 100 * pc.get(k, 0) / N
            print(f"  {TOPIC_LABELS[k]}: {pc.get(k, 0)} new nodes entered SME "
                  f"after t={t_conv} ({pct:.1f}%)")
    else:
        print("  N/A -- convergence not reached")


# -- Comparison table ---------------------------------------------------------

print(f"\n{'=' * 60}")
print("COMPARISON TABLE")
sep = "+" + "-" * 36 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+"
hdr = f"| {'Metric':<34} | {'mu=0':^7} | {'mu=1.0':^7} | {'mu=5.0':^7} |"
print(sep)
print(hdr)
print(sep)

def _row(label, vals):
    return f"| {label:<34} | {vals[0]:^7} | {vals[1]:^7} | {vals[2]:^7} |"

# Top holder fraction per topic
for k in range(M):
    vals = [f"{results[mu]['sme_fraction'][:, k].max()*100:.1f}%"
            for mu in MU_VALUES]
    print(_row(f"Top holder {TOPIC_LABELS[k]} (% steps)", vals))

print(sep)

# Ever-SME count per topic
for k in range(M):
    vals = [
        f"{int(np.any(results[mu]['SME_hist'][:, :, k], axis=0).sum())}/120"
        for mu in MU_VALUES
    ]
    print(_row(f"Ever-SME {TOPIC_LABELS[k]}", vals))

print(sep)

# Post-convergence rotation per topic
for k in range(M):
    vals = []
    for mu in MU_VALUES:
        r = results[mu]
        if r["t_conv"] is not None and k in r["post_conv_new_sme"]:
            vals.append(f"{100 * r['post_conv_new_sme'][k] / N:.1f}%")
        else:
            vals.append("N/A")
    print(_row(f"Post-conv rotation {TOPIC_LABELS[k]}", vals))

print(sep)

# Convergence timestep
vals = [str(results[mu]["t_conv"]) if results[mu]["t_conv"] is not None else "N/A"
        for mu in MU_VALUES]
print(_row("Convergence timestep", vals))

print(sep)

# Validation check
mu0_top = results[0.0]["sme_fraction"].max(axis=0).mean()
if mu0_top >= 0.90:
    print(f"\nValidation: mu=0 mean top-holder fraction = {mu0_top*100:.1f}% "
          f"-- consistent with Run 5 (97-100%). OK.")
else:
    print(f"\nWARNING: mu=0 mean top-holder fraction = {mu0_top*100:.1f}% -- "
          f"does NOT reproduce Run 5 (expected 97-100%). "
          f"Do NOT interpret mu>0 results until this is diagnosed.")


# -- Plots (3 columns x 2 rows) -----------------------------------------------

fig = plt.figure(figsize=(18, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle(
    "sim_v7.py -- Anti-Oligarchy Test\n"
    "N=120, M=3, T=500 | SBM(4x30, p_in=0.10, p_out=0.01) | "
    "eps=0.15  eta=0.15  noise=U(-0.3,0.3) | mu in {0, 1, 5}",
    fontsize=10
)

for col, mu in enumerate(MU_VALUES):
    r            = results[mu]
    sme_fraction = r["sme_fraction"]
    SME_hist     = r["SME_hist"]
    t_conv       = r["t_conv"]

    # Row 0: SME fraction heatmap
    ax_sme = fig.add_subplot(gs[0, col])
    im = ax_sme.imshow(sme_fraction.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for b in COMMUNITY_BOUNDARIES:
        ax_sme.axvline(b - 0.5, color="white", linewidth=1.0, linestyle="--")
    ax_sme.set_title(f"SME fraction  [mu={mu}]", fontsize=9)
    ax_sme.set_xlabel("Node", fontsize=8)
    ax_sme.set_ylabel("Topic", fontsize=8)
    ax_sme.set_yticks(range(M))
    ax_sme.set_yticklabels(TOPIC_LABELS, fontsize=7)
    plt.colorbar(im, ax=ax_sme, fraction=0.046, pad=0.04)

    # Row 1: Cumulative unique SME holders
    ax_turn = fig.add_subplot(gs[1, col])
    for k in range(M):
        ax_turn.plot(cumulative_unique_sme(SME_hist, k),
                     color=TOPIC_COLORS[k], label=TOPIC_LABELS[k])
    if t_conv is not None:
        ax_turn.axvline(t_conv, color="black", linestyle="--", linewidth=0.8,
                        label=f"conv t={t_conv}")
    ax_turn.axhline(0.6 * N, color="grey", linestyle=":", linewidth=0.8,
                    label="60% target")
    ax_turn.set_title(f"Cumul. unique SME holders  [mu={mu}]", fontsize=9)
    ax_turn.set_xlabel("Timestep", fontsize=8)
    ax_turn.set_ylabel("Unique nodes ever SME", fontsize=8)
    ax_turn.legend(fontsize=7)

plt.savefig(os.path.join(_HERE, "sim_v7_plots.png"),
            dpi=150, bbox_inches="tight")
print(f"\nPlots saved -> {os.path.join(_HERE, 'sim_v7_plots.png')}")

# -- CSV export (Session 52 backfill -- same values as the console printout above) --

import csv

csv_path = os.path.join(_HERE, "sim_v7_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "mu", "scope", "community", "topic", "entropy_t0", "entropy_t500", "t_conv",
        "ever_sme", "ever_sme_pct", "target_met",
        "top1_node", "top1_frac", "top2_node", "top2_frac", "top3_node", "top3_frac",
        "post_conv_new_sme", "post_conv_pct",
    ])
    for mu in MU_VALUES:
        r = results[mu]
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
            target_met = "PASS" if ever > 0.6 * N else "FAIL"
            row = [
                mu, "overall", "", TOPIC_LABELS[k],
                f"{entropy_hist[0, k]:.4f}", f"{entropy_hist[-1, k]:.4f}",
                t_conv if t_conv is not None else "NA",
                ever, f"{pct:.2f}", target_met,
            ]
            for node, frac in zip(top.tolist(), sme_fraction[top, k].tolist()):
                row += [node, f"{frac:.4f}"]
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
                    mu, "community", community_labels[i], TOPIC_LABELS[k],
                    "NA", "NA", t_conv if t_conv is not None else "NA",
                    ever, f"{100*ever/30:.2f}", "NA",
                    "NA", "NA", "NA", "NA", "NA", "NA",
                    "NA", "NA",
                ])
print(f"CSV saved -> {csv_path}")

plt.show()
