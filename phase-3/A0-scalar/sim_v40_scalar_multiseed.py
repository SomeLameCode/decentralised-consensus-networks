"""
sim_v40_scalar_multiseed.py — Phase 3 Series A0-scalar: Multi-Seed Robustness
Study (D=1 / scalar beliefs)

Series A0-scalar runs (Runs 24-32): scalar beliefs, 9 seeds — the same seed
set as Series A0 (D=5 vector, Runs 15-23), for direct comparability.

Closes the gap left by `phase3-plan-A0-amendment.md` (2026-06-18, APPROVED):
A0-scalar was numbered and scoped alongside A0 but never executed. This
script confirms (or refutes) whether Phase 1's original scalar SME-rotation
finding — the paper's single most central claim, measured at seed=42 only
(Run 2) — is seed-robust.

Scalar engine reused UNMODIFIED from
`workspace/simulation/phase-1/sim_v2-null-result/sim_v2.py` (Run 2):
  N=30, M=3, T=200, ALPHA=0.3, ETA=0.15, LAMBDA=0.1, EPSILON=0.15
  Graph: Watts-Strogatz(N=30, k=4, p=0.1)
Both graph topology and initial state are seeded per-seed, same convention
as `sim_v15_multiseed.py` (Series A0).

Convergence definition reused UNMODIFIED from `sim_v4.py`/`sim_v6.py`
(the original scalar post-convergence-rotation runs): first timestep where
belief entropy < ENTROPY_THRESHOLD (0.1) on ALL topics simultaneously.

post_convergence_rotation (the scalar-specific metric that matters here,
per the Session 47 prompt): of all agents that ever held SME status on a
topic (i.e. all "first-entry" events), what fraction first entered SME
status at or after that seed's global convergence timestep t_conv.

Outputs (written to workspace/simulation/phase-3/A0-scalar/):
  sim_v40_scalar_plots.png       — 3x3 grid, one panel per seed
  scalar-multiseed-summary.csv   — one row per seed with key metrics

Run from project root:
  python workspace/simulation/phase-3/A0-scalar/sim_v40_scalar_multiseed.py
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
import os

# -- Parameters (Run 2 exact, per sim_v2.py) ----------------------------------

N       = 30
M       = 3
T       = 200
ALPHA   = 0.3
ETA     = 0.15
LAMBDA  = 0.1
EPSILON = 0.15
SME_PCT = 90              # top-10% SME threshold
ENTROPY_THRESHOLD = 0.1   # entropy below this on all topics = converged (sim_v4/sim_v6 convention)

SEEDS = [1, 7, 13, 21, 42, 77, 99, 123, 256]

TOPIC_COLORS = ["#2196F3", "#4CAF50", "#FF5722"]
TOPIC_LABELS = [f"Topic {k + 1}" for k in range(M)]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Helper functions (unchanged from sim_v2.py) ------------------------------

def compute_trust(A, c_k):
    """
    Competence-weighted, normalised trust matrix for one topic.
    w_tilde_ij = a_ij * c_j; then row-normalise.
    Falls back to uniform-over-adjacency for isolated nodes.
    """
    W = A * c_k[np.newaxis, :]            # (N, N)
    row_sums = W.sum(axis=1, keepdims=True)
    fallback = A / np.where(A.sum(axis=1, keepdims=True) == 0, 1,
                            A.sum(axis=1, keepdims=True))
    return np.where(row_sums > 0, W / np.where(row_sums > 0, row_sums, 1), fallback)


def apply_bounded_confidence(W, b_k, epsilon):
    """
    Zero out trust to neighbours whose belief differs by more than epsilon (D-001).
    Re-normalise rows; nodes with no in-epsilon neighbours keep their own belief
    (social term becomes 0, belief decays toward innovation only).
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


# -- Single-seed simulation ----------------------------------------------------

def run_one_seed(run_num, seed):
    """
    Run the Run-2 scalar simulation for one seed.
    Returns a result dict containing full entropy history and summary metrics.
    """
    rng = np.random.default_rng(seed)

    # Graph: both topology and initial state seeded per-seed (A0 convention)
    G = nx.watts_strogatz_graph(N, k=4, p=0.1, seed=seed)
    A = nx.to_numpy_array(G)

    B = rng.uniform(0.0, 1.0, (N, M))    # beliefs[node, topic]
    C = rng.uniform(0.1, 0.9, (N, M))    # competence[node, topic]

    B_hist = np.empty((T + 1, N, M))
    SME_hist = np.zeros((T + 1, N, M), dtype=bool)

    B_hist[0] = B.copy()
    for k in range(M):
        threshold = np.percentile(C[:, k], SME_PCT)
        SME_hist[0, :, k] = C[:, k] >= threshold

    # Simulation loop (sim_v2.py, unmodified)
    for t in range(T):
        B_new = np.empty_like(B)
        C_new = np.empty_like(C)

        for k in range(M):
            W_k = compute_trust(A, C[:, k])
            W_k = apply_bounded_confidence(W_k, B[:, k], EPSILON)

            social = W_k @ B[:, k]
            u = rng.uniform(-0.05, 0.05, N)
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

        for k in range(M):
            threshold = np.percentile(C[:, k], SME_PCT)
            SME_hist[t + 1, :, k] = C[:, k] >= threshold

    # -- Derived metrics --------------------------------------------------------

    entropy_hist = np.array(
        [[belief_entropy(B_hist[t, :, k]) for k in range(M)] for t in range(T + 1)]
    )   # (T+1, M)

    sme_fraction = SME_hist.mean(axis=0)   # (N, M)
    ever_sme = [int(np.any(SME_hist[:, :, k], axis=0).sum()) for k in range(M)]
    top_holder = [float(sme_fraction[:, k].max()) for k in range(M)]

    # Global convergence: first t where entropy < threshold on ALL topics at once
    converged_steps = np.where(np.all(entropy_hist < ENTROPY_THRESHOLD, axis=1))[0]
    t_conv = int(converged_steps[0]) if len(converged_steps) > 0 else None
    converged = t_conv is not None

    # post_convergence_rotation[k] = fraction of a topic's ever-SME agents
    # whose first-ever SME entry occurred at or after t_conv
    post_convergence_rotation = [None] * M
    if converged:
        for k in range(M):
            pre_ever = set(np.where(np.any(SME_hist[:t_conv, :, k], axis=0))[0].tolist())
            post_ever = set(np.where(np.any(SME_hist[t_conv:, :, k], axis=0))[0].tolist())
            new_post = len(post_ever - pre_ever)
            post_convergence_rotation[k] = (
                new_post / ever_sme[k] if ever_sme[k] > 0 else None
            )

    return {
        "run_num":                   run_num,
        "seed":                      seed,
        "entropy_hist":              entropy_hist,
        "ever_sme":                  ever_sme,
        "top_holder":                top_holder,
        "post_convergence_rotation": post_convergence_rotation,
        "converged":                 converged,
        "t_conv":                    t_conv,
    }


# -- Run all seeds -------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 72)
print("sim_v40_scalar_multiseed.py -- Phase 3 Series A0-scalar | D=1 scalar | Runs 24-32")
print(f"N={N}  M={M}  T={T}  ALPHA={ALPHA}  ETA={ETA}  LAMBDA={LAMBDA}  EPSILON={EPSILON}")
print(f"Seeds: {SEEDS}")
print("=" * 72)

results = []
for i, seed in enumerate(SEEDS):
    run_num = 24 + i
    print(f"  Run {run_num:02d}  seed={seed:>3} ...", end=" ", flush=True)
    r = run_one_seed(run_num, seed)
    results.append(r)
    pcr_str = "  ".join(
        f"T{k+1}:{(r['post_convergence_rotation'][k]*100):.1f}%"
        if r['post_convergence_rotation'][k] is not None else f"T{k+1}:NA"
        for k in range(M)
    )
    print(f"done |  ever_SME={r['ever_sme']}  "
          f"conv={'Y' if r['converged'] else 'N'}  t_conv={r['t_conv']}  |  {pcr_str}")


# -- Console summary table ------------------------------------------------------

print(f"\n{'=' * 72}")
print(f"SUMMARY TABLE -- Series A0-scalar  N={N}  T={T}  ALPHA={ALPHA}  ETA={ETA}  EPSILON={EPSILON}")
print(f"{'Run':>4}  {'Seed':>5}  {'eSME_T1':>8}  {'eSME_T2':>8}  {'eSME_T3':>8}  "
      f"{'minSME':>7}  {'maxTop':>7}  {'conv':>5}  {'t_conv':>7}  "
      f"{'PCR_T1':>7}  {'PCR_T2':>7}  {'PCR_T3':>7}  {'minPCR':>7}")
print("-" * 130)
for r in results:
    pcr_vals = r['post_convergence_rotation']
    pcr_fmt = [f"{v*100:.1f}%" if v is not None else "NA" for v in pcr_vals]
    min_pcr = min([v for v in pcr_vals if v is not None], default=None)
    min_pcr_fmt = f"{min_pcr*100:.1f}%" if min_pcr is not None else "NA"
    print(f"{r['run_num']:>4}  {r['seed']:>5}  "
          f"{r['ever_sme'][0]:>8d}  {r['ever_sme'][1]:>8d}  {r['ever_sme'][2]:>8d}  "
          f"{min(r['ever_sme']):>7d}  {max(r['top_holder']):>7.3f}  "
          f"{'Y' if r['converged'] else 'N':>5}  "
          f"{str(r['t_conv']):>7}  "
          f"{pcr_fmt[0]:>7}  {pcr_fmt[1]:>7}  {pcr_fmt[2]:>7}  {min_pcr_fmt:>7}")


# -- Success criteria check ------------------------------------------------------

print(f"\n{'=' * 72}")
print("SUCCESS CRITERIA  (Session 47 prompt, adapted from phase3-plan-A0-amendment.md)")
print()

n_conv = sum(1 for r in results if r['converged'])
sc1_pass = n_conv == 9

n_sme_pass = sum(1 for r in results if min(r['ever_sme']) >= int(0.9 * N))
sc2_pass = n_sme_pass >= 8

n_pcr_pass = 0
for r in results:
    pcr_vals = [v for v in r['post_convergence_rotation'] if v is not None]
    if r['converged'] and len(pcr_vals) == M and min(pcr_vals) >= 0.70:
        n_pcr_pass += 1
sc3_pass = n_pcr_pass >= 8

print(f"  SC1  Convergence in 9/9 seeds (Section 4.4: structural, not seed-dependent):  "
      f"{n_conv}/9  {'PASS' if sc1_pass else 'FAIL'}")
if not sc1_pass:
    failing = [f"seed={r['seed']}" for r in results if not r['converged']]
    print(f"       FAILING SEEDS: {', '.join(failing)} did not converge by T={T}")

print(f"  SC2  ever-SME >=90% (>={int(0.9*N)}/{N}) in >=8/9 seeds:  "
      f"{n_sme_pass}/9  {'PASS' if sc2_pass else 'FAIL'}")
if not sc2_pass:
    failing = [f"seed={r['seed']} (min_ever_sme={min(r['ever_sme'])}/{N})"
               for r in results if min(r['ever_sme']) < int(0.9 * N)]
    print(f"       FAILING SEEDS: {', '.join(failing)}")

print(f"  SC3  Post-convergence rotation >=70% in >=8/9 seeds:  "
      f"{n_pcr_pass}/9  {'PASS' if sc3_pass else 'FAIL'}")
if not sc3_pass:
    failing = []
    for r in results:
        pcr_vals = [v for v in r['post_convergence_rotation'] if v is not None]
        if not r['converged']:
            failing.append(f"seed={r['seed']} (did not converge)")
        elif len(pcr_vals) < M or min(pcr_vals) < 0.70:
            min_v = min(pcr_vals) if pcr_vals else None
            min_v_fmt = f"{min_v*100:.1f}%" if min_v is not None else "NA"
            failing.append(f"seed={r['seed']} (min_PCR={min_v_fmt})")
    print(f"       FAILING SEEDS: {', '.join(failing)}")

overall_pass = sc1_pass and sc2_pass and sc3_pass
print(f"\n  A0-scalar overall: {'PASS' if overall_pass else 'FAIL -- review before proceeding'}")


# -- Write CSV --------------------------------------------------------------------

csv_path = os.path.join(OUTPUT_DIR, "scalar-multiseed-summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_num", "seed",
        "ever_sme_t1", "ever_sme_t2", "ever_sme_t3", "min_ever_sme",
        "top_holder_t1", "top_holder_t2", "top_holder_t3", "max_top_holder",
        "post_conv_rotation_t1", "post_conv_rotation_t2", "post_conv_rotation_t3",
        "min_post_conv_rotation",
        "converged", "t_conv",
    ])
    for r in results:
        pcr_vals = r['post_convergence_rotation']
        min_pcr = min([v for v in pcr_vals if v is not None], default=None)
        writer.writerow([
            r["run_num"],
            r["seed"],
            r["ever_sme"][0], r["ever_sme"][1], r["ever_sme"][2],
            min(r["ever_sme"]),
            f"{r['top_holder'][0]:.4f}", f"{r['top_holder'][1]:.4f}", f"{r['top_holder'][2]:.4f}",
            f"{max(r['top_holder']):.4f}",
            f"{pcr_vals[0]:.4f}" if pcr_vals[0] is not None else "NA",
            f"{pcr_vals[1]:.4f}" if pcr_vals[1] is not None else "NA",
            f"{pcr_vals[2]:.4f}" if pcr_vals[2] is not None else "NA",
            f"{min_pcr:.4f}" if min_pcr is not None else "NA",
            int(r["converged"]),
            r["t_conv"] if r["t_conv"] is not None else "NA",
        ])
print(f"\nCSV saved -> {csv_path}")


# -- Plots: 3x3 grid, one panel per seed -------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes_flat = axes.flatten()

fig.suptitle(
    f"sim_v40_scalar_multiseed.py — Phase 3 Series A0-scalar | Scalar Beliefs | 9 Seeds\n"
    f"N={N}, M={M}, T={T}, ALPHA={ALPHA}, ETA={ETA}, EPSILON={EPSILON}  |  "
    f"Dashed: entropy convergence threshold ({ENTROPY_THRESHOLD}) | Dotted: t_conv",
    fontsize=10
)

for idx, r in enumerate(results):
    ax = axes_flat[idx]
    for k in range(M):
        ax.plot(r["entropy_hist"][:, k],
                color=TOPIC_COLORS[k], linewidth=0.9, label=TOPIC_LABELS[k])
    ax.axhline(ENTROPY_THRESHOLD, color="black", linestyle="--",
               linewidth=0.7, alpha=0.8)
    if r["t_conv"] is not None:
        ax.axvline(r["t_conv"], color="grey", linestyle=":", linewidth=0.9, alpha=0.8)
    ax.set_title(
        f"Run {r['run_num']}  seed={r['seed']}  "
        f"conv={'Y' if r['converged'] else 'N'}  t_conv={r['t_conv']}",
        fontsize=8
    )
    ax.set_xlabel("Timestep", fontsize=7)
    ax.set_ylabel("Belief entropy", fontsize=7)
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "sim_v40_scalar_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plots saved -> {plot_path}")
print("\n-- sim_v40_scalar_multiseed.py complete --")
