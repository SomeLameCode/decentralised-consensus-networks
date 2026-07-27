# Phase 1 — Scalar-Belief Baseline (Runs 1–6)

Phase 1 is the project's proof-of-concept baseline: agents hold a single
scalar belief per topic (`b_i(k,t) ∈ [0,1]`), updated via competence-weighted
trust and a bounded-confidence filter (D-001) over a Watts-Strogatz
small-world graph (D-002; Run 5 onward also tests a Stochastic Block Model
topology). Runs 1–6 establish the model's three headline PoC findings —
SME rotation, topology invariance, and scale invariance — before the belief
representation is extended from a scalar to a D-dimensional unit vector in
Phase 2.

## Folder naming convention

Each run lives in its own `sim_vN-<short-description>/` folder, containing
exactly that run's script (`sim_vN.py`) and its output plot
(`sim_vN_plots.png`). Confirmed for all 6 folders — no exceptions, no extra
files:

```
phase-1/
├── sim_v1-baseline/          sim_v1.py, sim_v1_plots.png
├── sim_v2-null-result/       sim_v2.py, sim_v2_plots.png
├── sim_v3-widened-noise/     sim_v3.py, sim_v3_plots.png
├── sim_v4-n120-scaleup/      sim_v4.py, sim_v4_plots.png
├── sim_v5-sbm-topology/      sim_v5.py, sim_v5_plots.png
└── sim_v6-n10k-sparse/       sim_v6.py, sim_v6_plots.png
```

## Runs 1–6

| Run | Folder | Description |
|---|---|---|
| 1 | `sim_v1-baseline/` | Baseline scalar-belief PoC: N=30, M=3, T=200, ALPHA=0.3, ETA=0.05, LAMBDA=0.1, EPSILON=0.3, Watts-Strogatz(k=4, p=0.1). |
| 2 | `sim_v2-null-result/` | Tighter bounded-confidence filter (EPSILON=0.15, was 0.3) and higher innovation rate (ETA=0.15, was 0.05) — null result, still converges to full consensus. |
| 3 | `sim_v3-widened-noise/` | Root-cause fix for Runs 1–2's null result: widens innovation noise to Uniform(-0.3, 0.3) (was ±0.05), since the old noise cap made innovation's per-step contribution (±0.0075) negligible against the social pull term (~0.3). |
| 4 | `sim_v4-n120-scaleup/` | Scale-up to N=120, T=500. Tests the >60% ever-SME target and whether SME rotation continues *after* belief convergence (post-convergence rotation) — convergence defined as entropy < threshold on all topics. |
| 5 | `sim_v5-sbm-topology/` | Swaps topology to a Stochastic Block Model (4 communities × 30 nodes, N=120) to test whether block/community structure slows convergence or reduces SME participation relative to Run 4's Watts-Strogatz result. |
| 6 | `sim_v6-n10k-sparse/` | Scale to N=10,000 via a sparse (`scipy.sparse` CSR) implementation, SBM topology (500 communities × 20 nodes) — confirms the findings hold at two further orders of magnitude. |

Each description above is pulled directly from that script's own docstring,
not paraphrased from memory.

## How to reproduce

From the project root, after installing `workspace/simulation/requirements.txt`:

```
pip install -r workspace/simulation/requirements.txt

python workspace/simulation/phase-1/sim_v1-baseline/sim_v1.py
python workspace/simulation/phase-1/sim_v2-null-result/sim_v2.py
python workspace/simulation/phase-1/sim_v3-widened-noise/sim_v3.py
python workspace/simulation/phase-1/sim_v4-n120-scaleup/sim_v4.py
python workspace/simulation/phase-1/sim_v5-sbm-topology/sim_v5.py
python workspace/simulation/phase-1/sim_v6-n10k-sparse/sim_v6.py
```

Each script is self-contained and writes its own plot in place — no
shared setup or run order is required between them.

## What's saved vs. not

Each run saves its output plot (`sim_vN_plots.png`) only — **no raw data
CSV is written for any Phase 1 run.** All console-reported numbers (final
beliefs, entropy, SME fractions, etc.) can be reproduced exactly by
re-running the corresponding script: every run seeds both the graph
(`nx.*_graph(..., seed=...)`) and the initial state (`np.random.default_rng(seed)`
or equivalent), so re-execution is fully deterministic.

## Where this feeds into the paper

Phase 1 results are reported in Paper 1, Sections 4.1–4.5.
