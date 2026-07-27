# Phase 2 — Anti-Oligarchy Fix and Vector-Belief Extension (Runs 7–14)

Phase 2 has two parts, in sequence. Runs 7–10 diagnose and fix an
anti-oligarchy bug found in Phase 1's Run 5/6 SBM topology (three
degree-0 nodes were being scored as maximally competent). Runs 11–14 then
upgrade the belief representation from a Phase-1-style scalar
(`b_i(k,t) ∈ [0,1]`) to a `D=5` unit vector on the hypersphere
(`b_i(k,t) ∈ ℝ⁵`), and characterise how that changes the model's dynamics —
concluding that vector beliefs under Gaussian initialisation settle into a
**persistent diversity regime** rather than converging to consensus.

## Folder naming convention

Same pattern as Phase 1: each run lives in its own
`sim_vN-<short-description>/` folder containing that run's script
(`sim_vN.py`) and its output plot (`sim_vN_plots.png`). Confirmed for all
8 folders (Runs 7–14) — with one addition: Run 14's folder also contains
`run14-findings.md`, a written findings block (not just script + plot),
since Run 14 closed out Phase 2 with a specific adopted conclusion (see
below).

```
phase-2/
├── sim_v7-mu-trust-penalty/       sim_v7.py,  sim_v7_plots.png
├── sim_v8-degree-normalisation/   sim_v8.py,  sim_v8_plots.png
├── sim_v9-uniform-c0-init/        sim_v9.py,  sim_v9_plots.png
├── sim_v10-isolated-node-fix/     sim_v10.py, sim_v10_plots.png
├── sim_v11-vector-beliefs-d5/     sim_v11.py, sim_v11_plots.png
├── sim_v12-gaussian-init-fix/     sim_v12.py, sim_v12_plots.png
├── sim_v13-filter-off-kappa0/     sim_v13.py, sim_v13_plots.png
├── sim_v14-t1000-extended/        sim_v14.py, sim_v14_plots.png, run14-findings.md
└── _archive/                      README.md  (superseded, see below)
```

(`__pycache__/` also appears at `phase-2/` root from local script runs —
it's git-ignored and not part of this convention.)

## Runs 7–14

| Run | Folder | Description |
|---|---|---|
| 7 | `sim_v7-mu-trust-penalty/` | Anti-oligarchy test: three runs (mu=0, 1.0, 5.0) applying a trust-dispersion penalty `w_ij *= exp(-mu*w_ij)` on top of Run 5's SBM setup, testing whether it breaks up Run 5/6's hub-oligarchy pattern. |
| 8 | `sim_v8-degree-normalisation/` | Run 7 showed the mu penalty has no effect because the oligarchy compounds through competence, not trust — this run instead divides each node's peer-consistency delta by its degree, so hubs and leaves get equally-scaled signals. |
| 9 | `sim_v9-uniform-c0-init/` | Tests whether the three still-dominant nodes (16, 45, 85) owe their advantage to a lucky initial competence draw (H1) or a structural/bridging position (H2), by setting every node's initial competence to a uniform 0.5. |
| 10 | `sim_v10-isolated-node-fix/` | Root cause found: nodes 16/45/85 have degree=0 in the SBM draw, so their peer-consistency delta is identically 0 — which `minmax_unit()` was mapping to 1.0 (maximally competent). Fixes with a two-guard NaN-exclude-then-zero pattern so isolated nodes get no update instead of a false-maximal one. |
| 11 | `sim_v11-vector-beliefs-d5/` | First vector-belief run: upgrades scalar beliefs in [0,1] to unit vectors in ℝ⁵ (D=5), switching the bounded-confidence filter to a cosine-similarity threshold (KAPPA=0.5) and innovation noise to Gaussian. Direct comparison baseline: Run 1. |
| 12 | `sim_v12-gaussian-init-fix/` | Fixes a positive-orthant initialisation bias in Run 11 (uniform[0,1] init gave initial cosim ≈0.75) by initialising from N(0,1) and normalising, giving a uniform spherical distribution (cosim ≈0 at t=0). |
| 13 | `sim_v13-filter-off-kappa0/` | Run 12 showed KAPPA=0.5 nearly freezes social learning under the new Gaussian init (too few neighbour pairs pass the cosine filter). This run sets KAPPA=0.0 (filter off) to isolate whether the threshold, not the initialisation, was the root cause of the freeze. |
| 14 | `sim_v14-t1000-extended/` | Extends Run 13 from T=200 to T=1000 to let the network fully converge (or not) and test whether post-convergence rotation — the scalar-case finding — also holds for vector beliefs. **Result (see `run14-findings.md`): it does not converge at all within the tested horizon** — the network instead settles into a persistent diversity regime, adopted as Option B and reported as Phase 2's final finding, not a failure to reach the original convergence question. |

Each description above is pulled directly from that script's own docstring
(and, for Run 14, from `run14-findings.md`'s own result statement), not
paraphrased from memory.

## The `_archive/` subfolder

`_archive/README.md` is a **superseded early planning document, first
committed 2026-06-17** (confirmed via `git log`). It describes an earlier,
different naming scheme (e.g. `sim_v7_antioligarchy.py`,
`sim_v7_vectorbeliefs.py` as two separate planned scripts) that predates
the actual `sim_vN-<description>/` per-run folder reorganisation used by
every run above. **It is kept for historical reference only — it does not
describe the current structure and should not be used as guidance.**

## How to reproduce

From the project root, after installing `workspace/simulation/requirements.txt`:

```
pip install -r workspace/simulation/requirements.txt

python workspace/simulation/phase-2/sim_v7-mu-trust-penalty/sim_v7.py
python workspace/simulation/phase-2/sim_v8-degree-normalisation/sim_v8.py
python workspace/simulation/phase-2/sim_v9-uniform-c0-init/sim_v9.py
python workspace/simulation/phase-2/sim_v10-isolated-node-fix/sim_v10.py
python workspace/simulation/phase-2/sim_v11-vector-beliefs-d5/sim_v11.py
python workspace/simulation/phase-2/sim_v12-gaussian-init-fix/sim_v12.py
python workspace/simulation/phase-2/sim_v13-filter-off-kappa0/sim_v13.py
python workspace/simulation/phase-2/sim_v14-t1000-extended/sim_v14.py
```

Each script is self-contained and writes its own plot in place — no shared
setup or run order is required between them (though each run's docstring
describes what it changed relative to the previous one, for narrative
context).

## What's saved vs. not

Same convention as Phase 1: each run saves its output plot
(`sim_vN_plots.png`) only — **no raw data CSV is written for any Phase 2
run.** All console-reported numbers can be reproduced exactly by
re-running the corresponding script: every run seeds both the graph and
the initial state (seed=42 throughout), so re-execution is fully
deterministic.

## Where this feeds into the paper

Phase 2 results are reported in Paper 1, Section 4.6 (vector-belief
extension and the R⁵ geometry / diversity-regime finding), with the
isolated-node fix (Runs 9–10) reported in Section 4.3.
