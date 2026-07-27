# Phase 3 — Robustness, Dimensionality Sweep, and High-D Equilibrium (Runs 15–84)

Phase 3 is the largest and most structurally varied phase: multi-seed
robustness checks on the Phase-1/2 findings (Series A0, A0-scalar, A1),
a belief-dimensionality (D) sweep characterising five qualitatively
distinct dynamical regimes (Series A2), and a high-dimensional
equilibrium study locating a Pareto point between collective consensus
and residual belief diversity, then testing its own seed-robustness
(Series A3a/A3b/A3c).

## Folder naming convention

Each series gets its own named subfolder (not the `sim_vN-description`
per-run pattern used in Phase 1/2), containing that series' script(s),
a summary CSV, and its output plot(s). Confirmed for all 13 folders —
with two folders departing from the strict "one script, one CSV, one
plot" shape, noted plainly rather than smoothed over:

- **`A3b-fine-D-sweep/`** has one script (`sim_v21_A3b.py`) but two plots
  (`_subrun1_plots.png`, `_subrun2_plots.png`), since the script runs two
  distinct sub-experiments (a D sweep, then an ETA sensitivity check) in
  one pass and shares one summary CSV between them.
- **`A3c-seed-robustness/`** has two simulation scripts
  (`sim_v22_A3c_seedcheck.py` for the numeric results,
  `sim_v23_A3c_convergence_plots.py` — a later, deterministic rerun of
  the identical configs purely to capture per-timestep history for
  plotting, since `sim_v22` never saved it) plus a two-file
  traceability addition, `degree-verification.py` / `.csv` (see below).

```
phase-3/
├── A0-multiseed/          sim_v15_multiseed.py,        multiseed-summary.csv,        sim_v15_vector_plots.png
├── A0-scalar/             sim_v40_scalar_multiseed.py, scalar-multiseed-summary.csv, sim_v40_scalar_plots.png
├── A1-eta-sweep/          sim_v16_eta_sweep.py,         eta-sweep-summary.csv,        sim_v16_vector_plots.png
├── A1-longrun-T5000/      sim_v17_longrun.py,           longrun-summary.csv,          sim_v17_longrun_plots.png
├── A1-ultralong-T20000/   sim_v18_seed42_ultralong.py,  seed42-ultralong-summary.csv, sim_v18_seed42_ultralong_plots.png
├── A2-D-sweep/            sim_v19_D_sweep.py,           D-sweep-summary.csv,          sim_v19_D_sweep_plots.png
├── A2-D2-longrun/         sim_v19b_D2_longrun.py,       D2-longrun-summary.csv,       sim_v19b_D2_longrun_plots.png
├── A2-D5-longrun/         sim_v19c_D5_longrun.py,       D5-longrun-summary.csv,       sim_v19c_D5_longrun_plots.png
├── A2-D10-longrun/        sim_v19d_D10_longrun.py,      D10-longrun-summary.csv,      sim_v19d_D10_longrun_plots.png
├── A2-D20-longrun/        sim_v19e_D20_longrun.py,      D20-longrun-summary.csv,      sim_v19e_D20_longrun_plots.png
├── A3a-highD-sweep/       sim_v20_highD_sweep.py,       highD-sweep-summary.csv,      sim_v20_highD_sweep_plots.png
├── A3b-fine-D-sweep/      sim_v21_A3b.py,                A3b-summary.csv,             sim_v21_A3b_subrun1/2_plots.png
└── A3c-seed-robustness/   sim_v22_A3c_seedcheck.py + sim_v23_A3c_convergence_plots.py, A3c-summary.csv +
                            A3c-convergence-history.csv, sim_v23_A3c_convergence_plots.png,
                            + degree-verification.py / degree-verification.csv
```

## Series, run numbers, and purpose

Pulled directly from each script's own docstring — the same information
as Paper 1 Section 3's run table, framed here for a code reader rather
than a paper reader (spot-checked against the paper's own table, see
note below).

| Series | Folder | Run range | Purpose |
|---|---|---|---|
| A0 | `A0-multiseed/` | 15–23 | Multi-seed robustness check (9 seeds) on the D=5 vector-belief diversity-regime finding (Run 14). |
| A0-scalar | `A0-scalar/` | 24–32 | The same 9-seed robustness check, but for scalar beliefs (D=1) — closes a numbering gap reserved since 2026-06-18 but not actually executed until 2026-07-27 (Session 47). Tests whether Run 2's scalar SME-rotation/post-convergence-rotation finding, previously only checked at seed=42, holds across seeds. |
| A1 (ETA sweep) | `A1-eta-sweep/` | 33–50 | 6 ETA values × 3 seeds (21, 42, 123) at D=5 — finds the ETA threshold where the diversity regime becomes seed-stable rather than determined by initial geometry alone. |
| A1 (long run) | `A1-longrun-T5000/` | 51–53 | Extends the same 3 A1 seeds to T=5,000 at ETA=0.15, to test whether the diversity pattern is genuinely structural (persists given more time) or just slow convergence. |
| A1 (ultra-long) | `A1-ultralong-T20000/` | 54 | Extends the seed=42 case further to T=20,000 — reveals the "cascade" dynamic (long pre-cascade wandering, then a near-vertical phase transition per topic). |
| A2 (D sweep) | `A2-D-sweep/` | 55–59 | Belief-dimensionality sweep at seed=42, D ∈ {1, 2, 5, 10, 20}, one run number reserved per D (55→D=1, 56→D=2, 57→D=5, 58→D=10, 59→D=20). The script itself currently only actively executes D∈{1,2} (`D_VALUES = [1, 2]`) — D=5/10/20's own numbered runs are the three `A2-D*-longrun/` folders below, run separately. |
| A2 (D=2 long) | `A2-D2-longrun/` | extends 56 | Extends Run 56 (D=2) from T=2,000 to T=20,000 after the short run left Topic 1 plateaued short of convergence — this is the D=2 dataset Paper 1 Section 4.8 actually cites (t_conv=2,388), not the original short run. |
| A2 (D=5 long) | `A2-D5-longrun/` | 57 | D=5 at T=20,000 — an explicit validation re-run of Run 54 (`A1-ultralong-T20000/`), same seed/config/RNG order, that also serves as Series A2's own D=5 sweep checkpoint. Confirmed byte-identical to Run 54 (Session 49). |
| A2 (D=10 long) | `A2-D10-longrun/` | 58 | D=10 at T=50,000 — tests whether convergence timescales keep growing with D, or the relationship is non-monotone (finding: non-monotone — Topics 2/3 converge far *faster* than at D=5). |
| A2 (D=20 long) | `A2-D20-longrun/` | 59 | D=20 at T=100,000 — tests whether the D=10 speed-up continues (finding: no — settles into a stable "noise-floor plateau" instead, never crossing the convergence threshold). |
| A3a (high-D sweep) | `A3a-highD-sweep/` | 60–64 | D ∈ {20, 50, 100, 200, 500} at T=5,000, seed=42 — searches for the D where equilibrium cosim lands near 0.80 (the "Pareto point": strong collective direction with meaningful residual diversity). |
| A3b (fine D sweep + ETA sensitivity) | `A3b-fine-D-sweep/` | 65–69 | Sub-run 1 (65–67): fine D sweep {125, 150, 175} to locate the Pareto point precisely. Sub-run 2 (68–69, reusing Run 66 = D=150/ETA=0.15 as the third point): ETA sensitivity at D=150, testing whether lower ETA recovers rotation health without collapsing equilibrium cosim toward 1.0. |
| A3c (seed robustness) | `A3c-seed-robustness/` | 70–84 | Tests whether the D=150/175/200 Pareto-point equilibrium values (previously seed=42 only) hold across 5 new seeds (11, 99, 7, 3, 21) — 3 D values × 5 seeds = 15 runs. `sim_v23` then reruns the same 15 configs plus the seed=42 baseline (18 runs total, not new run numbers) purely to plot per-timestep convergence curves that `sim_v22` never saved. |

**Note on `_v6.md`:** this table was spot-checked against Section 3 of
the paper as instructed, but as of this writing **no `_v6.md` file
exists** in `_project-delivery/white-paper/PAPER-1-final/` — only
`_v1.md` through `_v5.md`. Every row above was checked against `_v5.md`
(the latest version actually present) instead, and matched exactly,
including the A0-scalar row (24–32) and the D=2 run using the T=20,000
long-run dataset rather than the original short run. Flagging the
missing `_v6.md` rather than assuming it exists, per this session's own
instruction to report discrepancies rather than reconcile them silently.

## The `degree-verification.py` / `.csv` addition

`A3c-seed-robustness/` also contains `degree-verification.py` and its
output `degree-verification.csv`, which are **not a numbered simulation
run** — they exist purely to make one specific paper claim independently
checkable. Paper 1 Section 4.9 states that seed 11's Watts-Strogatz graph
has an unusually low-degree node (min_degree=2) compared to every other
tested seed (min_degree=3), and uses this to explain a seed-11 outlier
result. `sim_v22_A3c_seedcheck.py` used each seed's degree sequence
internally (as an isolated-node guard) but never printed or saved the
minimum degree itself, so that specific claim had no standalone evidence
file. `degree-verification.py` regenerates the exact same six graphs
(identical N/k/p and seed, no simulation re-run) and asserts the claim
directly: seed 11 → min_degree=2, seeds 99/7/3/21/42 → min_degree=3.

## How to reproduce

From the project root, after installing `workspace/simulation/requirements.txt`:

```
pip install -r workspace/simulation/requirements.txt

python workspace/simulation/phase-3/A0-multiseed/sim_v15_multiseed.py
python workspace/simulation/phase-3/A0-scalar/sim_v40_scalar_multiseed.py
python workspace/simulation/phase-3/A1-eta-sweep/sim_v16_eta_sweep.py
python workspace/simulation/phase-3/A1-longrun-T5000/sim_v17_longrun.py
python workspace/simulation/phase-3/A1-ultralong-T20000/sim_v18_seed42_ultralong.py
python workspace/simulation/phase-3/A2-D-sweep/sim_v19_D_sweep.py
python workspace/simulation/phase-3/A2-D2-longrun/sim_v19b_D2_longrun.py
python workspace/simulation/phase-3/A2-D5-longrun/sim_v19c_D5_longrun.py
python workspace/simulation/phase-3/A2-D10-longrun/sim_v19d_D10_longrun.py
python workspace/simulation/phase-3/A2-D20-longrun/sim_v19e_D20_longrun.py
python workspace/simulation/phase-3/A3a-highD-sweep/sim_v20_highD_sweep.py
python workspace/simulation/phase-3/A3b-fine-D-sweep/sim_v21_A3b.py
python workspace/simulation/phase-3/A3c-seed-robustness/sim_v22_A3c_seedcheck.py
python workspace/simulation/phase-3/A3c-seed-robustness/sim_v23_A3c_convergence_plots.py
python workspace/simulation/phase-3/A3c-seed-robustness/degree-verification.py
```

Each script is self-contained. Unlike Phase 1/2, **every Phase 3 script
also writes a summary CSV** (`*-summary.csv` or `degree-verification.csv`)
alongside its plot — Phase 3's multi-seed/multi-D designs need per-run
numeric results in a comparable table, not just a plot per run. Several
long-horizon scripts (T=20,000–100,000) take materially longer to run
than Phase 1/2's scripts; `A2-D20-longrun/sim_v19e_D20_longrun.py`
(T=100,000) is the slowest.

All scripts are fully deterministic: every run seeds both the graph
(`nx.*_graph(..., seed=...)`) and the initial state
(`np.random.default_rng(seed)`), so re-running any script reproduces its
CSV/plot numbers exactly.

## Where this feeds into the paper

Phase 3 results are reported in Paper 1, Section 4.2a (Series A0-scalar),
and Sections 4.7–4.9 (Series A0/A1 timescale heterogeneity and cascade
dynamics, Series A2 dimensionality-regime structure, and Series A3
high-dimensional equilibrium / diversity–expertise trade-off,
respectively).
