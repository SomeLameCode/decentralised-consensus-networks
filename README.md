# Decentralised Consensus Networks

Agent-based simulations exploring how a decentralised network of agents
forms (or fails to form) shared beliefs, under competence-weighted trust
and bounded-confidence social learning over a graph.

Each agent ("node") holds a belief per topic — either a scalar in `[0, 1]`
or, from Phase 2 onward, a unit vector in `ℝ^D` — plus a competence score
per topic that rises when the agent's belief agrees with its trusted
neighbours. Agents only listen to neighbours whose belief is close enough
to their own (bounded confidence), and trust is weighted by neighbour
competence, so subject-matter expertise emerges and rotates endogenously
rather than being assigned.

The project is organised as a sequence of numbered simulation runs, grouped
into three phases:

| Phase | Runs | Focus |
|---|---|---|
| [`phase-1/`](phase-1/README.md) | 1–6 | Scalar-belief proof of concept: SME rotation, topology invariance, scale invariance (Watts-Strogatz and Stochastic Block Model graphs, up to N=10,000). |
| [`phase-2/`](phase-2/README.md) | 7–14 | Fixes an anti-oligarchy bug (isolated nodes scored as maximally competent), then upgrades beliefs from a scalar to a D=5 unit vector — finding that vector beliefs settle into a **persistent diversity regime** rather than full consensus. |
| [`phase-3/`](phase-3/README.md) | 15–84 | Multi-seed robustness checks, an ETA (innovation-rate) sweep, a belief-dimensionality sweep spanning five qualitatively distinct dynamical regimes, and a high-dimensional equilibrium study locating a "Pareto point" between collective consensus and residual belief diversity. |

Each phase's `README.md` is the authoritative record of what every run
tested and found, including its exact parameters — pulled directly from
that run's own script docstring.

## Repository layout

Every run lives in its own folder alongside that run's script, its output
plot, and (from Phase 3 onward) a summary CSV of the run's numeric results:

```
phase-1/sim_v1-baseline/       sim_v1.py, sim_v1_plots.png
phase-2/sim_v11-vector-beliefs-d5/   sim_v11.py, sim_v11_plots.png
phase-3/A2-D-sweep/            sim_v19_D_sweep.py, D-sweep-summary.csv, sim_v19_D_sweep_plots.png
```

Scripts are independent and self-contained — there's no shared setup
module and no required run order between them.

## Running a simulation

Install the dependencies actually imported across the scripts (there's no
package manifest — this is a research codebase, not a distributable
package):

```
pip install numpy networkx matplotlib scipy
```

Then run any script directly with Python 3, from anywhere:

```
python phase-1/sim_v1-baseline/sim_v1.py
```

Each script prints a console summary and writes its plot (and CSV, where
applicable) next to itself. Every run seeds both its graph
(`nx.*_graph(..., seed=...)`) and its RNG (`np.random.default_rng(seed)`),
so re-running any script reproduces its numbers exactly. Some Phase 3
long-horizon runs (T up to 100,000 timesteps) take noticeably longer than
the rest.

## Findings so far

- **SME rotation**: competence-based "expert" status is not permanently
  held by the same agents — it rotates across the network over time, and
  continues to rotate even after belief convergence (Phase 1).
- **An isolated-node bug**: nodes with zero graph degree were being scored
  as maximally competent, producing a false hub-oligarchy pattern — traced
  to a min-max normalisation edge case and fixed with an explicit
  NaN-exclude guard (Phase 2, Runs 9–10).
- **Scalar vs. vector beliefs are not a simple dimensionality upgrade**:
  under identical social-learning parameters, scalar beliefs converge
  readily, while D=5 vector beliefs on the hypersphere settle into a
  persistent diversity regime instead (Phase 2, Run 14).
- **A dimensionality-dependent regime structure**: sweeping belief
  dimension D reveals distinct dynamical regimes, including a non-monotone
  relationship between D and convergence speed, and a high-dimensional
  "Pareto point" (D≈150) balancing collective consensus against residual
  belief diversity, robust across seeds (Phase 3).

## License

[MIT](LICENSE)
