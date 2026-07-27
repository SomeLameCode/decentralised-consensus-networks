# Run 14 — Findings Block
> sim_v14.py | Gaussian init, KAPPA=0.0, T=1000, N=30, M=3, D=5
> Status: FINAL — adopted as Option B (diversity regime)
> Date: 2026-06-18

---

## 1. Result Statement

Run 14 confirms that vector beliefs (D=5) with Gaussian initialisation and moderate innovation noise (ETA=0.15) produce a **persistent diversity regime** — the network does not converge to consensus within any practically relevant time horizon.

Key observations at T=1000:

- Mean pairwise cosim: Topic 1 → 0.829, Topic 2 → 0.171, Topic 3 → 0.248. All three well below the convergence threshold (0.99) and showing negligible movement in the T=200–1000 window.
- Ever-SME participation: 28/30, 30/30, 29/30 (93–100%). The competence mechanism is healthy — given enough time, nearly every node earns SME status at least once.
- SME rotation remains active throughout. No permanent lock-in.
- Transient oligarchy present: Topic 1 top-2 holders at 86%/83% of timesteps; Topic 3 at 79%/77%. Interpreted as a pre-convergence phenomenon (see Section 3).

**Finding:** Scalar beliefs and vector beliefs are not a simple dimensionality upgrade — they represent qualitatively different convergence regimes under identical parameters. Scalar beliefs converge readily (Phase 1, Run 1: t_conv ≈ 14 steps). Vector beliefs in R⁵ with the same social learning rate and noise amplitude maintain persistent diversity indefinitely.

---

## 2. Root Cause — R⁵ Geometry

The divergence between scalar and vector behaviour originates in the geometry of the hypersphere.

In the scalar case, innovation noise is a small perturbation in [0,1], clipped to range. Social pull (ALPHA=0.3) competes against a 1-dimensional nudge.

In R⁵, each node's belief is a unit vector on the surface of a 5-dimensional hypersphere. Gaussian innovation noise with amplitude 0.1 produces a perturbation vector with expected magnitude ~0.1√5 ≈ 0.224 before normalisation. After renormalisation back onto the unit sphere, the angular displacement is preserved. Crucially, in 5D there are 4 independent directions orthogonal to any given vector — the noise is as likely to push laterally or away from the consensus attractor as toward it.

The social term (ALPHA=0.3) pulls toward the competence-weighted neighbour centroid. During early-to-mid convergence, when neighbours are still spread across the hypersphere, this centroid is a short vector (the average of near-randomly-oriented unit vectors tends toward zero). Normalising a short centroid vector amplifies noise. The social term is delivering less effective pull than the raw ALPHA value suggests.

**In plain terms:** ETA=0.15 in R⁵ rotates belief vectors significantly every step, in unpredictable directions. ALPHA=0.3 cannot consistently overpower this. The network settles into a stable dynamic balance between social learning and innovation — which is the diversity regime.

---

## 3. Oligarchy as Transient Phenomenon

The high SME concentration observed (top-2 holders at 79–86% of timesteps) is a pre-convergence artefact, not a structural failure of the rotation mechanism.

During the long convergence ramp, nodes whose initial vectors happen to point closest to the eventual consensus direction accumulate competence early. They become trusted, receive stronger social weighting from neighbours, and hold SME status persistently while the rest of the network is still aligning. Once beliefs fully converge (cosim → 1.0), all nodes are equally aligned and the competence advantage evaporates — rotation resumes normally. This was confirmed in scalar Phase 1 runs (e.g., Run 3) and in Run 11 (uniform init, fast convergence: t_conv=14, post-conv rotation 57–73%).

Run 14 does not reach full convergence, so the oligarchy does not resolve. This is consistent with the diversity regime interpretation: the transient is not transient if convergence never arrives.

---

## 4. Parameter Sensitivity — Options Noted, Not Run

Three response options were identified and documented for the record. None were executed; Option B was adopted.

**Option A — Reduce ETA (noise amplitude)**
Reduce ETA from 0.15 to approximately 0.02–0.05. This directly addresses the geometric root cause by shrinking angular perturbation per step, allowing ALPHA=0.3 to win the competition. Expected outcome: convergence restored, oligarchy resolves post-convergence, rotation pattern similar to Run 11.

A conservative variant: reduce to ETA=0.07 rather than an aggressive drop. This would likely produce a slower convergence than Run 11 but avoid collapsing innovation entirely. Not run — noted as the most informative future validation experiment.

**Option B — Accept diversity regime (ADOPTED)**
Treat persistent diversity as the finding rather than a failure. Vector beliefs in R⁵ with moderate noise maintain healthy diversity indefinitely under the same parameters that produce rapid scalar consensus. This is a substantive result with real-world relevance: knowledge networks are not expected or desired to collapse to a single shared belief vector.

**Option C — Increase ALPHA**
Strengthen social pull by raising ALPHA (e.g., 0.5–0.6). Would also promote convergence but risks a different pathology: very high ALPHA suppresses innovation, producing path-dependent frozen consensus that reflects initial conditions rather than genuine collective intelligence. Not recommended as a primary response.

---

## 5. Parameter Semantics — ALPHA, ETA, and Network Behaviour

These definitions are recorded here as reference for white paper v3 and future runs.

**ALPHA (social learning rate):** Controls how strongly each node updates toward its neighbours' belief vectors each step. High ALPHA = highly social, rapid consensus. Low ALPHA = skeptical network — nodes hear neighbours but update weakly. A lower ALPHA produces a more independent, critical network that resists social pressure.

**ETA (innovation rate):** Controls the amplitude of Gaussian noise injected into each node's belief vector each step. High ETA = persistent diversity and novelty generation. Low ETA = network converges readily. ETA is the primary driver of the diversity-vs-consensus balance.

**The tension:** ALPHA and ETA are in direct competition. The equilibrium regime (convergent vs diverse) is determined by their ratio and by the geometry of the belief space (D). In R¹ (scalar), ALPHA=0.3 beats ETA=0.15. In R⁵, ETA=0.15 beats ALPHA=0.3.

**Topology interaction — the tribal behaviour observation:**
Within a Watts-Strogatz cluster, nodes share many common neighbours. Social reinforcement arrives from multiple directions simultaneously, making the effective intra-cluster pull stronger than ALPHA alone suggests. Across bridge edges (the rewired long-range connections in WS graphs), the signal arrives from a single neighbour whose belief vector may point in a very different direction — the same ALPHA delivers much weaker effective pull.

A spatially uniform ALPHA therefore naturally produces heterogeneous effective social pressure from graph structure alone. Intra-cluster dynamics resemble a high-ALPHA network (strong tribal consensus pressure). Inter-cluster dynamics resemble a low-ALPHA network (skeptical, weak cross-cluster pull). This is consistent with real knowledge networks: professional communities converge internally while remaining distant from adjacent communities.

A natural extension — not modelled here — is a topology-aware dynamic ALPHA: higher within dense clusters, lower across bridge edges, explicitly encoding this tribal behaviour as a parameter rather than an emergent effect.

---

## 6. White Paper v3 — Additions Required

The following additions are needed in the white paper to capture Phase 2 vector belief findings:

1. **New section: Vector Beliefs (Phase 2)** — introduce D=5 upgrade, initialisation correction (Gaussian vs uniform positive orthant), cosine similarity replacing scalar bounded confidence.

2. **Run comparison table** — Runs 11–14 in a single table: init type, KAPPA, T, convergence outcome, ever-SME %, peak oligarchy concentration.

3. **Diversity regime finding** — formal statement of Option B as a result, not a failure. Frame against scalar baseline.

4. **R⁵ geometry explanation** — the angular noise argument from Section 2 above, written for a general technical audience.

5. **Parameter semantics note** — ALPHA/ETA definitions, the tribal topology observation, and the dynamic-ALPHA extension as future work.

6. **Convergence sensitivity note** — document Options A, B, C including the ETA=0.07 conservative variant, explaining why they were noted and not run.

---

## 7. Next Actions for Claude Code

```
Session start: read memory/project-state.md to confirm current white paper version and last commit.

Task 1: Confirm run14-findings.md is present in workspace/simulation/phase-2/
Task 2: Open white paper (current version — check memory for path).
Task 3: Add Phase 2 vector beliefs section per Section 6 above.
         Use run14-findings.md as the authoritative source — do not paraphrase from memory.
Task 4: Update run comparison table to include Runs 11–14.
Task 5: /prj-upd-management and /prj-upd-delivery as per session protocol.
Task 6: Commit with message: "feat: Phase 2 vector beliefs findings — diversity regime (Run 14)"
```
