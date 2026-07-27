# Phase 2 — Simulation

Phase 2 workstream. Three parallel directions:

## Anti-oligarchy mechanism
Test the trust dispersion penalty (mu) from model-spec.md.
Motivation: Run 5 and Run 6 showed hub nodes accumulating disproportionate
SME fraction when degree variance is high. The anti-oligarchy mechanism
was designed for exactly this scenario.
First script: sim_v7_antioligarchy.py

## Vector beliefs
Upgrade node beliefs from scalar [0,1] to semantic embedding vectors.
This is the step toward the full concept — nodes holding structured
knowledge representations rather than scalar opinion values.
First script: sim_v7_vectorbeliefs.py

## White paper
Draft is complete at _project-delivery/white-paper/white-paper-draft-v1.md
Next: review, refine, and prepare for external sharing.
