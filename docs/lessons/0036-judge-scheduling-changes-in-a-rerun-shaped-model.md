Summary: judge scheduling changes in a model of the competition rerun (about 110 games, 532 min each, all started together, 14 slots), never on public-25 runs (25 games x 121 min): ending a stuck game after 40 active minutes is +1.2 in the public-25 shape and -1.7 in the rerun model.

# Judge scheduling changes in a rerun-shaped model (2026-10-08)

What happened: docs/research/beat-tufa/time-allocation.md ported D′'s gate and priority (A·M·C + B·φ) into a simulator
(scratchpad timealloc/sim.py; replaying exp-073/073b/075 gives 49.9 / 57.6 / 43.9 against the observed 49.45 / 56.00 /
42.89) and compared give-up rules on identical draws. In the rerun shape, freed slot-time goes to games that would have
used it anyway at a lower value per token, and short cut-offs end so many games that most freed time is unused: every
cut-off at 40 minutes or less loses points. In the 25 x 121 shape, freed time goes to games parked on their last
levels, so the same rules look like gains (+1.20 at 40 min, +0.74 at 50).

How to apply:
- Any change to who gets the GPU (gate priority, slot counts, give-ups, caps, fresh-first rules) is evaluated in a
  rerun-shaped simulation first; a public-25 run cannot tell its sign.
- The rerun setting is in D′'s run cell under `if TRUE_SUBMISSION:` (concurrency 120, 532 min per game); the
  `532*60*concurrency//110` comment is the non-submission branch.
- Harness changes that leave the gate alone (prompts, tools, perception) are still read on public-25 runs; only
  their second-order effect on slot handovers (through tokens per turn and trims) carries this bias.
