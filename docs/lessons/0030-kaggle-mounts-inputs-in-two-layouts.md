Summary: Kaggle GPU sessions of the same kernel get inputs either at /kaggle/input/{datasets/<owner>,competitions}/<slug> or at /kaggle/input/<slug>; a notebook that hardcodes one layout fails at random, so resolve every input path against both (scripts/build_franzen_nb.py --input-fallback).

# Kaggle mounts inputs in two layouts (2026-10-07)

What happened: Franzen's and D''s notebooks hardcode the newer layout
(`/kaggle/input/datasets/dfranzen/pennyroyal-v253`, `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/...`).
Of nine GPU sessions of our copies pushed on 2026-10-07 between 21:18 and 22:47, five failed in cell 4
(`cp: cannot stat '/kaggle/input/datasets/dfranzen/taaf-kaggle-source-bundle-copy'`). A 600 s wait for the path
(exp-072a v5) changed nothing; its listing showed the inputs mounted the older way:
`/kaggle/input/taaf-kaggle-source-bundle-copy`, `/kaggle/input/pennyroyal-v253`,
`/kaggle/input/arc-prize-2026-arc-agi-3`, `/kaggle/input/models/dfranzen/...`. A CPU session of the same sources
(arc3-diag-mount) got the newer layout. Same kernel metadata and image either way; which layout a session gets is
not under our control as far as we can see.

How to apply:

- Every notebook we build passes `--input-fallback`: a helper at the top of cell 4 maps a missing
  `datasets/<owner>/<slug>` or `competitions/<slug>` path to `/kaggle/input/<slug>` when that exists, and the
  builder wraps every `/kaggle/input/...` literal in code cells with it (6 in Franzen's notebook: cell 4's four
  constants, the arc-agi wheel install, the offline environment files). Models were at
  `/kaggle/input/models/<owner>/...` in both layouts.
- `--wait-inputs` stays as a cheap guard (it lists /kaggle/input when something is still missing).
- The rental runner already creates both layouts (scripts/rental_box.py); the same check belongs in any new notebook.
- Our submitted copy of D' (2026-10-07) is unmodified and hardcodes the newer layout; hundreds of submissions of the
  same code have scored, so the rerun environment presumably uses that layout, but our next submission should carry
  the fallback.
