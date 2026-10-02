# Daniel Franzen's ARC-AGI-3 Milestone 2 notebook (unmodified copy)

`arc-agi-3-milestone-2-solution.ipynb` is a verbatim copy of the public Kaggle notebook
https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution (pulled 2026-10-02 with
`kaggle kernels pull`), Daniel Franzen's open-sourced Milestone 2 solution (public LB 27.89). His source repository,
https://github.com/da-fr/arc-agi-3-solution (commit 10882e3), is licensed under the Apache License 2.0; the notebook
builds on Tufa Labs' Duck harness (MIT) and serves Qwen3.8-Flash-Next (Intel W4A16 AutoRound, Albucino's MTP draft)
with John Pezzulli's Pennyroyal SGLang fork; third-party code and weights keep their own licenses.

Our arms are built from it by `scripts/build_franzen_nb.py`, which changes only the cells it names (demo game list,
per-game budget, harness environment knobs, SGLang launcher settings, and our harness patches as added `%%writefile`
cells applied after his) and records every change in the first markdown cell. Do not edit this copy. `bundle/` holds
what `scripts/franzen_tree.py` needs to rebuild the source bundle the notebook patches (see bundle/NOTICE.md).
