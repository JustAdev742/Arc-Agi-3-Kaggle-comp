# Tufa Labs' source bundle, as Daniel Franzen's notebook uses it (data for scripts/franzen_tree.py)

These files let `scripts/franzen_tree.py` rebuild, byte for byte, the Kaggle dataset
`dfranzen/taaf-kaggle-source-bundle-copy` (downloaded 2026-10-02; Franzen's unchanged copy of Tufa Labs' June 2026
Duck-harness bundle) from Franzen's GitHub repository https://github.com/da-fr/arc-agi-3-solution at commit 10882e3:

- `MANIFEST.json`: sha256 of every file of the dataset (`bundle`), of its `src/` after Franzen's harness patch
  (`notebook_src`), and of the files of his repository the rebuild reads (`his_repo`).
- `his-repo-to-bundle.diff`: the difference between his repository's `ARC3-Inference/` and `tufa-arc-agi-framework/`
  (with his patch reversed and `CONFIGURATION.md` removed) and the bundle's `src/`.
- `root/`: the dataset's top-level files, verbatim (pickled TAAF benchmark and deployment target, git status, preamble,
  setup/teardown commands).

The code in the bundle and in the delta is Tufa Labs' Duck harness and TAAF (MIT License; credit to Jeroen Cottaar and
Tufa Labs). Franzen's harness patch and repository are under the Apache License 2.0. Third-party code keeps its own
license. Do not edit these files by hand; regenerate them with `scripts/franzen_tree.py vendor --bundle DIR`.
