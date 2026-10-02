Summary: `git apply` run in a subdirectory of some other git repository applies paths from that repository's top, prints "Skipped patch" and exits 0; set GIT_CEILING_DIRECTORIES and count "Applied patch" lines whenever a patch must really land.

# git apply inside another repository skips silently (2026-10-02)

Reproduced in tests/test_franzen_tree.py: a directory `outer/sub` inside a git repository `outer`, a patch for
`d/a.txt` relative to `sub`, `git apply -v p.patch` run in `sub`: output "Skipped patch 'd/a.txt'.", exit code 0,
file unchanged. A notebook or script that checks only the exit code reports success.

Franzen's notebook runs `git apply` in /kaggle/taaf-kaggle-source-share/src, which is not inside a repository, so
it is safe on Kaggle. Local tooling is where it bites: a scratch tree created under a repository checkout (a
worktree, a runs/ folder) would silently stay unpatched.

How to apply: run `git apply` with `GIT_CEILING_DIRECTORIES=<parent of the target dir>` and check that the number
of "Applied patch" lines in `-v` output equals the number of files the patch names (scripts/franzen_tree.py
`git_apply`, and the cell that scripts/build_franzen_nb.py --patch adds, both do).
