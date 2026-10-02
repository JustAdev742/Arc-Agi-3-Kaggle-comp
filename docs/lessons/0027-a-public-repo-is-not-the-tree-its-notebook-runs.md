Summary: before patching a public solution, rebuild the exact tree its notebook runs (its attached bundle plus its own patch command) and check it by hash; the author's GitHub repo can differ from it in files a patch may touch.

# A public repo is not necessarily the tree its notebook runs (2026-10-02)

Franzen's GitHub repo holds "his patched harness", but his notebook patches Tufa's June Kaggle bundle, while the
repo is built on Tufa's later GitHub release. The two differ in 21 files, among them `inference/framework/run.py`
(48 lines, visible in his Kaggle log as "Hunk #1 succeeded at 467 (offset 48 lines)") and seven TAAF modules
(game_api.py among them). A patch written against the repo could apply locally and fail in the notebook, or
apply with different surrounding code.

What showed it: downloading the attached dataset and diffing it with the repo, then reproducing the offset lines
of the author's own Kaggle log by applying his patch to the download.

How to apply: for any public base, (1) download the exact inputs the notebook attaches, (2) replay its setup
commands locally, (3) diff the result with the author's repo before trusting the repo as "the code", and (4) keep
a hash manifest so a later rebuild proves it is the same tree (scripts/franzen_tree.py does this for Franzen's).
