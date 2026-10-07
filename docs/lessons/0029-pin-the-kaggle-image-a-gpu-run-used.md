Summary: a pushed kernel without `docker_image` runs Kaggle's latest image, which can change under a working notebook (Python 3.12 -> 3.13 by 2026-10-07 broke the M2 notebooks' cp312 wheels); pin the image digest that a GPU run of the same install cells used, never the one an upstream page lists without checking it is a GPU image.

# Pin the Kaggle image a GPU run used (2026-10-07)

What happened (exp-070/070d, 2026-10-07):

1. v1 of both copies (Franzen's notebook, the public D' notebook) had no `docker_image` in kernel-metadata.json.
   Kaggle ran them in its latest image, `gcr.io/kaggle-private-byod/python@sha256:2757e0c7...`, now Python 3.13.
   Pennyroyal's wheelhouse is cp312 only, so `uv pip install` failed in the install cell ("aiohttp==3.14.3 has
   no wheels with a matching Python ABI tag"). Same notebooks, same inputs: they had run days earlier.
2. v2 of the D' copy pinned the image D''s own metadata lists, `gcr.io/kaggle-images/python@sha256:e5452ce6...`.
   The server recorded the pin and the run had Python 3.12, but `torch.cuda.is_available()` was False:
   `gcr.io/kaggle-images/python` is the name of Kaggle's CPU image, and the session had no usable GPU.
3. v3 pins Franzen's image, `gcr.io/kaggle-private-byod/python@sha256:57e612b4...`, the one his v3 played 8.2 h
   on the RTX PRO 6000 in (2026-10-03) with the same install cells.

How to apply:

- Every kernel we push names `docker_image` (a digest, not a tag) and `docker_image_pinning_type: "original"`
  and `machine_shape` (scripts/build_franzen_nb.py `IMAGE`, scripts/copy_public_nb.py `--image`). The CLI sends
  both fields with the push (kaggle_api_extended.py `kernels_push`); its docs do not mention them.
- Take the image from a run that did what ours must do (GPU, same install cells), read with
  `kaggle kernels pull <kernel> -m`. After a push, `kaggle kernels pull <ours> -m` shows the image the server
  recorded; check it before spending a long run.
- A notebook submitted to the competition is rerun in the image of the version submitted, so a pinned version
  stays submittable after Kaggle moves its latest image.
- The rental runner (scripts/rental.py) uses `gcr.io/kaggle-gpu-images/python:v170`, a tag; check its Python
  version against the wheelhouse before the first rental (follow-up in docs/status.md).
