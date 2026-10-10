Summary: Kaggle's push API answers a bare "400 Client Error: Bad Request" on SaveKernel for a notebook of about 1.2 MB (902-927 KB went through) and for a title over 50 characters (54 refused, 46 accepted), so keep notebooks under ~0.9 MB and titles at 50 characters or fewer; scripts/push_eval.py checks the title before pushing.

# Kaggle refuses notebooks near 1 MB (2026-10-08)

What happened: exp-077 (D' + five harness patches + an FR-Spec map embedded as base64) was 1,195,948 bytes. Every
push returned `400 Client Error: Bad Request for url: .../KernelsApiService/SaveKernel`, with no reason. The same
notebook without the 270 KB map (901,831 bytes) passed validation at once: Kaggle answered "Maximum batch GPU
session count of 2 reached", the normal busy reply. The largest notebook pushed before was 832 KB.

How to apply:
- A 400 on SaveKernel with no message means "look at the notebook itself"; first check its size and its title.
- Titles (the slug with dashes as spaces) must be 50 characters or fewer: on 2026-10-10 exp-085's 54-character
  title was refused with the same bare 400 and the same notebook went through under a 42-character slug.
  scripts/push_eval.py refuses a longer title before it calls Kaggle (tests/test_push_eval.py).
- Keep built notebooks under ~0.9 MB (tests/test_build_franzen_nb.py asserts it for --hot-tokens). Franzen's base is
  ~0.8 MB, so patches and data must stay small.
- Ship binary data compactly (the FR-Spec map now travels as ~3 KB of zlib-compressed delta varints and is written
  in torch's zip layout with fixed metadata, so its sha256 is known at build time), or as a private Kaggle dataset.
- The 2-GPU-session limit answers pushes with "Maximum batch GPU session count of 2 reached"; it never queues them.
