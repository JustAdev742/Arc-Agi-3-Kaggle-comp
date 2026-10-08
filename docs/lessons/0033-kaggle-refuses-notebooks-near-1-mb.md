Summary: Kaggle's push API refuses a notebook of about 1.2 MB with a bare "400 Client Error: Bad Request" on SaveKernel (902-927 KB went through), so keep notebooks under ~0.9 MB: ship data as Kaggle datasets or compact encodings, not base64 blobs.

# Kaggle refuses notebooks near 1 MB (2026-10-08)

What happened: exp-077 (D' + five harness patches + an FR-Spec map embedded as base64) was 1,195,948 bytes. Every
push returned `400 Client Error: Bad Request for url: .../KernelsApiService/SaveKernel`, with no reason. The same
notebook without the 270 KB map (901,831 bytes) passed validation at once: Kaggle answered "Maximum batch GPU
session count of 2 reached", the normal busy reply. The largest notebook pushed before was 832 KB.

How to apply:
- A 400 on SaveKernel with no message means "look at the notebook itself"; first check its size.
- Keep built notebooks under ~0.9 MB (tests/test_build_franzen_nb.py asserts it for --hot-tokens). Franzen's base is
  ~0.8 MB, so patches and data must stay small.
- Ship binary data compactly (the FR-Spec map now travels as ~3 KB of zlib-compressed delta varints and is written
  in torch's zip layout with fixed metadata, so its sha256 is known at build time), or as a private Kaggle dataset.
- The 2-GPU-session limit answers pushes with "Maximum batch GPU session count of 2 reached"; it never queues them.
