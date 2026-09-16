# Upload nvidia/Qwen3.8-Flash-Next-NVFP4 to Kaggle (run on a machine with >= 140 GB free disk)

The Claude Code container has 18 GB of writable disk; the checkpoint is 132.7 GB with one 53.7 GB file, so it cannot
be staged there. On the workstation (128 GB RAM, large disk):

    pip install -U huggingface_hub kaggle
    hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir ./qwen3-8-flash-next-nvfp4
    cp dataset-metadata.json hf-provenance.json ./qwen3-8-flash-next-nvfp4/    # from this folder; set "downloaded" to today's date
    export KAGGLE_API_TOKEN=...                                                 # the same token as the repo's .kaggle/access_token
    kaggle datasets create -p ./qwen3-8-flash-next-nvfp4 --dir-mode skip       # private by default; ~1-3 h of upload

Then tell Claude the dataset slug (scottmahony/qwen3-8-flash-next-nvfp4) and it runs the serving check on the RTX.
Serving caveats recorded in docs/status.md: needs a vLLM newer than the 0.27.1 wheelhouse, PLE/n-gram embedding
offload to host RAM (51 GB), and NVFP4 kernels on SM120 (unverified).
