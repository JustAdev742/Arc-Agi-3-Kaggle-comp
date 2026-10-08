Summary: before serving a different checkpoint of the same model (a fine-tune, another quantization), diff its index, quantization config, tokenizer and special files against the one the notebook was built for, and resolve every file its index names: exp-076 v1 died in 52 s on a file the original's index never had.

# Check a swapped checkpoint offline before a GPU run (2026-10-08)

What happened: exp-076 swapped Intel's W4A16 Flash-Next for UkisAI's Swift-1.5 W4A16 (same architecture, tokenizer
and chat template; sha256 checked). Its Kaggle copy is one HF repo split over two model instances, so the builder
links both into one directory. The link step left out `model_mtp.safetensors`, reasoning "the draft is his
albucino MTP, the target never reads it". But Swift's `model.safetensors.index.json` names that file (31 mtp.*
tensors), and his launcher checks every indexed shard: "Missing/empty shard". Intel's index names MTP tensors too
(1,565, in another file), which is why nothing in his launcher had ever tripped on it.

What a few HTTP reads showed without a GPU (single files and safetensors headers through the Kaggle API, see
scratchpad tools/model_file.py and tools/st_header.py):
- tokenizer.json sha256 equal (so the FR-Spec map's tokenizer check holds), chat_template.jinja byte-identical;
- quantization: Swift's AutoRound keeps only routers, gates and a few projections in 16 bits, so its linear-attention
  projections are INT4 where Intel keeps them BF16 (a different kernel path in the server);
- config: new fields (number_of_conv_states, norm_topk_prob) and a different vision model_type string;
- index: 223,058 keys vs 224,280, MTP tensors fused BF16 vs per-expert.

How to apply:
- Download the candidate's small files (config.json, quantization_config.json, tokenizer files, chat template,
  model.safetensors.index.json) and diff them against the current checkpoint's before building the arm.
- Build any merged/symlinked model view from the index: every file the index names must be there.
- Give a test arm with a new checkpoint --fail-fast, so a server that cannot load it costs minutes, not the budget.
- The same care applies to vocabularies: the first ARC-tuned FR-Spec map dropped </think> and <|im_end|> because a
  frequency count of logged text never sees special tokens; protect the tokenizer's added tokens explicitly.
