Summary: a probe that replays 154 logged ARC agent requests (11 public games, images included) through the served model, greedy with logprobs, one at a time and then 8 in flight, so REAP-448's effect on what the model says can be read against the unpruned server's own run-to-run noise; built and tested on CPU (2026-10-08), not yet run on a GPU.

# Fidelity probe: does REAP-448 change what the model says on ARC requests?

Labels: **[verified]** = run in this session (command below), **[source]** = read in the Pennyroyal wheel or the
harness tree but not executed, **[estimate]** = derived.

## 1. Why

exp-073 (D' + REAP-448 + 14 streams, full length) scored 49.45 against Franzen v3's 45.6-47.5. It gained on games
where time ran out, but lost on games Franzen usually wins: vc33 -58.9, tn36 -49.9, tr87 -36.0
(docs/research_log.md, 2026-10-08 13:33). The public REAP-k448 build was calibrated on agentic text traffic, not
ARC boards. If the pruned experts carry image or grid reasoning, pruning costs quality. One game run cannot tell that
from draw noise: the per-game spread is ~17 points (lesson 0018).

The probe removes the game and the sampling from the question. It fixes the inputs (real requests, as sent) and
decodes greedily, then compares two servers token by token and logprob by logprob.

## 2. What it measures

For every sampled request, each run records:
- the generated token ids, with each chosen token's logprob;
- the top 5 alternatives at every position, with their ids;
- the finish reason, the reasoning, content and tool calls;
- usage, the server's speculative-decoding accept counts and the latency.

Two comparisons come out of it:

- **The floor.** This is the unpruned server against itself.
  - Within one run: the same requests one at a time (`seq`) and then 8 in flight (`conc`). This covers batching,
    a different prefix-cache structure, and any nondeterministic kernels.
  - Across two runs, optional: a second base run. This adds start-up effects such as autotuned kernel choices.
- **REAP.** The base run against the REAP run on the same pass: same requests, same order, same cache flushes, same
  stream count. Pruning is the only difference.

Why greedy works as a probe:
- With `temperature` 0, SGLang samples with `top_k` 1 (sampling_params.py:144-147) [source].
- When every request in the batch is greedy, logprobs are the plain log-softmax of the target's logits. This holds for
  every accepted speculative token too: `compute_spec_logprobs` in layers/logprob_processor.py works on the verify
  logits [source].
- With acceptance thresholds 1.0, the default, a draft token is kept only if it equals the target's argmax. The output
  is therefore the target's greedy output, whatever the MTP draft proposes. The probe refuses a server whose
  thresholds are not 1.0.

## 3. The pieces

| What | File | Status |
|---|---|---|
| Prompt sampler | `scripts/fidelity_sample.py` (stdlib, run with `python -I`; the logs are untrusted data) | [verified] |
| Sample (154 requests, 39.6 MB, sha256 `c80558419a58...`) | scratchpad `fidelity/dataset/arc3-fidelity-prompts/` (not in git) | [verified] deterministic |
| Its manifest and Kaggle metadata | `kaggle/fidelity/manifest.json`, `kaggle/fidelity/dataset-metadata.json` (private) | in git |
| Probe (runs in the notebook; also a CLI against any server) | `scripts/fidelity_probe.py` (stdlib) | [verified] against a fake server |
| Notebook option | `scripts/build_franzen_nb.py --probe DIR` | [verified] builds; its probe cells executed on CPU |
| Comparison | `scripts/fidelity_compare.py A.json B.json [C.json]` | [verified] on synthetic runs |
| Tests | `tests/test_fidelity.py` (14), `tests/test_build_franzen_nb.py` (5 new, 25 in all), `tests/fidelity_fakes.py` | 39 pass |

### 3.1 The sample

**Sources:**
- Franzen's Save & Run of 2026-09-30: 10 demo games at 25 min each (the run documented in engineering.md section 3).
- tn36 pass 0 of his v3 Save & Run (2026-10-03, 121 min per game). It was fetched alone with
  `kaggle kernels output dfranzen/arc-agi-3-milestone-2-solution --file-pattern '^tn36-[0-9a-f]+_p0_requests\.jsonl$'`
  (68 MB). tn36 is one of the three games REAP lost and is not among his demo games.

Both runs used the unpruned server. Each log line is a request snapshot written right before the call (messages,
tools, tool_choice, the harness's chat_template_kwargs), followed by a response record (usage, finish reason).

**Draw** [verified]:
- 14 answered requests per game. Each game's requests are cut into four consecutive quarters (early to late turns),
  and the draw within a quarter is systematic over prompt tokens, from a seeded offset.
- Late quarters mix long contexts with post-trim ones of ~55k tokens.
- Result: 44/44/33/33 requests per quarter.
- Prompt tokens: 5,474 to 118,992 (median 61,012). By context: <16k 20, 16-48k 38, 48-80k 56, >=80k 40.
- Images per request: 1 to 57 (median 20.5).
- 100 requests end with a fresh frame (a user turn with its image); 54 end with a tool result.
- 150 carry the logged reply for reference, taken from the next request's history.

**Exactly as sent** [verified]:
- The messages are the logged ones minus the harness's private `_arc3_control` key. This is the harness's
  `_strip_control_keys`; `_apply_summary_visibility` is a no-op with his settings.
- Images stay inline as base64 PNG data URLs.

### 3.2 How a request is replayed

The harness call (ARC3-Inference `inference/agent/tool_agent.py` `_chat_completion` and
`inference/utils/openai_compat.py` `build_chat_payload`, in the tree his notebook builds; Franzen's cell 4 sets the
values) [source]:
- `POST http://127.0.0.1:8001/v1/chat/completions`, `model` = `flashnext`, `stream` false;
- `temperature` 0.7, `top_p` 0.95, `top_k` 20, `max_tokens` 12288;
- `tools` and `tool_choice` "auto";
- `chat_template_kwargs` = `{"enable_thinking": true, "preserve_thinking": true}`. The first key comes from
  `build_chat_payload`; only the second is logged.
- No `separate_reasoning` (the server default is true, with the qwen3 reasoning parser) and no `seed`.

The probe sends the same request, with these changes:
- `temperature` 0 and `max_tokens` 192.
- `logprobs` true with `top_logprobs` 5.
- `return_token_ids` and `return_meta_info`. Both are fields of Pennyroyal's `ChatCompletionRequest` (protocol.py:890-892)
  [source]. They give exact token ids for every logprob and per-request `spec_accept_length`. A short warm-up request
  checks them. If the server rejects them, it falls back to plain OpenAI logprobs, compared by token string.

**Order of a run:**
1. Wait for `/health`.
2. Record `/server_info`, and refuse the run if the expert-count override or the acceptance thresholds do not match
   the arm.
3. Send the warm-up.
4. For each pass, flush the prefix cache (`/flush_cache`), then send the requests game by game in turn order: `seq`
   one at a time, then `conc` with 8 in flight.

Within a game, later requests extend earlier ones, so the cache serves their shared prefix. Both arms see the same
cache flow.

### 3.3 The notebook (`--probe`)

What the builder does:
- Cells up to and including the server launch stay as they are. On the base arm the launcher is byte for byte the
  same as without `--probe`.
- Right after cell 4, it adds two cells:
  - a `%%writefile /kaggle/arc3-fidelity-probe.py` cell holding the probe;
  - a cell that finds the dataset in either mount layout (lesson 0030) and checks the file's sha256 against the
    manifest. It waits up to 300 s, then stops before the server is started.
- The benchmark cell (`await bm.run(...)`) becomes the probe call. The cells after it (diagnostics) are dropped.
- The dataset joins the kernel's sources. The image stays pinned (lesson 0029).

What the probe cell does:
- Raises if the server process exits, or if `/health` is not up 30 minutes after the notebook started. A server
  failure therefore costs at most that, not a session.
- Caps its passes at 150 minutes.
- Writes `/kaggle/working/fidelity.json` after each pass.
- Refuses to run as a competition rerun.

What the builder refuses:
- `--probe` without `--input-fallback`;
- `--full25` or `--patch`, since the harness does not run;
- `--cfg SPEC_ACCEPT_SINGLE` or `SPEC_ACCEPT_ACC` other than 1.0.

The REAP arm is the same command plus `--reap-kept`. It differs from the base arm in:
- the three REAP `%%writefile` cells;
- the launcher's two REAP blocks;
- the probe call's arm label and build note;
- the kernel id (`tests/test_build_franzen_nb.py` pins this).

Both arms keep his 10 streams (`MAXREQ` 10), so pruning is the only difference.

## 4. Build, upload, run, compare

The scratchpad below is this session's: `SCRATCH=/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad`.

```bash
# 0. (only if the scratchpad copy is gone) regenerate the dataset folder: byte-identical (same seed)
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels output dfranzen/arc-agi-3-milestone-2-solution \
  -p $SCRATCH/fidelity/v3-tn36 --file-pattern '^tn36-[0-9a-f]+_p0_requests\.jsonl$'
.venv/bin/python -I scripts/fidelity_sample.py --logs $SCRATCH/m2/franzen-output --logs $SCRATCH/fidelity/v3-tn36 \
  --note "franzen-output: Save & Run of dfranzen/arc-agi-3-milestone-2-solution, 2026-09-30, 10 demo games x 25 min (kaggle/franzen/, Apache-2.0)" \
  --note "v3-tn36: tn36 pass 0 of the same notebook's v3 Save & Run (2026-10-03, 25 games x 4 passes, 121 min each, unpruned server), fetched with kaggle kernels output --file-pattern" \
  --out $SCRATCH/fidelity/dataset/arc3-fidelity-prompts --copy-meta kaggle/fidelity

# 1. upload the private dataset (owner); -t keeps the JSONL byte for byte (the notebook checks its sha256)
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle datasets create -t \
  -p $SCRATCH/fidelity/dataset/arc3-fidelity-prompts
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle datasets status scottmahony/arc3-fidelity-prompts  # until "ready"

# 2. build both arms (D' base, as the submission candidate; the probe ignores the harness, so --base franzen works too)
.venv/bin/python scripts/build_franzen_nb.py --base dprime --input-fallback --wait-inputs 120 --probe kaggle/fidelity \
  --out $SCRATCH/fidelity/nb/base --slug arc3-fidelity-base \
  --note "fidelity probe, base arm (docs/research/beat-tufa/fidelity-probe.md)"
.venv/bin/python scripts/build_franzen_nb.py --base dprime --input-fallback --wait-inputs 120 --probe kaggle/fidelity \
  --reap-kept kaggle/franzen/reap448_kept_experts.json \
  --out $SCRATCH/fidelity/nb/reap --slug arc3-fidelity-reap448 \
  --note "fidelity probe, REAP-448 arm (docs/research/beat-tufa/fidelity-probe.md)"
#   optional across-run floor: the base command again with --slug arc3-fidelity-base2 --out $SCRATCH/fidelity/nb/base2

# 3. push (lead; evening UTC window, lesson 0031)
.venv/bin/python scripts/push_eval.py $SCRATCH/fidelity/nb/base
.venv/bin/python scripts/push_eval.py $SCRATCH/fidelity/nb/reap

# 4. fetch the outputs and compare
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels output scottmahony/arc3-fidelity-base \
  -p runs/fidelity-base --file-pattern 'fidelity\.json$|serve\.log$|\.log$'
KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle kernels output scottmahony/arc3-fidelity-reap448 \
  -p runs/fidelity-reap448 --file-pattern 'fidelity\.json$|serve\.log$|\.log$'
.venv/bin/python scripts/fidelity_compare.py runs/fidelity-base/fidelity.json runs/fidelity-reap448/fidelity.json \
  [runs/fidelity-base2/fidelity.json] --json runs/fidelity-compare.json
```

`--probe` takes either the repo's `kaggle/fidelity/` (metadata only) or the dataset folder itself. Given the folder,
the builder also checks the local data against the manifest.

**What to check in the first minutes of each run's log:**
- the data-check cell (right after cell 4 and the probe-module cell): `data: ... (requests.jsonl), sha256 verified`;
- REAP arm, cell 12: the `arc3 REAP` patch line (reap-at-load.md section 7);
- probe cell:
  - `server healthy`;
  - `warm-up ok ... with extras ['return_meta_info', 'return_token_ids'] (token ids: True)`;
  - `pass seq: 154 requests, 1 in flight, prefix cache flushed: True`.

## 5. GPU cost [estimate]

Per arm:
- Server start: ~9 min. His run printed `READY after 478s`, 531 s after the notebook started.
- `seq` pass: ~8-10 min.
  - Prefill of ~2.5 M tokens the cache cannot serve (the sampler's estimate), at the 10-12k tokens/s measured per
    request in his serve.log ReqTimeStats: ~4 min.
  - 154 x <=192 decode tokens at batch 1: ~3-4 min.
  - Image preprocessing.
- `conc` pass: ~5-7 min (the same prefill, decode batched).

That is **about 25 minutes per arm**, 20-35 with Kaggle's variance. Base plus REAP comes to about 50 minutes, ~0.85
GPU-hours of the weekly 30 h. A second base run, for the across-run floor, adds ~25 minutes.

Bounds:
- A server that never comes up stops the notebook at 30 minutes.
- The passes stop at 150 minutes in any case.
- Each request times out at 900 s.

## 6. How to read the result

`fidelity_compare.py` prints:
- every pair of series, grouped as `floor` (base vs base), `reap` (base vs REAP, same pass), `reap-own` (REAP seq vs
  conc) and `*-mixed` (different passes);
- a headline comparing `base.seq vs reap448.seq` with the most similar floor: `base.seq vs base#2.seq` when a second
  base run is given, else `base.seq vs base.conc`;
- breakdowns by game, context length, images, last message (fresh frame or tool result) and turn quarter.

Per pair:
- **prefix share**: the share of generated positions that agree before the first divergence (1.0 = identical).
- **|dlp|**: the mean absolute difference of the chosen tokens' logprobs on the agreed prefix. This is the most direct
  measure of how much the model's distribution moved.
- **top-k |dlp|**: the same over tokens in both top-5 lists. This is the closest proxy for sampled behaviour.
- runner-up flips and top-5 set changes per 1,000 agreed positions.
- **near-tie share**: the share of divergences where both runs had their two candidates within 0.1 nats. Batching
  noise can only flip near ties; a model change also flips confident choices.

The top-1 token cannot flip on the agreed prefix (greedy), so a top-1 flip is the divergence itself.

**Reading rule** (judgment thresholds, set before any data):

| Outcome | Condition | Decision |
|---|---|---|
| **Close to the floor** | REAP's \|dlp\| <= 2x the floor's, and its prefix share not lower by more than 0.05 (paired bootstrap 95% interval) | Pruning changes the model's ARC outputs no more than serving noise does. exp-073's losses on vc33/tn36/tr87 are then most likely draw noise. Keep REAP-448 and confirm with the next LB draws. |
| **Shifts the model** | \|dlp\| ratio >= 3, or prefix share lower by more than 0.10 with the whole interval below zero | The pruned experts matter on these requests. Check where in the breakdowns (vc33/tn36/tr87, fresh frames with images, long contexts). Then weigh it with a score test at equal streams (REAP at 10 streams against base at 10), since +14% tokens may still outweigh a small quality loss. |
| **In between** | anything else | A small, detectable shift. Run the second base run first: the within-run floor lacks start-up effects, so part of the gap may be run-to-run noise. Then read the breakdowns. |

If the floor is exactly zero (serving turned out deterministic: every `seq`/`conc` pair identical), the ratio is
undefined and the rule falls back on the prefix share. Then also judge REAP's |dlp| in absolute terms: a few
thousandths of a nat is negligible next to sampling at temperature 0.7, a few tenths is not.

Also look at:
- the near-tie share of the REAP divergences against the floor's;
- the per-request accept length in each run (`spec_accept_length`; 2.64-2.67 measured under load);
- `finish_differs`, where one run ended in a tool call and the other did not.

## 7. Limitations

- **Greedy, not sampled.** Production samples at temperature 0.7, top-p 0.95, top-k 20.
  - The probe sees the argmax path and the top 5 along it.
  - A logprob shift of d nats moves a sampled probability by about d/0.7 in log space.
  - It does not measure how small shifts compound over thousands of sampled tokens and a game loop. That needs score
    runs.
- **First 192 tokens only.** With thinking on, these are mostly the opening of the reasoning. The tool-call code is
  reached only after short reasoning.
- **11 public games.** 10 come from his 25-minute demo run (early and middle levels); tn36 comes from a full-length v3
  pass. These are not the hidden games.
  - The prompts were produced by the unpruned server's own play. Both arms see the same prompts, and neither sees its
    own continuations.
  - 14 requests per game make the per-game rows indicative, not tests.
- **One run per arm.**
  - The REAP pair spans two GPU sessions. The within-run floor does not, so run-to-run effects such as autotune
    choices are counted against REAP unless a second base run is added.
  - Kernel nondeterminism is in both.
- **Top 5 and the chosen token only.** There is no full-vocabulary KL.
- **Not checked without a GPU:**
  - that the server accepts `return_meta_info` and `return_token_ids` on the chat endpoint (read in its protocol; the
    warm-up falls back if not);
  - the exact throughput;
  - that `/flush_cache` succeeds right after the warm-up (it retries for 30 s and records the result).

## 8. How this was checked (this session, CPU)

- **Sampling.** `python -I scripts/fidelity_sample.py --logs .../m2/franzen-output --logs .../fidelity/v3-tn36 --out ...`
  wrote 154 requests, 39.6 MB, sha256 `c80558419a58...`. A second run gave the same sha256. A check script confirmed:
  - no `_arc3_control` keys are left;
  - every request has `chat_template_kwargs` `{"preserve_thinking": true}`, `tool_choice` "auto", and logged finish
    `tool_calls`;
  - the context and quarter counts above.
- **Reading the logs.** The record format came from `scripts/franzen_report.py request_stats` and a survey of the logs:
  - request and response records carry the same messages;
  - a response follows each answered request;
  - the next request extends the previous one except at history trims.
- **Request parameters.** They come from the harness code, since the logs do not carry them. Pennyroyal's serving
  path was read in the wheel `sglang-0.5.19+gd00d88efc8d6`:
  - entrypoints/openai/serving_chat.py `_build_chat_response`: logprobs cover reasoning and content alike, because they
    are computed on all output tokens before the reasoning parser splits the text;
  - protocol.py `ChatCompletionRequest`;
  - sampling_params.py, layers/logprob_processor.py;
  - speculative/eagle_worker_common.py: verify calls `compute_spec_logprobs`;
  - entrypoints/http_server.py: `/health`, `/server_info`, `/flush_cache`.
  - A search of the runtime found no switch that turns off CUDA graphs or overlap scheduling for logprob requests.
- **Probe and comparison end to end.** `tests/fidelity_fakes.py` is an SGLang-like fake server. It provides logprobs,
  top-k, `meta_info`, extras rejection, slow health, server info and a "different model" mode. Against it, the real
  dataset ran through `scripts/fidelity_probe.py`, and the outputs went through `scripts/fidelity_compare.py`:
  309 requests per run (warm-up + 2 x 154), largest request 0.52 MB, 9.1 MB of output.
- **Built notebooks.** Both arms' module, data-check and probe cells were executed on CPU against the fake server
  with the real dataset: 154/154 per pass, 8 in flight in `conc`, extras accepted.
- **Tests and lint.** `.venv/bin/python -m pytest -q tests/test_build_franzen_nb.py tests/test_fidelity.py` passed
  (39), and `ruff check` was clean on all touched files.
