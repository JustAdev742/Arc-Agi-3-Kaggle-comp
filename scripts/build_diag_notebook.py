#!/usr/bin/env python
"""Build notebooks/diag/diag.ipynb: a bounded GPU diagnostics run for the rtx6000 accelerator.

What it measures (all printed and written to /kaggle/working/diag.json):
  driver/CUDA/GPU, preinstalled torch/vllm, disk; vLLM install time from the attached wheelhouse;
  vLLM start time with our flags; one tool-call completion with an image (latency, parsed call);
  throughput at 1 and 8 concurrent requests; a short REPL-agent smoke on two offline games.
Bounded by soft time checks plus a hard watchdog so it never burns more than ~35 min of quota.

Usage: .venv/bin/python scripts/build_diag_notebook.py --model-dataset saltb0x/qwen3-8-27b-fp8 \
           --wheels-dataset saltb0x/arc3-vllm-wheelhouse-v0271-cu129
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import COMP_DIR, code_cell, md_cell, package_tarball

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "diag"


def build(model_dataset: str, wheels_dataset: str, budget_min: int, smoke_s: int, attempts_json: str = "", title: str = "rtx6000",
          *, image: bool = True, efforts: str = "low,xhigh", smoke_config: str = "", effort_in_request: bool = False) -> dict:
    """``model_dataset`` is a Kaggle dataset ref (owner/slug) or a model ref (owner/slug/framework/variation/version);
    either way the notebook locates the folder holding config.json under /kaggle/input. ``image`` off means a
    text-only model: no image in the probe or the smoke. ``efforts`` are the reasoning efforts the chat probe tries
    (a model rejects the ones it does not know; that is recorded, not fatal). ``smoke_config`` is JSON merged into
    the REPL smoke's agent config (temperature, top_p, reasoning_effort, ...)."""
    tarball = package_tarball()
    effort_list = [e.strip() for e in efforts.split(",") if e.strip()]
    smoke_cfg = json.loads(smoke_config) if smoke_config.strip() else {}
    cells = [md_cell(f"# arc3 GPU diagnostics ({title})\n\nBounded probe of the serving stack. Not a submission.")]
    cells.append(code_cell(dedent(f"""\
        import os, sys, time, json, subprocess, threading, base64, io, tarfile, shutil, glob
        START = time.time(); BUDGET_S = {budget_min} * 60
        DIAG = {{'start': START}}
        def left(): return BUDGET_S - (time.time() - START)
        def sh(cmd, **kw):
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw); return (r.stdout + r.stderr).strip()
        def _watchdog():
            time.sleep(BUDGET_S + 240)
            print('WATCHDOG: hard stop', flush=True); json.dump(DIAG, open('/kaggle/working/diag.json', 'w'), indent=1, default=str); os._exit(0)
        threading.Thread(target=_watchdog, daemon=True).start()
        print(sh('nvidia-smi')); DIAG['nvidia_smi'] = sh('nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv')
        DIAG['nvcc'] = sh('nvcc --version | tail -1'); DIAG['python'] = sys.version
        DIAG['preinstalled'] = sh("pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|flashinfer|triton|xformers|numpy|pillow) '")
        DIAG['disk'] = sh('df -h /kaggle/working | tail -1'); DIAG['cpu'] = os.cpu_count(); DIAG['ram'] = sh('free -g | head -2 | tail -1')
        print(json.dumps({{k: DIAG[k] for k in ('nvidia_smi', 'nvcc', 'preinstalled', 'disk', 'cpu', 'ram')}}, indent=1))
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-warn-conflicts',
                               '--find-links', '{COMP_DIR}/arc_agi_3_wheels', 'arc-agi', 'python-dotenv'])
        """)))
    cells.append(code_cell(
        "ARC3_TGZ = '''" + tarball + "'''\n"
        + dedent("""\
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARC3_TGZ)), mode='r:gz') as tar:
            tar.extractall('/kaggle/working')
        sys.path.insert(0, '/kaggle/working'); os.environ['PYTHONPATH'] = '/kaggle/working'
        import arc3; print('arc3', arc3.__version__)
        """)))
    cells.append(code_cell(dedent(f"""\
        def find_mount(ref):
            owner, slug = ref.split('/')[:2]
            for p in ['/kaggle/input/' + slug, '/kaggle/input/datasets/' + owner + '/' + slug, '/kaggle/input/models/' + owner + '/' + slug]:
                if os.path.isdir(p): return p
            hits = glob.glob('/kaggle/input/**/' + slug, recursive=True)
            return hits[0] if hits else None
        def find_model(ref):
            # a Kaggle model mounts as .../<slug>/<framework>/<variation>/<version>/; a dataset as .../<slug>/; either
            # way the checkpoint is the folder that holds config.json (the first one below the mount, shallowest first)
            root = find_mount(ref)
            if root is None: return None
            cfgs = sorted(glob.glob(root + '/**/config.json', recursive=True), key=lambda p: p.count('/'))
            return os.path.dirname(cfgs[0]) if cfgs else root
        MODEL_DIR = find_model('{model_dataset}'); WHEELS = find_mount('{wheels_dataset}')
        print('MODEL_DIR', MODEL_DIR, 'WHEELS', WHEELS); DIAG['model_dir'] = MODEL_DIR; DIAG['wheels'] = WHEELS
        if WHEELS and not any(f.startswith('vllm') for f in os.listdir(WHEELS)):
            sub = [d for d in glob.glob(WHEELS + '/**/', recursive=True) if any(f.startswith('vllm') for f in os.listdir(d))]
            WHEELS = sub[0] if sub else WHEELS
        print('wheel dir', WHEELS, 'vllm wheels:', [f for f in os.listdir(WHEELS) if f.startswith('vllm')][:3] if WHEELS else None)
        t0 = time.time()
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', WHEELS, 'vllm'], capture_output=True, text=True)
        DIAG['pip_vllm_s'] = round(time.time() - t0, 1); DIAG['pip_vllm_rc'] = r.returncode
        print('pip vllm rc', r.returncode, 'in', DIAG['pip_vllm_s'], 's'); print((r.stdout + r.stderr)[-3000:])
        DIAG['versions_after'] = sh("pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|flashinfer|flashinfer-python|triton|xformers|numpy|pillow) '")
        print(DIAG['versions_after'])
        """)))
    cells.append(code_cell("ATTEMPTS_JSON = " + repr(attempts_json) + f"\nUSE_IMAGE = {image!r}\nEFFORTS = {effort_list!r}\nEFFORT_IN_REQUEST = {effort_in_request!r}\n" + dedent("""\
        from arc3 import serve
        from arc3.serve import start_vllm_with_fallback, build_vllm_command
        # attempts may name files inside the checkpoint folder (a reasoning-parser plugin) as {MODEL_DIR}/<file>
        ATTEMPTS = json.loads(ATTEMPTS_JSON.replace('{MODEL_DIR}', MODEL_DIR or '')) if ATTEMPTS_JSON.strip() else None
        vllm_proc = None; DIAG['vllm_ready'] = False
        if DIAG['pip_vllm_rc'] == 0 and MODEL_DIR and left() > 600:
            print(' '.join(build_vllm_command(MODEL_DIR)))
            print('GPU fraction free before start:', serve.gpu_fraction_available(), 'attempt ladder:', ATTEMPTS)
            t0 = time.time()
            vllm_proc, DIAG['vllm_ready'] = start_vllm_with_fallback(MODEL_DIR, log_path='/kaggle/working/vllm.log', port=8000,
                                                                     served_name='arc3-model', timeout_s=min(2400, left() - 420), attempts=ATTEMPTS)
            DIAG['last_start'] = dict(serve.LAST_START)
            DIAG['vllm_start_s'] = round(time.time() - t0, 1)
            DIAG['vllm_attempts'] = open('/kaggle/working/vllm.log').read().count('$ ')
            print('vLLM ready:', DIAG['vllm_ready'], 'after', DIAG['vllm_start_s'], 's, attempts', DIAG['vllm_attempts'])
            if not DIAG['vllm_ready']:
                print(open('/kaggle/working/vllm.log').read()[-6000:])
        else:
            print('skipping vLLM start')
        DIAG['gpu_after_start'] = sh('nvidia-smi --query-gpu=memory.used,memory.total --format=csv')
        """)))
    cells.append(code_cell(dedent("""\
        import concurrent.futures as cf
        from arc3.llm import ChatClient
        from arc3.prompts import TOOLS, SYSTEM_PROMPT
        from arc3.env import LocalEnv, make_arcade
        from arc3.perception import render_png
        if DIAG['vllm_ready']:
            client = ChatClient('http://127.0.0.1:8000/v1', model='arc3-model', effort_in_request=EFFORT_IN_REQUEST)
            arc = make_arcade(COMP_ENV := '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
            env = LocalEnv(arc, 'ls20'); png = base64.b64encode(render_png(env.frame.grid, scale=6)).decode(); env.close()
            parts = [{'type': 'text', 'text': 'Level 1/7. Inspect the board with objects() and take one action with act(...).'}]
            if USE_IMAGE:
                parts.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png}})
            msgs = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': parts}]
            for effort in EFFORTS:
                t0 = time.time()
                try:
                    r = client.chat(msgs, tools=TOOLS, max_tokens=2048, thinking=True, reasoning_effort=effort, timeout_s=600)
                    DIAG[f'chat_{effort}'] = {'latency_s': round(time.time() - t0, 1), 'prompt_tokens': r.prompt_tokens, 'completion_tokens': r.completion_tokens,
                                            'tool_calls': [tc.name for tc in r.tool_calls], 'finish': r.finish_reason,
                                            'reasoning_chars': len(r.reasoning), 'content_head': r.content[:300],
                                            'code_head': (r.tool_calls[0].arguments.get('code', '')[:300] if r.tool_calls else None)}
                except Exception as e:
                    DIAG[f'chat_{effort}'] = {'error': str(e)[:500]}
                print(effort, json.dumps(DIAG[f'chat_{effort}'], indent=1))
            # throughput: 1 then 8 concurrent short generations (never let a server error kill the notebook)
            def one():
                t = time.time(); rr = client.chat(msgs[:1] + [{'role': 'user', 'content': 'Count from 1 to 200 separated by spaces.'}], max_tokens=400, thinking=False, timeout_s=600)
                return rr.completion_tokens, time.time() - t
            for n in (1, 8):
                t0 = time.time()
                try:
                    with cf.ThreadPoolExecutor(n) as ex:
                        res = list(ex.map(lambda _: one(), range(n)))
                    wall = time.time() - t0; toks = sum(r[0] for r in res)
                    DIAG[f'throughput_{n}'] = {'wall_s': round(wall, 1), 'tokens': toks, 'tok_per_s': round(toks / wall, 1)}
                except Exception as e:
                    DIAG[f'throughput_{n}'] = {'error': str(e)[:500]}
                    DIAG['vllm_ready'] = False  # server is unusable; skip the REPL smoke
                print(n, DIAG[f'throughput_{n}'])
            DIAG['vllm_log_tail'] = open('/kaggle/working/vllm.log').read()[-2500:]
            DIAG['attention_backend'] = [l for l in open('/kaggle/working/vllm.log') if 'attention backend' in l][-1:]
        """)))
    cells.append(code_cell(dedent(f"""\
        if DIAG['vllm_ready'] and left() > {smoke_s} + 120:
            from arc3.eval import run_eval
            cfg = {{'base_url': 'http://127.0.0.1:8000/v1', 'model': 'arc3-model', 'context_tokens': 32768, 'reasoning_effort': 'low', 'max_output_tokens': 3072,
                   'image': {image!r}, 'effort_in_request': {effort_in_request!r}, **{smoke_cfg!r}}}
            s = run_eval('repl', 'ls20,vc33', time_budget_s={smoke_s}, max_actions=150, workers=1, runs_dir='/kaggle/working/runs',
                         environments_dir='/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files', run_name='diag-repl-smoke', config=cfg)
            DIAG['repl_smoke'] = {{'score': s['score'], 'levels': s['levels_completed'], 'actions': s['actions'], 'wall_s': s['wall_s'],
                                  'per_game': [{{'game': r['game_id'], 'levels': r['levels_completed'], 'actions': r['actions'], 'wall_s': r['wall_s'],
                                                'failure': r['failure'], 'stats': r['agent_stats']}} for r in s['results']]}}
            print(json.dumps(DIAG['repl_smoke'], indent=1, default=str))
        else:
            print('skipping REPL smoke (time left %.0fs)' % left())
        """)))
    cells.append(code_cell(dedent("""\
        if vllm_proc is not None:
            vllm_proc.terminate()
        DIAG['total_s'] = round(time.time() - START, 1)
        json.dump(DIAG, open('/kaggle/working/diag.json', 'w'), indent=1, default=str)
        print(json.dumps({k: v for k, v in DIAG.items() if k not in ('preinstalled', 'versions_after')}, indent=1, default=str))
        """)))
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": "nvidiaRtxPro6000", "isInternetEnabled": False, "isGpuEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat_minor": 4, "nbformat": 4, "cells": cells,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dataset", default="saltb0x/qwen3-8-27b-fp8")
    p.add_argument("--wheels-dataset", default="saltb0x/arc3-vllm-wheelhouse-v0271-cu129")
    p.add_argument("--budget-min", type=int, default=40)
    p.add_argument("--smoke-s", type=int, default=300)
    p.add_argument("--username", default="scottmahony")
    p.add_argument("--slug", default="arc3-gpu-diag")
    p.add_argument("--attempts-json", default="", help="JSON list of start_vllm attempt dicts (model_dir/label/env_extra allowed) replacing the default ladder")
    p.add_argument("--title", default="rtx6000")
    p.add_argument("--no-image", action="store_true", help="text-only model: no image in the probe or the REPL smoke")
    p.add_argument("--efforts", default="low,xhigh", help="comma-separated reasoning efforts for the chat probe")
    p.add_argument("--smoke-config", default="", help="JSON merged into the REPL smoke's agent config")
    p.add_argument("--effort-in-request", action="store_true", help="send reasoning_effort as the OpenAI request field (harmony models such as gpt-oss)")
    p.add_argument("--out", default=str(OUT_DIR), help="folder for diag.ipynb + kernel-metadata.json (git-ignored; push it with scripts/push_eval.py)")
    a = p.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diag.ipynb").write_text(json.dumps(build(a.model_dataset, a.wheels_dataset, a.budget_min, a.smoke_s, a.attempts_json, a.title,
                                                         image=not a.no_image, efforts=a.efforts, smoke_config=a.smoke_config,
                                                         effort_in_request=a.effort_in_request), indent=1))
    # a model ref (owner/slug/framework/variation/version, kernels_metadata.md) goes to model_sources, a dataset ref to dataset_sources
    is_model = a.model_dataset.count("/") >= 4
    meta = {"id": f"{a.username}/{a.slug}", "title": a.slug, "code_file": "diag.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [a.wheels_dataset] + ([] if is_model else [a.model_dataset]), "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-3"], "model_sources": [a.model_dataset] if is_model else []}
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[build_diag_notebook] wrote {out_dir}/diag.ipynb + kernel-metadata.json ({a.model_dataset}, {a.wheels_dataset})")


if __name__ == "__main__":
    main()
