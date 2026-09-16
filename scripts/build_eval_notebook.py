#!/usr/bin/env python
"""Build notebooks/eval/eval.ipynb: run the local harness on a split against local vLLM on Kaggle.

This is how experiments are measured until the local GPU box is online: the notebook installs
vLLM from the wheelhouse, starts the coordinator (and specialist) server, runs
``arc3.eval.run_eval`` with parallel games on the bundled environment_files, and writes
``runs/<name>/summary.json`` plus ``eval_result.json`` to /kaggle/working.

Usage: .venv/bin/python scripts/build_eval_notebook.py --agent repl --split dev --time-per-game 1200 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import COMP_DIR, code_cell, md_cell, package_tarball, vllm_setup_source  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notebooks" / "eval"


def build(a: argparse.Namespace) -> dict:
    tarball = package_tarball()
    cfg = json.loads(a.config)
    spec = a.specialist_dataset if a.agent == "council" else ""
    cells = [md_cell(f"# arc3 evaluation: {a.agent} on {a.split}\n\nNot a submission. Writes runs/ and eval_result.json.")]
    cells.append(code_cell(dedent(f"""\
        import os, sys, time, json, subprocess, threading, base64, io, tarfile
        START = time.time(); BUDGET_S = {a.budget_min} * 60
        os.environ['ARC3_AGENT'] = '{a.agent}'
        os.environ['ARC3_AGENT_CONFIG'] = json.dumps({cfg!r})
        def _watchdog():
            time.sleep(BUDGET_S)
            print('WATCHDOG: hard stop after %d min' % {a.budget_min}, flush=True); os._exit(0)
        threading.Thread(target=_watchdog, daemon=True).start()
        print(subprocess.run('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv', shell=True, capture_output=True, text=True).stdout)
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
    cells.append(code_cell(vllm_setup_source(a.model_dataset, a.wheels_dataset, spec)))
    cells.append(code_cell(dedent(f"""\
        from arc3.eval import run_eval
        result = {{'agent': os.environ['ARC3_AGENT'], 'split': '{a.split}', 'setup_s': round(time.time() - START, 1)}}
        if os.environ['ARC3_AGENT'] == '{a.agent}':
            cfg = json.loads(os.environ['ARC3_AGENT_CONFIG'])
            s = run_eval('{a.agent}', '{a.split}', seed={a.seed}, time_budget_s={a.time_per_game}, max_actions={a.max_actions},
                         workers={a.workers}, runs_dir='/kaggle/working/runs',
                         environments_dir='{COMP_DIR}/environment_files', run_name='{a.run_name}', config=cfg,
                         note='{a.note}')
            result.update({{k: s[k] for k in ('run_name', 'score', 'score_dev', 'score_val', 'levels_completed', 'levels_total',
                                            'games_solved', 'actions', 'wall_s', 'failures')}})
            result['per_game'] = [{{'game': r['game_id'], 'score': r['score'], 'levels': r['levels_completed'], 'win_levels': r['win_levels'],
                                   'actions': r['actions'], 'level_actions': r['level_actions'], 'wall_s': r['wall_s'], 'failure': r['failure'],
                                   'stats': r['agent_stats']}} for r in s['results']]
        else:
            result['error'] = 'model server did not start; agent fell back to ' + os.environ['ARC3_AGENT']
            print(open('/kaggle/working/vllm.log').read()[-4000:])
        result['total_s'] = round(time.time() - START, 1)
        json.dump(result, open('/kaggle/working/eval_result.json', 'w'), indent=1, default=str)
        print(json.dumps({{k: v for k, v in result.items() if k != 'per_game'}}, indent=1, default=str))
        for p_ in vllm_procs:
            p_.terminate()
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
    p.add_argument("--agent", default="repl", choices=["repl", "council", "explorer"])
    p.add_argument("--split", default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-per-game", type=int, default=1200)
    p.add_argument("--max-actions", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--budget-min", type=int, default=150)
    p.add_argument("--config", default='{"context_tokens": 32768, "reasoning_effort": "low", "max_output_tokens": 3072}')
    p.add_argument("--run-name", default=None)
    p.add_argument("--note", default="")
    p.add_argument("--model-dataset", default="saltb0x/qwen3-8-27b-fp8")
    p.add_argument("--wheels-dataset", default="saltb0x/arc3-vllm-wheelhouse-v0271-cu129")
    p.add_argument("--specialist-dataset", default="scottmahony/qwen3-vl-8b-instruct-nvfp4")
    p.add_argument("--username", default="scottmahony")
    p.add_argument("--slug", default="arc3-eval")
    a = p.parse_args()
    a.run_name = a.run_name or f"kaggle-{a.agent}-{a.split}-s{a.seed}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval.ipynb").write_text(json.dumps(build(a), indent=1))
    ds = [d for d in (a.wheels_dataset, a.model_dataset, a.specialist_dataset if a.agent == "council" else "") if d]
    meta = {"id": f"{a.username}/{a.slug}", "title": a.slug, "code_file": "eval.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": ds, "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-3"],
            "model_sources": []}
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[build_eval_notebook] wrote {OUT_DIR}/eval.ipynb ({a.agent} on {a.split}, {a.time_per_game}s/game, workers {a.workers}, datasets {ds})")


if __name__ == "__main__":
    main()
