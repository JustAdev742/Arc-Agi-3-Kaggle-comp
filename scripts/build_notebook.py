#!/usr/bin/env python
"""Build notebooks/submission.ipynb from arc3/ + agent/my_agent.py.

Mirrors the Kaggle starter's notebook shape (install wheel -> write agent -> run the
framework against the gateway -> dummy submission when not a competition rerun) and adds:
  * the whole ``arc3`` package embedded as a base64 tarball (no extra dataset needed);
  * optional vLLM start-up from attached datasets (model weights + wheelhouse);
  * an offline smoke run of the explorer on the competition's bundled environment files
    during "Save & Run All", so a broken package fails loudly before a real submission.

Usage:  .venv/bin/python scripts/build_notebook.py [--accelerator rtx6000|t4|cpu] [--agent repl|explorer]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"
COMP = "arc-prize-2026-arc-agi-3"
COMP_DIR = f"/kaggle/input/competitions/{COMP}"

_ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    # "nvidiaRtxPro6000" is the id the Milestone-1 2nd/3rd place notebooks carry; the starter's
    # "nvidiaRtx6000" is unknown to Kaggle and silently falls back to T4 (diag run, 2026-09-16).
    "rtx6000": {"name": "nvidiaRtxPro6000", "gpu": True},
}


def package_tarball() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted((ROOT / "arc3").rglob("*.py")):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "outputs": [], "execution_count": None, "source": src}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def _mount_expr(ref: str) -> str:
    """Python expression (as source) resolving a dataset ref to its mount path or None."""
    if not ref:
        return "None"
    owner, slug = ref.split("/", 1)
    return (f"next((p for p in ['/kaggle/input/{slug}', '/kaggle/input/datasets/{owner}/{slug}'] if os.path.isdir(p)), None)")


def specialist_refs(specialist_dataset: str) -> list[str]:
    """``--specialist-dataset`` takes a comma-separated preference list; the notebook tries them in order."""
    return [d.strip() for d in (specialist_dataset or "").split(",") if d.strip()]


def vllm_setup_source(model_dataset: str, wheels_dataset: str, specialist_dataset: str = "") -> str:
    """Notebook cell source: install vLLM from the wheelhouse, start the coordinator server (and, for the
    council arm, the specialist server through ``arc3.serve.specialist_attempts``: each attached specialist
    checkpoint with tuned then conservative flags, sized from the GPU memory the coordinator left), and
    publish ARC3_AGENT_CONFIG. The council only shares the coordinator model when every attempt failed."""
    spec_exprs = ", ".join(_mount_expr(r) for r in specialist_refs(specialist_dataset)) or ""
    return dedent(f"""\
        # Local model server(s). Datasets are attached in kernel-metadata.json.
        MODEL_DIR = {_mount_expr(model_dataset)}
        WHEELS = {_mount_expr(wheels_dataset)}
        SPECIALIST_DIRS = [d for d in [{spec_exprs}] if d]
        vllm_procs = []
        cfg = json.loads(os.environ.get('ARC3_AGENT_CONFIG', '{{}}'))
        if os.environ['ARC3_AGENT'] in ('repl', 'council'):
            if MODEL_DIR is None:
                print('no model dataset attached -> falling back to the rules agent'); os.environ['ARC3_AGENT'] = 'rules'
            else:
                if WHEELS:
                    t0 = time.time()
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-warn-conflicts', '--find-links', WHEELS, 'vllm'])
                    print('vllm installed in %.0fs' % (time.time() - t0))
                from arc3 import serve
                two = os.environ['ARC3_AGENT'] == 'council' and bool(SPECIALIST_DIRS) and not cfg.get('specialist')
                proc, ok = serve.start_vllm_with_fallback(MODEL_DIR, log_path='/kaggle/working/vllm.log', port=8000, served_name='arc3-model',
                                                          gpu_mem=0.60 if two else 0.90, timeout_s=1500)
                print('vllm ready:', ok, dict(serve.LAST_START), 'after %.0fs' % (time.time() - START))
                if not ok:
                    print(open('/kaggle/working/vllm.log').read()[-3000:]); os.environ['ARC3_AGENT'] = 'rules'
                else:
                    vllm_procs.append(proc)
                    cfg.update({{'base_url': 'http://127.0.0.1:8000/v1', 'model': 'arc3-model'}})
                    if two:
                        print('specialist checkpoints:', SPECIALIST_DIRS, 'GPU fraction free:', serve.gpu_fraction_available())
                        ladder = serve.specialist_attempts(SPECIALIST_DIRS, gpu_mem=0.30)
                        proc2, ok2 = serve.start_vllm_with_fallback(SPECIALIST_DIRS[0], log_path='/kaggle/working/vllm-specialist.log',
                                                                    attempts=ladder, timeout_s=1200)
                        print('specialist vllm ready:', ok2, dict(serve.LAST_START), 'after %.0fs' % (time.time() - START))
                        if ok2:
                            vllm_procs.append(proc2)
                            cfg['specialist'] = {{'base_url': 'http://127.0.0.1:8001/v1', 'model': 'arc3-specialist'}}
                            cfg['specialist_start'] = dict(serve.LAST_START)
                        else:
                            log2 = open('/kaggle/working/vllm-specialist.log').read()
                            print('\\n'.join(l for l in log2.splitlines() if l.startswith('#') or 'Error' in l or 'error' in l)[-3000:])
                            print('SPECIALIST SERVER FAILED on every attempt; council will use the coordinator model for specialist roles')
                            cfg['specialist_start'] = dict(serve.LAST_START)
                    os.environ['ARC3_AGENT_CONFIG'] = json.dumps(cfg)
        print('agent:', os.environ['ARC3_AGENT'], 'config keys:', sorted(cfg))
        """)


def build(accelerator: str, agent: str, model_dataset: str, wheels_dataset: str, budget_s: int, smoke_s: int = 300,
          specialist_dataset: str = "") -> dict:
    if accelerator not in _ACCELERATORS:
        raise SystemExit(f"unknown accelerator {accelerator}")
    agent_body = AGENT_SRC.read_text()
    tarball = package_tarball()

    cells = [md_cell("# ARC Prize 2026 — ARC-AGI-3 submission\n\nGenerated by `scripts/build_notebook.py`; edit the repo, not this file.")]
    cells.append(code_cell(dedent(f"""\
        import os, time, json, sys, subprocess, base64, io, tarfile
        START = time.time()
        os.environ['ARC3_START_EPOCH'] = str(START)
        os.environ.setdefault('ARC3_TIME_BUDGET_S', '{budget_s}')
        os.environ.setdefault('ARC3_AGENT', '{agent}')
        RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))
        print('competition rerun:', RERUN)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-warn-conflicts',
                               '--find-links', '{COMP_DIR}/arc_agi_3_wheels', 'arc-agi', 'python-dotenv'])
        """)))
    cells.append(code_cell(
        "ARC3_TGZ = '''" + tarball + "'''\n"
        + dedent("""\
        os.makedirs('/kaggle/working', exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARC3_TGZ)), mode='r:gz') as tar:
            tar.extractall('/kaggle/working')
        sys.path.insert(0, '/kaggle/working')
        os.environ['PYTHONPATH'] = '/kaggle/working' + os.pathsep + os.environ.get('PYTHONPATH', '')
        import arc3; print('arc3', arc3.__version__, 'unpacked at /kaggle/working/arc3')
        """)))
    cells.append(code_cell("%%writefile /tmp/my_agent.py\n" + agent_body))
    cells.append(code_cell(vllm_setup_source(model_dataset, wheels_dataset, specialist_dataset)))
    cells.append(code_cell(dedent(f"""\
        if RERUN:
            subprocess.check_call('curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games', shell=True)
            subprocess.check_call('cp -r {COMP_DIR}/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents', shell=True)
            subprocess.check_call('cp /tmp/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py', shell=True)
            open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w').write('''from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {{'random': Random, 'myagent': MyAgent}}
        ''')
            open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w').write('''SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        ''')
            env = dict(os.environ, MPLBACKEND='agg')
            rc = subprocess.call([sys.executable, 'main.py', '--agent', 'myagent'], cwd='/kaggle/working/ARC-AGI-3-Agents', env=env)
            print('framework exit code', rc, 'elapsed %.0fs' % (time.time() - START))
        else:
            # Save & Run All: prove the *configured* agent plays real games offline (REPL against the local
            # vLLM when it started, explorer otherwise), then emit the dummy submission.
            env_files = '{COMP_DIR}/environment_files'
            if os.path.isdir(env_files):
                from arc3.eval import run_eval
                smoke_agent = os.environ['ARC3_AGENT']
                cfg = json.loads(os.environ.get('ARC3_AGENT_CONFIG', '{{}}'))
                smoke_s = int(os.environ.get('ARC3_SMOKE_S', '{smoke_s}'))
                run_eval(smoke_agent, 'ls20,vc33', time_budget_s=smoke_s, max_actions=200, workers=1,
                         runs_dir='/kaggle/working/runs', environments_dir=env_files, run_name='save-and-run-smoke', config=cfg)
            else:
                print('no bundled environment_files; skipping offline smoke')
            import pandas as pd
            pd.DataFrame([['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet('/kaggle/working/submission.parquet', index=False)
        for p_ in vllm_procs:
            p_.terminate()
        """)))
    accel = _ACCELERATORS[accelerator]
    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python", "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": accel["name"], "isInternetEnabled": False, "isGpuEnabled": accel["gpu"],
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat_minor": 4, "nbformat": 4, "cells": cells,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accelerator", default="rtx6000", choices=sorted(_ACCELERATORS))
    p.add_argument("--agent", default="repl", choices=["repl", "council", "explorer", "random"])
    p.add_argument("--specialist-dataset", default="scottmahony/qwen3-vl-8b-instruct-fp8,scottmahony/qwen3-vl-8b-instruct-nvfp4",
                   help="comma-separated owner/slug list of specialist model datasets, tried in order (council arm only)")
    p.add_argument("--model-dataset", default="saltb0x/qwen3-8-27b-fp8",
                   help="owner/slug of the model weights dataset (default: official Qwen3.8-27B-FP8 snapshot, "
                        "verified byte-for-byte against HF sha 017b9c7a on 2026-09-16)")
    p.add_argument("--wheels-dataset", default="saltb0x/arc3-vllm-wheelhouse-v0271-cu129",
                   help="owner/slug of the vLLM wheelhouse dataset (vLLM 0.27.1, CUDA 12.9, built for the ARC3 duck harness)")
    p.add_argument("--budget-s", type=int, default=9 * 3600)
    p.add_argument("--smoke-s", type=int, default=300, help="per-game seconds for the Save & Run All offline smoke")
    p.add_argument("--out", default=str(NOTEBOOK_PATH))
    a = p.parse_args()
    spec = a.specialist_dataset if a.agent == 'council' else ''
    nb = build(a.accelerator, a.agent, a.model_dataset, a.wheels_dataset, a.budget_s, a.smoke_s, spec)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(nb, indent=1))
    print(f"[build_notebook] wrote {a.out} (accelerator={a.accelerator}, agent={a.agent}, {len(json.dumps(nb)) // 1024} KB)")
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        meta["enable_gpu"] = _ACCELERATORS[a.accelerator]["gpu"]
        ds = [d for d in (a.model_dataset, a.wheels_dataset, *specialist_refs(spec)) if d]
        if ds:
            meta["dataset_sources"] = ds
        METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
