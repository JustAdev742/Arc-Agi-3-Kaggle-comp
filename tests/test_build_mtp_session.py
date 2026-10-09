"""scripts/build_mtp_session.py: the session-A notebook of the MTP draft fine-tune (plan sections 10.4 and 11).

D' up to its launcher (the same cells scripts/build_franzen_nb.py builds), cell 12 cut before the server starts, our
scripts shipped compressed and written back byte for byte, the dump servers' arguments derived from the launcher's
real cell, the step cells (GO and NO-GO) run against the real module with stub scripts, the kernel metadata, the
notebook's size and the session's time budget. CPU only."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_franzen_nb as bf  # noqa: E402
import build_mtp_session as bm  # noqa: E402
import hc_dump_driver as dd  # noqa: E402
import mtp_session_a as sa  # noqa: E402

from tests.test_mtp_session_a import Bed, launcher_args  # noqa: E402

REAP_KEPT = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("nb")
    result = bm.build(out)
    nb = json.loads(result["notebook"].read_text())
    return result, nb, ["".join(c["source"]) for c in nb["cells"]], json.loads((out / "kernel-metadata.json").read_text())


def _compiles(src: str) -> None:
    compile("\n".join(line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))), "cell", "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def _index(cells: list[str], needle: str) -> int:
    hits = [i for i, s in enumerate(cells) if needle in s]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def test_every_code_cell_compiles_and_the_notebook_stays_small(built):
    result, nb, cells, _ = built
    assert result["bytes"] < 900_000  # Kaggle refuses a notebook near 1 MB with a bare HTTP 400 (lesson 0033)
    for cell, src in zip(nb["cells"], cells):
        if cell["cell_type"] == "code" and not src.startswith("%%writefile"):
            _compiles(src)
    assert nb["metadata"]["kaggle"]["accelerator"] == "nvidiaRtxPro6000"
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_the_notebook_is_dprime_up_to_its_launcher(built, tmp_path):
    _, _, cells, _ = built
    bf.build(tmp_path, "ref", base="dprime", input_fallback=True, wait_inputs=120, reap_kept=REAP_KEPT, compact=True)
    ref = ["".join(c["source"]) for c in json.loads((tmp_path / "ref.ipynb").read_text())["cells"]]
    launch_ref, launch = _index(ref, bf.LAUNCH_ANCHOR), _index(cells, bf.LAUNCH_ANCHOR)
    ours = [s for s in cells[:launch] if not (s.startswith(bm.SA_BEGIN) or s.startswith(bm.MAP_BEGIN)
                                              or s.startswith("## Session A (ours)")
                                              or s.startswith(bf.COMPACT_BEGIN + ": /kaggle/arc3-mtp/"))]
    assert ours[1:] == ref[1:launch_ref]  # every D' cell (and the REAP file cells) as build_franzen_nb.py builds them
    assert cells[0].startswith(ref[0].split("\n\n**Our arm")[0]) and "**Our session-A notebook" in cells[0]
    head = ref[launch_ref].split(bm.LAUNCH_CUT)[0]
    assert cells[launch].startswith(head) and cells[launch][len(head):].startswith(bm.SA_BEGIN)
    for gone in ("subprocess.Popen(args", "precache_model_thread.start()", "READY after", "def show_log_tail"):
        assert gone not in cells[launch]
    # nothing of D' after the launcher (benchmark, monitor, diagnostics)
    assert not any("await bm.run(" in s or "diagnostics.html" in s or "NOTEBOOK_START_EPOCH" in s
                   for s in cells[launch + 1:])
    order = [_index(cells, s) for s in (bf.SETUP_ANCHOR, "os.makedirs('/kaggle/arc3-mtp'", "SESSION.storage()",
                                        "def precache(", bm.MAP_BEGIN, bf.LAUNCH_ANCHOR, "SESSION.go('A1')",
                                        "SESSION.replica_gate()", "SESSION.write_draft()", "SESSION.finish()")]
    assert order == sorted(order)


def test_shipped_files_round_trip_byte_for_byte(built, tmp_path):
    _, _, cells, _ = built
    shipped = {}
    for src in cells:
        written = bf.franzen_tree.written_file(src)
        if written is not None:
            shipped[written[0]] = (src, written[1])
    expected = {f"{bm.SCRIPTS_DIR}/{n}": (ROOT / "scripts" / n).read_text() for n in bm.SCRIPTS}
    expected.update({bf.REAP_FILES["script"]: bf.REAP_SCRIPT.read_text(), bf.REAP_FILES["kept"]: REAP_KEPT.read_text(),
                     bf.REAP_FILES["meta"]: REAP_KEPT.with_suffix(".meta.json").read_text()})
    assert set(shipped) == set(expected) | {"/kaggle/harness-changes.patch"}
    for path, text in expected.items():
        src, got = shipped[path]
        assert src.startswith(bf.COMPACT_BEGIN) and got == text, path
        out = tmp_path / Path(path).name
        exec(src.replace(repr(path), repr(str(out))), {})  # what the notebook writes
        assert out.read_bytes() == text.encode("utf-8"), path
        broken = src.replace(hashlib.sha256(text.encode()).hexdigest(), "0" * 64)
        with pytest.raises(RuntimeError, match="corrupted in the notebook"):
            exec(broken.replace(repr(path), repr(str(out))), {})


def test_the_training_map_is_the_one_build_franzen_nb_ships_and_the_config_pins_it(built, tmp_path):
    result, _, cells, _ = built
    cell = cells[_index(cells, bm.MAP_BEGIN)]
    out = tmp_path / "arc3-hot-tokens.pt"
    exec(cell.replace(repr(bm.ARC_MAP_FILE), repr(str(out))), {})
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert result["config"]["train"] == dict(result["config"]["train"], map=bm.ARC_MAP_FILE, map_sha256=sha)
    assert bf._read_hot_ids(out) == bf._read_hot_ids(bm.DEFAULT_MAP)
    bf.build(tmp_path / "h", "h", base="dprime", hot_tokens=bm.DEFAULT_MAP)  # the --hot-tokens arm's file
    launch = next(s for s in ("".join(c["source"]) for c in json.loads((tmp_path / "h" / "h.ipynb").read_text())["cells"])
                  if bf.LAUNCH_ANCHOR in s)
    assert f'TOKEN_MAP_SHA = "{sha}"' in launch
    # the launcher here keeps the wheelhouse's generic map (the reference run's): its own assert checks it
    launch = cells[_index(cells, bf.LAUNCH_ANCHOR)]
    assert f'TOKEN_MAP_SHA = "{bm.GENERIC_MAP_SHA256}"' in launch and 'find_unique(WHEELHOUSE_DIR, "hot_tokens_64k.pt"' \
        in launch and result["config"]["probe"]["generic_map_sha256"] == bm.GENERIC_MAP_SHA256


def _launcher_args(cell: str, tmp_path: Path) -> list[str]:
    """Run the launcher cell's constants and its argument assembly (with stubs for the paths and helpers)."""
    consts = cell[cell.index('PREFIX = "/tmp/sgl-intel"'):cell.index("\n\ndef run(")]
    section = cell[cell.index("# ---- assemble Pennyroyal / AutoRound launch arguments ----"):cell.index(bm.SA_BEGIN)]
    model, wheels = tmp_path / "model", tmp_path / "wheels"
    model.mkdir()
    wheels.mkdir()
    (model / "chat_template.jinja").write_text("{{ x }}")
    ns = {"Path": Path, "WORKING_DIR": tmp_path, "SERVED_MODEL_NAME": "flashnext", "SERVED_MODEL_PORT": 8001,
          "MODEL_DIR": str(model), "DRAFT_VIEW": "/tmp/sgl-intel/draft-view-x", "WHEELHOUSE_DIR": str(wheels),
          "find_unique": lambda root, name, **kw: Path(root) / name}
    exec(consts, ns)
    ns["sha256"] = lambda p: ns["TOKEN_MAP_SHA"] if Path(p).name == "hot_tokens_64k.pt" else ns["TOKENIZER_SHA"]
    exec(section, ns)
    return ns["args"]


def test_the_dump_servers_are_the_launchers_args_plus_and_minus_the_listed_flags(built, tmp_path):
    _, _, cells, _ = built
    args = _launcher_args(cells[_index(cells, bf.LAUNCH_ANCHOR)], tmp_path)
    assert args[1] == "serve" and "--speculative-token-map" in args and "--json-model-override-args" in args
    _, items = sa.split_flags(args)
    speculative = [[f, *v] for f, v in items if f.startswith("--speculative-")]
    assert len(speculative) == 12  # NEXTN's ten flags, the FR-Spec map and REAP's draft override
    reap = ["--json-model-override-args", '{"text_config": {"num_experts": 448}}']
    a2, a6 = sa.dump_server_args(args, reap=False), sa.dump_server_args(args, reap=True)
    assert reap == next([f, *v] for f, v in items if not f.startswith("--speculative-")
                        and f.endswith("override-args"))
    removed = [[f, *v] for f, v in items if f.startswith("--speculative-") or f == "--json-model-override-args"]
    assert sa.args_diff(args, a2) == {"removed": removed,
                                      "added": [["--disable-cuda-graph"], ["--disable-radix-cache"]],
                                      "changed": [["--max-running-requests", "10", "->", "1"]]}
    assert sa.args_diff(args, a6) == {"removed": speculative,
                                      "added": [["--disable-cuda-graph"], ["--disable-radix-cache"]],
                                      "changed": [["--max-running-requests", "10", "->", "1"]]}
    # D' already prefills in 8,192-token chunks; the flags only a prefix cache uses stay (inert without one: plan 11)
    assert a2[a2.index("--chunked-prefill-size"):][:2] == ["--chunked-prefill-size", "8192"]
    for flag, value in (("--mamba-radix-cache-strategy", "extra_buffer"), ("--schedule-policy", "lpm"),
                        ("--page-size", "64")):
        assert a2[a2.index(flag) + 1] == value and a6[a6.index(flag) + 1] == value
    for server, drop in ((a2, ("--json-model-override-args",)), (a6, ())):  # word for word, in the launcher's order
        kept = [[f, *(["1"] if f == "--max-running-requests" else v)] for f, v in items
                if not f.startswith("--speculative-") and f not in drop]
        assert server == args[:2] + [w for item in kept for w in item] + ["--disable-cuda-graph",
                                                                          "--disable-radix-cache"]


def test_the_step_cells_run_a_go_session_to_the_end(built, tmp_path):
    _run_step_cells(built, tmp_path, "go")


def test_the_step_cells_stop_cleanly_at_a5_on_no_go(built, tmp_path):
    bed, session = _run_step_cells(built, tmp_path, "no-go")
    state = json.loads((bed.working / "session-a.json").read_text())
    assert (state["verdict"], state["stopped_at"], state["exit_code"]) == ("no-go", "A5", 2)
    assert [c["script"] for c in bed.calls()] == ["sglang_hc_dump_patch.py", "mtp_probe_dump.py", "mtp_replica.py"]
    assert not (bed.working / "mtp-train").exists() and not (bed.working / "mtp-draft").exists()
    assert (bed.working / "replica-check.json").is_file() and session.server is None


def _run_step_cells(built, tmp_path: Path, verdict: str):
    """The notebook's own cells after the launcher, executed in order against the real module (stub scripts)."""
    _, nb, cells, _ = built
    bed = Bed(tmp_path, verdict=verdict)
    session = bed.session()
    session.find_inputs(root=str(bed.input), poll_s=0.1)
    session.storage()
    args = launcher_args(sglang=sys.executable, port=bed.port)
    args[1:2] = [str(bed.scripts / "sglang.py"), "serve"]
    ns = {"SESSION": session, "args": args, "env": dict(os.environ, ARC3_REAP_KEPT_EXPERTS="/k/kept.json"),
          "LOG": bed.working / "serve.log", "SERVED_MODEL_PORT": bed.port, "SERVED_MODEL_NAME": "flashnext",
          "PYTHON": sys.executable, "VENV": str(bed.venv), "MODEL_DIR": str(tmp_path / "target"),
          "DRAFT_MODEL_DIR": str(tmp_path / "albucino"), "tok": bed.generic, "precache_model_thread": None}
    launch = _index(cells, bf.LAUNCH_ANCHOR)
    steps = [s for c, s in zip(nb["cells"][launch + 1:], cells[launch + 1:]) if c["cell_type"] == "code"]
    assert [s.split(":")[0].split()[-1] for s in steps] == [f"A{i}" for i in range(1, 12)] + ["end"]
    for src in steps:
        exec(src, ns)
    state = json.loads((bed.working / "session-a.json").read_text())
    if verdict == "go":
        assert state["verdict"] == "complete" and ns["SESSION_A_GO"] is True
        assert [x["step"] for x in state["steps"]][-8:] == ["A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"]
        assert (bed.working / "mtp-draft" / "arc3-draft-manifest.json").is_file()
    else:
        assert ns["SESSION_A_GO"] is False
    return bed, session


def test_the_setup_cell_holds_the_session_config(built):
    result, _, cells, _ = built
    src = cells[_index(cells, "SESSION_A_CONFIG = {")]
    tree = ast.parse(src)
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id == "SESSION_A_CONFIG")
    config = ast.literal_eval(value)
    assert config == result["config"]
    assert src.index("SESSION.find_inputs()") < src.index("SESSION.storage()")
    assert "if TRUE_SUBMISSION:\n    raise RuntimeError" in src
    inputs = config["inputs"]
    manifest = json.loads((bm.PROBE_DIR / "manifest.json").read_text())
    assert inputs["prompts"] == {"kind": "dataset", "id": "scottmahony/arc3-fidelity-prompts",
                                 "files": {"requests.jsonl": manifest["files"][0]["sha256"]}}
    assert inputs["reference"]["kind"] == "kernel" and len(inputs["reference"]["files"]["fidelity.json"]) == 2
    for sha, local in zip(inputs["reference"]["files"]["fidelity.json"], ("fidelity-base", "fidelity-base2")):
        path = ROOT / "runs" / local / "kernel-output" / "fidelity.json"
        if path.is_file():  # runs/ is not in every checkout
            assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    logs = inputs["logs"]
    assert logs["kind"] == "kernel" and logs["id"] == "scottmahony/arc3-dprime-reap448-r14-full"
    games = {dd.LOG_RE.match(n)["game"] for n in logs["files"]}
    assert len(logs["files"]) == 25 and set(dd.HOLDOUT) <= games and all(v > 1_000_000 for v in logs["files"].values())
    assert config["dump"]["holdout_games"] == sorted(dd.HOLDOUT)
    assert config["server"] == {"reap": True, "num_experts": 448, "gpu_free_mib": 3000}
    assert set(config["build"]["scripts"]) == set(bm.SCRIPTS)


def test_the_kernel_metadata_mounts_every_input(built, tmp_path):
    _, _, _, meta = built
    assert meta["id"] == "scottmahony/arc3-mtp-session-a" and meta["title"] == "arc3 mtp session a"
    assert meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"]
    assert meta["docker_image"] == bf.IMAGE and meta["docker_image_pinning_type"] == "original"
    assert meta["machine_shape"] == bf.MACHINE_SHAPE
    assert meta["dataset_sources"] == bf.SOURCES["dataset_sources"] + ["scottmahony/arc3-fidelity-prompts"]
    assert meta["model_sources"] == bf.SOURCES["model_sources"]
    assert meta["competition_sources"] == bf.SOURCES["competition_sources"]
    assert meta["kernel_sources"] == ["scottmahony/arc3-fidelity-base", "scottmahony/arc3-dprime-reap448-r14-full"]
    # the fallback: the logs from a dataset the owner uploads
    result = bm.build(tmp_path, "s", logs_dataset="scottmahony/arc3-exp073-request-logs")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["scottmahony/arc3-fidelity-base"]
    assert meta["dataset_sources"][-1] == "scottmahony/arc3-exp073-request-logs"
    assert result["config"]["inputs"]["logs"]["kind"] == "dataset"
    with pytest.raises(SystemExit, match="owner/slug"):
        bm.build(tmp_path, "s", logs_dataset="not a slug")
    # both from one dataset: listed once, no kernel sources left
    result = bm.build(tmp_path, "s", logs_dataset="me/session-a-inputs", reference_dataset="me/session-a-inputs")
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == [] and meta["dataset_sources"].count("me/session-a-inputs") == 1
    assert result["config"]["inputs"]["reference"]["files"] == bm.REFERENCE["files"]


def test_the_session_fits_its_deadline_and_kaggles(built, tmp_path):
    result, _, _, _ = built
    config = result["config"]
    assert bm.worst_case_minutes(config) <= config["session_hours"] * 60 <= 8 * 60  # Kaggle stops a session at 12 h
    assert config["working_limit_gb"] == 19.5 and "/kaggle/working" not in config["storage"]["candidates"]
    with pytest.raises(SystemExit, match="cover the steps' budgets"):
        bm.build(tmp_path, "s", settings=["train.minutes=600"])
    with pytest.raises(SystemExit, match="at most 10"):
        bm.build(tmp_path, "s", settings=["session_hours=12"])


def test_set_changes_one_entry_with_its_type(tmp_path):
    result = bm.build(tmp_path, "s", settings=["storage.train_gb=30", "dump.holdout_games=[\"ar25\"]"])
    assert result["config"]["storage"]["train_gb"] == 30.0 and result["config"]["dump"]["holdout_games"] == ["ar25"]
    assert any(c == "session config storage.train_gb=30" for c in result["changes"])
    for bad, why in (("storage.nope=1", "no such entry"), ("nope.x=1", "no such section"),
                     ("train.minutes=\"long\"", "expected int"), ("storage.train_gb", "expected KEY=VALUE")):
        with pytest.raises(SystemExit, match=why):
            bm.build(tmp_path, "s", settings=[bad])


def test_without_reap_the_training_dump_serves_the_unpruned_target(tmp_path):
    result = bm.build(tmp_path, "s", reap_kept=None, settings=["server.reap=false"])
    cells = ["".join(c["source"]) for c in json.loads(result["notebook"].read_text())["cells"]]
    assert result["config"]["server"] == {"reap": False, "num_experts": None, "gpu_free_mib": 3000}
    assert not any(bf.REAP_BEGIN in s for s in cells)
    with pytest.raises(SystemExit, match="needs --no-reap"):
        bm.build(tmp_path, "s", settings=["server.reap=false"])


def test_the_setup_cell_loads_the_shipped_module_by_path_and_builds_the_session(built, tmp_path):
    _, _, cells, _ = built
    src = cells[_index(cells, "SESSION_A_CONFIG = {")]
    module = f"{bm.SCRIPTS_DIR}/{bm.MODULE}"
    assert repr(module) in src
    run = src.replace(repr(module), repr(str(ROOT / "scripts" / bm.MODULE)))
    run = run.replace("SESSION.find_inputs()\n", "").replace("SESSION.storage()  # A0\n", "")  # need /kaggle/input
    ns = {"sys": sys, "WORKING_DIR": tmp_path, "NOTEBOOK_START_TIME": 1_000.0, "TRUE_SUBMISSION": False}
    exec(run, ns)
    state = json.loads((tmp_path / "session-a.json").read_text())
    assert state["verdict"] == "running" and state["build"]["builder"] == "scripts/build_mtp_session.py"
    assert ns["SESSION"].deadline == 1_000.0 + ns["SESSION_A_CONFIG"]["session_hours"] * 3600
    with pytest.raises(RuntimeError, match="not a submission"):
        exec(run, dict(ns, TRUE_SUBMISSION=True))
