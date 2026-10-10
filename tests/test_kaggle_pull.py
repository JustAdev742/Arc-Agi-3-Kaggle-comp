"""scripts/kaggle_pull.py: paced listing of a kernel's output, with backoff on 429 (fake SDK, no network)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import kaggle_pull  # noqa: E402


class _Http429(Exception):
    def __init__(self):
        super().__init__("429 Client Error: Too Many Requests")
        self.response = types.SimpleNamespace(status_code=429)


class _FakeApi:
    """Pages of 2 file names; the listing call fails with 429 on the given call numbers."""

    def __init__(self, names, refuse=()):
        self.names, self.refuse, self.calls, self.requests = names, set(refuse), 0, []

    def _set_paging(self, request, page_size, token):
        request.page_size, request.page_token = page_size, token or ""

    def build_kaggle_client(self):
        api = self

        class Client:
            def __enter__(self):
                def list_output(request):
                    api.calls += 1
                    api.requests.append((request.page_size, request.page_token))
                    if api.calls in api.refuse:
                        raise _Http429()
                    start = int(request.page_token or 0)
                    files = [types.SimpleNamespace(file_name=n, url=f"https://x/{n}")
                             for n in api.names[start:start + 2]]
                    nxt = str(start + 2) if start + 2 < len(api.names) else ""
                    return types.SimpleNamespace(files=files, next_page_token=nxt, log="the log")
                kernels = types.SimpleNamespace(kernels_api_client=types.SimpleNamespace(
                    list_kernel_session_output=list_output))
                return types.SimpleNamespace(kernels=kernels)

            def __exit__(self, *exc):
                return False

        return Client()


@pytest.fixture(autouse=True)
def _sdk(monkeypatch):
    module = types.ModuleType("kagglesdk.kernels.types.kernels_api_service")
    module.ApiListKernelSessionOutputRequest = lambda: types.SimpleNamespace()
    for name in ("kagglesdk", "kagglesdk.kernels", "kagglesdk.kernels.types"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "kagglesdk.kernels.types.kernels_api_service", module)


def test_pages_are_listed_with_pauses_and_a_refused_page_is_retried():
    api = _FakeApi(["a", "b", "c", "d", "e"], refuse={2, 3})
    sleeps, logs = [], []
    pages = list(kaggle_pull.list_pages(api, "me", "k", 200, 5.0, sleep=sleeps.append, log=logs.append))
    assert [[f.file_name for f in files] for files, _ in pages] == [["a", "b"], ["c", "d"], ["e"]]
    assert [log for _, log in pages] == ["the log", None, None]
    assert api.calls == 5 and all(size == 200 for size, _ in api.requests)
    assert sleeps == [0, 5.0, 0, 60, 120, 5.0, 0]  # page 2 refused twice: 60 s, then 120 s
    assert len(logs) == 2 and "429" in logs[0]


def test_a_page_still_refused_after_the_last_wait_raises():
    api = _FakeApi(["a", "b", "c"], refuse=set(range(2, 20)))
    with pytest.raises(_Http429):
        list(kaggle_pull.list_pages(api, "me", "k", 100, 1.0, sleep=lambda s: None, log=lambda m: None))
    assert api.calls == 1 + 1 + len(kaggle_pull.WAITS)
