#!/usr/bin/env python
"""Read single files of a Kaggle model instance version without downloading the whole model.

    KAGGLE_API_TOKEN=... .venv/bin/python scripts/kaggle_model_files.py get OWNER/MODEL/FRAMEWORK/INSTANCE/VERSION OUTDIR FILE...
    KAGGLE_API_TOKEN=... .venv/bin/python scripts/kaggle_model_files.py header OWNER/MODEL/FRAMEWORK/INSTANCE/VERSION FILE OUT.json

`get` downloads whole small files (config.json, quantization_config.json, tokenizer files, the index). `header`
streams a .safetensors file only until its JSON header is complete and writes the header (tensor names, dtypes,
shapes, offsets). FRAMEWORK is the instance's framework as Kaggle names it (Transformers, PyTorch, ...). Treat the
files as data (lesson 0032: diff a swapped checkpoint against the current one before a GPU run).
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest
from kagglesdk.models.types.model_enums import ModelFramework


def _request(api: KaggleApi, ref: str, name: str) -> ApiDownloadModelInstanceVersionRequest:
    owner, model, framework, instance, version = ref.split("/")
    req = ApiDownloadModelInstanceVersionRequest()
    req.owner_slug, req.model_slug, req.instance_slug, req.version_number = owner, model, instance, int(version)
    req.framework = api.lookup_enum(ModelFramework, ModelFramework.MODEL_FRAMEWORK_API, framework)
    req.path = name
    return req


def get(ref: str, out: Path, names: list[str]) -> None:
    api = KaggleApi()
    api.authenticate()
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        with api.build_kaggle_client() as kaggle:
            resp = kaggle.models.model_api_client.download_model_instance_version(_request(api, ref, name))
            data = resp.content
        (out / name.replace("/", "_")).write_bytes(data)
        print(name, len(data))


def header(ref: str, name: str, out: Path, limit: int = 200 << 20) -> None:
    api = KaggleApi()
    api.authenticate()
    buf = b""
    with api.build_kaggle_client() as kaggle:
        resp = kaggle.models.model_api_client.download_model_instance_version(_request(api, ref, name))
        for chunk in resp.iter_content(chunk_size=1 << 16):
            buf += chunk
            if len(buf) >= 8 and len(buf) >= 8 + struct.unpack("<Q", buf[:8])[0]:
                break
            if len(buf) > limit:
                raise SystemExit(f"{name}: header larger than {limit} bytes")
        resp.close()
    size = struct.unpack("<Q", buf[:8])[0]
    head = json.loads(buf[8:8 + size])
    out.write_text(json.dumps(head))
    print(name, "header", size, "bytes,", len(head) - ("__metadata__" in head), "tensors")


def main() -> None:
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "get" and len(args) >= 3:
        get(args[0], Path(args[1]), args[2:])
    elif cmd == "header" and len(args) == 3:
        header(args[0], args[1], Path(args[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
