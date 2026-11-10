"""Kaggle through its REST API with the standard library only, for a Mac without the kaggle CLI.
The token comes from $KAGGLE_CONFIG_DIR/access_token (default ~/.kaggle) and is never printed.

    python scripts/kaggle_rest.py dataset-create OWNER SLUG FILE...      # a new private dataset
    python scripts/kaggle_rest.py dataset-version OWNER SLUG FILE...     # replaces its files (old versions deleted:
                                                                         # never while a kernel that mounts it is queued)
    python scripts/kaggle_rest.py tpu-push OWNER SLUG KERNEL.py DATASET... # a private TPU v5e-8 script kernel, internet on
    python scripts/kaggle_rest.py status OWNER SLUG
    python scripts/kaggle_rest.py logs OWNER SLUG [LINES]                # the session's log so far
    python scripts/kaggle_rest.py delete OWNER SLUG                      # also drops a queued session (no cancel by slug)
    python scripts/kaggle_rest.py get PATH                               # GET https://www.kaggle.com/api/v1/PATH

Kaggle allows one queued or running TPU session per account.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://www.kaggle.com/api/v1"
RPC = "https://api.kaggle.com/v1/kernels.KernelsApiService"


def token():
    folder = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (folder / "access_token").read_text().strip()


def call(url, body=None, raw=False):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    request.add_header("Authorization", f"Bearer {token()}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            out = response.read()
    except urllib.error.HTTPError as error:
        return {"http_error": error.code, "body": error.read().decode()[:3000]}
    if raw:
        return out
    try:
        return json.loads(out or b"{}")
    except ValueError:
        return {"text": out.decode()[:3000]}


def upload_blob(path):
    path = Path(path)
    meta = call(f"{API}/blobs/upload", {"type": "dataset", "name": path.name, "contentLength": path.stat().st_size, "lastModifiedEpochSeconds": int(path.stat().st_mtime)})
    assert "token" in meta, meta
    request = urllib.request.Request(meta["createUrl"], data=path.read_bytes(), method="PUT")
    request.add_header("Content-Type", "application/octet-stream")
    with urllib.request.urlopen(request, timeout=3600) as response:
        response.read()
    return meta["token"]


def dataset_create(owner, slug, files):
    tokens = [{"token": upload_blob(f)} for f in files]
    return call(f"{API}/datasets/create/new", {"ownerSlug": owner, "slug": slug, "title": slug, "licenseName": "CC0-1.0", "isPrivate": True, "convertToCsv": False, "files": tokens, "categoryIds": []})


def dataset_version(owner, slug, files):
    tokens = [{"token": upload_blob(f)} for f in files]
    return call(f"{API}/datasets/create/version/{owner}/{slug}", {"versionNotes": "update", "deleteOldVersions": True, "convertToCsv": False, "files": tokens, "categoryIds": []})


def tpu_push(owner, slug, path, datasets):
    body = {"slug": f"{owner}/{slug}", "newTitle": slug, "text": Path(path).read_text(), "language": "python", "kernelType": "script", "isPrivate": True, "enableGpu": False, "enableTpu": True, "enableInternet": True, "machineShape": "TpuV5E8", "datasetDataSources": [d if "/" in d else f"{owner}/{d}" for d in datasets], "kernelDataSources": [], "competitionDataSources": [], "modelDataSources": [], "categoryIds": []}
    return call(f"{RPC}/SaveKernel", body)


def logs(owner, slug):
    raw = call(f"{RPC}/GetKernelSessionLogsStream", {"userName": owner, "kernelSlug": slug}, raw=True)
    try:
        return "".join(e.get("data", "") for e in json.loads(raw) if isinstance(e, dict))
    except (ValueError, TypeError):
        return ""


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == "logs":
        lines = logs(args[0], args[1]).splitlines()
        print("\n".join(lines[-int(args[2]) if len(args) > 2 else -40 :]))
        return
    out = {
        "dataset-create": lambda: dataset_create(args[0], args[1], args[2:]),
        "dataset-version": lambda: dataset_version(args[0], args[1], args[2:]),
        "tpu-push": lambda: tpu_push(args[0], args[1], args[2], args[3:]),
        "status": lambda: call(f"{RPC}/GetKernelSessionStatus", {"userName": args[0], "kernelSlug": args[1]}),
        "delete": lambda: call(f"{RPC}/DeleteKernel", {"userName": args[0], "kernelSlug": args[1]}),
        "get": lambda: call(f"{API}/{args[0]}"),
    }[command]()
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
