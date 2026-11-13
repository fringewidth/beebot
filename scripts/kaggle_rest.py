"""Kaggle through its REST API with the standard library only, for a Mac without the kaggle CLI.
The token comes from $KAGGLE_CONFIG_DIR/access_token (default ~/.kaggle) and is never printed.

    python scripts/kaggle_rest.py dataset-create OWNER SLUG FILE...      # a new private dataset
    python scripts/kaggle_rest.py dataset-version OWNER SLUG FILE...     # replaces its files (old versions deleted:
                                                                         # never while a kernel that mounts it is queued)
    python scripts/kaggle_rest.py tpu-push OWNER SLUG KERNEL.py DATASET... # a private TPU v5e-8 script kernel, internet on
    python scripts/kaggle_rest.py cpu-push OWNER SLUG KERNEL.py        # a private CPU-only script kernel (tests; no GPU quota)
    python scripts/kaggle_rest.py status OWNER SLUG
    python scripts/kaggle_rest.py logs OWNER SLUG [LINES]                # the session's log so far
    python scripts/kaggle_rest.py delete OWNER SLUG                      # also drops a queued session (no cancel by slug)
    BEEBOT_CONTROL_RUN=NAME python scripts/kaggle_rest.py tpu-push ...   # also fills the kernel's __CONTROL_URL__ (see run_control.py)
    python scripts/kaggle_rest.py get PATH                               # GET https://www.kaggle.com/api/v1/PATH
    python scripts/kaggle_rest.py handoff SOURCE SLUG TARGET [RUN]       # the TPU run's second session (handoff below)

An account is OWNER (token in ~/.kaggle) or OWNER=CONFIG_DIR (a teammate's token folder), e.g.
    python scripts/kaggle_rest.py handoff rvitmpolicyboot=~/.kaggle-friends/rvitm beebot-think-tpu fringewidth

Kaggle allows one queued or running TPU session per account. A kernel appears to mount the dataset
versions current when it was pushed (unconfirmed for a queued kernel), so a dataset that changes later
is only sure to reach it through a new push; deleting a version a queued kernel was pushed against
leaves it without that source. An account that has never run a TPU gets a 4-core CPU machine instead.
"""

import json
import os
import shutil
import sys
import time
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
    """Upload a file (streamed from disk) for a dataset; its blob token."""
    path = Path(path)
    meta = call(f"{API}/blobs/upload", {"type": "dataset", "name": path.name, "contentLength": path.stat().st_size, "lastModifiedEpochSeconds": int(path.stat().st_mtime)})
    assert "token" in meta, meta
    with open(path, "rb") as f:
        request = urllib.request.Request(meta["createUrl"], data=f, method="PUT")
        request.add_header("Content-Type", "application/octet-stream")
        request.add_header("Content-Length", str(path.stat().st_size))
        with urllib.request.urlopen(request, timeout=7200) as response:
            response.read()
    return meta["token"]


def dataset_create(owner, slug, files):
    tokens = [{"token": upload_blob(f)} for f in files]
    return call(f"{API}/datasets/create/new", {"ownerSlug": owner, "slug": slug, "title": slug, "licenseName": "CC0-1.0", "isPrivate": True, "convertToCsv": False, "files": tokens, "categoryIds": []})


def dataset_version(owner, slug, files):
    tokens = [{"token": upload_blob(f)} for f in files]
    return call(f"{API}/datasets/create/version/{owner}/{slug}", {"versionNotes": "update", "deleteOldVersions": True, "convertToCsv": False, "files": tokens, "categoryIds": []})


def tpu_push(owner, slug, path, datasets):
    text = Path(path).read_text()
    if os.environ.get("BEEBOT_CONTROL_RUN"):  # a control topic for the run (scripts/run_control.py), put in the pushed copy only
        sys.path.insert(0, str(Path(__file__).parent))
        import run_control

        text = run_control.inject(text, os.environ["BEEBOT_CONTROL_RUN"])
    body = {"slug": f"{owner}/{slug}", "newTitle": slug, "text": text, "language": "python", "kernelType": "script", "isPrivate": True, "enableGpu": False, "enableTpu": True, "enableInternet": True, "machineShape": "TpuV5E8", "datasetDataSources": [d if "/" in d else f"{owner}/{d}" for d in datasets], "kernelDataSources": [], "competitionDataSources": [], "modelDataSources": [], "categoryIds": []}
    return call(f"{RPC}/SaveKernel", body)


def cpu_push(owner, slug, path):
    """A private CPU-only script kernel, internet on (no GPU or TPU quota): for testing the kill switch."""
    text = Path(path).read_text()
    if os.environ.get("BEEBOT_CONTROL_RUN"):
        sys.path.insert(0, str(Path(__file__).parent))
        import run_control

        text = run_control.inject(text, os.environ["BEEBOT_CONTROL_RUN"])
    body = {"slug": f"{owner}/{slug}", "newTitle": slug, "text": text, "language": "python", "kernelType": "script", "isPrivate": True, "enableGpu": False, "enableTpu": False, "enableInternet": True, "datasetDataSources": [], "kernelDataSources": [], "competitionDataSources": [], "modelDataSources": [], "categoryIds": []}
    return call(f"{RPC}/SaveKernel", body)


def logs(owner, slug, seconds=30):
    """The session's log so far. A running session's log is an event stream that stays open
    ("data: {...}" lines): read for `seconds`, then keep what arrived; a finished one's is a JSON
    list (one entry per line, '[' or ',' first)."""
    request = urllib.request.Request(f"{RPC}/GetKernelSessionLogsStream", data=json.dumps({"userName": owner, "kernelSlug": slug}).encode(), method="POST")
    request.add_header("Authorization", f"Bearer {token()}")
    request.add_header("Content-Type", "application/json")
    raw, end = b"", time.time() + seconds
    try:
        with urllib.request.urlopen(request, timeout=seconds) as response:
            while time.time() < end:
                piece = response.read1(1 << 16)
                if not piece:
                    break
                raw += piece
    except Exception:  # the stream's timeout or a cut chunk: what arrived is kept
        pass
    text = []
    for line in raw.decode(errors="replace").splitlines():
        line = line.strip().removeprefix("data:").strip().lstrip("[,").rstrip("]")
        try:
            text.append(json.loads(line).get("data", ""))
        except (ValueError, AttributeError):
            pass
    return "".join(text)


def account(spec):
    """Act as OWNER or OWNER=CONFIG_DIR (a teammate's token folder); the owner's name."""
    owner, _, folder = spec.partition("=")
    if folder:
        os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser(folder)
    else:
        os.environ.pop("KAGGLE_CONFIG_DIR", None)
    return owner


def output_files(owner, slug):
    """[(file name, signed url)] of a finished kernel's output, every page."""
    files, page = [], None
    while True:
        body = {"userName": owner, "kernelSlug": slug, "pageSize": 100, **({"pageToken": page} if page else {})}
        out = call(f"{RPC}/ListKernelSessionOutput", body)
        assert "http_error" not in out, out
        files += [(f["fileName"], f["url"]) for f in out.get("files", [])]
        page = out.get("nextPageToken")
        if not page:
            return files


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=7200) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 24)
    print(f"downloaded {path} ({path.stat().st_size / 1e9:.2f} GB)", flush=True)


def handoff(source, slug, target, run, kernel, folder):
    """The TPU run's second session: the first session's (source's) resume.pt, config.json,
    kernel.json, history.jsonl and latest policy_step*.pt downloaded to folder, published as the
    target account's private dataset beebot-think-resume, and the target's kernel slug (deleted
    first if it only waits in the queue) pushed with ROLE second, which resumes it."""
    owner = account(source)
    status = call(f"{RPC}/GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug}).get("status", "QUEUED")
    assert status not in ("QUEUED", "RUNNING"), f"{owner}/{slug} is {status}: wait for it to finish"
    files = dict(output_files(owner, slug))
    wanted = [f"runs/{run}/{name}" for name in ("resume.pt", "config.json", "kernel.json", "history.jsonl")]
    steps = sorted((f for f in files if f.startswith(f"runs/{run}/policy_step") and f.endswith(".pt")), key=lambda f: int(f.rsplit("step", 1)[1][:-3]))
    assert f"runs/{run}/resume.pt" in files and f"runs/{run}/config.json" in files, f"no resume.pt and config.json for {run} in {owner}/{slug}'s output: {sorted(files)[:20]}"
    local = []
    for name in [f for f in wanted if f in files] + steps[-1:]:
        path = Path(folder) / Path(name).name
        download(files[name], path)
        local.append(path)
    owner = account(target)
    exists = "http_error" not in call(f"{API}/datasets/view/{owner}/beebot-think-resume")
    out = dataset_version(owner, "beebot-think-resume", local) if exists else dataset_create(owner, "beebot-think-resume", local)
    assert out.get("status", "Ok") == "Ok" and not out.get("error"), out
    for _ in range(120):  # until the new version is ready
        view = call(f"{API}/datasets/view/{owner}/beebot-think-resume")
        if view.get("totalBytes", 0) >= sum(p.stat().st_size for p in local) * 0.9 and all(v.get("status") == "Ready" for v in view.get("versions", [])[:1]):
            break
        time.sleep(30)
    print(f"published {owner}/beebot-think-resume: {[p.name for p in local]}", flush=True)
    status = call(f"{RPC}/GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug})
    assert status.get("status") != "RUNNING", f"{owner}/{slug} is running: not replaced"
    if "http_error" not in status:  # an existing kernel (queued, or an earlier session): replaced
        call(f"{RPC}/DeleteKernel", {"userName": owner, "kernelSlug": slug})
    text = Path(kernel).read_text().replace('ROLE = os.environ.get("BEEBOT_ROLE", "first")', 'ROLE = os.environ.get("BEEBOT_ROLE", "second")')
    assert 'ROLE = os.environ.get("BEEBOT_ROLE", "second")' in text, "the kernel has no ROLE line"
    pushed = Path(folder) / "kernel_second.py"
    pushed.write_text(text)
    print(json.dumps(tpu_push(owner, slug, pushed, ["beebot-src-tpu", "beebot-think-data", "beebot-think-resume"]), indent=1))
    return call(f"{RPC}/GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug})


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == "handoff":
        source, slug, target = args[:3]
        run = args[3] if len(args) > 3 else "think-08b-tpu"
        print(json.dumps(handoff(source, slug, target, run, "scripts/kaggle_tpu/think_kernel.py", f"runs/kaggle/handoff-{slug}")))
        return
    if command == "logs":
        lines = logs(args[0], args[1]).splitlines()
        print("\n".join(lines[-int(args[2]) if len(args) > 2 else -40 :]))
        return
    out = {
        "dataset-create": lambda: dataset_create(args[0], args[1], args[2:]),
        "dataset-version": lambda: dataset_version(args[0], args[1], args[2:]),
        "cpu-push": lambda: cpu_push(args[0], args[1], args[2]),
        "tpu-push": lambda: tpu_push(args[0], args[1], args[2], args[3:]),
        "status": lambda: call(f"{RPC}/GetKernelSessionStatus", {"userName": args[0], "kernelSlug": args[1]}),
        "delete": lambda: call(f"{RPC}/DeleteKernel", {"userName": args[0], "kernelSlug": args[1]}),
        "get": lambda: call(f"{API}/{args[0]}"),
    }[command]()
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
