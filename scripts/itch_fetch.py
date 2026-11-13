"""Download the .vox files of free itch.io asset packs (the public "Download" flow of a pay-what-you-want page at 0).

    nice -n 10 .venv/bin/python -I scripts/itch_fetch.py data/goal_sources/itch_padadu padadu voxel-pack-02-nature ...

For each pack slug of an itch.io user: open the page, ask for the download page, list the files, fetch every .vox
(and .txt licence or readme file) from the CDN link itch.io returns. Writes <out>/<slug>/<name> and
<out>/manifest.json (slug, licence text found on the page, files). Never executes anything it downloads.
"""

import html
import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}


def main():
    out = Path(sys.argv[1])
    user = sys.argv[2]
    slugs = sys.argv[3:]
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = list(UA.items())

    def get(url, data=None):
        request = urllib.request.Request(url, data=data, headers=UA)
        with opener.open(request, timeout=60) as response:
            return response.read()

    manifest = []

    def fetch(slug):
        base = f"https://{user}.itch.io/{slug}"
        page = get(base).decode("utf8", "replace")
        csrf = re.search(r'name="csrf_token" value="([^"]*)"', page).group(1)
        text = html.unescape(re.sub(r"<[^>]*>", " ", page))
        licence = sorted(set(re.findall(r"(CC0|CC-BY[^ ,.)]*|public domain)", text, flags=re.I)))
        post = urllib.parse.urlencode({"csrf_token": csrf}).encode()
        url = json.loads(get(base + "/download_url", post))["url"]
        listing = get(url).decode("utf8", "replace")
        ids = re.findall(r'data-upload_id="(\d+)"', listing)
        names = re.findall(r'<strong[^>]*class="name"[^>]*>([^<]*)<', listing)
        token = url.split("/download/")[1]
        entries = list(zip(ids, names))
        print(slug, len(ids), len(names), len(listing), flush=True)
        folder = out / slug
        folder.mkdir(parents=True, exist_ok=True)
        files = []
        for upload_id, name in entries:
            name = html.unescape(name)
            if not name.lower().endswith((".vox", ".txt", ".md", ".zip")):
                continue
            reply = json.loads(get(f"{base}/file/{upload_id}?download_token={token}", post))
            data = get(reply["url"])
            (folder / re.sub(r"[^\w .()+-]", "_", name)).write_bytes(data)
            files.append(name)
            time.sleep(0.3)
        return {"slug": slug, "user": user, "licence_mentions": licence, "files": files}

    for slug in slugs:
        try:
            manifest.append(fetch(slug))
        except Exception as error:
            print(slug, "failed", error, flush=True)
    path = out / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(old + manifest, indent=1))


if __name__ == "__main__":
    main()
