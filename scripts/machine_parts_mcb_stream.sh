#!/bin/bash
# Stream MCB dataset A (Box shared link, 2.5 GB) straight into the converter: nothing is stored on disk.
cd "$(dirname "$0")/.." || exit 1
curl -sL "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=lwvmxbu8v75g5hd1ulaloszte0l1t1n9&file_id=f_721820761571" \
  | nice -n 10 .venv/bin/python -I scripts/machine_parts_mcb.py - --per-class 120 --procs 2
