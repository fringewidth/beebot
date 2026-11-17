#!/bin/zsh
# Push a prepared kernel directory once a slot frees on the account: waits until WAIT_SLUG is no longer
# queued or running (Kaggle allows two GPU sessions per account), then runs `kaggle kernels push -p DIR`.
#   zsh v1/scripts/push_when_free.sh OWNER/WAIT_SLUG KERNEL_DIR   # ops:<node>
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
while uvx --with six kaggle kernels status "$1" 2>/dev/null | grep -qE "RUNNING|QUEUED"; do sleep 300; done
uvx --with six kaggle kernels push -p "$2"
