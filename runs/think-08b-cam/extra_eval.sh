#!/bin/zsh
# Transfer gap on the segment's checkpoint; the copy in scripts/transfer_gap_hook.sh is the source.
exec "${0:A:h}/../../scripts/transfer_gap_hook.sh" "$@"
