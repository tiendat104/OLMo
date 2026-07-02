# wandb-logs

Dedicated branch for relaying offline wandb run folders from the NPU server
(where `wandb sync` is unreliable due to VPN/proxy instability) to a machine
with stable internet access for syncing to wandb.ai.

Workflow:
1. On the NPU server, copy an offline run folder (e.g.
   `wandb/wandb/offline-run-<timestamp>-<id>`) into a checkout of this
   branch, commit, and push.
2. On a machine with stable internet, pull this branch and run
   `wandb sync <folder>` on the copied run directory.

This branch intentionally has no shared history with `main` /
`npu_device_support` to avoid bloating the code repo with binary log data.
