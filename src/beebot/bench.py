"""Time training iterations on each device, with and without gradient checkpointing.

    uv run bench
"""

import time

import numpy as np
import torch

from beebot.ca import loss, rollout
from beebot.layout import sample_batch
from beebot.model import NCA

SIZE = 32
BATCH_SIZE = 32
N_STEPS = 96
WARMUP = 2
TIMED = 5


def time_iterations(device, use_checkpoint):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = NCA().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    batch = sample_batch(rng, SIZE, (1, 9), (2, 12), BATCH_SIZE).to(device)

    def iteration():
        result = rollout(model, batch, N_STEPS, 0.75, 0.5, use_checkpoint=use_checkpoint)
        value = loss(result, 0.5, 0.5, 1.0, 0.5)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        value.item()  # wait for the device to finish

    for _ in range(WARMUP):
        iteration()
    start = time.time()
    for _ in range(TIMED):
        iteration()
    return (time.time() - start) / TIMED


def main():
    for device in ("mps",):
        for use_checkpoint in (False,):
            seconds = time_iterations(torch.device(device), use_checkpoint)
            print(f"{device:4s} checkpoint={use_checkpoint!s:5s} {seconds:.2f} s/iteration", flush=True)


if __name__ == "__main__":
    main()
