"""Can a swarm that sees the cheese catch it? Every robot steps down the distance to the cheese."""
import numpy as np
import torch
import torch.nn.functional as F

from beebot import chase
from beebot.ca import N_NOTES, OFFSETS


class Homing:
    def __init__(self, sharpness):
        self.sharpness = sharpness

    def __call__(self, mass, cheese, walls, notes):
        b, h, w = mass.shape
        # Walking distance to the cheese through open floor, by repeated dilation.
        dist = torch.full((b, h, w), 1e3)
        front = cheese > 0
        dist[front] = 0
        open_ = walls == 0
        for d in range(1, h + w):
            grown = F.max_pool2d(F.pad(front.float()[:, None], (1, 1, 1, 1), mode="circular"), 3, stride=1)[:, 0] > 0
            new = grown & ~front & open_
            if not new.any():
                break
            dist[new] = d
            front = front | new
        logits = []
        for dy, dx in OFFSETS:
            logits.append(-self.sharpness * torch.roll(dist, (-dy, -dx), dims=(1, 2)))
        return torch.stack(logits, dim=1), torch.zeros(b, N_NOTES, h, w)


for size in (22, 30):
    batch = chase.sample_batch(np.random.default_rng(1), size, 16)
    for sharpness in (1.0, 4.0):
        torch.manual_seed(0)
        with torch.no_grad():
            result = chase.rollout(Homing(sharpness), batch, chase.STEPS_PER_CELL * size, chase.FIRE_PROB, chase.TAU)
        done = (result.left == 0).any(dim=0).float().mean().item()
        print(f"size {size} sharpness {sharpness}: caught {done:.0%}, final deficit {result.deficit[-1].mean():.2f}, fog {result.fog_share.mean():.2f}")
