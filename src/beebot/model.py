"""The learned rule (an MLP shared by every square) and a hand-written baseline."""

import torch
from torch import nn

from beebot.ca import N_NOTES, OFFSETS, look

# mass, cheese, and the pheromones
N_CHANNELS = 2 + N_NOTES


class NCA(nn.Module):
    """An MLP applied to every square's 3x3 neighborhood.

    An MLP whose first layer reads the 3x3 window is exactly a 3x3 convolution,
    and its later layers act per square, which is exactly a 1x1 convolution.
    So the whole MLP is written as convolutions, one forward call for the whole grid.
    Circular padding makes the window wrap around the torus edges.
    """

    def __init__(self, hidden=96, n_hidden_layers=2, stay_bias=4.0):
        super().__init__()
        layers = [nn.Conv2d(N_CHANNELS, hidden, kernel_size=3, padding=1, padding_mode="circular"), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, len(OFFSETS) + N_NOTES, kernel_size=1)

        # Start close to "do nothing": most mass stays put and nothing is written.
        # stay_bias = 4 keeps about 87% of the mass in place.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = stay_bias

    def forward(self, mass, cheese, notes):
        x = torch.cat([mass[:, None], cheese[:, None], notes], dim=1)
        out = self.head(self.body(x))
        return out[:, : len(OFFSETS)], out[:, len(OFFSETS) :]


class StickyBaseline:
    """Hand-written rule to beat: wander at random, and stay put once touching the cheese.

    Away from the cheese, mass spreads evenly to every neighbor, so the swarm
    diffuses outward. A square next to the cheese keeps nearly all its mass, so
    mass that drifts onto the ring sticks there until the ring fills.
    """

    def __init__(self, stick=8.0, wander_stay=0.0):
        self.stick = stick
        self.wander_stay = wander_stay

    def __call__(self, mass, cheese, notes):
        touching = torch.stack([look(cheese, dy, dx) for dy, dx in OFFSETS[1:]]).amax(dim=0)
        logits = torch.zeros(mass.shape[0], len(OFFSETS), *mass.shape[1:], device=mass.device)
        logits[:, 0] = self.wander_stay + self.stick * touching
        return logits, torch.zeros_like(notes)
