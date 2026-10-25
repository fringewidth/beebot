"""The learned rule (an MLP shared by every square) and a hand-written baseline."""

import torch
from torch import nn

from beebot.ca import N_NOTES, OFFSETS, look

# mass, smell, and the notes
N_CHANNELS = 2 + N_NOTES


class NCA(nn.Module):
    """An MLP applied to every square's 3x3 neighborhood.

    An MLP whose first layer reads the 3x3 window is exactly a 3x3 convolution,
    and its later layers act per square, which is exactly a 1x1 convolution.
    So the whole MLP is written as convolutions, one forward call for the whole grid.
    Circular padding makes the window wrap around the torus edges.
    """

    def __init__(self, hidden=96, n_hidden_layers=2, stay_bias=4.0, fold_change=False):
        super().__init__()
        # Fold-change sensing: each square also reads log(neighbor smell / own smell) for its
        # 8 neighbors. Those ratios don't depend on how far away the cheese is, so a
        # far-off square sees the same kind of signal as a nearby one.
        self.fold_change = nn.Conv2d(len(OFFSETS) - 1, hidden, kernel_size=1) if fold_change else None
        layers = [nn.Conv2d(N_CHANNELS, hidden, kernel_size=3, padding=1, padding_mode="circular"), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, len(OFFSETS) + N_NOTES, kernel_size=1)

        # Start close to "do nothing": most mass stays put and notes don't change.
        # stay_bias = 4 keeps about 87% of the mass in place.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[0] = stay_bias

    def forward(self, mass, smell, notes):
        x = torch.cat([mass[:, None], smell[:, None], notes], dim=1)
        h = self.body[0](x)
        if self.fold_change is not None:
            log_smell = torch.log(smell.clamp_min(1e-30))
            ratios = torch.stack([look(log_smell, dy, dx) - log_smell for dy, dx in OFFSETS[1:]], dim=1)
            h = h + self.fold_change(ratios)
        out = self.head(self.body[1:](h))
        return out[:, : len(OFFSETS)], out[:, len(OFFSETS) :]


class GreedyBaseline:
    """Hand-written rule to beat: walk toward stronger smell, avoid squares that are already full.

    It scores each destination by how close it is to the cheese (recovered from
    the smell as -log(smell), which is distance / smell_length) minus a penalty
    for the mass already there.
    """

    def __init__(self, smell_length, closeness_weight=10.0, crowding_weight=0.5):
        self.smell_length = smell_length
        self.closeness_weight = closeness_weight
        self.crowding_weight = crowding_weight

    def __call__(self, mass, smell, notes):
        distance = -torch.log(smell.clamp_min(1e-30)) * self.smell_length
        scores = []
        for dy, dx in OFFSETS:
            if (dy, dx) == (0, 0):
                crowd = torch.relu(mass - 1)
            else:
                crowd = look(mass, dy, dx)
            scores.append(-self.closeness_weight * look(distance, dy, dx) - self.crowding_weight * crowd)
        return torch.stack(scores, dim=1), torch.zeros_like(notes)
