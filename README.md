# beebot

One model that steers a robot swarm from plain English.

A pretrained language model (Qwen3-0.6B with LoRA) reads an instruction as text and the world as a grid
of patch tokens, and gives every robot its next moves. One policy steers about 100 robots: eat a named
cheese, clear the board, fill a box, move as a group, and take a new command mid-run. It was trained by
behaviour cloning on scripted expert demonstrations, about 40 T4 hours in all, and runs the same in a
MuJoCo physics world and on the raw grid. The checkpoint is public on Kaggle
([fringewidth/beebot-2d-policy](https://www.kaggle.com/datasets/fringewidth/beebot-2d-policy)); it is
undertrained, since training stopped when compute ran out with the validation loss still falling. The next steps are a 3D version with modular cubes (in
progress) and real hardware.

| dir | what |
| --- | --- |
| [`v1/`](v1/) | The 2D language-steered policy: world, tasks, experts, the LM policy, training, the demo |
| [`v2/`](v2/) | The 3D voxel world of pivoting cubes and a policy that thinks in text before acting (in progress) |
| [`hardware/`](hardware/) | The cube module: MuJoCo models, design searches, CAD (nothing built yet) |
| [`v0/`](v0/) | Where it started: neural cellular automata with one local rule per square |

`AGENTS.md` summarises the project so far; each part has its own notes and write-ups in `docs/`.

## Running

```
uv sync                                   # everything; --no-group cad --no-group hardware skips the CAD and hardware packages
uv run pytest -q
```

To steer the 2D policy in a browser, download `config.json` and `policy_latest.pt` from the Kaggle dataset
into one folder (the Qwen3-0.6B base model comes from Hugging Face on first run), then:

```
uv run python v1/scripts/swarm_steer.py --init_run <that folder> --device cuda   # or cpu; the default is mps (Mac)
```

and open http://127.0.0.1:8790.

`beebot` is one Python package split over `v0/src`, `v1/src`, `v2/src` and `hardware/src`; run commands
from the repository root.
