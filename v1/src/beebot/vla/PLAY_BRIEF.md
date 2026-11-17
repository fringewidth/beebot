# Play session brief

You are playing with a simulated robot swarm by writing small Python programs that steer it.
There is no task and no score. The goal is varied, interesting play: try many different things
the swarm could do in this world, including odd ones, and avoid repeating yourself.

## The world

- A square grid of side N = view["size"] (16 now; your programs will later be re-run on bigger
  grids, up to 48, so never hard-code 16: compute positions from N). Row 0 is the top, column 0
  the left.
- The swarm is mass spread over squares (about 16 to 28 units at N = 16, growing with the area).
  A square holds at most 2.
- Gray walls (terrain) block movement.
- Cheeses are square colored blocks (red, green, blue or yellow), view["cheese_side"] squares on a
  side (N / 8), one to three per scene.
  - Mass pressing on one side of a cheese pushes it one square at a time, if enough mass
    presses at once (about 0.75 units per square of the cheese's side) and the square beyond is
    free. Pushes from opposite sides cancel out.
  - A cheese is eaten (disappears) when most of the squares around it are covered with mass. A
    cheese pushed into a corner can no longer be eaten.
- Boxes are colored floor rectangles (zero to two per scene). Mass and cheeses can move over them.

## How you steer

You place up to 4 attractor points every step. Each square of mass walks toward its nearest
attractor along the shortest path. Options per attractor:

- "row", "col": where it is. Placing it on a cheese (in normal mode) makes mass surround that
  cheese.
- "straight": true makes mass go straight through cheeses toward the point instead of around
  them. That is how you push: first gather behind a cheese (normal mode), then switch to straight
  mode with the point on the far side.
- "share": a fraction of the swarm this attractor should get, e.g. four attractors with share 0.25
  split the swarm evenly. Without shares, mass simply goes to whichever attractor is nearest.

## The program

Write a file defining:

```python
def act(view, memory):
    # view["step"]        int, from 0
    # view["size"]        N, the grid's side
    # view["cheese_side"] a cheese's side in squares
    # view["mass"]        NxN numpy array
    # view["walls"]       NxN numpy bool array
    # view["total_mass"]  float
    # view["cheeses"]     list of {"index", "color", "row", "col", "eaten"}; row/col of the top-left square
    # view["boxes"]       list of {"color", "row", "col", "height", "width"}
    # memory              dict you can keep state in across steps of one episode
    memory["phase"] = "short name of what you're doing now"
    return [{"row": 3, "col": 12, "straight": False, "share": None}]
```

- Set memory["phase"] whenever your program switches to a new thing it is doing.
- Only these modules can be imported: math, random, itertools, collections, heapq, functools,
  numpy (already available as np). No files, no network.
- Each call must return quickly (under a second).
- An episode is 64 steps at N = 16 (4 * N in general). Programs should react to the scene: colors,
  positions and counts differ every time, and some scenes have walls in the way. Write timing in
  terms of N too (e.g. "after N steps"), not fixed step counts.

## The loop

1. Write a program to a new file in this directory (program_001.py, program_002.py, ...).
1. Run exactly this command, with your file name in place of `<program file>`, and nothing piped
   or redirected: `PLAY_RUN_COMMAND`
   It takes about a minute; that pacing is deliberate.
1. Read the summary (what happened in each scene and phase), and keep a one-line note per program
   in notes.md of what you tried and what happened.
1. Write the next program. Try something you haven't tried yet. Ideas to range over: gathering
   in places, splitting into groups, lines and shapes, eating some cheeses but not others,
   pushing cheeses to places or into boxes, moving between boxes, avoiding things, sequences of
   several of these, reacting to colors or positions, half the swarm doing one thing and half
   another. Mix them; vary order and targets.

Only write files in this directory and only run the command above, exactly as written: no `cd`,
no `&&`, no pipes. Never stop to ask questions or wait for a reply: nobody is watching live.
Decide for yourself and keep going.
