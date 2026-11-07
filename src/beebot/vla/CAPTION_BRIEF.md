You are labelling recordings of a simulated robot swarm with the instructions that would have
produced them. Each recording is a short window of time from one episode.

## The world

- A square grid seen from above. Row 0 is the top, column 0 the left.
- The swarm is many small robots, at most one per square. The grids below show every square, one
  character each: `o` is a robot, `#` a wall (terrain), `.` empty floor.
- Cheeses are square coloured blocks, shown as capital letters: R red, G green, B blue, Y yellow.
  The swarm can push a cheese by pressing on one side, and eats a cheese by covering most of the
  squares around it (the cheese then disappears).
- Boxes are coloured floor rectangles, shown as small letters (r, g, b, y) where no robots cover
  them. Robots and cheeses can move over boxes.

## What you get, per window

- The facts: what a program measured happened between the window's first and last frame. They
  are reliable; treat them as the truth.
- Three grids: the first frame, the middle frame and the last frame.

## What to write

For each window, write four instructions that someone could have given the swarm and that the
recording would carry out:

1. short: 2 to 6 words, the way a person types a quick command;
1. medium: one plain sentence;
1. detailed: one or two sentences with the specifics (which cheese, where, how many groups);
1. casual: how a real person types into a chat box without trying, for example "go eat the blue
   one", "ok now everyone to the left pls", "push that red block down", "split up". Lowercase,
   fragments, filler words and slang are welcome; a small typo now and then is fine (but spell
   the tagged words correctly). Keep only the main point. Most real commands have no filler at
   all: don't open them all the same way ("ok ...") or close them all the same way ("... pls").
   Tags still apply, as in every level.

The people giving these commands don't know our vocabulary. "Cheese" is our word; across windows,
also call cheeses blocks, squares, pieces, targets, things, or just "the blue one". Likewise the
swarm can be "the swarm", "everyone", "the robots", "you", or left out. Vary this in every level,
most of all in short and casual.

Rules:

- Write commands to the swarm ("Push the red cheese to the left"), not descriptions ("The swarm
  pushed..."). Vary the wording across windows and across the four levels; don't reuse one
  template.
- Say only what the facts support. Use the grids to choose natural words (for example "the
  cheese near the wall"), but never contradict the facts.
- Never use numbers of rows, columns, squares or steps, coordinates, or percentages. Rough words
  are fine ("most of the swarm", "a small group").
- Tag every spatial word or phrase: places, directions, sides, corners, edges, "center",
  "nearest", "above", "toward the bottom", like <d>top left corner</d>, <d>up</d>,
  <d>to the right</d>. Put the whole phrase inside one tag. Don't use phrasal verbs that hide a
  direction word outside a tag ("gather up", "back off"); say "gather" or "move away" instead.
- Tag every colour word: <c>red</c>.
- If you say something the facts can't confirm (shapes such as lines or rings, "between",
  "half the swarm", order or timing of events, speed, intent like "protect" or "hide"), wrap that
  claim in <u>...</u>. Use such claims only when the grids clearly show them.
- If a window's facts are thin, the short instruction can stay generic ("Move <d>down</d>").

## Output

Reply with one JSON object and nothing else:

{"windows": [{"id": "<window id>", "captions": ["<short>", "<medium>", "<detailed>", "<casual>"]}, ...]}

with one entry per window, in the order given, ids copied exactly.
