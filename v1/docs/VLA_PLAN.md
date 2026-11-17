# Language-conditioned swarm policy: plan of record (2026-11-03)

Goal: a policy that takes a natural-language command and a top-down view of a grid world, and
steers a swarm to carry it out, including commands never used in training. It must train on one
laptop (see Hardware).


## Hardware

Apple M4 Pro, 24 GB unified memory, PyTorch 2.14 with the MPS backend. Training starts only once
the GPU is otherwise idle. Budget: one to a few nights in total, across all runs.

## World

- 16x16 bounded grid (no wraparound).
- Swarm: continuous mass on squares, 16 to 28 units in total, starting as one or two blobs.
- Terrain: walls from smoothed random noise thresholded at 0 to 25% density; pockets cut off from
  the main open region become wall. Every cheese and box must be reachable.
- Cheeses: 1 to 3 rigid 2x2 blocks, distinct colors from a palette of 4.
  - Pushable: net flow pressing on a cheese is summed as a force; along the dominant axis, if it
    beats a friction threshold, the cheese moves one square. Mass on the squares it moves into is
    shoved one square further; if there is no room (wall, edge, other cheese), the cheese stays.
    Opposing pushes cancel.
  - Edible: eaten (removed) when at least 6 of its ring squares, and at least 75% of its open ring
    squares, hold at least 0.3 mass.
- Floor boxes: 0 to 2 colored rectangles that mass can enter; visual only.
- Observation per step: 10 channels on the 16x16 grid: mass, wall, cheese mask, cheese RGB (3),
  box mask, box RGB (3).
- Action per step: a 2D vector on every square (2x16x16). A square sends |v| (clipped to 1) of
  its mass along v, split between the two of its 8 neighbor directions that bracket v's angle.
  Flows into walls or off the grid stay put; flows into a cheese stay put and count as push force.
  Mass is conserved exactly.
- Scene sampler: about 50% of scenes have at least 2 cheeses and 2 boxes, so cluttered test
  scenes are in distribution.
- The environment is batched in torch (all episodes step together); no per-episode Python
  loop in data generation or evaluation.

## Data

1. Play, three sources, all using the same control (up to 4 attractor points, optionally in
   straight-line mode; a low-level controller turns them into the velocity field). Stored: state
   and applied field per step, never the player's intent.
   - Main source: LLM-authored play programs. Long-running LLM sessions in tmux (one provider's
     small model, alternating with a second provider's small model) write short controller
     programs: a function from the current state to attractor commands, with whatever internal
     phases the model wants ("go behind the red cheese, then push it toward the box, then
     scatter"). Each program is run by the batched simulator on about 10 random scenes with
     random seeds. The session sees a short text outcome summary per program and writes the next
     one, asked to keep trying things it hasn't tried. One LLM turn yields about 10 episodes, so
     5,000 episodes need about 500 turns, instead of 40k turn-by-turn decisions. Sessions restart
     with fresh context every 20 programs, keeping a running list of what was already tried.
     Each program runs in its own subprocess with a time limit, receiving numpy copies of the
     state; timeouts and crashes count as errors. A timeout or crash ends only that
     episode; the rest of the batch continues. The LLM sessions get no shell access and no file
     access beyond one play directory outside the repository: they can write a program file there
     and invoke one runner command, nothing else.
   - Turn-by-turn LLM play: a small set (about 100 episodes), deciding every 8 steps, fresh
     context per episode. Kept for messier, reactive behavior.
   - Human teleop: 1 to 2 hours, about 40 to 80 episodes. Small; kept as a held-out source for
     checking that the policy isn't only fitting LLM-program style.
   - Scripted random player: ablation only.
   - Pilot first (1 hour): measure programs per hour per session, the rate-limit ceiling, the
     share of programs that run without error, behavior coverage (pushes, eats, splits, box
     entries per 100 episodes, and distinct behavior clusters per 100 programs; clusters are k-means with a fixed k
     and seed over per-window extractor fact vectors, the same method in the pilot and the full
     collection), and 
     caption calls per hour with its rate-limit ceiling. The target volumes (3,000 to 8,000
     episodes, number of captioned windows) are set from the measured rates.
1. Storage: mass float16 per step, action int8-quantized per step, object positions and eaten
   flags per step. All observation frames are pre-rendered once (10 channels x 256 squares x
   fp16 = 5 KB per step; 2.5 GB at 500k steps) and held in unified memory, so batch construction
   is index gathers only.
1. Windows, 8 to 48 steps:
   - program play: from one phase boundary (logged by the runner) to a later one; phases shorter
     than 8 steps merge with the next. Phase changes are used only to cut windows, never as labels;
   - turn-by-turn play: on the 8-step decision points, so multiples of 8;
   - human teleop: random cuts.
1. Fact extraction: a programmatic extractor runs on every window: swarm centroid shift, number
   and positions of mass blobs, mass inside each box, each cheese's displacement and eaten flag,
   terrain contact. Windows where nothing happened are dropped.
1. Captions: a small LLM (alternating between two providers, low to medium effort) receives each
   window's start, middle and end as text grids plus the extracted facts, and writes three
   instructions at varying detail that would produce what happened. 5 windows per call. About
   40k windows from main play plus 10k from scripted play (for its ablation), so about 10k calls
   and 150k captions. The scripted-ablation windows are captioned last, so they can be dropped
   if the captioning rate falls short.
   - The extractor states movement in direction words under one fixed convention (row 0 is
     "top"). The captioner wraps every spatial term it uses in tags (<d>top left</d>) and every
     colour word (<c>red</c>); tags are stripped before embedding. Claims the captioner adds that the facts can't support (lines,
     rings, "between", "half the swarm") are tagged <u>...</u>.
   - Audit gate, two parts, before training:
     - human audit (by the user, about 1.5 hours): windows shown as rendered clips next to their
       facts and captions, stratified so each fact type (centroid move, split, box entry, push,
       eat, and the <u> claims as a separate line) has at least 25 instances. Covers both the
       extractor and the captioner. Pass rule: at most 2 errors per fact type; <u> claims pass at
       80% or better.
     - automatic audit on a 500-window slice: captions against facts, with direction-word accuracy
       and tag recall (spatial and colour words left untagged) as their own lines, and untagged
       unverifiable claims counted as failures.
     If the captioner fails, fix the prompt and re-audit on the 500-window slice. If the extractor
     changes, also repeat a 50-window stratified human check. Only then re-run the full pass.
   - <u> claims are kept in training captions if they pass the human audit (80% or better),
     otherwise stripped.
   - Before training, the test prompt list is mapped against what captions can carry (the
     extractor's facts plus any <u> content that was kept), recording in advance which prompts
     depend on concepts no caption carries.
1. Text embeddings: a frozen pretrained text embedding model (Qwen3-Embedding-0.6B, 1024-d) embeds
   every caption once before training; training never runs it. Captions and test prompts get
   the same instruction prefix.
1. Text pre-flight, before any training: a linear probe on frozen embeddings of templated
   minimal pairs (colour, direction, verb, object), role-swap pairs ("push the red cheese to the
   blue box" vs "push the blue cheese to the red box") and quantity pairs ("half the swarm" vs
   "all of it"), with held-out templates; each pair type has its own 90% threshold. If any type
   fails, switch to the token-level fallback: precomputed token-level hidden states (stored
   randomly projected to 128-d in int8: 600k variants x 30 tokens x 128 bytes, about 2.3 GB),
   reduced to the same single 64-d z by a small attention-pooling head (a learned query over
   the token states). Everything downstream is unchanged: one z token, shared LayerNorm and
   dropout, null token, contrastive loss, source probe. The pre-flight probe is re-run on the
   pooled head's output (trained on the probe data) before committing. If a pair type still fails
   there, it is recorded as a known limit and test prompts that depend on it are tagged.
1. Augmentation:
   - Goal half: the 8 rotations and reflections, and colour permutations.
   - Caption half, failing closed:
     - colour permutation only if every colour word in the caption is tagged and in the palette
       (no untagged colour word from a broad colour lexicon anywhere); tagged words are swapped;
     - rotation or reflection only if no word from a broad direction lexicon appears outside tags,
       and every tag parses under the rewriter's mapping (no handedness terms, no unmappable spans
       like "far side"); only tagged spans are rewritten, so "gather up" is never touched;
     - captions that fail either check are used unaugmented.
     - The share of captions that pass each check is logged. If rotation coverage is low
       (phrasal verbs like "gather up" put direction words outside tags), the captioner is told
       to avoid such phrasal verbs. About 4 variants per caption are embedded offline (about
     600k embeddings, 2 to 4 hours).

## Model (about 4M parameters)

- Goal encoder: a ViT over the window's start and end frames (20 channels), patch size 2 so 64
  tokens, width 128, 3 layers, mean pool, linear to a 64-d goal vector z.
- Text head: MLP 1024 -> 256 -> 64 over the frozen embedding, producing z in the same space.
- Both z sources pass through the same final LayerNorm and the same dropout, so the policy
  can't tell them apart by their statistics.
- Null goal: 10% of samples replace z with a learned null token; at test time, classifier-free
  guidance with weight w (swept over 1 to 3 on a validation prompt set).
- Backbone: the current observation as 64 patch tokens (patch 2) plus one z token; width 192,
  4 layers, 6 heads of 32, MLP 768, full bidirectional attention.
- Action expert: generates a chunk of the next 8 action fields as 128 tokens (64 squares-patches
  x 2 time halves; each token 2x2 squares, 2 components, 4 steps = 32 numbers). Width 128, MLP 512,
  4 layers, separate weights from the backbone. Its q/k/v project to 6 heads of 32 (192) so they
  can join the backbone's; at each layer its queries attend to the concatenation of that
  backbone layer's keys and values and its own; the backbone never attends to it (it runs once
  per decision and its keys and values are cached). The noise level is concatenated with each
  token's action embedding and mixed by an MLP.
- Output head: predicts a 32-number velocity per token.

## Training

- Generative objective (flow matching / rectified flow): a = true action chunk, e ~ N(0, I),
  x_t = t a + (1 - t) e, regress the predicted velocity onto a - e. t = 0.999 (1 - b) with
  b ~ Beta(1.5, 1).
- Loss weighting: every open square is supervised. Weight 1 on the reachable set (for chunk
  step k, the step-0 mass dilated by k + 1 squares) and 0.1 elsewhere, since the field away from
  mass carries the player's intent and steps 2 onward run open loop. Wall squares weight 0.
  The stored field is the controller's output on every square, including empty ones (checked by
  a test).
- Sampling: the current state s is drawn from [w0, w1 - 1]; chunk steps after w1 get loss
  weight 0, so states right before completion are trained on. The goal is the window's end.
- Each batch is 256 samples: 128 conditioned on z from the goal encoder, 128 on z from the text
  head. Goal-half windows are drawn from all play, not only captioned windows.
- Auxiliary: a sigmoid contrastive loss between goal z and text z on the caption half, weight 0.1,
  with negatives masked when two windows share a caption, overlap in time, or come from the same
  episode, or come from the same program. If the source probe (below) finds the sources about 100% separable, the weight is raised
  and Gaussian noise is added to goal z.
- AdamW, lr 3e-4, cosine decay, grad clip 1.0.
- Diagnostics logged during training:
  - flow loss with the true z vs a shuffled z, per half (the gap shows whether z is used at all);
  - a linear source classifier on z (goal vs text);
  - text-to-goal retrieval top-k on held-out windows;
  - closed-loop success on a fixed held-out prompt and scene set every 20 minutes, which (not
    validation loss) selects the checkpoint;
  - target switches per episode.
- Runs and comparisons:
  - Main: 3 hours, 2 seeds.
  - Ablations at 90 minutes each (text encoder trained from scratch; caption half only; scripted
    play only; plain MSE regression head), compared against main's 90-minute checkpoint, not its
    final one. The text-from-scratch ablation also gets 2 seeds.
  - Validation and test splits are by program and by scene (no program and no scene layout in
    more than one split).
  - In the scripted-only ablation, both halves use that run's scripted play only.

## Inference and evaluation

- Per decision: run the backbone once (conditional and null goal for guidance), then 10 Euler
  steps of the action expert from noise; execute the first 4 of the 8 fields, then decide again.
  Episodes of 48 steps, so 12 decisions.
- Test prompts written before any caption is seen (the user's list plus LLM-written ones), on fresh
  scenes. Any prompt that names a specific object is run only on scenes with at least 2 cheeses
  and 2 boxes, so naming matters.
- Every success rate is reported next to a shuffled-prompt baseline and a null-prompt baseline
  on the same scenes and seeds.
- Success is "first reached at any step"; "still held at step 48" is reported separately.
- Success from hand checkers where possible, otherwise an LLM judge. The judge is validated
  against the checkers per prompt category, not overall.
- Novelty: each test prompt carries a hand-labelled novelty tag (paraphrase, new combination,
  new concept) as well as its embedding distance to the nearest training caption.
- Results stratified by wall density.
- Checkpoint choice and the guidance weight are selected on the validation prompts and scenes;
  reported numbers come only from the test prompts and scenes, which are disjoint from them.

## Known limits (accepted)

Only behaviors present in play; no long chains of subgoals; no improvement beyond the play's
quality (no reinforcement learning yet). Captions lean toward what the fact extractor can see,
which limits the labelling, not the model.
