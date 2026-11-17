# Related work: generalist and language-conditioned swarm control

Survey from 2026-11-03, done by a research agent with web search. Source tags: [OV] read from
the alphaxiv overview; [ABS] abstract only (no overview existed); [SRCH] from search snippets or
secondary summaries, less reliable; [MEM] recalled from memory, not re-fetched. Absence claims
come from targeted searches, so they are likely but not guaranteed.

## 1. Language, LLM or VLM driven swarm and multi-robot control

The common pattern: the LLM is never the low-level policy. It writes code or a target, and a
classical or RL controller executes it.

| Work | Method | Generality shown | Setting |
| --- | --- | --- | --- |
| Morad, Shankar, Blumenkamp, Prorok, Language-Conditioned Offline RL for Multi-Robot Navigation, 2024 (ICRA 2025), arXiv 2407.20164 [OV] | Frozen text embedding (GTE-Base best) computed once; GNN policy on local obs plus embedding; offline RL on 20 to 90 min of random single-robot data recombined into multi-agent samples, 396 language tasks | Unseen commands, about 7 cm error; navigation goals only | 5 real RoboMasters |
| GenSwarm (Westlake WindyLab and others), 2025, arXiv 2503.23875 [OV] | Multi-LLM pipeline: constraints, skill graph, hierarchical Python; lint, sim test, VLM or human video critique, auto-deploy | 10 tasks zero-shot (aggregation, flocking, shaping, encircling, coverage, pursuit, ...), 81% with o1-mini | Real wheeled robots |
| LLM2Swarm, Strobel, Dorigo, Fritz, 2024, arXiv 2410.11387 [ABS] | LLM synthesises controllers, or runs on each robot for dialogue and anomaly detection | Conceptual demos | ARGoS sim |
| Zhu, Dorigo, Heinrich, online code generation for swarms, 2025, arXiv 2510.04774 [ABS] | Stuck swarm queries an LLM for code at runtime (SoNS hierarchy) | 85% mission success | 6 real, 30+ sim |
| LGPF, Liu et al., IROS 2024 [SRCH] | LLM turns a pattern description into a target; MARL forms it | Patterns | Sim and real |
| LAMARL, 2025, arXiv 2506.01538 [ABS] | LLM writes prior policy and reward for MAPPO | Shape assembly only | Sim and real |
| ZeroCAP, 2024, arXiv 2404.02318 [ABS] | Training-free VLM plus segmentation gives per-robot goals | Surround, cage, infill | |
| CLIPSwarm, 2024, arXiv 2403.13467 [ABS] | Search over drone positions maximising CLIP similarity | Contour shapes | AirSim |
| Gen-Swarms, 2024, arXiv 2408.15899 [ABS] | Text-conditioned flow-matching point-cloud generator; flow trajectories become drone motion | 3D shapes | Sim |
| SwarmGPT, 2024 (RA-L 2025), arXiv 2412.08428 [OV] | GPT-4 waypoints plus a safety filter | Choreography, 200 sim and 20 real drones | |
| FlockGPT, 2024, arXiv 2405.05872 [ABS] | LLM writes a signed distance function; flocking moves onto it | 6 shapes | Sim |
| LLM-Flock, 2025, arXiv 2505.06513 [ABS] | One LLM per robot, consensus merging, collision checker | Formations | Real Crazyflies |
| CommandSwarm, 2026, arXiv 2605.07764 [ABS] | LoRA-tuned 7 to 14B LLM emits XML behavior trees over whitelisted primitives | Parser validity 0 to 72% | |
| Web-of-Drones, 2026, arXiv 2605.03788 [ABS] | LLM agent with MCP tools | Finds LLMs unreliable without grounding tools | ArduPilot sim |
| SwarmBench, 2025, arXiv 2505.04364 [OV] | Each agent a stateless LLM, zero-shot | LLMs struggle with long-range planning | Grid sim |

No published end-to-end VLA controls a swarm; drone VLAs are single-UAV, and multi-robot
vision-language work covers small teams (LEMMA, CoNavBench) [SRCH]. Text-to-crowd work in
graphics (CrowdMoGen, arXiv 2407.06188 [OV]; Gen-C, arXiv 2504.01924 [ABS]) uses an LLM planner
plus a motion generator, not closed-loop control.

## 2. Generalist or multi-task swarm policies without language

- Rubenstein, Cornejo, Nagpal, Science 2014 [MEM]: arbitrary-shape self-assembly with hand-written
  rules on 1,024 Kilobots. The origin of shape formation from a supplied target.
- Sun et al., mean-shift shape assembly, Nature Communications 2023 [SRCH]: hand-designed, any
  target as a grid image, 50 real robots; the usual benchmark for later RL shape assembly.
- Tolstaya et al., CoRL 2019 [MEM]; Li, Gama, Ribeiro, Prorok, arXiv 1912.06095 [ABS/SRCH]: GNN
  policies imitating centralised experts, transfer across team sizes.
- Batra et al., quadrotor swarms with end-to-end RL, CoRL 2021, arXiv 2109.07735 [OV]: attention
  over neighbors, trained on 8, scales to 128 in sim, zero-shot to real.
- MAPF-GPT, AAAI 2025, arXiv 2409.00134 [OV]: transformer (2 to 85M) imitating a MAPF solver on
  1B observation-action pairs, zero-shot on new maps, quality scales with size.
- MARL-GPT, AAMAS 2026, arXiv 2604.05943 [OV]: one GPT-style actor-critic trained offline across
  SMACv2, GRF and POGEMA with a universal tokenizer; no language.
- SwarmDiff, 2025, arXiv 2505.15679 [OV]: conditional diffusion transformer generating
  macroscopic swarm density trajectories, then optimal transport and MPC for individual robots.
  Flow-Opt, arXiv 2510.09204 [OV]: flow matching for joint multi-robot trajectories plus a safety
  filter. Both are planners, not closed-loop policies.
- Mattson and Brown, arXiv 2305.16148 [SRCH]: self-supervised embeddings of swarm behaviors plus
  novelty search; a behavior space, not a conditioned policy.
- Gap: no single learned swarm policy conditioned on a goal or behavior code that switches
  between qualitatively different collective behaviors.

## 3. Field or density-level control

- Model-based roots: Berman et al. T-RO 2009 [MEM]; Bandyopadhyay, Chung, Hadaegh T-RO 2017 [MEM];
  Elamvazhuthi and Berman survey [MEM].
- Zheng, Han, Lin, mean-field feedback, arXiv 2006.11462 [ABS]: density estimate in, velocity
  field out, toward a target density.
- Sinigaglia, Manzoni, Braghin, Berman, arXiv 2205.12592 [ABS]: advection-diffusion PDE with a
  velocity-field control, one target density per solve.
- Cui et al., mean-field control, arXiv 2209.07420 [ABS] and ICLR 2024, arXiv 2307.06175 [OV]:
  RL over distributions.
- Becker lab [SRCH]: steering particle or Kilobot swarms and manipulating objects with one global
  input (light, magnetic field); RL version on real magnetic particles (IROS 2022).
- Falk et al., learning to control active matter, arXiv 2105.04641 [ABS]: RL chooses a
  spatiotemporal activity field for directed transport.
- Gap: no learned policy mapping a rasterised swarm, object and terrain image to a per-cell
  velocity field across many tasks.

## 4. Neural cellular automata and general local rules

- Growing NCA, Distill 2020 [MEM]: one rule per target.
- Goal-Guided NCA, Sudhakaran, Najarro, Risi, arXiv 2205.06806 [OV]: a goal encoding injected into
  hidden channels; one rule grows and morphs between several targets, interpolates, controls
  pattern locomotion, and transfers from 64x64 to 192x192 grids. Pixels only.
- Graph NCA, NeurIPS 2021, arXiv 2110.14237: GNN as the rule, including imitation of boids.
- Physical NCA bricks (Walker et al. arXiv 2203.07548 [ABS]; Nature Communications 2026 [SRCH]):
  stationary bricks classifying shape.
- Gap: multi-goal NCAs have not been shown on mobile agents; text-conditioned NCAs exist only for
  images and textures.

## 5. Pretraining, datasets, benchmarks

- DeepFleet, Amazon Robotics, arXiv 2508.08574 [OV]: foundation models on fleet logs from hundreds
  of thousands of warehouse robots; an image-floor (rasterised fleet) variant underperformed
  robot-centric and graph variants. Prediction, not commanded control.
- VMAS, arXiv 2207.03530 [OV]: vectorised 2D multi-robot simulator with transport and pushing
  scenarios; POGEMA; Melting Pot [MEM].
- Gap: no large swarm pretraining dataset and no language-annotated swarm-behavior dataset.

## Closest to our project

1. Morad et al. 2024: same conditioning mechanism (frozen text embedding into one learned
   multi-robot policy, unseen commands), narrow tasks.
1. GenSwarm 2025: same breadth of behaviors from language, but code per task, not one policy.
1. Goal-Guided NCA 2022: one rule plus a goal vector for many targets, on pixels.
1. MAPF-GPT and MARL-GPT: one imitation-trained model across many instances, no language.

Gaps the survey could not find filled, each of which this project touches: an end-to-end
language-conditioned swarm policy over qualitatively different behaviors; a per-cell velocity
field policy on a rasterised world; hindsight-relabelled play in swarm settings; a flow-matching
action head for closed-loop swarm control; a language-annotated swarm dataset; multi-task swarm
object pushing.
