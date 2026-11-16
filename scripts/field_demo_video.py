"""Video of the field demo from scripts/field_demo_record.py's recordings: a title card, then per
instruction a short card and a side-by-side segment (execution in MuJoCo on the left; policy input
and action field on the right), then an end card. 1920x1080, 30 fps, H.264, through ffmpeg.

With the trained policy's recordings (field_demo_record.py --policy lm) an instruction is shown only
if it succeeds in at least --min_share of its rollouts and the run shown succeeds (tasks.score, as
steer.validate counts it). With the dummy policy's recordings the frames carry a mock-up note.

Recordings made with MuJoCo physics (field_demo_record.py --physics mujoco) carry every robot's and
cheese's continuous position at several frames per policy step ("path"); the execution view shows
those instead of moving robots in straight lines between squares. A steering instruction (one
whose commands are switched mid-motion) gets its own layout: the instruction typed in large at the
top, the execution view larger and the action field beside it.

The look is deliberately plain, like a lab's supplementary video: MuJoCo's default checker floor
and gradient sky, matplotlib default figures, white Open Sans overlay text on black.

    uv run python scripts/field_demo_video.py --recordings runs/field-demo/recordings.json --out runs/field-demo
"""

import argparse
import base64
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

RECORDINGS = Path("runs/field-demo/recordings.json")
FONT = Path("assets/fonts/OpenSans.ttf")
WIDTH, HEIGHT = 1920, 1080
FPS = 30
SHRINK_SECONDS = 0.5
TITLE_SECONDS = 5.0
CARD_SECONDS = 1.5
EXTRA_CARD_SECONDS = 3.0
HOLD_SECONDS = 0.7
END_SECONDS = 6.0
SWITCH_CAPTION_STEPS = 12  # how long "new instruction" stays up
VIEW = (40, 140, 1140, 870)  # 3D panel: left, top, width, height
POLICY = (1220, 140)  # policy-input figure: left, top (700 x 320)
TOKENS_TOP = 466
FIELD = (1220, 604)  # action-field figure: left, top (700 x 460)
STEER_VIEW = (40, 170, 1160, 870)  # steering layout: 3D panel
STEER_FIELD = (1220, 230)  # steering layout: action-field figure (700 x 480)
PAIR_COLUMNS = (40, 980)  # paired layout: left MuJoCo, right the raw grid
PAIR_VIEW = (900, 860)  # each world view: width, height (non-steering)
PAIR_TOP, PAIR_TOP_STEER = 150, 190  # top of the world views (steering leaves room for the live instruction)
TYPE_SECONDS = 0.5
HIGHLIGHT_SECONDS = 1.5  # a panel's command label stays bright this long after it changes  # a new command is typed in over this long, ending as it takes over
MOCK_NOTE = "MOCK-UP: driven by a hand-coded dummy policy, not the trained model"
CHANNEL_TITLES = ("0 robots", "1 walls", "2 cheese", "3 cheese colour R", "4 cheese colour G", "5 cheese colour B", "6 box", "7 box colour R", "8 box colour G", "9 box colour B")
PATCH = 4
POLICY_LINES = (
    "Policy: Qwen3-0.6B with a rank-64 LoRA, trained by supervised learning on expert demonstrations.",
    "Centralized: it reads the whole 32 x 32 grid and the instruction, and outputs",
    "a distribution over 9 moves for every square; one move per robot is sampled from it.",
)


def font(size, weight=b"Regular"):
    face = ImageFont.truetype(str(FONT), size)
    face.set_variation_by_name(weight)
    return face


def grid(blob, dtype, shape):
    return np.frombuffer(base64.b64decode(blob), dtype).reshape(shape)


def robots_of(step):
    return np.array(step["robots"]).reshape(-1, 3)


def square_xy(row, col, size):
    """World (x, y) of a square's centre: columns run along +x, row 0 is the far side (+y)."""
    return col + 0.5, size - row - 0.5


# ---------------------------------------------------------------------------------------------
# MuJoCo scene


def scene_xml(episode, size, side):
    walls = grid(episode["walls"], np.uint8, (size, size))
    geoms = []
    for r, c in zip(*np.nonzero(walls)):
        x, y = square_xy(r, c, size)
        geoms.append(f'<geom type="box" pos="{x} {y} 0.5" size="0.5 0.5 0.5" rgba="0.55 0.55 0.55 1"/>')
    for box in episode["boxes"]:
        r, c, h, w = box["rect"]
        rgba = " ".join(f"{v:.2f}" for v in box["rgb"]) + " 1"
        top, bottom, left, right = size - r, size - r - h, c, c + w
        for x0, y0, x1, y1 in ((left, top, right, top), (left, bottom, right, bottom), (left, bottom, left, top), (right, bottom, right, top)):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hx, hy = max(abs(x1 - x0) / 2, 0.08), max(abs(y1 - y0) / 2, 0.08)
            geoms.append(f'<geom type="box" pos="{cx} {cy} 0.01" size="{hx} {hy} 0.01" rgba="{rgba}"/>')
    border = []
    for x, y, hx, hy in ((size / 2, -0.15, size / 2 + 0.3, 0.15), (size / 2, size + 0.15, size / 2 + 0.3, 0.15), (-0.15, size / 2, 0.15, size / 2), (size + 0.15, size / 2, 0.15, size / 2)):
        border.append(f'<geom type="box" pos="{x} {y} 0.2" size="{hx} {hy} 0.2" rgba="0.3 0.3 0.3 1"/>')
    cheeses = []
    for k, cheese in enumerate(episode["cheeses"]):
        rgba = " ".join(f"{v:.2f}" for v in cheese["rgb"]) + " 1"
        half = side / 2
        cheeses.append(f'<body name="cheese{k}" mocap="true" pos="0 0 {half}"><geom name="cheese{k}" type="box" size="{half} {half} {half}" rgba="{rgba}"/></body>')
    robots = []
    for k in range(episode["robot_count"]):
        legs = []
        for angle in (90, 210, 330):
            lx, ly = 0.26 * math.cos(math.radians(angle)), 0.26 * math.sin(math.radians(angle))
            legs.append(f'<geom type="capsule" fromto="{lx:.3f} {ly:.3f} 0.02 {lx:.3f} {ly:.3f} 0.24" size="0.035" rgba="0.1 0.1 0.1 1"/>')
        robots.append(
            f'<body name="robot{k}" mocap="true" pos="0 0 0">'
            '<geom type="cylinder" pos="0 0 0.3" size="0.4 0.06" rgba="0.85 0.85 0.82 1"/>'
            '<geom type="cylinder" pos="0 0 0.4" size="0.09 0.05" rgba="0.15 0.15 0.15 1"/>'
            + "".join(legs)
            + "</body>"
        )
    return f"""
<mujoco>
  <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="checker" type="2d" builtin="checker" rgb1="0.82 0.82 0.82" rgb2="0.68 0.68 0.68" width="512" height="512"/>
    <material name="floor" texture="checker" texrepeat="16 16"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="{size / 2} {size * 0.2} 40" dir="0 0.3 -1" directional="true" castshadow="true"/>
    <geom type="plane" pos="{size / 2} {size / 2} 0" size="{size * 0.75} {size * 0.75} 0.1" material="floor"/>
    {"".join(border)}
    {"".join(geoms)}
    {"".join(cheeses)}
    {"".join(robots)}
  </worldbody>
</mujoco>"""


class Arena:
    """The episode's MuJoCo scene; pose(t) places robots and cheeses at fractional step t."""

    def __init__(self, episode, size, side, frames_per_step, view=VIEW):
        self.episode, self.size, self.side = episode, size, side
        self.seconds_per_step = frames_per_step / FPS
        self.model = mujoco.MjModel.from_xml_string(scene_xml(episode, size, side))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, view[3], view[2])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = [size / 2, size / 2 + 0.5, 0]
        self.camera.distance = 42 * max(1.0, (VIEW[2] / VIEW[3]) / (view[2] / view[3]))  # back off for narrower views so the arena still fits
        self.camera.azimuth = 90
        self.camera.elevation = -50
        self.robot_ids = [self.model.body(f"robot{k}").mocapid[0] for k in range(episode["robot_count"])]
        self.cheese_ids = [self.model.body(f"cheese{k}").mocapid[0] for k in range(len(episode["cheeses"]))]
        self.cheese_geoms = [self.model.geom(f"cheese{k}").id for k in range(len(episode["cheeses"]))]
        self.eaten_step = {}
        for step in episode["steps"]:
            for k in step["eaten"]:
                self.eaten_step[k] = step["t"]

    def positions(self, t):
        """(robots (N, 2), cheeses (K, 2)) as continuous (row, col) at fractional step t: the
        recorded physical path when there is one, else straight lines between squares."""
        steps = self.episode["steps"]
        i = min(int(t), len(steps) - 1)
        f = t - i
        n = self.episode["robot_count"]
        if "path" in steps[i]:
            path = np.frombuffer(base64.b64decode(steps[i]["path"]), np.float16).reshape(-1, n + len(self.cheese_ids), 2).astype(float)
            frame = path[min(round(f * len(path)), len(path) - 1)] if i < len(steps) - 1 else path[0]
            return frame[:n], frame[n:]
        j = min(i + 1, len(steps) - 1)
        a, b = robots_of(steps[i])[:, :2], robots_of(steps[j])[:, :2]
        ca = np.array([c[:2] for c in steps[i]["cheeses"]], float).reshape(-1, 2)
        cb = np.array([c[:2] for c in steps[j]["cheeses"]], float).reshape(-1, 2)
        return a * (1 - f) + b * f, ca * (1 - f) + cb * f

    def pose(self, t):
        robots, cheeses = self.positions(t)
        for k, mocap in enumerate(self.robot_ids):
            x, y = square_xy(robots[k, 0], robots[k, 1], self.size)
            self.data.mocap_pos[mocap] = [x, y, 0.0]
        half = self.side / 2
        for k, mocap in enumerate(self.cheese_ids):
            row, col = cheeses[k]
            scale = 1.0
            if k in self.eaten_step:  # shrinks to nothing over SHRINK_SECONDS from the step it is eaten on
                elapsed = (t - (self.eaten_step[k] - 1)) * self.seconds_per_step
                scale = float(np.clip(1 - elapsed / SHRINK_SECONDS, 0.0, 1.0))
            x, y = square_xy(row + half - 0.5, col + half - 0.5, self.size)
            self.data.mocap_pos[mocap] = [x, y, half * scale - (0 if scale > 0 else 5)]
            self.model.geom_size[self.cheese_geoms[k]] = [max(half * scale, 1e-3)] * 3
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()



class CellView:
    """The raw grid world as the policy steps it, drawn flat: one cell per square, no physics and
    no interpolation (each step every robot jumps one square or stays), like a cellular automaton."""

    def __init__(self, episode, size, view=VIEW):
        self.episode, self.size = episode, size
        self.cell = min(view[2], view[3]) // size
        self.width, self.height = view[2], view[3]
        self.walls = grid(episode["walls"], np.uint8, (size, size))

    def pose(self, t):
        step = self.episode["steps"][min(int(t), len(self.episode["steps"]) - 1)]
        size, cell = self.size, self.cell
        colours = np.full((size, size, 3), 236, np.uint8)
        colours[self.walls > 0] = 90
        for (r, c, alive), cheese in zip(step["cheeses"], self.episode["cheeses"]):
            if alive:
                colours[r : r + 4, c : c + 4] = (np.array(cheese["rgb"]) * 255).astype(np.uint8)
        robots = robots_of(step)
        colours[robots[:, 0], robots[:, 1]] = 20
        image = Image.fromarray(colours).resize((size * cell, size * cell), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        for k in range(size + 1):  # cell borders
            draw.line((k * cell, 0, k * cell, size * cell), fill=(200, 200, 200), width=1)
            draw.line((0, k * cell, size * cell, k * cell), fill=(200, 200, 200), width=1)
        for box in self.episode["boxes"]:
            r, c, h, w = box["rect"]
            draw.rectangle((c * cell - 2, r * cell - 2, (c + w) * cell + 2, (r + h) * cell + 2), outline=(40, 40, 40), width=8)  # dark rim: a white box shows on the light floor
            draw.rectangle((c * cell, r * cell, (c + w) * cell, (r + h) * cell), outline=tuple(int(255 * v) for v in box["rgb"]), width=4)
        frame = Image.new("RGB", (self.width, self.height), "black")
        frame.paste(image, ((self.width - size * cell) // 2, (self.height - size * cell) // 2))
        return np.asarray(frame)


# ---------------------------------------------------------------------------------------------
# Matplotlib figures


def figure_image(fig):
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


class PolicyFigure:
    """The 10 observation channels at one step, matplotlib defaults (viridis)."""

    def __init__(self, episode, size):
        self.episode, self.size = episode, size
        self.fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.2), dpi=100)
        self.images = []
        for c, ax in enumerate(axes.flat):
            self.images.append(ax.imshow(np.zeros((size, size)), vmin=0, vmax=1, interpolation="nearest"))
            ax.set_title(CHANNEL_TITLES[c], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for k in range(1, size // PATCH):  # the 4 x 4 patches the policy reads as tokens
            axes.flat[0].axhline(k * PATCH - 0.5, color="white", linewidth=0.4, alpha=0.5)
            axes.flat[0].axvline(k * PATCH - 0.5, color="white", linewidth=0.4, alpha=0.5)
        self.fig.suptitle("observation tensor 10 x 32 x 32 (values in [0, 1])", fontsize=9)
        self.fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86, wspace=0.12, hspace=0.22)

    def draw(self, step):
        for c, image in enumerate(self.images):
            image.set_data(grid(self.episode["obs_blobs"][step["obs"][c]], np.uint8, (self.size, self.size)) / 255)
        return figure_image(self.fig)


class FieldFigure:
    """Expected move per square (arrows, length = how peaked the distribution is) and each robot's
    sampled move, over faint occupancy, walls, cheeses and boxes."""

    def __init__(self, episode, size, code_vectors, dummy, figsize=(7.0, 4.6)):
        self.episode, self.size, self.dummy = episode, size, dummy
        self.code_vectors = np.array(code_vectors)
        self.text = None
        self.fig = plt.figure(figsize=figsize, dpi=100)
        side = min(0.9, 0.6 * figsize[0] / figsize[1])  # a square plot, as tall as fits with its tick labels
        self.width = side * figsize[1] / figsize[0]
        self.ax = self.fig.add_axes((0.035, 0.97 - side, self.width, side))
        self.walls = grid(episode["walls"], np.uint8, (size, size))
        cols, rows = np.meshgrid(np.arange(size), np.arange(size))
        self.cols, self.rows = cols, rows

    def draw(self, step):
        ax = self.ax
        ax.clear()
        size = self.size
        backdrop = np.ones((size, size, 3))
        robots = robots_of(step)
        backdrop[robots[:, 0], robots[:, 1]] = 0.86
        backdrop[self.walls > 0] = 0.45
        ax.imshow(backdrop, interpolation="nearest", extent=(-0.5, size - 0.5, size - 0.5, -0.5))
        for box in self.episode["boxes"]:
            r, c, h, w = box["rect"]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), w, h, fill=False, edgecolor=box["rgb"], linewidth=1.5))
        for (r, c, alive), cheese in zip(step["cheeses"], self.episode["cheeses"]):
            if alive:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 4, 4, facecolor=cheese["rgb"], alpha=0.6, edgecolor="none"))
        field = grid(step["field"], np.int8, (2, size, size)) / 127
        open_squares = self.walls == 0
        ax.quiver(self.cols[open_squares], self.rows[open_squares], field[0][open_squares], field[1][open_squares], angles="xy", scale_units="xy", scale=1 / 0.9, width=0.003, color="C0", label="the policy's expected\nmove per square")
        moving = robots[robots[:, 2] != 8]
        vectors = self.code_vectors[moving[:, 2]]
        ax.quiver(moving[:, 1], moving[:, 0], vectors[:, 0], vectors[:, 1], angles="xy", scale_units="xy", scale=1 / 0.6, width=0.006, color="C3", zorder=3, label="sampled move (each robot)")
        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(size - 0.5, -0.5)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=8, frameon=True)
        mean_peak = grid(step["pmax"], np.uint8, (size, size))[robots[:, 0], robots[:, 1]].mean() / 255
        if self.text is not None:
            self.text.remove()
        source = "dummy policy\n\n" if self.dummy else ""
        top = 1 - 0.26 * 4.6 / self.fig.get_figheight()
        self.text = self.fig.text(0.065 + self.width, top, f"{source}arrow length 1:\nalways the same move\n\nmean max p on robots:\n{mean_peak:.2f}", fontsize=8, va="top")
        return figure_image(self.fig)


# ---------------------------------------------------------------------------------------------
# Frames


class Video:
    def __init__(self, path):
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(path)],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def add(self, image, count=1):
        data = image.convert("RGB").tobytes()
        for _ in range(count):
            self.process.stdin.write(data)
        self.frames += count

    def close(self):
        self.process.stdin.close()
        self.process.wait()


def card(lines, note=""):
    """A plain black card; lines are (text, size, weight, gap above)."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)
    faces = [font(size, weight) for _, size, weight, _ in lines]
    heights = [face.getbbox("Ag")[3] + gap for (_, _, _, gap), face in zip(lines, faces)]
    y = (HEIGHT - sum(heights)) / 2
    for (text, _, _, gap), face in zip(lines, faces):
        y += gap
        width = draw.textlength(text, font=face)
        draw.text(((WIDTH - width) / 2, y), text, fill="white", font=face)
        y += face.getbbox("Ag")[3]
    if note:
        draw.text((30, HEIGHT - 50), note, fill=(200, 200, 200), font=font(22))
    return image


def commands_of(episode):
    if "commands" in episode:
        return episode["commands"]
    return [{"text": episode["instruction"], "token_ids": episode["token_ids"], "tokens": episode["tokens"]}]


def command_at(episode, i):
    return episode["steps"][i].get("command", 0)


def caption_for(episode, t):
    """A short event caption for step t, or ''."""
    steps = episode["steps"]
    i = int(t)
    for k in range(max(i - SWITCH_CAPTION_STEPS, 1), i + 1):
        if command_at(episode, k) != command_at(episode, k - 1):
            return f"new instruction given at step {k}"
    for k in range(max(i - 6, 0), i + 1):
        if steps[k]["eaten"]:
            return f"{episode['cheeses'][steps[k]['eaten'][0]]['colour']} cheese eaten: enough of its ring is covered"
    for k in range(max(i - 3, 0), i + 1):
        if steps[k]["pushed"]:
            return "cheese pushed: the robots pressing on it beat the friction threshold"
    return ""


def fitted(draw, text, size, weight, width):
    """The largest font up to size at which text fits in width."""
    while size > 20 and draw.textlength(text, font=font(size, weight)) > width:
        size -= 2
    return font(size, weight)


def compose(episode, number, total, t, view, policy, field, note, physics="grid"):
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)
    steps = len(episode["steps"]) - 1
    commands = commands_of(episode)
    j = command_at(episode, min(int(t), steps))
    step_text = f"Task {number}/{total}    Step {min(int(t), steps)} / {steps}"
    face = font(36)
    step_width = draw.textlength(step_text, font=face)
    draw.text((WIDTH - 40 - step_width, 32), step_text, fill="white", font=face)
    prefix = f"Instruction {j + 1}/{len(commands)}" if len(commands) > 1 else "Instruction"
    header = f"{prefix}: “{commands[j]['text']}”"
    draw.text((40, 26), header, fill="white", font=fitted(draw, header, 44, b"SemiBold", WIDTH - 140 - step_width))

    label = font(28, b"SemiBold")
    draw.text((VIEW[0], 98), "Execution (MuJoCo physics)" if physics == "mujoco" else "Execution (physics simulation, MuJoCo render)", fill="white", font=label)
    image.paste(Image.fromarray(view), (VIEW[0], VIEW[1]))
    draw.rectangle((VIEW[0] - 1, VIEW[1] - 1, VIEW[0] + VIEW[2], VIEW[1] + VIEW[3]), outline="white")
    caption = caption_for(episode, t)
    if caption:
        face = font(30)
        width = draw.textlength(caption, font=face)
        x, y = VIEW[0] + (VIEW[2] - width) / 2, VIEW[1] + VIEW[3] - 60
        draw.rectangle((x - 12, y - 4, x + width + 12, y + 44), fill="black")
        draw.text((x, y), caption, fill="white", font=face)

    draw.text((POLICY[0], 98), "Policy input", fill="white", font=label)
    image.paste(policy, POLICY)
    pieces = " | ".join(p.strip() or "_" for p in commands[j]["tokens"])
    ids = " ".join(str(i) for i in commands[j]["token_ids"])
    small = font(20)
    draw.text((POLICY[0], TOKENS_TOP), f"instruction tokens (Qwen3 tokenizer): {pieces}", fill="white", font=fitted(draw, f"instruction tokens (Qwen3 tokenizer): {pieces}", 20, b"Regular", 690))
    draw.text((POLICY[0], TOKENS_TOP + 27), f"token ids: [{ids}]", fill="white", font=small)
    draw.text((POLICY[0], TOKENS_TOP + 54), f"grid: 64 patches of 4 x 4 squares -> 64 tokens after the instruction tokens", fill="white", font=small)
    draw.text((FIELD[0], FIELD[1] - 42), "Action field (policy output per square)", fill="white", font=label)
    image.paste(field, FIELD)
    if note:
        draw.text((VIEW[0], HEIGHT - 52), note, fill=(200, 200, 200), font=font(22))
    return image


def switches(episode):
    """[(step, command index)]: the step each command takes over at, the first at 0."""
    out = [(0, command_at(episode, 0))]
    for i in range(1, len(episode["steps"])):
        if command_at(episode, i) != command_at(episode, i - 1):
            out.append((i, command_at(episode, i)))
    return out


def typed(episode, t, seconds_per_step):
    """(text, typing): the instruction overlay at fractional step t. A new command is typed in
    letter by letter over TYPE_SECONDS, ending at the step it takes over; the previous one is
    cleared when the typing starts."""
    commands = commands_of(episode)
    lead = TYPE_SECONDS / seconds_per_step  # in steps
    text = commands[command_at(episode, 0)]["text"]
    for at, j in switches(episode)[1:]:
        if t >= at:
            text = commands[j]["text"]
        elif t >= at - lead:
            new = commands[j]["text"]
            return new[: int(len(new) * (t - (at - lead)) / lead)], True
    return text, False


def compose_steering(episode, number, total, t, view, field, note, seconds_per_step):
    """The steering layout: the live instruction large at the top, the execution view larger,
    the action field beside it and the commands given so far below it."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)
    steps = len(episode["steps"]) - 1
    step_text = f"Task {number}/{total}    Step {min(int(t), steps)} / {steps}"
    face = font(36)
    step_width = draw.textlength(step_text, font=face)
    draw.text((WIDTH - 40 - step_width, 32), step_text, fill="white", font=face)
    draw.text((40, 18), "Instruction (changed live):", fill=(200, 200, 200), font=font(28))
    text, typing = typed(episode, t, seconds_per_step)
    big = font(64, b"SemiBold")
    shown = f"\u201c{text}" + ("" if typing else "\u201d")
    draw.text((40, 58), shown, fill="white", font=big)
    if typing:  # a text cursor after the letters typed so far
        x = 40 + draw.textlength(shown, font=big) + 6
        draw.rectangle((x, 72, x + 5, 138), fill="white")

    image.paste(Image.fromarray(view), (STEER_VIEW[0], STEER_VIEW[1]))
    draw.rectangle((STEER_VIEW[0] - 1, STEER_VIEW[1] - 1, STEER_VIEW[0] + STEER_VIEW[2], STEER_VIEW[1] + STEER_VIEW[3]), outline="white")
    caption = caption_for(episode, t)
    if caption and not caption.startswith("new instruction"):
        face = font(30)
        width = draw.textlength(caption, font=face)
        x, y = STEER_VIEW[0] + (STEER_VIEW[2] - width) / 2, STEER_VIEW[1] + STEER_VIEW[3] - 60
        draw.rectangle((x - 12, y - 4, x + width + 12, y + 44), fill="black")
        draw.text((x, y), caption, fill="white", font=face)

    label = font(28, b"SemiBold")
    draw.text((STEER_FIELD[0], STEER_FIELD[1] - 42), "Action field (policy output per square)", fill="white", font=label)
    image.paste(field, STEER_FIELD)
    y = STEER_FIELD[1] + field.height + 30
    draw.text((STEER_FIELD[0], y), "Instructions given (MuJoCo physics, no cuts):", fill="white", font=label)
    commands = commands_of(episode)
    given = [(at, j) for at, j in switches(episode) if at <= t]
    for at, j in given:
        y += 40
        current = (at, j) == given[-1]
        draw.text((STEER_FIELD[0], y), f"step {at}: \u201c{commands[j]['text']}\u201d", fill="white" if current else (140, 140, 140), font=font(26, b"SemiBold" if current else b"Regular"))
    if note:
        draw.text((STEER_VIEW[0], HEIGHT - 36), note, fill=(200, 200, 200), font=font(22))
    return image


def compose_pair(worlds, number, total, t, note, steering, seconds_per_step):
    """The paired layout: the same scene run in MuJoCo (left) and on the raw grid (right).
    worlds: [(episode, view array, title)]."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(image)
    lead = worlds[0][0]
    task_text = f"Task {number}/{total}"
    face = font(36)
    task_width = draw.textlength(task_text, font=face)
    draw.text((WIDTH - 40 - task_width, 32), task_text, fill="white", font=face)
    if steering:
        draw.text((40, 14), "Instruction (changed live):", fill=(200, 200, 200), font=font(28))
        text, typing = typed(lead, min(t, len(lead["steps"]) - 1), seconds_per_step)
        big = font(56, b"SemiBold")
        shown = f"\u201c{text}" + ("" if typing else "\u201d")
        draw.text((40, 50), shown, fill="white", font=big)
        if typing:
            x = 40 + draw.textlength(shown, font=big) + 6
            draw.rectangle((x, 64, x + 5, 122), fill="white")
    else:
        header = f"Instruction: \u201c{commands_of(lead)[0]['text']}\u201d"
        draw.text((40, 26), header, fill="white", font=fitted(draw, header, 44, b"SemiBold", WIDTH - 140 - task_width))
    top = PAIR_TOP_STEER if steering else PAIR_TOP
    for left, (episode, view, title) in zip(PAIR_COLUMNS, worlds):
        label = font(30, b"SemiBold")
        draw.text((left, top - 44), title, fill="white", font=label)
        if steering:  # each world switches when its own swarm has acted on the command
            i = min(int(t), len(episode["steps"]) - 1)
            status = f"\u201c{commands_of(episode)[command_at(episode, i)]['text']}\u201d"
            since = max(at for at, _ in switches(episode) if at <= i)
            fresh = since > 0 and (t - since) * seconds_per_step < HIGHLIGHT_SECONDS  # this world just took the new command
            status_face = fitted(draw, status, 26, b"SemiBold" if fresh else b"Regular", PAIR_VIEW[0] - draw.textlength(title, font=label) - 24)
            draw.text((left + PAIR_VIEW[0] - draw.textlength(status, font=status_face), top - 40), status, fill="white" if fresh else (200, 200, 200), font=status_face)
        image.paste(Image.fromarray(view), (left, top))
        draw.rectangle((left - 1, top - 1, left + PAIR_VIEW[0], top + view.shape[0]), outline="white")
    if note:
        draw.text((40, HEIGHT - 36), note, fill=(200, 200, 200), font=font(22))
    return image


def render_pair(video, stills, episode, partner, number, total, data, frames_per_step, note, steering):
    """One task in the paired layout; both worlds advance one policy step per frames_per_step frames,
    and the one that finishes first holds its last state."""
    size, side = data["size"], data["cheese_side"]
    top = PAIR_TOP_STEER if steering else PAIR_TOP
    view = (0, 0, PAIR_VIEW[0], PAIR_VIEW[1] - (top - PAIR_TOP))
    worlds = [(episode, Arena(episode, size, side, frames_per_step, view), "Physics simulation (MuJoCo)"), (partner, CellView(partner, size, view), "Cellular automaton")]
    steps = max(len(e["steps"]) for e, _, _ in worlds) - 1
    wanted = {"mid": float(steps // 2), "final": float(steps)}
    for i in range(steps + 1):
        sub = frames_per_step if i < steps else int(HOLD_SECONDS * FPS)
        for k in range(sub):
            t = i + (k / frames_per_step if i < steps else 0.0)
            views = [(e, arena.pose(min(t, len(e["steps"]) - 1)), title) for e, arena, title in worlds]
            frame = compose_pair(views, number, total, t, note, steering, frames_per_step / FPS)
            video.add(frame)
            for name, at in wanted.items():
                if k == 0 and int(at) == i:
                    stills[f"{number:02d}-{name}"] = frame


def paired(shown, grid_data):
    """Per shown (episode, entry): (grid episode, grid entry) for the same scene (seed) run on the raw
    grid, both runs successful (the chosen MuJoCo run first, then its other runs), or (None, grid
    entry) when no seed succeeds in both. The MuJoCo episode may change to the paired seed's run."""
    grids = {e["instruction"]: e for e in grid_data["instructions"]}
    out = []
    for episode, entry in shown:
        grid_entry = grids.get(episode["instruction"])
        found = None
        if grid_entry is not None:
            by_seed = {r["seed"]: r for r in grid_entry["runs"] if r["success"]}
            for run in [episode] + [r for r in entry["runs"] if r is not episode]:
                if run["success"] and run.get("seed") in by_seed:
                    found = (run, by_seed[run["seed"]])
                    break
        if found is None:
            print("no scene succeeds in both worlds:", episode["instruction"])
            out.append((episode, entry, None, grid_entry))
        else:
            out.append((found[0], entry, found[1], grid_entry))
    return out


def success_rates(path):
    """{instruction: (successes, rollouts)} from a field_demo_record log's success_rate lines."""
    rates = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "success_rate" in row:
            k, n = row["success_rate"].split("/")
            rates[row["instruction"]] = (int(k), int(n))
    return rates


def shown_and_dropped(data, choices, rates, min_share=0.5):
    """(shown, dropped): per instruction (episode, its entry). The trained policy's instructions are
    shown only if at least min_share of their rollouts succeed (and at least one; for a steering
    instruction, complete its last command) and the chosen run succeeds. rates, when given, replaces the recordings' own success counts (a larger earlier
    measurement)."""
    if not data.get("instructions"):
        return [(episode, None) for episode in data["episodes"]], []
    shown, dropped = [], []
    for k, entry in enumerate(data["instructions"]):
        if entry["instruction"] in rates:
            entry = {**entry, "successes": rates[entry["instruction"]][0], "rollouts": rates[entry["instruction"]][1]}
        run = entry["runs"][choices.get(k, 0)]
        wins = entry["final_successes"] if entry.get("steering") else entry["successes"]  # steering: the last command completed
        if wins >= max(min_share * entry["rollouts"], 1) and run["success"]:  # the run shown: every command met (steering: took effect)
            shown.append((run, entry))
        else:
            dropped.append((run, entry))
    return shown, dropped


def trimmed(episode, after):
    """The episode cut off after steps past the moment its last instruction was met."""
    segments = episode.get("segments")
    if not segments or segments[-1]["met_at"] is None:
        return episode
    end = min(segments[-1]["met_at"] + after, len(episode["steps"]) - 1)
    return {**episode, "steps": episode["steps"][: end + 1]}


def still_steps(episode):
    """Steps to save as stills, by name: the middle of the run, the end, and just after each new
    instruction."""
    steps = len(episode["steps"]) - 1
    out = {"mid": float(steps // 2), "final": float(steps)}
    for i in range(1, steps + 1):
        if command_at(episode, i) != command_at(episode, i - 1):
            out[f"switch{command_at(episode, i)}"] = float(min(i + 3, steps))
    return out


def intro_card(episode, entry, number, total, physics="grid", grid_entry=None):
    commands = commands_of(episode)
    walls = int(grid(episode["walls"], np.uint8, (32, 32)).sum())
    scene = f"{episode['robot_count']} robots, {len(episode['cheeses'])} cheeses, {len(episode['boxes'])} boxes, {'walls' if walls else 'no walls'}"
    lines = [(f"Task {number} of {total}", 36, b"Regular", 0)]
    where = " under MuJoCo physics" if physics == "mujoco" else ""
    if entry is not None and entry.get("steering"):
        lines.append(("Real-time steering", 58, b"SemiBold", 24))
        lines.append(("One swarm, no cuts; each instruction is replaced while the swarm is still moving,", 30, b"Regular", 24))
        lines.append(("once it has taken effect (half the swarm in the box; the cheese eaten):", 30, b"Regular", 6))
        lines += [(f"{k + 1}. \u201c{c['text']}\u201d", 44, b"SemiBold", 14) for k, c in enumerate(commands)]
        lines.append((scene, 28, b"Regular", 36))
        if grid_entry is not None:
            lines.append((f"The last instruction is completed in {entry['final_successes']}/{entry['rollouts']} rollouts under MuJoCo and {grid_entry['final_successes']}/{grid_entry['rollouts']} on the raw grid (fresh scenes).", 28, b"Regular", 10))
        else:
            lines.append((f"The last instruction is completed in {entry['final_successes']}/{entry['rollouts']} rollouts on fresh scenes{where}.", 28, b"Regular", 10))
        return card(lines)
    if len(commands) > 1:
        lines.append((f"{len(commands)} instructions, given one after another to the same swarm:", 34, b"Regular", 24))
        lines += [(f"{k + 1}. “{c['text']}”", 48, b"SemiBold", 16) for k, c in enumerate(commands)]
    else:
        lines.append((f"Instruction: “{commands[0]['text']}”", 58, b"SemiBold", 24))
    lines.append((scene, 28, b"Regular", 36))
    if entry is not None and grid_entry is not None:
        lines.append((f"Succeeds in {entry['successes']}/{entry['rollouts']} rollouts under MuJoCo and {grid_entry['successes']}/{grid_entry['rollouts']} on the raw grid (fresh scenes).", 28, b"Regular", 10))
        lines.append(("Shown: one scene, run in both worlds.", 28, b"Regular", 6))
    elif entry is not None:
        lines.append((f"Succeeds in {entry['successes']}/{entry['rollouts']} rollouts on fresh scenes{where}; shown: one of the successful runs.", 28, b"Regular", 10))
    return card(lines)


def beyond_the_teacher(directory):
    """(episode, meta) of a move-37 candidate (scripts/move37_search.py)."""
    data = json.loads((directory / "episode.json").read_text())
    meta = json.loads((directory / "meta.json").read_text())
    return data["instructions"][0]["runs"][0], meta


def beyond_line(meta):
    confirm = meta.get("confirm") or {"successes": 1, "runs": 1}
    expert = "the scripted expert failed" if meta["expert_steps"] is None else f"the scripted expert took {meta['expert_steps']} steps"
    return f"Beyond the teacher, \u201c{meta['instruction']}\u201d: policy succeeded in {confirm['successes']}/{confirm['runs']} runs in {meta['policy_steps']} steps; {expert}"


def beyond_card(episode, meta, number, total):
    confirm = meta.get("confirm") or {"successes": 1, "runs": 1}
    if meta["expert_steps"] is None:
        result = f"The scripted expert failed this scene; the policy succeeded ({confirm['successes']}/{confirm['runs']} runs)."
    else:
        result = f"The policy needed {meta['policy_steps']} steps, the scripted expert {meta['expert_steps']} ({confirm['successes']}/{confirm['runs']} policy runs succeed)."
    lines = [(f"Task {number} of {total}", 36, b"Regular", 0), ("Beyond the teacher", 58, b"SemiBold", 24)]
    commands = commands_of(episode)
    if len(commands) > 1:
        lines += [(f"{k + 1}. \u201c{c['text']}\u201d", 40, b"Regular", 12) for k, c in enumerate(commands)]
    else:
        lines.append((f"Instruction: \u201c{commands[0]['text']}\u201d", 44, b"Regular", 16))
    lines.append((result, 30, b"Regular", 36))
    return card(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", default=str(RECORDINGS))
    parser.add_argument("--out", required=True, help="directory for field-demo.mp4")
    parser.add_argument("--stills", default="", help="directory for the stills (default: <out>/stills)")
    parser.add_argument("--frames_per_step", type=int, default=0, help="video frames per policy step (default 3; with MuJoCo recordings, the recorded frames per step)")
    parser.add_argument("--name", default="field-demo.mp4", help="video file name in --out")
    parser.add_argument("--min_share", type=float, default=0.5, help="show an instruction only if at least this share of its rollouts succeed (and at least one)")
    parser.add_argument("--success_log", default="", help="a field_demo_record log whose success_rate lines give the k/N shown (default: the recordings' own)")
    parser.add_argument("--trim", type=int, default=4, help="end each run this many steps after its last instruction is met (skips the holding steps); -1 keeps them")
    parser.add_argument("--extra", nargs="*", default=[], help="move-37 candidate directories (scripts/move37_search.py: episode.json, meta.json) to append as 'beyond the teacher' segments")
    parser.add_argument("--grid", default="", help="raw-grid recordings (field_demo_record.py --physics grid, same instructions): every task then shows the same scene run in MuJoCo and on the grid (a cellular automaton) side by side")
    parser.add_argument("--title_card", action="store_true", help="open with the title card (title, policy, physics)")
    parser.add_argument("--end_card", action="store_true", help="end with the card of success rates and definitions")
    parser.add_argument("--steer_frames", type=int, default=5, help="video frames per policy step in the real-time steering task (slower, so each command and the change it causes can be seen)")
    parser.add_argument("--choices", default="", help="instruction=run pairs (indices into recordings' instructions and their runs), e.g. 3=1,5=2; default the first run")
    args = parser.parse_args()
    out = Path(args.out)
    stills_dir = Path(args.stills) if args.stills else out / "stills"
    out.mkdir(parents=True, exist_ok=True)
    stills_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.recordings).read_text())
    size, side = data["size"], data["cheese_side"]
    physics = data.get("physics", "grid")
    frames_per_step = args.frames_per_step or data.get("frames_per_step") or 3
    choices = dict(tuple(int(v) for v in pair.split("=")) for pair in args.choices.split(",") if pair)
    shown, dropped = shown_and_dropped(data, choices, success_rates(args.success_log) if args.success_log else {}, args.min_share)
    if args.grid:
        pairs = paired(shown, json.loads(Path(args.grid).read_text()))
    else:
        pairs = [(episode, entry, None, None) for episode, entry in shown]
    if args.trim >= 0:
        pairs = [(trimmed(episode, args.trim), entry, partner and trimmed(partner, args.trim), grid_entry) for episode, entry, partner, grid_entry in pairs]
    total = len(shown) + len(args.extra)
    segments = []  # (episode, intro card, end-card line, raw-grid episode or None)
    for number, (episode, entry, partner, grid_entry) in enumerate(pairs, 1):
        line = f"{episode['instruction']}: {entry['successes']}/{entry['rollouts']}" if entry else episode["instruction"]
        if entry and grid_entry:
            line = f"{episode['instruction']}: MuJoCo {entry['successes']}/{entry['rollouts']}, raw grid {grid_entry['successes']}/{grid_entry['rollouts']}"
        if entry and entry.get("steering"):
            line = f"real-time steering, {episode['instruction']}: last instruction completed in {entry['final_successes']}/{entry['rollouts']}"
            if grid_entry:
                line = f"real-time steering, {episode['instruction']}: last instruction completed, MuJoCo {entry['final_successes']}/{entry['rollouts']}, raw grid {grid_entry['final_successes']}/{grid_entry['rollouts']}"
        segments.append((episode, intro_card(episode, entry, number, total, physics, grid_entry if partner else None), line, partner))
    for number, directory in enumerate(args.extra, len(shown) + 1):
        episode, meta = beyond_the_teacher(Path(directory))
        segments.append((episode, beyond_card(episode, meta, number, total), beyond_line(meta), None))
    dummy = data.get("policy", "dummy") == "dummy"
    note = MOCK_NOTE if dummy else ""
    video = Video(out / args.name)
    stills = {}

    title_lines = [("Language-Steered Swarm Control with Per-Square Action Fields", 56, b"Bold", 0), ("Supplementary video", 36, b"Regular", 30)]
    if dummy:
        title_lines.append(("Mock-up: all rollouts below use a hand-coded dummy policy.", 30, b"SemiBold", 50))
    else:
        title_lines += [(line, 30, b"Regular", 50 if k == 0 else 8) for k, line in enumerate(POLICY_LINES)]
    if physics == "mujoco":
        title_lines.append(("Execution: rigid-body physics in MuJoCo; the policy sees the scene snapped to the grid.", 30, b"Regular", 8))
    if args.grid:
        title_lines.append(("Beside it, the same scene on the raw grid the policy was trained in: a cellular automaton.", 30, b"Regular", 8))
    if args.title_card:
        title = card(title_lines, note)
        video.add(title, int(TITLE_SECONDS * FPS))
        stills["00-title"] = title

    for number, (episode, intro, _, partner) in enumerate(segments, 1):
        print("task", number, episode["instruction"], flush=True)
        steering = number <= len(shown) and bool(shown[number - 1][1]) and shown[number - 1][1].get("steering")
        video.add(intro, int((EXTRA_CARD_SECONDS if number > len(shown) or steering or partner else CARD_SECONDS) * FPS))
        if partner is not None:
            render_pair(video, stills, episode, partner, number, total, data, args.steer_frames if steering else frames_per_step, note, steering)
            continue
        arena = Arena(episode, size, side, frames_per_step, STEER_VIEW if steering else VIEW)
        per_step = frames_per_step
        policy_figure = PolicyFigure(episode, size)
        field_figure = FieldFigure(episode, size, data["code_vectors"], dummy, (7.0, 4.8) if steering else (7.0, 4.6))
        steps = len(episode["steps"]) - 1
        wanted = still_steps(episode)
        if steering:
            wanted.update({f"typing{j}": at - 0.5 * TYPE_SECONDS * FPS / frames_per_step for at, j in switches(episode)[1:]})
        for i in range(steps + 1):
            policy = None if steering else policy_figure.draw(episode["steps"][i])
            field = field_figure.draw(episode["steps"][i])
            sub = per_step if i < steps else int(HOLD_SECONDS * FPS)
            for k in range(sub):
                t = i + (k / per_step if i < steps else 0.0)
                if steering:
                    frame = compose_steering(episode, number, total, t, arena.pose(t), field, note, frames_per_step / FPS)
                else:
                    frame = compose(episode, number, total, t, arena.pose(t), policy, field, note, physics)
                video.add(frame)
                for name, at in wanted.items():
                    if abs(t - at) <= 0.5 / per_step + 1e-9 and (k == 0 or float(at) != int(at)):
                        stills[f"{number:02d}-{name}"] = frame
        plt.close("all")

    end_lines = []
    if dummy:
        stepped = "run in MuJoCo physics" if physics == "mujoco" else "stepped on the real grid physics"
        end_lines += [("Mock-up", 48, b"Bold", 0), (f"Moves are sampled per square from a hand-coded dummy policy, {stepped}.", 30, b"Regular", 30)]
    else:
        rollouts = shown[0][1]["rollouts"] if shown else 0
        where = " under MuJoCo physics" if physics == "mujoco" and not args.grid else ""
        end_lines.append((f"Shown: selected successful runs; success over {rollouts} rollouts per instruction{where} (fresh scenes):", 30, b"SemiBold", 0))
        for k, (_, _, line, _) in enumerate(segments):
            end_lines.append((line, 26, b"Regular", 16 if k == 0 else 4))
        end_lines.append(("Success: the task's own check (score >= 0.99 at the end of the run, no cheese eaten that was not asked for).", 24, b"Regular", 30))
        if physics == "mujoco":
            end_lines.append(("Robots and cheeses are rigid bodies in MuJoCo; each policy step their positions are snapped to the 32 x 32 grid the policy reads.", 24, b"Regular", 6))
            end_lines.append(("Eating is judged by the grid rule (enough of a cheese's ring covered by robots) on the snapped positions.", 24, b"Regular", 6))
        if args.grid:
            end_lines.append(("Raw grid: the world the policy was trained in; each step every robot moves one square or stays, and a cheese moves when enough robots push it.", 24, b"Regular", 6))
    if args.end_card:
        end = card(end_lines, note)
        video.add(end, int(END_SECONDS * FPS))
        stills["99-end"] = end
    video.close()
    for _, entry in dropped:
        print("not shown (below half or the chosen run fails):", entry["instruction"])
    for name, image in stills.items():
        image.save(stills_dir / f"{name}.png")
    print("wrote", out / args.name, round(video.frames / FPS, 1), "s;", len(stills), "stills in", stills_dir)


if __name__ == "__main__":
    main()
