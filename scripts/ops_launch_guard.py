#!/usr/bin/env python3
"""PreToolUse hook on Bash (.claude/settings.json): no tracked job starts without an ops node.

Reads the hook input JSON on stdin ({"tool_name": "Bash", "tool_input": {"command": ..., "run_in_background": ...}}).
Exits 2 with a message on stderr (Claude Code blocks the call and shows Claude the message) when the command:

1. starts a tracked job (training, evaluation, demo generation, MuJoCo measurement, Haiku generation, a Kaggle
   push or the Mac's training chain) without naming an open ops node: an `OPS_NODE=<id>` prefix or a trailing
   `# ops:<id>` comment, where ~/ops/projects/beebot/nodes/<id>.yaml exists and is not done or dropped;
2. detaches a long-lived process (nohup, caffeinate, setsid, disown, or a tracked job sent to the background) whose
   script or output file lives under /tmp or /private/tmp, which a reboot wipes.

Anything else passes: tests, --help, `python -c`, reads of these files, dry runs. Any error in the guard itself
lets the command through (exit 0). Tests: tests/test_ops_launch_guard.py.
"""

import json
import os
import re
import shlex
import sys
from pathlib import Path

NODES = Path(os.environ.get("OPS_HOME", Path.home() / "ops")) / "projects" / "beebot" / "nodes"
WRAPPERS = {"nohup", "caffeinate", "time", "nice", "exec", "env", "setsid", "command", "stdbuf", "ionice", "sudo"}
WRAPPER_VALUES = {"nice": {"-n"}, "caffeinate": {"-w", "-t"}, "env": {"-u"}, "ionice": {"-c", "-n"}, "sudo": {"-u"}}
KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "{", "!"}
DETACHERS = {"nohup", "caffeinate", "setsid", "disown"}
SHELLS = {"sh", "bash", "zsh"}
SEPARATORS = {";", "&", "&&", "||", "|", "(", ")", "\n", ";;", "|&"}
REDIRECTS = {">", ">>", "&>", "&>>", "2>", "2>>", "1>", "1>>", ">|"}
TMP = re.compile(r"^(/private)?/tmp(/|$)")
NODE_ID = r"([A-Za-z0-9][A-Za-z0-9_.-]*)"

# Tracked modules: module -> subcommands that count (None: every run but --help).
MODULES = {
    "beebot.vla.voxel.think_train": None,
    "beebot.vla.voxel.basics": None,
    "beebot.vla.voxel.demos": None,
    "beebot.hardware.closed_loop": {"measure", "episode"},
    "beebot.hardware.cube_outcomes": {"measure", "calibrate", "right"},
    "beebot.vla.voxel.haiku": {"export", "generate", "verify", "testset", "pilot"},
    "beebot.vla.voxel.haiku_frame": {"generate"},
}


def strip_heredocs(text, scripts=None):
    """Drop heredoc bodies, which are data, not commands. Bodies fed to a shell or written to a .sh/.zsh file are
    scripts about to run: they go to `scripts` when given."""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        tags = re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
        script = re.search(r">\s*\S+\.(sh|zsh|bash)\b|(^|[\s;&|(])(ba|z)?sh(\s+-s)?\s*<<", line)
        i += 1
        for tag in tags:
            start = i
            while i < len(lines) and lines[i].strip() != tag:
                i += 1
            if script and scripts is not None:
                scripts.append("\n".join(lines[start:i]))
            i += 1
    return "\n".join(out)


def unquoted_newlines(text):
    """Turn newlines outside quotes into command separators (after the newline, so a `# comment` ends first)."""
    out, quote, i = [], None, 0
    text = text.replace("\\\n", " ")
    while i < len(text):
        c = text[i]
        if c == "\\" and quote != "'" and i + 1 < len(text):
            out.append(text[i : i + 2])
            i += 2
            continue
        if quote:
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c == "\n":
            out.append("\n;")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def tokens(text):
    lexer = shlex.shlex(unquoted_newlines(text), posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace = " \t\r\n"
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def simple_commands(text):
    """Split a command line into simple commands: lists of words, with the redirection targets separated."""
    commands, words, targets, background = [], [], [], []
    stream = tokens(strip_heredocs(text))
    i = 0
    while i < len(stream):
        token = stream[i]
        if token in SEPARATORS:
            if words or targets:
                commands.append((words, targets, token == "&"))
            words, targets = [], []
        elif token in REDIRECTS or token in {"<", "<<", "<<<", ">&"} or re.fullmatch(r"\d?>>?", token):
            if i + 1 < len(stream) and token not in {"<<", "<<<"}:
                targets.append(stream[i + 1])
            i += 1
        else:
            words.append(token)
        i += 1
    if words or targets:
        commands.append((words, targets, False))
    return commands


def assignments(text):
    return dict(re.findall(r"(?:^|[\s;&|(])(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=([^\s;&|()]+)", text))


def expand(word, names):
    return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: names.get(m.group(1), m.group(0)).strip("'\""), word)


def unwrap(words):
    """Drop env assignments and wrappers (nohup, caffeinate -i, uv run, ...) in front of the real program."""
    i = 0
    while i < len(words):
        word = words[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", word) or word in KEYWORDS:
            i += 1
        elif os.path.basename(word) in WRAPPERS:
            valued = WRAPPER_VALUES.get(os.path.basename(word), set())
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in valued else 1
        elif word == "uv" and i + 1 < len(words) and words[i + 1] == "run":
            i += 2
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in {"--with", "--python", "-p", "--project", "--directory"} else 1
        else:
            break
    return words[i:]


def python_target(words):
    """(module, script, rest) for a python invocation, else None."""
    if not words or not re.fullmatch(r"python(\d+(\.\d+)?)?", os.path.basename(words[0])):
        return None
    i = 1
    while i < len(words):
        word = words[i]
        if word == "-m" and i + 1 < len(words):
            return words[i + 1], None, words[i + 2 :]
        if word in {"-c", "-"}:
            return None
        if word.startswith("-"):
            i += 1
            continue
        return None, word, words[i + 1 :]
    return None


def tracked(words):
    """A short description of the tracked job these words start, or None."""
    words = unwrap(words)
    if not words:
        return None
    program = os.path.basename(words[0])
    if program == "kaggle" and words[1:3] == ["kernels", "push"]:
        return "kaggle kernels push"
    if words[0].endswith("mac_chain.sh") or (program in SHELLS and len(words) > 1 and words[1].endswith("mac_chain.sh")):
        return "scripts/mac_chain.sh"
    target = python_target(words)
    if target is None:
        return None
    module, script, rest = target
    if script:
        match = re.search(r"beebot/(vla/voxel|hardware)/(\w+)\.py$", script)
        module = f"beebot.{match.group(1).replace('/', '.')}.{match.group(2)}" if match else None
    if module not in MODULES or {"-h", "--help"} & set(rest) or "--dry-run" in rest:
        return None
    commands = MODULES[module]
    if commands is not None and not (rest and rest[0] in commands):
        return None
    return f"{module} {rest[0]}" if commands else module


def node_problem(text):
    text = re.sub(r"['\"]", " ", text)
    named = re.findall(rf"(?:^|[\s;&|(])(?:export\s+)?OPS_NODE=['\"]?{NODE_ID}", text) + re.findall(rf"#\s*ops:\s*{NODE_ID}", text)
    if not named:
        return "it names no ops node"
    node_id = named[-1]
    path = NODES / f"{node_id}.yaml"
    if not path.exists():
        return f"ops node {node_id!r} does not exist in {NODES} (run `python3 ~/ops/ops.py board` to pull, or add it)"
    status = re.search(r"^status:\s*(\w+)", path.read_text(), re.M)
    if status and status.group(1) in {"done", "dropped"}:
        return f"ops node {node_id!r} is {status.group(1)}"
    return None


def problems(command, background=False, context=""):
    found = []
    names = assignments(context + "\n" + command)
    jobs = []
    shell_inner = []
    scripts = []
    strip_heredocs(command, scripts)
    for script in scripts:
        jobs += [job for words, _, _ in simple_commands(script) if (job := tracked(words))]
    for words, targets, sent_back in simple_commands(command):
        bare = unwrap(words)
        if bare and os.path.basename(bare[0]) in SHELLS and len(bare) > 2 and bare[1] in {"-c", "-lc"}:
            shell_inner.append(bare[2])
        job = tracked(words)
        if job:
            jobs.append(job)
        program = unwrap(words)
        prefix = words[: len(words) - len(program)]
        if any(os.path.basename(w) in DETACHERS for w in prefix) or sent_back or (background and job):
            paths = [expand(t, names) for t in targets]
            target = python_target(program) if program else None
            if target and target[1]:
                paths.append(expand(target[1], names))
            elif program:
                paths += [expand(w, names) for w in program[:2] if "/" in w and (w is program[0] or os.path.basename(program[0]) in SHELLS)]
            under_tmp = [p for p in paths if TMP.match(p)]
            if under_tmp:
                found.append(f"a detached process uses {', '.join(under_tmp)}; /tmp and /private/tmp are wiped at reboot (2026-11-11 lost the chain scripts there). Put scripts in scripts/ and logs in logs/")
    for inner in shell_inner:
        found += problems(inner, background, context + "\n" + command)
    if jobs:
        problem = node_problem(context + "\n" + command)
        if problem:
            found.append(
                f"this starts a tracked job ({', '.join(sorted(set(jobs)))}) but {problem}. Add or claim its node first "
                f"(`python3 ~/ops/ops.py add ..` / `claim ..`), then name it: prefix `OPS_NODE=<id> ` or end the command with `# ops:<id>`"
            )
    return list(dict.fromkeys(found))


def main():
    try:
        event = json.load(sys.stdin)
        if event.get("tool_name") != "Bash":
            return 0
        command = event.get("tool_input", {}).get("command") or ""
        found = problems(command, bool(event.get("tool_input", {}).get("run_in_background")))
    except Exception as error:  # the guard must never break a session
        print(f"ops_launch_guard: skipped ({error})", file=sys.stderr)
        return 0
    if not found:
        return 0
    print("ops launch guard: blocked.\n- " + "\n- ".join(found), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
