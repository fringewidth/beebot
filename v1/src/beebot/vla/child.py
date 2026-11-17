"""Runs one play program for one episode, inside a sandboxed subprocess. Started by play.py.

Protocol, one JSON object per line: the first line from the parent is {"source": ...}; then each
line is a view, answered by {"commands": [...], "phase": ...} or {"error": ...}. The program
must define act(view, memory) returning a list of up to 4 attractors {"row", "col", "straight"}.
"""

import json
import sys

ALLOWED_MODULES = {"math", "random", "itertools", "collections", "heapq", "functools", "numpy"}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"module {name!r} is not available to play programs")
    return __import__(name, globals, locals, fromlist, level)


def main():
    import builtins

    import numpy as np

    blocked = {"open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "help", "__import__"}
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in blocked}
    safe_builtins["__import__"] = safe_import
    namespace = {"__builtins__": safe_builtins, "np": np}

    first = json.loads(sys.stdin.readline())
    try:
        exec(first["source"], namespace)
        act = namespace["act"]
    except Exception as error:  # report and stop: the program doesn't load
        print(json.dumps({"error": f"load: {type(error).__name__}: {error}"}), flush=True)
        return

    memory = {}
    for line in sys.stdin:
        view = json.loads(line)
        view["mass"] = np.array(view["mass"])
        view["walls"] = np.array(view["walls"], dtype=bool)
        try:
            raw = act(view, memory) or []
            commands = [
                {
                    "row": int(c["row"]),
                    "col": int(c["col"]),
                    "straight": bool(c.get("straight", False)),
                    "share": None if c.get("share") is None else float(c["share"]),
                }
                for c in list(raw)[:4]
            ]
            reply = {"commands": commands, "phase": str(memory.get("phase", ""))[:80]}
        except Exception as error:
            reply = {"error": f"act: {type(error).__name__}: {error}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
