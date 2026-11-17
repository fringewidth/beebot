"""Helpers every trainer shares: dataclass configs as command-line flags, and a clock that stops while the machine sleeps."""

import argparse
import dataclasses
import time


def awake_seconds():
    """A clock that stops while the machine sleeps, so a laptop's sleep doesn't eat a run's time budget.

    CLOCK_UPTIME_RAW on macOS; on Linux, CLOCK_MONOTONIC already excludes suspend.
    """
    return time.clock_gettime(getattr(time, "CLOCK_UPTIME_RAW", time.CLOCK_MONOTONIC))


def parse_config(config_class):
    """An instance of the dataclass config_class with every field settable as --name."""
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(config_class):
        if field.type is bool:
            parser.add_argument(f"--{field.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=field.default)
        else:
            parser.add_argument(f"--{field.name}", type=field.type, default=field.default)
    return config_class(**vars(parser.parse_args()))
