import importlib.util
import math
import os

spec = importlib.util.spec_from_file_location("transfer_gap", os.path.join(os.path.dirname(__file__), "..", "scripts", "transfer_gap.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def row(category, grid, mujoco, gp=None, mp=None):
    return {"category": category, "grid_score": grid, "mujoco_score": mujoco, "grid_pass": grid >= 0.5 if gp is None else gp, "mujoco_pass": mujoco >= 0.5 if mp is None else mp}


def test_compare_gap_is_grid_minus_mujoco_per_category_and_overall():
    rows = [row("side", 1.0, 0.5), row("side", 0.5, 0.5), row("steps", 0.8, 0.0), row("steps", 0.0, 0.0)]
    table = T.compare(rows)
    assert table["side"]["n"] == 2
    assert math.isclose(table["side"]["gap"], 0.25)
    assert math.isclose(table["steps"]["grid_score"], 0.4)
    assert math.isclose(table["steps"]["gap"], 0.4)
    assert math.isclose(table["all"]["gap"], (1.0 + 0.5 + 0.8 + 0.0) / 4 - (0.5 + 0.5) / 4)
    assert table["all"]["grid_pass"] == 0.75
    assert table["all"]["mujoco_pass"] == 0.5


def test_a_negative_gap_when_mujoco_does_better():
    assert T.compare([row("side", 0.2, 0.6)])["all"]["gap"] < 0


def test_disagreements_count_non_reached_moves_by_pivot_class():
    moves = [
        {"primitive": "roll", "detail": "reached"},
        {"primitive": "roll", "detail": "stayed"},
        {"primitive": "roll", "detail": "knock"},
        {"primitive": "climb", "detail": "off"},
        {"primitive": "", "detail": "refused"},
        {"primitive": "", "detail": "blocked"},
    ]
    per, total = T.disagreements(moves)
    assert per["roll"] == {"moves": 3, "disagree": 2, "harmful": 1, "detail": {"reached": 1, "stayed": 1, "knock": 1}}
    assert per["climb"]["harmful"] == 1
    assert per["none"]["moves"] == 2 and per["none"]["harmful"] == 0
    assert total == {"moves": 6, "disagree": 5, "harmful": 2}


def test_summary_rows_carry_the_dashboard_fields():
    rows = [row("side", 1.0, 0.5), row("steps", 0.5, 0.5)]
    moves = [{"primitive": "roll", "detail": "reached"}, {"primitive": "roll", "detail": "off"}]
    lines = T.summary_rows(rows, moves, "runs/x/policy_step593.pt", {"grid": 1.0, "mujoco": 2.0})
    last = lines[-1]
    assert [l["category"] for l in lines[:-1]] == ["side", "steps"]
    assert last["run"] == "transfer" and last["step"] == 593 and last["checkpoint"].endswith("policy_step593.pt")
    assert last["transfer_gap"] == 0.25 and last["transfer_grid_score"] == 0.75 and last["transfer_mujoco_score"] == 0.5
    assert last["transfer_harmful_per_move"] == 0.5 and last["transfer_disagree_share"] == 0.5


def test_step_of():
    assert T.step_of("runs/a/policy_step723.pt") == 723
    assert T.step_of("") is None
