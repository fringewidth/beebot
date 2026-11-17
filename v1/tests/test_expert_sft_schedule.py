from beebot.vla.expert_sft import eval_due, schedule_fraction


def test_schedule_over_steps_ignores_time():
    assert schedule_fraction(1, 9999.0, 800, 1.0) == 0.0
    assert schedule_fraction(401, 0.0, 800, 1.0) == 0.5


def test_schedule_over_time_without_max_steps():
    assert schedule_fraction(5, 30.0, 0, 1.0) == 0.5
    assert schedule_fraction(5, 120.0, 0, 1.0) == 1.0


def test_eval_every_steps_skips_the_last_step():
    assert [s for s in range(1, 801) if eval_due(s, 0.0, 1e9, 100, 800)] == [100, 200, 300, 400, 500, 600, 700]


def test_eval_by_time_without_step_interval():
    assert eval_due(7, 61.0, 60.0, 0, 0)
    assert not eval_due(7, 59.0, 60.0, 0, 0)
