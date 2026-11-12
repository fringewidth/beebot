from beebot.vla.voxel import suite as S
from beebot.vla.voxel.basics import PHRASES, board_region, episodes, feasible


def test_directions_map_onto_the_board_from_where_the_user_stands():
    assert board_region(("left",), "front") == "left"  # the default viewer: the board's own names
    assert board_region(("away", "left"), "front") == "back left"
    assert board_region(("away",), "left") == "right"  # at the board's left, looking across it
    assert board_region(("left",), "left") == "back"
    assert board_region(("toward", "right"), "back") == "back left"


def test_episodes_say_the_text_as_given_and_pin_camera_and_user():
    eps = episodes(per=1)
    assert len(eps) == len(PHRASES)
    for e, (category, text, _, job) in zip(eps, PHRASES):
        assert e.texts == [text] and e.as_said and e.user == e.key["place"] and e.camera == e.key["camera"]
        assert e.task.spec["kind"] == job and e.key["category"] == category


def test_every_episode_passes_the_suites_own_draw_filters():
    """As suite.make_motion draws: no task already half done at the start, every move with room
    for its steps (most scenes the eval draws have none for a three-step move: they are redrawn)."""
    eps = episodes(per=2)
    for e in eps:
        assert feasible(e.task, e.world, e.meta, 0) and S.has_room(e.task, e.world, 0) and not S.already_done(e.task, e.world, e.meta, 0)
    assert any(e.key["redraws"] for e in eps if e.key["category"] == "steps")
