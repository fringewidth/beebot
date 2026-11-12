from beebot.vla.voxel import suite as S
from beebot.vla.voxel.basics import PHRASES, board_region, episodes, feasible
from beebot.vla.voxel.camera_frame import SIDES, world_word


def test_directions_map_onto_the_board_from_where_the_user_stands():
    assert board_region(("left",), "front") == "left"  # the default viewer: the board's own names
    assert board_region(("away", "left"), "front") == "back left"
    assert board_region(("away",), "left") == "right"  # at the board's left, looking across it
    assert board_region(("left",), "left") == "back"
    assert board_region(("toward", "right"), "back") == "back left"


def test_the_user_is_at_the_camera_and_the_words_turn_back_onto_the_board():
    """The camera-frame words of a user at the camera's side k name the world sides
    camera_frame.world_word gives: the scoring turns them back with the camera's pose."""
    said = {"toward": "front", "right": "right", "away": "back", "left": "left"}  # each user direction as a camera-frame side
    for k, side in enumerate(SIDES):
        for direction, word in said.items():
            assert board_region((direction,), side) == world_word(word, k), (side, direction)


def test_episodes_say_the_text_as_given_and_pin_the_camera():
    eps = episodes(per=2)
    assert len(eps) == 2 * len(PHRASES)
    for e, (category, text, directions, job) in zip(eps, PHRASES + PHRASES):
        assert e.texts == [text] and e.as_said and e.camera == e.key["camera"] and "place" not in e.key
        assert e.task.spec["kind"] == job and e.key["category"] == category
        assert e.key["region"] == board_region(directions, e.camera)
    assert len({e.camera for e in eps}) > 1


def test_every_episode_passes_the_suites_own_draw_filters():
    """As suite.make_motion draws: no task already half done at the start, every move with room
    for its steps (most scenes the eval draws have none for a three-step move: they are redrawn)."""
    eps = episodes(per=2)
    for e in eps:
        assert feasible(e.task, e.world, e.meta, 0) and S.has_room(e.task, e.world, 0) and not S.already_done(e.task, e.world, e.meta, 0)
    assert any(e.key["redraws"] for e in eps if e.key["category"] == "steps")
