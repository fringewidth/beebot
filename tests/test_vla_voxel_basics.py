from beebot.vla.voxel.basics import PHRASES, board_region, episodes


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
