from beebot.vla.voxel.haiku_frame import mapping, problems, user_side, user_text_words


def entry(user="left", camera="front", place="left"):
    return {"camera": camera, "place": place, "user": user, "task": "push the red cheese to the right of the floor", "sub_goals": ["push the red cheese to the right of the floor"]}


def row(text, statement, first="push the red cheese to the right of the floor"):
    e = entry()
    thought = f"you asked: {text}. i see a red cheese. the camera looks from the board's {e['camera']} and you stand to the left of the camera, so you face the board from its {e['user']}. {statement}. first: {first}."
    return {"text": text, "thought": thought, "switches": []}


def test_user_side_and_mapping():
    assert user_side("front", "behind") == "front"
    assert user_side("front", "right") == "right"
    assert user_side("front", "left") == "left"
    assert user_side("left", "opposite") == "right"
    assert mapping("front") == {"toward": "front", "right": "right", "away": "back", "left": "left"}
    assert mapping("left") == {"toward": "left", "right": "front", "away": "right", "left": "back"}


def test_user_words_map_onto_the_board():
    table = mapping("front")
    assert user_text_words("push it to the near right corner", table) == {"front", "right"}
    assert user_text_words("one goes northwest", table) == {"back", "left"}
    assert user_text_words("move toward the red cheese", table) == set()


def test_checks_pass_a_right_row_and_catch_wrong_ones():
    e = entry()  # a user at the board's left: away from them is the board's right
    assert problems(e, row("push the red cheese away from me", "away from me is the board's right")) == []
    assert "wrong_statement" in problems(e, row("push the red cheese away from me", "away from me is the board's left"))
    assert "text_directions" in problems(e, row("push the red cheese toward me", "toward me is the board's right"))
    assert "first_directions" in problems(e, row("push the red cheese away from me", "away from me is the board's right", first="push the red cheese to the left of the floor"))
    assert "first_user_words" in problems(e, row("push the red cheese away from me", "away from me is the board's right", first="push the red cheese right, away from me"))
