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


def test_spanish_and_french_direction_words_map_onto_the_board():
    front, back = mapping("front"), mapping("back")
    assert user_text_words("empuja el bloque rayado hacia tu izquierda, al lado derecho del suelo", back) == {"left", "right"}  # run-s6:s6-0002822:casual names both sides
    assert user_text_words("pousse le fromage vers le coin arrière droit", front) == {"back", "right"}
    assert user_text_words("pon el cubo en la esquina de adelante a la derecha", front) == {"front", "right"}
    assert user_text_words("empuja el queso hacia adelante", front) == {"back"}  # forward: away from the user
    assert user_text_words("empuja el queso hacia mi izquierda", front) == {"left"}  # "mi izquierda" is my left, not toward me
    assert user_text_words("pousse le fromage dans le coin près de moi à gauche", front) == {"front", "left"}
    assert user_text_words("empuja el queso cerca del cubo azul a la izquierda", front) == {"left"}  # near a thing: no direction
    assert user_text_words("empuja el queso a la esquina delantera izquierda", front) == {"front", "left"}
    assert user_text_words("déplace le fromage vers le coin sud-est", front) == {"front", "right"}
    assert user_text_words("divide en dos: uno al norte, otro al oeste", front) == {"back", "left"}
    assert user_text_words("split em in 2, one sw and one nw", front) == {"front", "back", "left"}
    assert user_text_words("pousse le fromage, ne le mange pas", front) == set() and user_text_words("que se quede quieto", front) == set()


def test_english_words_that_are_no_direction():
    table = mapping("front")
    assert user_text_words("push the cheese with the least cells to my left", table) == {"left"}  # "least" is no east
    assert user_text_words("shove the big block toward my left side of the floor", table) == {"left"}  # toward my left is left
    assert user_text_words("toward and right", table) == {"front", "right"}  # the prompt's own name for the toward direction


def test_checks_pass_a_right_row_and_catch_wrong_ones():
    e = entry()  # a user at the board's left: away from them is the board's right
    assert problems(e, row("push the red cheese away from me", "away from me is the board's right")) == []
    assert "wrong_statement" in problems(e, row("push the red cheese away from me", "away from me is the board's left"))
    assert "text_directions" in problems(e, row("push the red cheese toward me", "toward me is the board's right"))
    assert "first_directions" in problems(e, row("push the red cheese away from me", "away from me is the board's right", first="push the red cheese to the left of the floor"))
    assert "first_user_words" in problems(e, row("push the red cheese away from me", "away from me is the board's right", first="push the red cheese right, away from me"))
    assert problems(e, row("empuja el queso rojo lejos de mí", "lejos de mí is the board's right")) == []
    assert "text_directions" in problems(e, row("empuja el queso rojo hacia mí", "hacia mí is the board's right"))
    assert "wrong_statement" in problems(e, row("empuja el queso rojo lejos de mí", "izquierda is the board's right"))  # their left is the board's back
    assert problems(e, row("push the red cheese away from me", "the board's right is away from you")) == []  # written the other way round
    assert "wrong_statement" in problems(e, row("push the red cheese away from me", "the board's right is your left"))
    assert problems(e, row("push the red cheese away from me", "away from me is the board's right and your left is the board's back")) == []


def test_frame_choices_take_camera_rows_and_views_must_take_a_side():
    import numpy as np
    import pytest

    from beebot.vla.voxel.think_data import ORDER, frame_choice, quarter_turns, takes_side
    from beebot.vla.voxel.views import side_azimuth, side_of

    choice = frame_choice({"rung": "plain", "text": "go left", "thought": "t", "camera": "back", "pair_id": "x:plain"})
    assert choice["rung"] == "frame-plain" and choice["instruction"] == "go left" and choice["camera"] == "back" and "user" not in choice
    with pytest.raises(ValueError, match="board-frame row"):
        frame_choice({"rung": "plain", "text": "go left", "thought": "t", "camera": "back", "place": "right", "user": "left", "pair_id": "x:plain"})
    assert takes_side(lambda obs, rng, side=None: None) and not takes_side(lambda obs, rng: None)
    rng = np.random.default_rng(0)
    for k, side in enumerate(ORDER):
        for _ in range(20):
            azimuth = side_azimuth(side, rng)
            assert side_of(azimuth) == side and quarter_turns(azimuth) == k


BOARD_ROWS = [  # verified rows of data/vla/haiku/frame (2026-11-11), as Haiku wrote them
    {"rung": "plain", "text": "split into two groups: one heads to the near corner on my right, the other builds a 3-high tower", "thought": "you asked: split into two groups: one heads to the near corner on my right, the other builds a 3-high tower. i see a flat floor and a 3-high tower on the left. the camera looks from the board's left and you stand behind the camera, so you face the board from its left. your right is the board's front, and toward you is the board's left, so the near corner on your right is the board's front left. first: split into two groups: one should go to the front left, the other should build a tower 3 high.", "switches": [], "id": "run-s6:s6-0002043", "pair_id": "run-s6:s6-0002043:plain", "camera": "left", "place": "behind", "user": "left"},
    {"rung": "casual", "text": "shove the yellow cheese south to the edge of the floor, dont eat it", "thought": "you asked: shove the yellow cheese south to the edge of the floor, dont eat it. i see low hills and a yellow cheese on the left. the camera looks from the board's right and you stand to the right of the camera, so you face the board from its back. south is toward you, which is the board's back. first: push the yellow cheese to the back of the floor without eating it.", "switches": [], "id": "run-s2:s2-0001664", "pair_id": "run-s2:s2-0001664:casual", "camera": "right", "place": "right", "user": "back"},
    {"rung": "plain", "text": "push the red cheese to the near left corner of the floor, and don't eat it", "thought": "you asked: push the red cheese to the near left corner of the floor, and don't eat it. i see red cheese at the board's front left and white cheese at the board's back. the camera looks from the board's right and you stand opposite the camera, facing it, so you face the board from its left. your near left is the board's back left. first: push the red cheese to the back left of the floor without eating it.", "switches": [], "id": "x:s0-1", "pair_id": "x:s0-1:plain", "camera": "right", "place": "opposite", "user": "left"},
    {"rung": "casual", "text": "shove the cyan cheese over to your right, and don't eat it.", "thought": "you asked: 'shove the cyan cheese over to your right, and don't eat it.' i see a flat floor and a cyan cheese at the back. the camera looks from the board's front and you face the board from its front. your right is the board's right. first: push the cyan cheese to the right of the floor without eating it.", "switches": [], "id": "x:s0-2", "pair_id": "x:s0-2:casual", "camera": "front", "place": "behind", "user": "front"},
]


def test_camera_rows_drop_the_board_mapping_and_say_the_rest_from_the_camera():
    from beebot.vla.voxel.haiku_frame import camera_problems, camera_row

    rows = [camera_row(row) for row in BOARD_ROWS]
    assert [row["camera"] for row in rows] == ["left", "back", "left", "front"]  # the camera goes where the user stood
    assert rows[0]["thought"] == "you asked: split into two groups: one heads to the near corner on my right, the other builds a 3-high tower. i see a flat floor and a 3-high tower at the front. first: split into two groups: one should go to the front right, the other should build a tower 3 high."
    assert rows[1]["thought"] == "you asked: shove the yellow cheese south to the edge of the floor, dont eat it. i see low hills and a yellow cheese on the right. first: push the yellow cheese to the front of the floor without eating it."
    assert rows[2]["thought"] == "you asked: push the red cheese to the near left corner of the floor, and don't eat it. i see red cheese at the front right and white cheese at the left. first: push the red cheese to the front left of the floor without eating it."
    assert rows[3]["thought"] == "you asked: shove the cyan cheese over to your right, and don't eat it. i see a flat floor and a cyan cheese at the back. first: push the cyan cheese to the right of the floor without eating it."  # the quote is the instruction as written
    for row in rows:
        assert set(row) == {"rung", "text", "thought", "switches", "id", "pair_id", "camera"}
        assert "board" not in row["thought"] and "camera" not in row["thought"] and "you face" not in row["thought"]
    near_right = {"task": "split into two groups: one should go to the front left, the other should build a tower 3 high", "sub_goals": ["split into two groups: one should go to the front left, the other should build a tower 3 high"]}
    assert camera_problems(near_right, rows[0]) == []  # "near ... my right" is the camera frame's front right
    assert camera_problems(near_right, dict(rows[0], camera="right")) == ["first_directions", "text_directions"]
    assert camera_problems(near_right, dict(rows[0], thought=BOARD_ROWS[0]["thought"])) == ["board_words_left", "first_directions"]
