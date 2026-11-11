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


def test_frame_choices_carry_their_camera_and_views_must_take_a_side():
    import numpy as np

    from beebot.vla.voxel.think_data import ORDER, frame_choice, takes_side, user_turns
    from beebot.vla.voxel.views import side_azimuth, side_of

    choice = frame_choice({"rung": "plain", "text": "go left", "thought": "t", "camera": "back", "place": "right", "user": "left", "pair_id": "x:plain"})
    assert choice["rung"] == "frame-plain" and choice["instruction"] == "go left" and choice["camera"] == "back"
    assert takes_side(lambda obs, rng, side=None: None) and not takes_side(lambda obs, rng: None)
    rng = np.random.default_rng(0)
    for side in ORDER:
        for _ in range(20):
            azimuth = side_azimuth(side, rng)
            assert side_of(azimuth) == side
            assert ORDER[user_turns(azimuth, "right")] == user_side(side, "right")
