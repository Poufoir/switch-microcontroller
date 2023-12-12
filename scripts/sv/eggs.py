from __future__ import annotations

import argparse
import time

import serial

from scripts.engine import (
    always_matches,
    do,
    make_vid,
    Press,
    require_tesseract,
    run,
    SERIAL_DEFAULT,
    Press_and_Wait,
    Wait,
    match_box_text,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=SERIAL_DEFAULT)
    parser.add_argument("--boxes", type=int, default=1)
    args = parser.parse_args()

    require_tesseract()

    start_time = 0.0
    egg_count = 0

    def set_start(vid: object, ser: object) -> None:
        nonlocal start_time
        start_time = time.monotonic()

    def increment_egg_count(vid: object, ser: object) -> None:
        nonlocal egg_count
        egg_count += 1
        print(f"DEBUG: You have {egg_count} eggs currently")

    def end_egg_timer(frame: object) -> bool:
        return time.monotonic() > start_time + 30 * 60

    states = {
        "RESET": (
            always_matches,
            do(
                Press_and_Wait("H", wait_duration=1),
                Press_and_Wait("X", wait_duration=1),
                Press_and_Wait("A", wait_duration=6),
                # Enter Game
                Press_and_Wait("A", wait_duration=1),
                Press_and_Wait("A", wait_duration=25),
                # After that, you are in pokemon game
                Press_and_Wait("A", wait_duration=35),
            ),
            "INITIAL",
        ),
        "INITIAL": (
            (
                always_matches,
                do(
                    Press_and_Wait("h", 1.9, 0.2),
                    Press_and_Wait("w", wait_duration=0.1),
                    # After that, dialog with Recette du Fil starts
                    Press_and_Wait("w", 1, 3),
                ),
                "RECETTE_DU_FILS",
            ),
        ),
        "RECETTE_DU_FILS": (
            always_matches,
            do(
                Press_and_Wait("A", wait_duration=2),
                # Select compote du fils
                Press_and_Wait("s", wait_duration=0.1),
                Press_and_Wait("A", wait_duration=2),
                # Select PL payement
                Press_and_Wait("s", wait_duration=0.1),
                Press_and_Wait("A", wait_duration=2),
                Press_and_Wait("A", wait_duration=25),
                # Accept effects
                Press_and_Wait("A", wait_duration=2),
                Press_and_Wait("A", wait_duration=2),
                set_start,
            ),
            "GO_TO_OUTSIDE",
        ),
        "GO_TO_OUTSIDE": (
            always_matches,
            do(
                Press("d", 1.5),
                Press("w", 1),
                Press("d", 0.8),
                Press("w", 0.5),
                Press_and_Wait("L", wait_duration=0.5),
            ),
            "MENU",
        ),
        "MENU": (
            (
                always_matches,
                do(
                    Press_and_Wait("X", wait_duration=1.5),
                    Press("d"),
                    Press("s"),
                    Press_and_Wait("s", wait_duration=2),
                    Press_and_Wait("A", wait_duration=10),
                ),
                "FIND_BASKET",
            ),
        ),
        "FIND_BASKET": (
            always_matches,
            do(
                Press_and_Wait("d", 0.1, 0.2),
                Press_and_Wait("L", wait_duration=0.2),
                Press_and_Wait("w", 0.4, 0.5),
                Press_and_Wait("a", 0.1, 0.2),
                Press_and_Wait("L", wait_duration=0.2),
                Press_and_Wait("w", 0.7, 0.2),
                Press_and_Wait("z", 0.1, 0.2),
                Press_and_Wait("L", wait_duration=0.2),
                Press_and_Wait("w", 0.5, 0.5),
            ),
            "CHECK_BASKET",
        ),
        "CHECK_BASKET": (
            always_matches,
            do(Press_and_Wait("A", wait_duration=1)),
            "VERIFY_BASKET",
        ),
        "VERIFY_BASKET": (
            (
                # We are near the basket to interract so there is a dialog box
                match_box_text(),
                do(
                    Press_and_Wait("A", wait_duration=2),
                    Press_and_Wait("A", wait_duration=2),
                ),
                "VERIFY_EGG",
            ),
            # if it fails, reset
            (always_matches, do(Press("B")), "RESET"),
        ),
        "VERIFY_EGG": (
            (
                # There is an egg and you must accept it
                match_box_text(),
                do(
                    increment_egg_count,
                    Press_and_Wait("A", wait_duration=2),
                    Press_and_Wait("A", wait_duration=2),
                    Press_and_Wait("A", wait_duration=2),
                ),
                "VERIFY_ANOTHER_EGG",
            ),
            (always_matches, do(Press("B")), "WAIT"),
        ),
        "VERIFY_ANOTHER_EGG": (
            (
                match_box_text(),
                do(
                    increment_egg_count,
                    Press_and_Wait("A", wait_duration=2),
                    Press_and_Wait("A", wait_duration=2),
                    Press_and_Wait("A", wait_duration=2),
                ),
                "VERIFY_ANOTHER_EGG",
            ),
            (always_matches, do(Press("B")), "WAIT"),
        ),
        "WAIT": (
            (end_egg_timer, do(Press("B")), "EXIT"),
            (always_matches, do(Wait(30)), "CHECK_BASKET"),
        ),
    }

    with serial.Serial(args.serial, 9600) as ser:
        run(vid=make_vid(), ser=ser, initial="INITIAL", states=states)


if __name__ == "__main__":
    raise SystemExit(main())
