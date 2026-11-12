"""Graded frames, per-frame codes to submission labels, and writing the file.

Codes are possession_metrics' own -- H 0, A 1, D 2, X 3 -- so a code array from here can be scored
directly without translating it.
"""

import os

import numpy as np

from .repo import games, pm


def graded_blocks(game=games.GAME):
    """Half-open [(start, stop)] video-frame spans of the two halves -- the only graded frames."""
    return [(first, last + 1) for _, first, last in games.half_video_blocks(game)]


def to_codes(owner, dead, n_frames, blocks):
    """Submission codes from a per-frame owner (0 home, 1 away, -1 none) and a dead-ball mask.

    Outside the blocks every frame is X: it is never graded, so nothing is gained by guessing. Inside
    them dead ball wins over possession, and a frame with neither is X.
    """
    codes = np.full(n_frames, pm.X, dtype=np.int8)
    inside = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        inside[start:stop] = True
    codes[inside & (owner == 0)] = pm.H
    codes[inside & (owner == 1)] = pm.A
    codes[inside & dead] = pm.D
    return codes


def to_labels(codes):
    return np.array(pm.LABELS)[codes]


def write(path, codes):
    """Write the frame,label CSV in the baseline's format (starter/baseline.write_submission), with
    "\\n" line endings on every platform.

    The baseline's writer opens the file in text mode, which writes "\\r\\n" on Windows, and the
    submission has to be the same file wherever it is regenerated (SUBMISSION.md).
    """
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(to_labels(codes)))
    return path


def check(path, n_frames=games.GAMES[games.GAME]["n_video_frames"]):
    """None if the file passes the official validation, else the one-line reason it does not."""
    return pm.validate_submission(pm.read_label_csv(path), n_frames)
