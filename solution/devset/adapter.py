"""Run the possession pipeline, unchanged, on a degraded DFL match.

possession.pipeline.label_frames takes a Tracking, the graded blocks and the tags, and has no file
paths in it, so a degraded DFL match goes straight in. One input is specific to the competition
match and is read from config rather than passed: KNOWN_DEAD_SPANS, the published span before the
second-half kick-off (frames 110808-113742), which deadball.detect always adds. On a dev match those
frame numbers fall in the middle of the second half, so the span would be a false two-minute dead
ball. The adapter empties it for the duration of each call and puts it back afterwards. Nothing in
solution/possession is edited.

PIPELINES are the variants scored by default: the current pipeline, and the current pipeline with
one of its stages switched off by label_frames' own flags. The current pipeline is M4: the rules'
stoppages, with the team in open play from the learned model (possession/learned.py, segment.py),
trained on the calibration matches alone (train_model.py). "M2c: no learned team" is the rules
alone, the pipeline before M4, and "M2b: no M2c rules, no learned team" the one before M2c. The other
switches act under M4, so they also change the evidence the model reads: "no foul stoppages" switches
off M2b's slowdown test, leaving M3b's crossing-or-tag confirmation (M2b's join stays). If the dev
set orders things sensibly, each switch should cost a little, and smooth50 should sit well below all
of them.
"""

import contextlib

from possession import config, pipeline

from . import baselines

CURRENT = "M4 (current)"
SMOOTH50 = "baseline_smooth50"

PIPELINES = (
    (CURRENT, {}),
    ("M2c: no learned team", {"learned_model": False}),
    ("M2b: no M2c rules, no learned team", {"learned_model": False, "restart_laws": False,
                                            "stoppage_starts": False}),
    ("no restart laws", {"restart_laws": False}),
    ("no stoppage starts", {"stoppage_starts": False}),
    ("no foul stoppages", {"foul_stoppages": False}),
    ("no boundary placement", {"boundaries": False}),
    ("no duel rule", {"duels": False}),
    ("no keeper split", {"split_keepers": False}),
)


@contextlib.contextmanager
def no_known_dead_spans():
    """config.KNOWN_DEAD_SPANS emptied inside the block, restored however the block exits."""
    saved = config.KNOWN_DEAD_SPANS
    config.KNOWN_DEAD_SPANS = ()
    try:
        yield
    finally:
        config.KNOWN_DEAD_SPANS = saved


def label(tracking, blocks, tags=(), **flags):
    """(codes, info) from label_frames on a dev match."""
    with no_known_dead_spans():
        return pipeline.label_frames(tracking, blocks, tags, **flags)


def smooth50(tracking, blocks):
    return baselines.smooth50(tracking, blocks)
