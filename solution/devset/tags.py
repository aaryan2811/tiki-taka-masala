"""A Smart Tagging stand-in for a DFL match: set-piece tags made from DFL's set-piece events, then
degraded to look like the competition's.

The pipeline confirms a slowdown as a dead ball by a line crossing or a Smart Tagging set piece, and
on the real match 27 of its 55 stoppages rest on a tag alone (DESIGN.md section 4). A dev set
without tags measures only the crossing path; one with DFL's events as tags would measure a far
cleaner signal than the real one -- DFL has a hand-coded event for 94-95% of stoppages. So the
events are degraded to what the competition's tags are (params.json "tags", fitted by fit.py):

  - how many    per set-piece type, as many as the competition match has, chosen at random: 26
                throw-ins, 12 corners, 10 goal kicks, 6 kick-offs, but only 5 free kicks against
                ~20 in a DFL match. A type the competition match never shows (a penalty) is kept at
                the overall rate.
  - which team  "N/A" for every throw-in and kick-off, as there; otherwise the event's team.
  - when        each window starts where the competition's do relative to the pipeline's restart
                (a median 5 s before it), drawn from the fitted offsets and applied to the true
                restart: the end of the dead ball nearest the event. Windows are 35 s long,
                kick-offs 5 s, as there.

What it cannot reproduce: the competition's tags are inferred, so they may include false positives;
nothing in the data says which ones, so none are added.
"""

import numpy as np

from possession.runs import true_runs

from . import degrade

CODE = "SET PIECES"
NO_TEAM = "N/A"
KICK_OFF = "Kick Off"

# The restart a set-piece event belongs to: the end of a dead ball from 12 s before the event to 6 s
# after it. The audit found 96 of 101 and 90 of 96 dead runs with an event within 6 s before to 12 s
# after their end; the event time minus the run's end has a median of -0.5 to +0.4 s.
RESTART_BEFORE = 12 * 25  # frames
RESTART_AFTER = 6 * 25  # frames


def restart_of(half, row):
    """The frame play restarts for an event at `row`: the nearest dead-ball end in the window above,
    else the event's own row."""
    _, stops = true_runs(half.status == 0)
    near = stops[(stops >= row - RESTART_BEFORE) & (stops <= row + RESTART_AFTER)]
    return int(near[np.argmin(np.abs(near - row))]) if len(near) else int(row)


def kept_events(set_pieces, tag_params, rng):
    """The subset of set-piece events that get a tag: per type, at most the competition's count."""
    per_type = tag_params["per_type"]
    total = sum(entry["count"] for entry in per_type.values())
    overall = min(1.0, total / float(max(1, len(set_pieces))))
    kept = []
    for kind in sorted({piece[2] for piece in set_pieces}):
        events = [piece for piece in set_pieces if piece[2] == kind]
        if kind in per_type:
            n_keep = min(len(events), per_type[kind]["count"])
        else:
            n_keep = int(rng.binomial(len(events), overall))
        chosen = rng.choice(len(events), size=n_keep, replace=False)
        kept += [events[index] for index in sorted(chosen)]
    return sorted(kept)


def synthetic(match, blocks, params, seed):
    """[{code, start_frame, end_frame, start_s, end_s, tags}] for one match, the format
    starter/load_smart_tagging.load() returns, on degrade.layout's frame axis."""
    rng = np.random.default_rng(seed)
    tag_params = params["tags"]
    per_type = tag_params["per_type"]
    default_window = int(np.median([entry["window_frames"] for entry in per_type.values()]))
    out = []
    for half_index, row, kind, team in kept_events(match.set_pieces, tag_params, rng):
        half, (block_start, _) = match.halves[half_index], blocks[half_index]
        entry = per_type.get(kind, {"no_team_share": 0.0, "window_frames": default_window})
        start = max(0, block_start + restart_of(half, row) + int(round(float(
            degrade.sample(tag_params["start_offset_frames"], rng)))))
        stop = start + entry["window_frames"]
        name = NO_TEAM if team < 0 or rng.random() < entry["no_team_share"] else \
            match.team_names[team]
        label = kind if kind != KICK_OFF else "{} ({})".format(
            KICK_OFF, "Start" if row == 0 else "After Goal")
        out.append({"code": CODE, "start_frame": start, "end_frame": stop,
                    "start_s": start / 25.0, "end_s": stop / 25.0,
                    "tags": {"Team": name, "Type": label,
                             "Half": "{} Half".format("1st" if half_index == 0 else "2nd")}})
    return sorted(out, key=lambda tag: tag["start_frame"])
