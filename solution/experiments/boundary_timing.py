"""Where our possession changes fall against the truth, and what the tracking can say about touches.
Reproduces the numbers in DESIGN.md section 4, "M3: boundary timing".

    python solution/experiments/boundary_timing.py

Needs the match data (like run.py). Three parts:

  per change   every true change in the public window against the nearest predicted change of the
               same kind, with stage 5 off (M2) and on (M3).
  oracle       how much spellF1 perfect timing could buy: each M2 change moved onto the nearest true
               change of the same kind within W frames. Uses the public truth, so it is a diagnosis
               and never a setting.
  contacts     whether a touch shows in the ball's own motion, over the whole match and without any
               ground truth: velocity change with a player near the ball against with nobody within
               5 m, frame by frame and across short proximity runs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import atd, boundary, config, control, labels, pipeline, repo  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession.features import nearest_by_team, velocity  # noqa: E402
from possession.repo import pm  # noqa: E402
from possession.runs import true_runs, value_runs  # noqa: E402

FAR = 5.0  # m: nobody can touch a ball this far from every player
NEAR = 1.5  # m
SPAN = 5  # frames either side for the frame-by-frame velocity change
MAX_RUN = 12  # frames: proximity runs this short are touches or passes going by, not dribbles


def changes(codes, runs):
    """[(frame, from, to)] for every label change inside the graded runs."""
    out = []
    for start, stop in runs:
        for frame in np.flatnonzero(codes[start + 1:stop] != codes[start:stop - 1]) + start + 1:
            out.append((int(frame), int(codes[frame - 1]), int(codes[frame])))
    return out


def nearest_offset(true_change, predicted, within=250):
    """Frames from a true change to the nearest predicted change of the same kind; None if there is
    none within `within` frames (10 s), which means the prediction has no counterpart for it."""
    frame, before, after = true_change
    offsets = [f - frame for f, b, a in predicted if b == before and a == after]
    offsets = [o for o in offsets if abs(o) <= within]
    return min(offsets, key=abs) if offsets else None


def snapped(codes, gt, runs, window):
    """codes rebuilt with each change moved onto the nearest true change of the same kind within
    window, segments keeping their order and labels."""
    truth = changes(gt, runs)
    out = codes.copy()
    for run_start, run_stop in runs:
        segments = pm.segments(codes, [(run_start, run_stop)])
        bounds = [run_start]
        for (start, _, code), (_, _, before) in zip(segments[1:], segments[:-1]):
            same = [f for f, b, a in truth if b == before and a == code and abs(f - start) <= window]
            bounds.append(min(same, key=lambda f: abs(f - start)) if same else start)
        bounds.append(run_stop)
        for (_, _, code), low, high in zip(segments, bounds[:-1], bounds[1:]):
            out[low:high] = code
    return out


def main():
    tracking = atd.load(verbose=False)
    tags = repo.load_smart_tagging.load()
    blocks = labels.graded_blocks()
    gt = pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT))
    runs = pm.valid_runs(gt)

    m2, _ = pipeline.label_frames(tracking, blocks, tags, boundaries=False, learned_model=False)
    m3, info = pipeline.label_frames(tracking, blocks, tags, learned_model=False)
    m2, m3 = pm.fill_abstentions(m2, runs), pm.fill_abstentions(m3, runs)

    print("per change  (offset of the nearest predicted change of the same kind, frames)")
    print("  {:>6}  {:<5} {:>6} {:>6}".format("truth", "", "M2", "M3"))
    predicted = (changes(m2, runs), changes(m3, runs))
    for true_change in changes(gt, runs):
        offsets = [nearest_offset(true_change, p) for p in predicted]
        print("  {:>6}  {}->{} {:>6} {:>6}".format(
            true_change[0], pm.LABELS[true_change[1]], pm.LABELS[true_change[2]],
            *["--" if o is None else "{:+d}".format(o) for o in offsets]))

    print("\noracle  (M2 changes moved onto the truth within W frames)")
    for window in (0, 25, 50, 100, 200):
        result = pm.evaluate(gt, snapped(m2, gt, runs, window), bootstrap_n=0)
        print("  W {:>3}  spellF1 {:.3f}".format(window, result["spell_f1_mean"]))

    ball, _ = ball_stage.repair(tracking.ball)
    distance, _ = nearest_by_team(ball, tracking.players, tracking.team_of_slot)
    nearest = np.min(np.where(np.isnan(distance), np.inf, distance), axis=1)
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True

    raw = tracking.ball
    raw_velocity = velocity(raw)
    seen = np.concatenate(([0], np.cumsum(~np.isnan(raw[:, 0]))))
    index = np.arange(SPAN + 1, tracking.n_frames - SPAN - 1)
    change = np.full(tracking.n_frames, np.nan)
    change[index] = np.hypot(*(raw_velocity[index + SPAN] - raw_velocity[index - SPAN]).T)
    whole = np.zeros(tracking.n_frames, dtype=bool)
    whole[index] = seen[index + SPAN + 2] - seen[index - SPAN - 1] == 2 * SPAN + 3
    usable = graded & whole & np.isfinite(nearest)
    print("\ncontacts, frame by frame  (raw ball seen throughout, velocity change over +/-{} "
          "frames, m/s)".format(SPAN))
    for name, mask in (("nobody within {:.0f} m".format(FAR), usable & (nearest > FAR)),
                       ("player within {:.1f} m".format(NEAR), usable & (nearest < NEAR))):
        print("  {:<20} n {:>6}  median {:4.1f}  p90 {:4.1f}  p95 {:4.1f}".format(
            name, int(mask.sum()), *np.percentile(change[mask], [50, 90, 95])))

    ball_velocity = velocity(ball)
    near = control.on_ball_team(distance)
    short = [(start, stop) for start, stop, team in value_runs(near)
             if team != control.NOBODY and graded[start] and stop - start <= MAX_RUN]
    across = np.array([boundary.velocity_change(ball_velocity, a, b) for a, b in short])
    across = across[~np.isnan(across)]

    rng = np.random.RandomState(0)
    lengths = np.array([b - a for a, b in short])
    far = []
    for start, stop in zip(*true_runs(graded & (nearest > FAR))):
        for frame in range(start + config.CONTACT_SPAN, stop - config.CONTACT_SPAN - MAX_RUN, 7):
            far.append(boundary.velocity_change(ball_velocity, frame,
                                                frame + int(rng.choice(lengths))))
    far = np.array(far)
    far = far[~np.isnan(far)]
    print("\ncontacts, across a run  (velocity change from just before to just after, m/s)")
    print("  proximity runs of up to {} frames: n {}; windows as long with nobody within {:.0f} m: "
          "n {}, median {:.1f}, p75 {:.1f}".format(MAX_RUN, len(across), FAR, len(far),
                                                   np.median(far), np.percentile(far, 75)))
    for threshold in (2.0, 3.0, config.CONTACT_MIN_CHANGE, 5.0, 6.0):
        print("  change >= {:.0f} m/s: {:.0%} of runs, {:.0%} of far windows".format(
            threshold, np.mean(across >= threshold), np.mean(far >= threshold)))

    shifts = np.array([old - new for old, new in info["moved"]])
    print("\nstage 5 on the whole match: {} of {} changes moved, median {:.0f} frames, "
          "90th percentile {:.0f}".format(len(shifts), info["changes"], np.median(shifts),
                                          np.percentile(shifts, 90)))


if __name__ == '__main__':
    main()
