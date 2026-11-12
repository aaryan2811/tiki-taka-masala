"""The stages composed: tracking in, submission codes out.

Kept free of file paths and printing so the same call runs on the real match from run.py and on
synthetic tracking in the tests.
"""

import numpy as np

from . import ball as ball_stage
from . import boundary, contest, control, deadball, keepers, labels, learned, segment
from .features import nearest_by_team, relative_speed


def rules(tracking, blocks, tags=(), boundaries=False, split_keepers=False, duels=False,
          foul_stoppages=True, stoppage_starts=True, restart_laws=True):
    """Stages 1-3 and 5, the hand-written rules (M1, M2, M2b, M2c): a dict of what each produced.

    tracking   the tracking every stage saw (keeper slots split only if split_keepers)
    ball       the repaired ball track, and interpolated, the frames the repair filled
    distance, slot, relative
               each team's nearest player to the ball, and how fast the ball moves past them
    near, on_ball
               per frame, the team nearest the ball within CONTROL_RADIUS, without and with the
               speed condition (control.NOBODY otherwise)
    touches, void
               the controlled touches that count, and the ones the duel rule took out
    stoppages  deadball.DeadBall, sorted
    owner, moved
               the per-frame team in possession, and the changes stage 5 moved

    label_frames documents the flags.
    """
    if split_keepers:
        tracking = keepers.split(tracking, blocks)
    ball, interpolated = ball_stage.repair(tracking.ball)
    distance, slot = nearest_by_team(ball, tracking.players, tracking.team_of_slot)
    relative = relative_speed(ball, tracking.players, slot)

    near = control.on_ball_team(distance)
    on_ball = control.on_ball_team(distance, relative)
    touch_list = control.touches(on_ball)
    void = []
    if duels:
        touch_list, void = contest.resolve(touch_list, distance)
    stoppages = deadball.detect(tracking, ball, blocks, tags, foul_stoppages, stoppage_starts,
                                restart_laws)
    owner = control.possession(touch_list, tracking.n_frames, blocks, stoppages)
    moved = []
    if boundaries:
        owner, moved = boundary.place_changes(owner, near, ball, stoppages)
    return {"tracking": tracking, "ball": ball, "interpolated": interpolated,
            "distance": distance, "slot": slot, "relative": relative, "near": near,
            "on_ball": on_ball, "touches": touch_list, "void": void, "stoppages": stoppages,
            "owner": owner, "moved": moved}


def label_frames(tracking, blocks, tags=(), boundaries=False, split_keepers=False, duels=False,
                 foul_stoppages=True, stoppage_starts=True, restart_laws=True, learned_model=True,
                 model=None):
    """(codes, info) for every frame of the recording.

    tags are the Smart Tagging instances (starter/load_smart_tagging.load()); without them the
    dead-ball detector confirms stoppages from line crossings and slowdowns alone.
    foul_stoppages=False confirms dead balls by crossings and tags only (M2's rule), to measure what
    it does.

    boundaries, split_keepers and duels are off, and the submission does not use them. They are
    M3's and M3b's rules (moving each change to the decisive touch, reading a keeper slot by the half
    it stands in, voiding a duel taken straight back), and they came out of comparing our output
    with the tactical video, which the organisers' clarification of the rules does not allow. They
    were withdrawn on 2026-11-12 (DESIGN.md section 4, "Withdrawn: the rules from the video") and
    stay only as switches for the experiments that measured them. stoppage_starts=False and
    restart_laws=False switch off M2c's two rules (deadball.detect): earlier stoppage starts, and
    the restart team from where the players stand.

    learned_model replaces the team in possession on every frame of play with M4's: the learned
    per-frame evidence (learned.py) segmented into possessions between the stoppages (segment.py).
    The stoppages and their restart teams stay as the rules found them. learned_model=False keeps
    the rules' team (M2c). model is a model dict as learned.load returns it; None reads the shipped
    one.

    info carries the intermediate numbers worth printing on every run, so a change in any stage
    shows up as a change in a number rather than only in the score.
    """
    found = rules(tracking, blocks, tags, boundaries, split_keepers, duels, foul_stoppages,
                  stoppage_starts, restart_laws)
    n_frames = tracking.n_frames
    owner, stoppages = found["owner"], found["stoppages"]
    if learned_model:
        model = model if model is not None else learned.load()
        play = learned.span_mask(n_frames, [(start, stop) for start, stop, _ in
                                            learned.play_stretches(n_frames, stoppages, blocks)])
        log_odds = np.zeros(n_frames)  # only play is decoded, so only play is scored
        log_odds[play] = learned.log_odds(learned.evidence(found, blocks)[play], model)
        owner = segment.decode(log_odds, owner, stoppages, blocks, model["switch_cost"],
                               model["min_frames"])
    dead = deadball.dead_mask(n_frames, stoppages)
    codes = labels.to_codes(owner, dead, n_frames, blocks)

    info = {
        "ball_raw": float(np.mean(~np.isnan(found["tracking"].ball[:, 0]))),
        "ball_repaired": float(np.mean(~np.isnan(found["ball"][:, 0]))),
        "interpolated_frames": int(found["interpolated"].sum()),
        "on_ball_frames": int(np.sum(found["on_ball"] != control.NOBODY)),
        "touches": len(found["touches"]),
        "void": found["void"],
        "changes": int(np.sum((owner[1:] != owner[:-1]) & (owner[1:] != control.NOBODY)
                              & (owner[:-1] != control.NOBODY))),
        "moved": found["moved"],
        "stoppages": stoppages,
        "learned": bool(learned_model),
    }
    return codes, info
