"""Where M2c's numbers come from: who restarts a stoppage by the laws of the game, and when it
started. Reproduces the measurements next to the M2c settings in possession/config.py and the rule
tables in DESIGN.md section 4, "M2c".

    python solution/experiments/restart_rules.py --dfl-dir /path/to/dfl

Calibration matches only by default (devset/dfl.CALIBRATION), degraded as run_devset.py degrades
them, 2 seeds, with synthetic tags. The rules are measured on M2b's own stoppages, each paired with
the truth stoppage it overlaps most (experiments/stoppage_ceiling.py), so that "before" is M2b and
"after" changes one thing. Five parts:

  last touch  the law the brief started from: a throw-in goes to the team that did not touch the
              ball last, a corner or goal kick by who touched it last. Three readings of the last
              touch before a crossing, none of them better than a coin on this tracking.
  team        the restart team by rule, by the line the ball crossed and by restart type: M2b's
              (nearest player at the kick, else the tag), the law rule alone (thrower_team,
              goal_line_team), M2c's (law rule, then M2b's), and the team the pipeline carries
              through an unknown restart (the one it had before the stoppage).
  geometry    what the law rules read: how far inside its touchline the thrower stands and how far
              along it from the crossing; attackers in the penalty area at corners and goal kicks.
  speed       collective speed in dead time and open play (OPEN_PLAY_SPEED), and at the true start
              of stoppages with no crossing (STOP_SPEED); how late the slowdown begins.
  starts      what out_since and whistle move: starts moved, by how much, the truth's dead and
              play frames newly marked dead, and starts more than 1 s / 3 s before the truth's.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from devset import adapter, degrade, dfl, tags, truth  # noqa: E402  (needs the paths above)
from devset.run_devset import match_seed, table  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession import boundary, config, contest, control, deadball, keepers  # noqa: E402
from possession.features import nearest_by_team, relative_speed, velocity  # noqa: E402
from possession.repo import pm  # noqa: E402
from possession.runs import true_runs  # noqa: E402
from stoppage_ceiling import M2B, pair, restart_type, truth_stoppages  # noqa: E402

FPS = config.FPS
LAST_TOUCH_BACK = 10 * FPS  # how far before a crossing a last touch is looked for
VERDICTS = ("right", "wrong", "unknown")


class Seen(object):
    """One degraded match as the dead-ball stage sees it, plus M2b's output and the truth."""

    def __init__(self, match, params, seed):
        self.match_id, self.seed = match.match_id, seed
        tracking, blocks = degrade.to_tracking(match, params, match_seed(match.match_id, seed))
        tag_list = tags.synthetic(match, blocks, params, match_seed(match.match_id, seed))
        self.codes, info = adapter.label(tracking, blocks, tag_list, **M2B)
        self.stoppages = info["stoppages"]
        self.with_laws = adapter.label(tracking, blocks, tag_list, stoppage_starts=False,
                                       restart_laws=True, learned_model=False)[1]["stoppages"]
        self.raw = truth.variants(match)["raw"]
        self.oracle = truth_stoppages(self.raw, blocks)
        self.pieces = [(blocks[half][0] + row, kind) for half, row, kind, _ in match.set_pieces]

        split = keepers.split(tracking, blocks)
        self.tracking, self.blocks = split, blocks
        self.ball, _ = ball_stage.repair(split.ball)
        distance, slot = nearest_by_team(self.ball, split.players, split.team_of_slot)
        self.near = control.on_ball_team(distance)
        on_ball = control.on_ball_team(distance, relative_speed(self.ball, split.players, slot))
        self.touches, _ = contest.resolve(control.touches(on_ball), distance)
        self.contacts = boundary.contacts(self.near, velocity(self.ball, FPS))
        self.collective = deadball.collective_speed(split.speed)
        self.crossings = deadball.line_crossings(self.ball, split.pitch)
        graded = np.zeros(split.n_frames, dtype=bool)
        for start, stop in blocks:
            graded[start:stop] = True
        self.graded = graded
        self.candidates = deadball.slow_runs(self.collective, graded)
        self.home_low = keepers.home_defends_low_x(split.n_frames, blocks,
                                                   split.home_attacks_right_first_half)

    def crossing_of(self, stoppage):
        """The crossing M2b started this stoppage at, or None."""
        return stoppage.start if stoppage.start in set(self.crossings.tolist()) else None

    def slowdown_of(self, stoppage):
        """The slowdown candidate this stoppage came from (the first one overlapping it)."""
        for start, stop in self.candidates:
            if start < stoppage.stop and stop > stoppage.start:
                return start, stop
        return stoppage.start, stoppage.stop

    def carried(self, stoppage):
        """The team the pipeline has before the stoppage, which an unknown restart carries."""
        code = self.codes[stoppage.start - 1]
        return int(code) if code in (pm.H, pm.A) else deadball.NOBODY

    def law(self, stoppage, crossing):
        """The law rule's team alone, exactly as detect asks for it."""
        frame = min(stoppage.stop, self.tracking.n_frames - 1)
        players, teams, pitch = self.tracking.players, self.tracking.team_of_slot, \
            self.tracking.pitch
        if crossing is None:
            return deadball.thrower_team(players, teams, pitch, frame), "no crossing"
        line, end, x = deadball.crossed_line(self.ball, pitch, crossing)
        if line == deadball.GOAL_LINE:
            return deadball.goal_line_team(players, teams, pitch, frame, end,
                                           self.home_low[crossing]), line
        return deadball.thrower_team(players, teams, pitch, frame, end, x), line


def verdict(team, truth_team):
    if team == deadball.NOBODY:
        return "unknown"
    return "right" if team == truth_team else "wrong"


def counts(verdicts):
    tally = Counter(verdicts)
    return " / ".join(str(tally[v]) for v in VERDICTS)


# -- last touch ------------------------------------------------------------------------------------

def last_of(runs, frame, back=LAST_TOUCH_BACK):
    """Team of the last run [(start, stop, team)] beginning in the `back` frames before frame."""
    chosen = [team for start, _, team in runs if frame - back <= start < frame]
    return chosen[-1] if chosen else deadball.NOBODY


def by_last_touch(line, toucher, defending):
    """The restart team the laws give from who touched the ball last."""
    if toucher == deadball.NOBODY:
        return deadball.NOBODY
    if line == deadball.TOUCHLINE:
        return 1 - toucher
    return 1 - defending if toucher == defending else defending


def last_touch_part(seen):
    rows = []
    for match in seen:
        near_runs = [(f, f + 1, int(match.near[f])) for f in np.flatnonzero(match.near >= 0)]
        for ours, theirs in pair(match.stoppages, match.oracle):
            crossing = match.crossing_of(ours)
            if crossing is None:
                continue
            line, end, _ = deadball.crossed_line(match.ball, match.tracking.pitch, crossing)
            defending = 0 if bool(match.home_low[crossing]) == (end == 0) else 1
            row = {"line": line}
            for name, runs in (("nearest within 2 m", near_runs),
                               ("velocity-change contact", match.contacts),
                               ("controlled touch", match.touches)):
                team = by_last_touch(line, last_of(runs, crossing), defending)
                row[name] = verdict(team, theirs.team)
            rows.append(row)
    out = []
    for name in ("nearest within 2 m", "velocity-change contact", "controlled touch"):
        out.append([name] + [counts(row[name] for row in rows if row["line"] == line)
                             for line in (deadball.TOUCHLINE, deadball.GOAL_LINE)])
    return "## Last touch before a crossing, read three ways (right / wrong / unknown)\n\n" + table(
        ["last touch", "touchline crossings", "goal-line crossings"], out)


# -- team ------------------------------------------------------------------------------------------

def team_part(seen):
    rows = []
    for match in seen:
        law_pairs = dict((theirs.start, ours) for ours, theirs in pair(match.with_laws,
                                                                       match.oracle))
        for ours, theirs in pair(match.stoppages, match.oracle):
            crossing = match.crossing_of(ours)
            team, line = match.law(ours, crossing)
            after = law_pairs.get(theirs.start)
            rows.append({"line": line, "type": restart_type(match.pieces, theirs.stop),
                         "M2b": verdict(ours.team, theirs.team),
                         "law rule": verdict(team, theirs.team),
                         "M2c": verdict(after.team, theirs.team) if after else "unpaired",
                         "carried": verdict(match.carried(ours), theirs.team),
                         "carried, M2b unknown": verdict(match.carried(ours), theirs.team)
                         if ours.team == deadball.NOBODY else None,
                         "law rule, M2b unknown": verdict(team, theirs.team)
                         if ours.team == deadball.NOBODY else None})
    lines = []
    for line in (deadball.TOUCHLINE, deadball.GOAL_LINE, "no crossing"):
        for kind in sorted(set(row["type"] for row in rows if row["line"] == line)) + ["all"]:
            chosen = [row for row in rows if row["line"] == line
                      and (kind == "all" or row["type"] == kind)]
            lines.append([line if kind == "all" else "", kind, len(chosen)]
                         + [counts(row[name] for row in chosen)
                            for name in ("M2b", "law rule", "M2c")]
                         + [counts(row[name] for row in chosen if row[name] is not None)
                            for name in ("carried, M2b unknown", "law rule, M2b unknown")])
    lines.append(["all", "all", len(rows)] + [counts(row[name] for row in rows)
                                             for name in ("M2b", "law rule", "M2c")]
                 + [counts(row[name] for row in rows if row[name] is not None)
                    for name in ("carried, M2b unknown", "law rule, M2b unknown")])
    flips = Counter((row["M2b"], row["law rule"]) for row in rows
                    if row["M2b"] != "unknown" and row["law rule"] != "unknown")
    return ("## Restart team by rule, on M2b's stoppages (right / wrong / unknown)\n\n" + table(
        ["crossing", "restart type", "pairs", "M2b: nearest, then tag", "law rule alone",
         "M2c: law rule, then M2b's", "carried, where M2b is unknown",
         "law rule, where M2b is unknown"], lines)
        + "\n\nWhere both name a team, M2b's verdict against the law rule's: "
        + ", ".join("{} / {}: {}".format(a, b, n) for (a, b), n in sorted(flips.items())))


# -- geometry --------------------------------------------------------------------------------------

def geometry_part(seen):
    inside, along, attackers = [], [], {}
    stops = []
    for match in seen:
        players, teams = match.tracking.players, match.tracking.team_of_slot
        length, width = match.tracking.pitch
        for ours, theirs in pair(match.stoppages, match.oracle):
            stops.append((ours.stop - theirs.stop) / float(FPS))
            kind = restart_type(match.pieces, theirs.stop)
            crossing = match.crossing_of(ours)
            frame = min(ours.stop, match.tracking.n_frames - 1)
            if crossing is None:
                continue
            line, end, x = deadball.crossed_line(match.ball, match.tracking.pitch, crossing)
            if line == deadball.TOUCHLINE and kind == "Throw-in":
                span = players[max(0, frame - config.THROW_WINDOW):frame + 1]
                y = span[..., 1]
                gap = np.where((y > width / 2) == bool(end), np.minimum(y, width - y), np.nan)
                gap = np.where(teams[None, :] == theirs.team, gap, np.nan)
                if not np.isnan(gap).all():
                    at = np.unravel_index(np.nanargmin(gap), gap.shape)
                    inside.append(float(gap[at]))
                    along.append(float(abs(span[at][0] - x)))
            if line == deadball.GOAL_LINE and kind in ("Corner Kick", "Goal Kick"):
                defending = 0 if bool(match.home_low[crossing]) == (end == 0) else 1
                px, py = players[frame, :, 0], players[frame, :, 1]
                with np.errstate(invalid="ignore"):
                    box = ((np.abs(px - (length if end else 0.0)) <= config.PENALTY_AREA_DEPTH)
                           & (np.abs(py - width / 2) <= config.PENALTY_AREA_HALF_WIDTH))
                attackers.setdefault(kind, []).append(int(np.sum(box & (teams == 1 - defending))))
    q = (10, 25, 50, 75, 90)
    lines = ["## What the law rules read", "",
             "Throw-ins after a touchline crossing ({}): the throwing team's player nearest that "
             "touchline in the {} frames up to our restart".format(len(inside),
                                                                   config.THROW_WINDOW),
             "- metres inside the line, q10 / q25 / median / q75 / q90: " + " / ".join(
                 "{:+.2f}".format(v) for v in np.percentile(inside, q)),
             "- within {} m: {:.0%}".format(config.THROW_LINE,
                                            np.mean(np.array(inside) <= config.THROW_LINE)),
             "- metres along the line from the crossing, median / q90 / max: {:.1f} / {:.1f} / "
             "{:.1f}; within {:.0f} m: {:.0%}".format(
                 np.median(along), np.percentile(along, 90), np.max(along), config.THROW_REACH,
                 np.mean(np.array(along) <= config.THROW_REACH)),
             "",
             "Attackers in the penalty area at the restart, after a goal-line crossing:"]
    for kind, values in sorted(attackers.items()):
        lines.append("- {} ({}): {}".format(kind, len(values), ", ".join(
            "{} x{}".format(k, n) for k, n in sorted(Counter(values).items()))))
    lines += ["", "Our restart minus the truth's, all pairs (s), q25 / median / q75: " + " / ".join(
        "{:+.1f}".format(v) for v in np.percentile(stops, (25, 50, 75)))]
    return "\n".join(lines)


# -- speed -----------------------------------------------------------------------------------------

def speed_part(seen):
    dead, play, dead_max, play_max = [], [], [], []
    at_start, lag = [], {}
    for match in seen:
        graded, raw, collective = match.graded, match.raw, match.collective
        dead.append(collective[graded & (raw == pm.D)])
        play.append(collective[graded & (raw != pm.D)])
        for start, stop in zip(*true_runs(graded & (raw == pm.D))):
            if stop - start >= 10 * FPS:
                dead_max.append(np.nanmax(collective[start + 2 * FPS:stop]))
        for start, stop in zip(*true_runs(graded & (raw != pm.D))):
            for at in range(start, stop - 10 * FPS, 10 * FPS):
                play_max.append(np.nanmax(collective[at:at + 10 * FPS]))
        for ours, theirs in pair(match.stoppages, match.oracle):
            if "oob" in ours.evidence:
                continue
            slow_start, _ = match.slowdown_of(ours)
            lag.setdefault(ours.evidence, []).append((slow_start - theirs.start) / float(FPS))
            if theirs.start < slow_start:
                at_start.append(float(collective[theirs.start]))
    dead, play = np.concatenate(dead), np.concatenate(play)
    q = (5, 10, 25, 50, 75, 90, 95)
    lines = ["## Collective speed (m/s)", "",
             table(["", "q5", "q10", "q25", "median", "q75", "q90", "q95"], [
                 ["dead frames"] + ["{:.2f}".format(v) for v in np.nanpercentile(dead, q)],
                 ["open-play frames"] + ["{:.2f}".format(v) for v in np.nanpercentile(play, q)],
                 ["highest in a stoppage of 10 s or more, after its first 2 s ({})".format(
                     len(dead_max))] + ["{:.2f}".format(v) for v in np.percentile(dead_max, q)],
                 ["highest in 10 s of open play ({})".format(len(play_max))]
                 + ["{:.2f}".format(v) for v in np.percentile(play_max, q)],
                 ["at the true start, no crossing, truth before our slowdown ({})".format(
                     len(at_start))] + ["{:.2f}".format(v) for v in np.percentile(at_start, q)]]),
             "",
             "- stoppages of 10 s or more staying under {} m/s after their first 2 s: "
             "{:.0%}".format(config.OPEN_PLAY_SPEED,
                             np.mean(np.array(dead_max) < config.OPEN_PLAY_SPEED)),
             "- 10 s stretches of open play reaching {} m/s: {:.0%}".format(
                 config.OPEN_PLAY_SPEED, np.mean(np.array(play_max) >= config.OPEN_PLAY_SPEED)),
             "- slowdown start minus the true start, no crossing (s), q25 / median / q75: "
             + "; ".join(
                 "{} {}".format(evidence, " / ".join("{:+.1f}".format(v) for v in
                                                    np.percentile(values, (25, 50, 75))))
                 for evidence, values in sorted(lag.items()))]
    return "\n".join(lines)


# -- starts ----------------------------------------------------------------------------------------

def starts_part(seen):
    out = []
    for name in ("out_since", "out_since, 120 s back", "whistle"):
        moved, gained, lost, early = [], 0, 0, Counter()
        for match in seen:
            ball, pitch = match.ball, match.tracking.pitch
            paired = dict((id(ours), theirs) for ours, theirs in pair(match.stoppages,
                                                                       match.oracle))
            for ours in match.stoppages:
                theirs = paired.get(id(ours))
                if ours.evidence == "known":
                    continue
                if name.startswith("out_since"):
                    back = config.OUT_SINCE_BACK * (2 if "120" in name else 1)
                    new = deadball.out_since(ball, pitch, match.collective, match.crossings,
                                             ours.start, back=back)
                elif match.crossing_of(ours) is None:
                    new = deadball.whistle(match.collective, match.slowdown_of(ours)[0])
                else:
                    new = None
                if new is None or new >= ours.start:
                    continue
                span = match.raw[new:ours.start]
                gained += int(np.sum(span == pm.D))
                lost += int(np.sum(span != pm.D))
                moved.append((ours.start - new) / float(FPS))
                if theirs is not None:
                    for limit in (1, 3):
                        early[limit] += (new - theirs.start) / float(FPS) < -limit
                        before = (ours.start - theirs.start) / float(FPS)
                        early[(limit, "before")] += before < -limit
        out.append([name, len(moved),
                    "{:.1f} / {:.1f} / {:.1f}".format(*np.percentile(moved, (10, 50, 90)))
                    if moved else "-", gained, lost,
                    "{:.0%}".format(gained / float(gained + lost)) if moved else "-",
                    "{} / {} (from {} / {})".format(early[1], early[3], early[(1, "before")],
                                                    early[(3, "before")])])
    return "## What the start rules move, each alone on M2b's stoppages\n\n" + table(
        ["rule", "starts moved", "moved by (s), q10 / median / q90", "truth dead frames added",
         "truth play frames added", "dead share", "moved starts > 1 s / > 3 s early"], out) \
        + "\n\n" + last_sighting(seen)


def last_sighting(seen, near_line=10.0, outward=2.0):
    """The signal not used: the ball's last sighting before a slowdown with no crossing, and, where
    it was heading out within near_line metres of a line at outward m/s or more, where the straight
    line would have crossed against the true start."""
    ago, misses = [], []
    for match in seen:
        ball, (length, width) = match.ball, match.tracking.pitch
        ball_velocity = velocity(ball, FPS)
        for ours, theirs in pair(match.stoppages, match.oracle):
            if "oob" in ours.evidence or ours.evidence == "known":
                continue
            slow_start, _ = match.slowdown_of(ours)
            seen_at = np.flatnonzero(~np.isnan(ball[:slow_start, 0]))
            if not len(seen_at):
                continue
            last = int(seen_at[-1])
            ago.append((slow_start - last) / float(FPS))
            (x, y), (vx, vy) = ball[last], ball_velocity[last]
            lines = ((x, -vx), (length - x, vx), (y, -vy), (width - y, vy))
            distance, speed = min(lines, key=lambda line: line[0])
            if distance <= near_line and speed >= outward:
                misses.append((last + distance / speed * FPS - theirs.start) / float(FPS))
    return ("The ball's last sighting before a slowdown with no crossing ({} stoppages): q25 / "
            "median / q75 {} s before it. Heading out within {:.0f} m of a line at {:.0f} m/s or "
            "more ({}): the straight-line crossing minus the true start, q10 / median / q90: {} s; "
            "more than 3 s off: {}.".format(
                len(ago), " / ".join("{:.1f}".format(v) for v in np.percentile(ago, (25, 50, 75))),
                near_line, outward, len(misses),
                " / ".join("{:+.1f}".format(v) for v in np.percentile(misses, (10, 50, 90))),
                int(np.sum(np.abs(misses) > 3))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--matches", nargs="+", default=list(dfl.CALIBRATION),
                        help="match ids (default: the calibration matches)")
    parser.add_argument("--seeds", type=int, default=2, help="degradation repeats per match")
    args = parser.parse_args()

    params = degrade.load_params()
    seen = []
    for match_id in args.matches:
        match = dfl.load(args.dfl_dir, match_id, verbose=False)
        seen += [Seen(match, params, seed) for seed in range(args.seeds)]
        print("done {}".format(match_id), file=sys.stderr)
    print("# {} ({} seeds)\n".format(", ".join(args.matches), args.seeds))
    for part in (last_touch_part, team_part, geometry_part, speed_part, starts_part):
        print(part(seen))
        print()


if __name__ == "__main__":
    main()
