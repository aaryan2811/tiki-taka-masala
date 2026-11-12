"""What our stoppages cost, and what is worth fixing: oracle ceilings, and each of our stoppages set
against the truth's. Reproduces the diagnosis in DESIGN.md section 4, "M2c".

    python solution/experiments/stoppage_ceiling.py --dfl-dir /path/to/dfl
    python solution/experiments/stoppage_ceiling.py --dfl-dir /path/to/dfl --seeds 5

Runs on the DFL calibration matches (devset/dfl.CALIBRATION) unless --matches names others: the
held-out five are kept for one check per milestone, and measuring on them would spend them. Each
match is degraded as run_devset.py degrades it, with synthetic tags. Two parts:

  ceilings  M2b (the pipeline with M2c's two rules off) with one thing replaced by the DFL truth,
            scored with the competition's metric under each truth variant, and M2c for
            comparison (learned_model=False throughout: these are the rules). The truth goes in
            through deadball.detect, which is swapped for a function returning DeadBalls made from
            it; everything else runs as usual.
            The truth's restart team is its team on the first frame after the dead ball.
              - our stoppages, with the truth's restart team, start, stop, or several of them, taken
                from the truth stoppage each overlaps most;
              - every truth stoppage, with and without its restart team;
              - our stoppages, with the truth's team on every frame of play (an open-play ceiling).
  pairs     each of our stoppages paired with the truth stoppage it overlaps most: restart team
            right / wrong / unknown, and start and stop as ours minus the truth's in seconds, by the
            restart DFL's events name (an event within 5 s of the truth's restart) and by evidence:
            for M2b, then for M2c, so M2c's before and after sit side by side.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devset import adapter, degrade, dfl, score, tags, truth  # noqa: E402  (needs the path above)
from devset.run_devset import match_seed, table  # noqa: E402
from possession import config, deadball  # noqa: E402
from possession.repo import pm  # noqa: E402

FPS = config.FPS
# label_frames flags: the pipeline before M2c, and M2c itself, both without M4's learned team
M2B = {"stoppage_starts": False, "restart_laws": False, "learned_model": False}
M2C = {"learned_model": False}
EVENT_REACH = 5 * FPS  # a set-piece event this close to the truth's restart names its type
EARLY = (1 * FPS, 3 * FPS)  # starts this much before the truth's are counted apart


def truth_stoppages(raw, blocks):
    """DeadBall per run of D in the raw truth, the team being the truth's on the frame after it
    (NOBODY when the half ends dead)."""
    out = []
    for block_start, block_stop in blocks:
        dead = raw[block_start:block_stop] == pm.D
        edges = np.flatnonzero(np.diff(np.concatenate(([0], dead.astype(np.int8), [0]))))
        for start, stop in zip(block_start + edges[::2], block_start + edges[1::2]):
            team = int(raw[stop]) if stop < block_stop and raw[stop] in (pm.H, pm.A) \
                else deadball.NOBODY
            out.append(deadball.DeadBall(start, stop, team, "truth"))
    return out


def overlap(a, b):
    return min(a.stop, b.stop) - max(a.start, b.start)


def pair(ours, oracle):
    """[(our stoppage, the truth stoppage it overlaps most)] for each of ours that overlaps one."""
    out = []
    for stoppage in ours:
        best = max(oracle, key=lambda other: overlap(stoppage, other))
        if overlap(stoppage, best) > 0:
            out.append((stoppage, best))
    return out


def with_stoppages(stoppages, tracking, blocks, tag_list):
    """Codes from the pipeline with deadball.detect returning `stoppages` instead of detecting."""
    saved = deadball.detect
    deadball.detect = lambda *args, **kwargs: stoppages
    try:
        return adapter.label(tracking, blocks, tag_list, **M2B)[0]
    finally:
        deadball.detect = saved


def team_in_play(codes, raw):
    """codes with every frame of play given the truth's team: the truth's H or A, carried forward
    over its dead balls."""
    team = np.where(np.isin(raw, (pm.H, pm.A)), raw, -1)
    seen = np.flatnonzero(team >= 0)
    index = np.maximum.accumulate(np.where(team >= 0, np.arange(len(team)), -1))
    index[index < 0] = seen[0]
    out = codes.copy()
    play = (codes != pm.D) & (codes != pm.X)
    out[play] = team[index][play]
    return out


def ceilings(tracking, blocks, tag_list, raw):
    """{name: codes} for every ceiling, M2b first."""
    codes, info = adapter.label(tracking, blocks, tag_list, **M2B)
    oracle = truth_stoppages(raw, blocks)
    pairs = pair(info["stoppages"], oracle)

    def mixed(start, stop, team, only=lambda s, t: True):
        return [deadball.DeadBall(t.start if start and only(s, t) else s.start,
                                  t.stop if stop else s.stop, t.team if team else s.team,
                                  s.evidence) for s, t in pairs]

    found = [t for t in oracle if any(overlap(s, t) > 0 for s in info["stoppages"])]
    no_team = [deadball.DeadBall(t.start, t.stop, deadball.NOBODY, t.evidence) for t in oracle]
    out = {"M2b": codes}
    for name, stoppages in (
            ("our stoppages, truth restart team", mixed(False, False, True)),
            ("our stoppages, truth start", mixed(True, False, False)),
            ("... only where we have no crossing", mixed(
                True, False, False, lambda s, t: "oob" not in s.evidence)),
            ("... only where ours is 10 s or more late", mixed(
                True, False, False, lambda s, t: s.start - t.start >= 10 * FPS)),
            ("our stoppages, truth stop", mixed(False, True, False)),
            ("our stoppages, truth start and stop", mixed(True, True, False)),
            ("truth stoppages we find, with their team", found),
            ("every truth stoppage, no restart team", no_team),
            ("every truth stoppage, with its team", oracle)):
        out[name] = with_stoppages(stoppages, tracking, blocks, tag_list)
    out["our stoppages, truth team in play"] = team_in_play(codes, raw)
    return out


def restart_type(pieces, frame):
    """The type of the set-piece event nearest `frame` within EVENT_REACH, else "no event"."""
    near = [(abs(at - frame), kind) for at, kind in pieces if abs(at - frame) <= EVENT_REACH]
    return min(near)[1] if near else "no event"


def pair_rows(stoppages, oracle, pieces):
    """One dict per pair: restart type, evidence, team verdict, start and stop offsets (s)."""
    rows = []
    for ours, theirs in pair(stoppages, oracle):
        verdict = ("unknown" if ours.team == deadball.NOBODY else
                   "right" if ours.team == theirs.team else "wrong")
        rows.append({"type": restart_type(pieces, theirs.stop), "evidence": ours.evidence,
                     "team": verdict, "start": (ours.start - theirs.start) / float(FPS),
                     "stop": (ours.stop - theirs.stop) / float(FPS)})
    return rows


def pair_table(rows, key):
    """Markdown table of the pairs grouped by `key`, plus all of them."""
    out = []
    for group in sorted(set(row[key] for row in rows)) + ["all"]:
        chosen = [row for row in rows if group == "all" or row[key] == group]
        start = np.array([row["start"] for row in chosen])
        stop = np.array([row["stop"] for row in chosen])
        out.append([group, len(chosen),
                    " / ".join(str(sum(row["team"] == verdict for row in chosen))
                               for verdict in ("right", "wrong", "unknown")),
                    " / ".join("{:+.1f}".format(v) for v in np.percentile(start, (25, 50, 75))),
                    " / ".join(str(int(np.sum(start < -early / float(FPS)))) for early in EARLY),
                    " / ".join("{:+.1f}".format(v) for v in np.percentile(stop, (25, 50, 75)))])
    return table([key, "pairs", "team right / wrong / unknown",
                  "start, ours - truth (s): q25 / median / q75",
                  "starts > 1 s / > 3 s early", "stop, ours - truth (s): q25 / median / q75"], out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--matches", nargs="+", default=list(dfl.CALIBRATION),
                        help="match ids (default: the calibration matches)")
    parser.add_argument("--seeds", type=int, default=2, help="degradation repeats per match")
    args = parser.parse_args()

    params = degrade.load_params()
    scored, rows = {}, {"M2b": [], "M2c": []}
    for match_id in args.matches:
        match = dfl.load(args.dfl_dir, match_id, verbose=False)
        truths = truth.variants(match)
        for seed in range(args.seeds):
            tracking, blocks = degrade.to_tracking(match, params, match_seed(match_id, seed))
            tag_list = tags.synthetic(match, blocks, params, match_seed(match_id, seed))
            oracle = truth_stoppages(truths["raw"], blocks)
            pieces = [(blocks[half][0] + row, kind) for half, row, kind, _ in match.set_pieces]
            for name, codes in ceilings(tracking, blocks, tag_list, truths["raw"]).items():
                for variant, gt in truths.items():
                    scored.setdefault((name, variant), [[] for _ in range(args.seeds)])[seed] \
                        .append(score.Scored(gt, codes))
            rows["M2b"] += pair_rows(adapter.label(tracking, blocks, tag_list, **M2B)[1]
                                     ["stoppages"], oracle, pieces)
            codes, info = adapter.label(tracking, blocks, tag_list, **M2C)
            rows["M2c"] += pair_rows(info["stoppages"], oracle, pieces)
            for variant, gt in truths.items():
                scored.setdefault(("M2c", variant), [[] for _ in range(args.seeds)])[
                    seed].append(score.Scored(gt, codes))
        print("done {}".format(match_id), file=sys.stderr)

    variants = [name for name, _ in truth.VARIANTS]
    names = list(dict.fromkeys(name for name, _ in scored))
    lines = []
    for name in names:
        cells = [score.mean_summary(scored[(name, variant)]) for variant in variants]
        lines.append([name] + ["{:.3f}".format(cell["spell_f1"]) for cell in cells]
                     + ["{:.2f} ({:.2f})".format(cells[-1]["pred_per_min"],
                                                 cells[-1]["gt_per_min"]),
                        "{:.0f} ({:.0f})".format(cells[0]["pred_stoppages"],
                                                 cells[0]["gt_stoppages"])])
    print("## Ceilings: {} ({} seeds, tags), spellF1\n".format(", ".join(args.matches), args.seeds))
    print(table(["pipeline"] + variants + ["poss/min, merged3s (truth)",
                                           "stoppages, all seeds (truth)"], lines))
    for title, chosen in rows.items():
        print("\n## Our stoppages against the truth's: {}, {} pairs\n".format(title, len(chosen)))
        print(pair_table(chosen, "type"))
        print()
        print(pair_table(chosen, "evidence"))


if __name__ == "__main__":
    main()
