"""Open play: what the team in possession is worth, and three cheap routes to it that failed.
Reproduces the diagnosis at the start of DESIGN.md section 4, "M4".

    python3 -I solution/experiments/open_play.py --dfl-dir /path/to/dfl
    python3 -I solution/experiments/open_play.py --dfl-dir /path/to/dfl --sweep

Runs on the DFL calibration matches (devset/dfl.CALIBRATION) unless --matches names others: the
held-out five are kept for one check per milestone. Each match is degraded as run_devset.py degrades
it, 2 seeds, with synthetic tags, and M2c (the rules, learned_model=False) runs on it. Four parts:

  ceilings      M2c with the merged 3 s truth's team on its play frames: all of them, only those
                with the ball seen, only those with it unseen (stoppage_ceiling.team_in_play). Upper
                bounds, since the merged truth's own team goes in.
  accuracy      on frames both M2c and the merged 3 s truth call play, how often M2c names the
                truth's team, with the ball seen and unseen.
  pass chains   two velocity-change contacts in a row by one team (boundary.contacts), at most
                CHAIN_GAP apart, with the ball seen between: how often the truth agrees, and how
                often where M2c names the other team. Then a crude rule that hands the rest of
                M2c's spell to the chain's team.
  team shape    a gradient-boosting classifier on team shape (centroids along the attack direction,
                spreads, speeds, tracked counts) and on how long ago and where the ball was last
                seen, with and without M2c's owner. Trained on one calibration match, tested on the
                other. Its 3 s-smoothed team is pasted onto unseen-ball stretches of play of at
                least 2 s or 5 s.
  --sweep       M2c with other controlled-touch thresholds: radius, maximum relative speed and
                minimum frames (config.CONTROL_*), seed 0.
"""

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from devset import adapter, degrade, dfl, score, tags, truth  # noqa: E402  (needs the paths above)
from devset.run_devset import match_seed, table  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession import boundary, config, control, keepers  # noqa: E402
from possession.features import nearest_by_team, team_shape, velocity  # noqa: E402
from possession.repo import pm  # noqa: E402
from possession.runs import true_runs  # noqa: E402
from stoppage_ceiling import team_in_play  # noqa: E402

FPS = config.FPS
CHAIN_GAP = 3 * FPS  # frames between the two contacts of a chain, at most
SMOOTH = 3 * FPS  # the classifier's probability is averaged over this many frames
PASTE_MIN = (2 * FPS, 5 * FPS)  # unseen stretches at least this long get the classifier's team
SWEEP = list(itertools.product((2.0, 2.5, 3.0), (3.0, 5.0, 8.0), (3, 5)))


class Seen(object):
    """One degraded match as the touch rule sees it, plus M2c's codes."""

    def __init__(self, match, params, seed):
        tracking, blocks = degrade.to_tracking(match, params, match_seed(match.match_id, seed))
        tag_list = tags.synthetic(match, blocks, params, match_seed(match.match_id, seed))
        self.codes, _ = adapter.label(tracking, blocks, tag_list, learned_model=False)
        self.tracking, self.blocks = keepers.split(tracking, blocks), blocks
        self.ball, _ = ball_stage.repair(self.tracking.ball)
        distance, self.slot = nearest_by_team(self.ball, self.tracking.players,
                                              self.tracking.team_of_slot)
        self.contacts = boundary.contacts(control.on_ball_team(distance),
                                          velocity(self.ball, FPS))


def chains(seen):
    """[(first start, second start, team, distinct slots)]: consecutive contacts by one team at most
    CHAIN_GAP apart with the ball seen between them. distinct is whether the two contacts were made
    by different track slots, a pass rather than one player's dribble."""
    out = []
    visible = ~np.isnan(seen.ball[:, 0])
    for (start1, stop1, team1), (start2, _, team2) in zip(seen.contacts, seen.contacts[1:]):
        if team1 != team2 or start2 - stop1 > CHAIN_GAP or not visible[stop1:start2].all():
            continue
        distinct = seen.slot[start1, team1] != seen.slot[start2, team2]
        out.append((start1, start2, team1, bool(distinct)))
    return out


def apply_chains(codes, chain_list, distinct_only=True):
    """codes with M2c's spell at each chain's second contact handed to the chain's team, from the
    chain's first contact to the end of that spell."""
    out = codes.copy()
    for start1, start2, team, distinct in chain_list:
        if distinct_only and not distinct:
            continue
        current = out[start2]
        if current not in (pm.H, pm.A) or current == team:
            continue
        stop = start2
        while stop < len(out) and out[stop] == current:
            stop += 1
        out[start1:stop] = np.where(np.isin(out[start1:stop], (pm.H, pm.A)), team,
                                    out[start1:stop])
    return out


def shape_features(seen, with_owner):
    """(n, k) per-frame features for the team-shape classifier, along the home team's attack."""
    tracking, n = seen.tracking, seen.tracking.n_frames
    shape = team_shape(tracking.players, tracking.speed, tracking.team_of_slot)
    home_low = keepers.home_defends_low_x(n, seen.blocks, tracking.home_attacks_right_first_half)
    sign = np.where(home_low, 1.0, -1.0)  # +x is home's attack when home defends low x
    middle = tracking.pitch[0] / 2.0
    along = (shape["centroid"][:, :, 0] - middle) * sign[:, None]
    visible = ~np.isnan(seen.ball[:, 0])
    last = np.maximum.accumulate(np.where(visible, np.arange(n), -1))
    since = np.where(last >= 0, np.arange(n) - last, 10000) / float(FPS)
    last_x = np.where(last >= 0, (seen.ball[np.maximum(last, 0), 0] - middle) * sign, 0.0)
    columns = [along[:, 0], along[:, 1], along[:, 0] - along[:, 1],
               shape["spread"][:, 0, 0], shape["spread"][:, 0, 1],
               shape["spread"][:, 1, 0], shape["spread"][:, 1, 1],
               shape["mean_speed"][:, 0], shape["mean_speed"][:, 1],
               shape["n_tracked"][:, 0].astype(float), shape["n_tracked"][:, 1].astype(float),
               np.minimum(since, 60.0), last_x]
    if with_owner:
        columns.append(np.where(seen.codes == pm.H, 1.0, np.where(seen.codes == pm.A, -1.0, 0.0)))
    return np.column_stack(columns).astype(np.float32)


def running_mean(values, width=SMOOTH):
    return np.convolve(values, np.ones(width) / width, mode="same")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--matches", nargs="+", default=list(dfl.CALIBRATION),
                        help="match ids (default: the calibration matches)")
    parser.add_argument("--seeds", type=int, default=2, help="degradation repeats per match")
    parser.add_argument("--sweep", action="store_true", help="also sweep the touch thresholds")
    args = parser.parse_args()

    params = degrade.load_params()
    seen, truths = {}, {}
    for match_id in args.matches:
        match = dfl.load(args.dfl_dir, match_id, verbose=False)
        truths[match_id] = truth.variants(match)
        for seed in range(args.seeds):
            seen[(match_id, seed)] = Seen(match, params, seed)
        print("prepared {}".format(match_id), file=sys.stderr)

    scored = {}

    def add(name, key, codes):
        for variant, gt in truths[key[0]].items():
            scored.setdefault((name, variant), [[] for _ in range(args.seeds)])[key[1]].append(
                score.Scored(gt, codes))

    accuracy = {"seen": [0, 0], "unseen": [0, 0]}
    chain_rows = []
    for key, one in seen.items():
        codes, gt = one.codes, truths[key[0]]["merged3s"]
        play = np.isin(codes, (pm.H, pm.A))
        both = play & np.isin(gt, (pm.H, pm.A))
        unseen = np.isnan(one.ball[:, 0])
        for name, mask in (("seen", both & ~unseen), ("unseen", both & unseen)):
            accuracy[name][0] += int(np.sum(codes[mask] == gt[mask]))
            accuracy[name][1] += int(mask.sum())
        oracle = team_in_play(codes, gt)
        add("M2c", key, codes)
        add("truth's team, all play", key, oracle)
        add("truth's team, ball unseen only", key, np.where(play & unseen, oracle, codes))
        add("truth's team, ball seen only", key, np.where(play & ~unseen, oracle, codes))
        chain_list = chains(one)
        chain_rows += [(distinct, codes[second] == team, gt[second] == team,
                        gt[second] in (pm.H, pm.A))
                       for _, second, team, distinct in chain_list]
        add("pass chains (distinct slots)", key, apply_chains(codes, chain_list, True))
        add("pass chains (any)", key, apply_chains(codes, chain_list, False))

    classifier = classify(seen, truths, add)

    variants = [name for name, _ in truth.VARIANTS]
    rows = []
    for name in dict.fromkeys(name for name, _ in scored):
        cells = [score.mean_summary(scored[(name, variant)]) for variant in variants]
        rows.append([name] + ["{:.3f}".format(cell["spell_f1"]) for cell in cells]
                    + ["{:.2f} ({:.2f})".format(cells[-1]["pred_per_min"],
                                                cells[-1]["gt_per_min"])])
    print("## Open play: {} ({} seeds, tags), spellF1\n".format(", ".join(args.matches),
                                                                args.seeds))
    print(table(["pipeline"] + variants + ["poss/min, merged3s (truth)"], rows))

    total = sum(n for _, n in accuracy.values())
    print("\n## Team on play frames (M2c and the merged 3 s truth both call it play)\n")
    for name, (right, n) in accuracy.items():
        print("- ball {}: {:.1%} of play frames, M2c's team right on {:.1%}".format(
            name, n / float(total), right / float(n)))
    for name, (right, right_smoothed, n) in classifier.items():
        print("- classifier ({}) on unseen-ball play frames: right on {:.1%} per frame, {:.1%} "
              "smoothed over 3 s".format(name, right / float(n), right_smoothed / float(n)))

    chain_rows = np.array(chain_rows, dtype=bool)
    print("\n## Pass chains: two contacts in a row by one team, at most {:.0f} s apart, ball seen "
          "between\n".format(CHAIN_GAP / float(FPS)))
    for name, chosen in (("distinct slots", chain_rows[:, 0]), ("same slot", ~chain_rows[:, 0])):
        sub = chain_rows[chosen]
        flips = sub[~sub[:, 1] & sub[:, 3]]
        print("- {}: {} chains; the truth agrees on {:.1%}; {} where M2c names the other team, and "
              "the truth agrees with the chain on {:.1%}".format(
                  name, len(sub), sub[sub[:, 3]][:, 2].mean(), len(flips),
                  flips[:, 2].mean() if len(flips) else 0.0))

    if args.sweep:
        print("\n" + sweep(args.dfl_dir, args.matches, params))


def classify(seen, truths, add):
    """Leave one match out: train the team-shape classifier on every other match and paste its team
    onto the held-out match's unseen-ball stretches. {name: [right, right smoothed, frames]} on the
    unseen-ball play frames of the test matches."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # only this part needs it
    from threadpoolctl import threadpool_limits

    match_ids = sorted(set(match_id for match_id, _ in seen))
    out = {}
    for test_id in match_ids:
        for with_owner in (False, True):
            name = "with M2c's owner" if with_owner else "shape only"
            features, target = [], []
            for key, one in seen.items():
                if key[0] == test_id:
                    continue
                gt = truths[key[0]]["merged3s"]
                mask = np.isin(one.codes, (pm.H, pm.A)) & np.isin(gt, (pm.H, pm.A))
                features.append(shape_features(one, with_owner)[mask][::5])
                target.append((gt[mask] == pm.H)[::5])
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                   random_state=0)
            with threadpool_limits(limits=1):  # one summation order, so a rerun repeats it
                model.fit(np.concatenate(features), np.concatenate(target))
            for key, one in seen.items():
                if key[0] != test_id:
                    continue
                gt, codes = truths[test_id]["merged3s"], one.codes
                probability = model.predict_proba(shape_features(one, with_owner))[:, 1]
                team = np.where(probability > 0.5, pm.H, pm.A)
                smoothed = np.where(running_mean(probability) > 0.5, pm.H, pm.A)
                play = np.isin(codes, (pm.H, pm.A))
                unseen = np.isnan(one.ball[:, 0])
                mask = play & np.isin(gt, (pm.H, pm.A)) & unseen
                counts = out.setdefault(name, [0, 0, 0])
                counts[0] += int(np.sum(team[mask] == gt[mask]))
                counts[1] += int(np.sum(smoothed[mask] == gt[mask]))
                counts[2] += int(mask.sum())
                for shortest in PASTE_MIN:
                    pasted = codes.copy()
                    for start, stop in zip(*true_runs(play & unseen)):
                        if stop - start >= shortest:
                            pasted[start:stop] = smoothed[start:stop]
                    add("classifier ({}) on unseen stretches >= {} s".format(
                        name, shortest // FPS), key, pasted)
        print("classifier tested on {}".format(test_id), file=sys.stderr)
    return out


def sweep(dfl_dir, match_ids, params):
    """Markdown table: spellF1 of M2c with each set of controlled-touch thresholds, seed 0."""
    data = []
    for match_id in match_ids:
        match = dfl.load(dfl_dir, match_id, verbose=False)
        tracking, blocks = degrade.to_tracking(match, params, match_seed(match_id, 0))
        tag_list = tags.synthetic(match, blocks, params, match_seed(match_id, 0))
        data.append((tracking, blocks, tag_list, truth.variants(match)))
    saved = (config.CONTROL_RADIUS, config.CONTROL_MAX_RELATIVE_SPEED, config.CONTROL_MIN_FRAMES)
    on_ball_team, touches = control.on_ball_team, control.touches
    rows = []
    for radius, relative, frames in SWEEP:
        # Only the controlled-touch call changes; the contacts of stage 5 keep CONTROL_RADIUS.
        control.on_ball_team = (lambda distance, speeds=None, radius=radius, relative=relative:
                                on_ball_team(distance) if speeds is None else
                                on_ball_team(distance, speeds, radius, relative))
        control.touches = lambda on_ball, frames=frames: touches(on_ball, frames)
        scored = {}
        try:
            for tracking, blocks, tag_list, truths in data:
                codes, _ = adapter.label(tracking, blocks, tag_list, learned_model=False)
                for variant, gt in truths.items():
                    scored.setdefault(variant, []).append(score.Scored(gt, codes))
        finally:
            control.on_ball_team, control.touches = on_ball_team, touches
        summaries = {variant: score.summary(chosen) for variant, chosen in scored.items()}
        mark = " (M2c)" if (radius, relative, frames) == saved else ""
        rows.append(["{}{}".format(radius, mark), relative, frames]
                    + ["{:.3f}".format(summaries[v]["spell_f1"])
                       for v in ("raw", "merged2s", "merged3s")]
                    + ["{:.2f}".format(summaries["merged3s"]["pred_per_min"])])
    return "## Controlled-touch thresholds, seed 0, tags\n\n" + table(
        ["radius m", "max relative m/s", "min frames", "raw", "merged2s", "merged3s",
         "poss/min merged3s"], rows)


if __name__ == "__main__":
    main()
