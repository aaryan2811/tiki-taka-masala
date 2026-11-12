"""Regenerate the submission from the provided data. The one command the repository promises.

    python solution/run.py                     # writes evaluation/results/submission.csv
    python solution/run.py --name my_variant   # same pipeline, another file name
    python solution/run.py --rules-only --name rules_only   # the rules alone, without M4's model

The first run parses the raw tracking and caches it under solution/cache/ (gitignored). The written
file is validated with the official checks before this exits.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession import atd, labels, learned, pipeline, repo  # noqa: E402  (needs the path above)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="submission", help="submission name, without .csv")
    parser.add_argument("--out", default=repo.RESULTS_DIR, help="where to write")
    parser.add_argument("--data-dir", default=None, help="data bundle root (default: games.py)")
    parser.add_argument("--refresh", action="store_true", help="re-parse the raw tracking")
    parser.add_argument("--rules-only", action="store_true",
                        help="keep the rules' team in open play (M2c), without M4's learned model")
    args = parser.parse_args()

    tracking = atd.load(data_dir=args.data_dir, refresh=args.refresh)
    print("teams: H = {}, A = {}".format(*tracking.team_names))
    print("tracking: {} frames, {} track slots, {} fields per ball group".format(
        tracking.n_frames, len(tracking.team_of_slot), tracking.ball_fields.shape[1]))

    tags = repo.load_smart_tagging.load(data_dir=args.data_dir)
    codes, info = pipeline.label_frames(tracking, labels.graded_blocks(), tags,
                                        learned_model=not args.rules_only)
    print("ball located on {:.1%} of frames, {:.1%} after repair ({} interpolated)".format(
        info["ball_raw"], info["ball_repaired"], info["interpolated_frames"]))
    print("goalkeeper slots read by the half they stand in: {}".format(len(tracking.keeper_slots)))
    print("on the ball: {} frames, {} controlled touches ({} more void: duels the other team took "
          "straight back)".format(info["on_ball_frames"], info["touches"], len(info["void"])))
    shifts = sorted((old - new) / 25.0 for old, new in info["moved"])
    print("changes of possession: {}, {} moved back to the decisive touch (median {:.1f} s)".format(
        info["changes"], len(shifts), shifts[len(shifts) // 2] if shifts else 0.0))
    if info["learned"]:
        model = learned.load()
        print("team in open play: M4's learned model ({} trees, trained on DFL {}, CC BY 4.0), "
              "switch cost {}, possessions of at least {} frames".format(
                  len(model["trees"]), " and ".join(model["trained_on"]), model["switch_cost"],
                  model["min_frames"]))
    else:
        print("team in open play: the rules' (--rules-only)")

    stoppages = info["stoppages"]
    evidence = {}
    for stoppage in stoppages:
        evidence[stoppage.evidence] = evidence.get(stoppage.evidence, 0) + 1
    seconds = [(s.stop - s.start) / 25.0 for s in stoppages]
    print("dead balls: {} ({}), median {:.1f} s, restart team unknown for {}".format(
        len(stoppages), ", ".join("{} {}".format(n, k) for k, n in sorted(evidence.items())),
        sorted(seconds)[len(seconds) // 2] if seconds else 0.0,
        sum(s.team < 0 for s in stoppages)))

    path = labels.write(os.path.join(args.out, args.name + ".csv"), codes)
    reason = labels.check(path)
    if reason:
        print("\nFAIL  {}: {}".format(os.path.relpath(path, repo.ROOT), reason))
        sys.exit(1)
    print("\nOK    {}".format(os.path.relpath(path, repo.ROOT)))
    print("score it:  python solution/evaluate.py {}".format(os.path.relpath(path, repo.ROOT)))


if __name__ == '__main__':
    main()
