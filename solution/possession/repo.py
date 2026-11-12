"""The repository's own modules, made importable from inside this package.

games.py, evaluation/possession_metrics.py and starter/baseline.py are plain modules rather than a
package, so they are reached the same way the starter scripts reach them: by putting their folders
on sys.path. Doing it once, here, keeps every other module free of path handling.

Everything is reused rather than copied. In particular the metrics are always the official
possession_metrics -- a local re-implementation would be a second definition that could drift.
"""

import os
import sys

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SOLUTION_DIR)
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

for _path in (os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation"), ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import baseline  # noqa: E402,F401  (needs the path above)
import games  # noqa: E402,F401
import load_smart_tagging  # noqa: E402,F401
import possession_metrics as pm  # noqa: E402,F401
