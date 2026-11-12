"""Run-length helpers. Every stage reasons about stretches of frames rather than single frames."""

import numpy as np


def true_runs(mask):
    """Half-open [start, stop) spans where a boolean array is True, as two int arrays."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def value_runs(values):
    """[(start, stop, value)] maximal constant runs of an integer array."""
    values = np.asarray(values)
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(bounds[i]), int(bounds[i + 1]), int(values[bounds[i]]))
            for i in range(len(bounds) - 1)]
