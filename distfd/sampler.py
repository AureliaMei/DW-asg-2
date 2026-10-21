"""Sampler (systematic sampling) and SamplingValidate.

An FD that fails on a sample fails on the whole table, so failing candidates
are discarded without touching the full data. Candidates that hold on the
sample still need GlobalValidate.
"""
import math

import numpy as np

from .partitions import holds


def sample_size(n_rows, ratio, min_rows):
    return min(n_rows, max(min_rows, math.ceil(ratio * n_rows)))


def systematic_indices(n_rows, n, seed):
    """k = integer closest to N/n, random start r in [1, k], then every k-th row until n rows."""
    k = max(1, round(n_rows / n))
    r = int(np.random.default_rng(seed).integers(1, k + 1))
    return np.arange(r - 1, n_rows, k)[:n]


def sampling_validate(cands, cache):
    """Split candidates into (hold on the sample, fail on the sample)."""
    survivors, rejected = [], []
    for x, a in cands:
        (survivors if holds(cache.get(x), cache.get(x + (a,))) else rejected).append((x, a))
    return survivors, rejected
