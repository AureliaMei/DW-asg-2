"""DisTFD driver (paper Algorithm 1)."""
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .balance import Router, assignment, skewness
from .cache import PartitionCache
from .lattice import Lattice, level_order
from .sampler import sample_size, sampling_validate, systematic_indices
from .spark_validate import build_blocks, global_validate


@dataclass
class Config:
    sampling: bool = True
    cache: bool = True
    balance: bool = True
    two_way: bool = True
    sort_attributes: bool = True  # Pre_processing order; False keeps the schema order
    max_lhs: int = 3
    sample_ratio: float = 0.10
    sample_min: int = 50
    skew_t: float = 0.1
    nodes: int = 4
    seed: int = 0
    cache_capacity: int = 4096


def value_frequencies(data):
    return [Counter(data[:, a].tolist()) for a in range(data.shape[1])]


def discover(sc, codes, cfg):
    """Return (minimal FDs as {(frozenset X, A)}, stats dict)."""
    t0 = time.perf_counter()
    n_rows, n_attrs = codes.shape
    # D = D1, ..., Dn: the table split horizontally over the nodes
    rows = sc.parallelize(list(enumerate(map(tuple, codes.tolist()))), cfg.nodes).persist()

    # Exact per-attribute key frequencies from the full data: constants (∅ -> A) and Algorithm 4
    pairs = rows.flatMap(lambda t: [((a, v), 1) for a, v in enumerate(t[1])]).reduceByKey(lambda x, y: x + y)
    freq = [Counter() for _ in range(n_attrs)]
    for (a, v), c in pairs.collect():
        freq[a][v] = c
    constants = [a for a in range(n_attrs) if len(freq[a]) == 1]

    # D' = Sampler(D, n)
    sample = None
    if cfg.sampling:
        idx = set(systematic_indices(n_rows, sample_size(n_rows, cfg.sample_ratio, cfg.sample_min), cfg.seed).tolist())
        sample = np.array(rows.filter(lambda t: t[0] in idx).map(lambda t: t[1]).collect(), dtype=np.int32)
        sample = sample.reshape(-1, n_attrs)

    # Pre_processing (Algorithm 2): cardinality and skew from the sample; skewed attributes last
    stats_freq = value_frequencies(sample) if sample is not None else freq
    stats_n = len(sample) if sample is not None else n_rows
    skew = [skewness(f, stats_n) for f in stats_freq]
    skewed = {a for a in range(n_attrs) if skew[a] > cfg.skew_t}
    if cfg.sort_attributes:
        order = sorted(range(n_attrs), key=lambda a: (a in skewed, -len(stats_freq[a]), a))
    else:
        order = list(range(n_attrs))
    rank = {a: i for i, a in enumerate(order)}

    routers = {}
    for a in order:
        plan = assignment(freq[a], cfg.nodes)[0] if cfg.balance and a in skewed else None
        routers[a] = Router(a, cfg.nodes, plan)
    blocks = build_blocks(sc, rows, order, n_attrs, cfg.nodes, routers)

    lattice = Lattice(n_attrs, cfg.max_lhs, constants)
    sample_cache = PartitionCache(sample, enabled=cfg.cache, capacity=cfg.cache_capacity) if sample is not None else None
    levels = []
    for level in level_order(cfg.max_lhs, cfg.two_way):
        cands = lattice.candidates(level)
        if sample_cache is not None:
            survivors, rejected = sampling_validate(cands, sample_cache)
        else:
            survivors, rejected = cands, []
        # each candidate is validated after redistributing on its first LHS attribute in `order`
        by_attr = {}
        for cid, (x, a) in enumerate(survivors):
            home = min(x, key=rank.__getitem__)
            by_attr.setdefault(home, []).append((cid, x, a))
        failed = global_validate(sc, blocks, by_attr, cfg.cache, cfg.cache_capacity)
        valid = [c for i, c in enumerate(survivors) if i not in failed]
        invalid = rejected + [survivors[i] for i in failed]
        lattice.record(level, valid, invalid)
        levels.append({"level": level, "candidates": len(cands), "pruned_by_sample": len(rejected),
                       "validated": len(survivors), "valid": len(valid)})
    blocks.unpersist()
    rows.unpersist()
    fds = lattice.minimal()
    stats = {
        "seconds": time.perf_counter() - t0,
        "constants": len(constants),
        "skewed_attributes": len(skewed),
        "sample_rows": 0 if sample is None else len(sample),
        "levels": levels,
        "config": asdict(cfg),
    }
    return fds, stats
