"""Redistribution and GlobalValidate (paper Algorithm 5) on Spark RDDs.

For each attribute Z the rows are redistributed so that rows with the same
Z value land on the same node. A candidate X -> A with Z in X is then checked
on the node: every class of Π_X lies inside one Z value, so it is complete
there, and a local |Π_X| = |Π_XA| check is a global one.

A skewed key that Assignment split over several nodes breaks this. For those
rows each node emits its local classes of X with their single A value, and
classes with the same X value are merged across nodes (Eq. 2). The FD fails
if a merged class has more than one A value.
"""
import numpy as np

from .cache import PartitionCache
from .partitions import holds


def build_blocks(sc, rows, order, n_attrs, n_nodes, routers):
    """One RDD block (attr, rows array, split-key mask) per (attribute, node)."""
    rdds = []
    for a in order:
        router = routers[a]

        def to_block(it, a=a, split=router.split):
            arr = np.array([r for _, r in it], dtype=np.int32).reshape(-1, n_attrs)
            mask = np.isin(arr[:, a], list(split)) if split else np.zeros(len(arr), bool)
            return [(a, arr, mask)]

        rdds.append(
            rows.map(lambda t, router=router: (router(*t), t))
            .partitionBy(n_nodes, lambda d: d)
            .values()
            .mapPartitions(to_block)
        )
    return sc.union(rdds).persist()


def _check_block(block, cands, use_cache, capacity):
    a, arr, split = block
    out = []
    whole = arr[~split]
    failed = set()
    if len(whole):
        cache = PartitionCache(whole, enabled=use_cache, capacity=capacity)
        for cid, x, rhs in cands:
            if not holds(cache.get(x), cache.get(x + (rhs,))):
                failed.add(cid)
                out.append(("fail", cid))
    part = arr[split]
    if len(part):
        cache = PartitionCache(part, enabled=use_cache, capacity=capacity)
        for cid, x, rhs in cands:
            if cid in failed:
                continue
            px = cache.get(x)
            if not holds(px, cache.get(x + (rhs,))):
                out.append(("fail", cid))
                continue
            # local classes Π_X/Z_ij: one row per class carries the class's X value and A value
            _, first = np.unique(px.labels, return_index=True)
            cols = list(x)
            for r in first:
                out.append((("merge", cid, part[r, cols].tobytes()), int(part[r, rhs])))
    return out


def global_validate(sc, blocks, cands_by_attr, use_cache, capacity):
    """Return the ids of candidates that fail on the full data."""
    if not cands_by_attr:
        return set()
    bc = sc.broadcast(cands_by_attr)

    def run(block):
        cands = bc.value.get(block[0])
        return _check_block(block, cands, use_cache, capacity) if cands else []

    recs = blocks.flatMap(run).persist()
    failed = set(recs.filter(lambda r: r[0] == "fail").map(lambda r: r[1]).collect())
    merged = (
        recs.filter(lambda r: r[0] != "fail")
        .map(lambda r: (r[0], {r[1]}))
        .reduceByKey(lambda s, t: s | t)  # Eq. 2: merge classes with the same X value
    )
    failed |= set(merged.filter(lambda kv: len(kv[1]) > 1).map(lambda kv: kv[0][1]).collect())
    recs.unpersist()
    bc.unpersist()
    return failed
