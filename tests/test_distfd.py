from itertools import combinations, product
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from distfd.balance import Router, assignment
from distfd.distfd import Config, discover
from distfd.lattice import level_order
from distfd.truth import tane


def brute_force(data, max_lhs):
    """All minimal non-trivial FDs with |LHS| <= max_lhs, by direct checking."""
    n = data.shape[1]

    def holds(x, a):
        seen = {}
        for row in data:
            k = tuple(row[list(x)])
            if seen.setdefault(k, row[a]) != row[a]:
                return False
        return True

    valid = {(frozenset(x), a) for k in range(max_lhs + 1) for x in combinations(range(n), k)
             for a in range(n) if a not in x and holds(x, a)}
    return {(x, a) for x, a in valid if not any((x - {b}, a) in valid for b in x)}


def toy(seed, n_rows=200, n_cols=6):
    """Random table with planted FDs, a constant column and a skewed column."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 12, n_rows)
    b = rng.integers(0, 5, n_rows)
    skew = np.where(rng.random(n_rows) < 0.6, 0, rng.integers(1, 30, n_rows))
    cols = [a, b, (a * 7 + b) % 9, skew, np.zeros(n_rows, int), (a + skew) % 4]
    return np.column_stack(cols[:n_cols]).astype(np.int32)


def test_level_order():
    assert level_order(3, True) == [1, 3, 2]
    assert level_order(4, True) == [1, 4, 2, 3]
    assert level_order(3, False) == [1, 2, 3]


def test_assignment_splits_heavy_keys():
    plan, loads = assignment({0: 70, 1: 10, 2: 10, 3: 10}, 4)
    assert len(plan[0]) > 1
    assert sum(c for _, c in plan[0]) == 70
    assert sum(loads) == 100
    r = Router(0, 4, plan)
    assert 0 in r.split


def test_paper_example_1(sc):
    # A -> B holds on R1 and on R2 but not on R1 ∪ R2.
    r1 = [(1, 1, 0), (2, 2, 1)]
    r2 = [(1, 3, 2), (3, 4, 3)]
    codes = np.array(r1 + r2, dtype=np.int32)
    for part in (r1, r2):
        assert (frozenset({0}), 1) in brute_force(np.array(part), 1)
    fds, _ = discover(sc, codes, Config(max_lhs=2, sampling=False))
    assert (frozenset({0}), 1) not in fds
    assert fds == brute_force(codes, 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_brute_force(sc, seed):
    codes = toy(seed)
    truth = brute_force(codes, 3)
    cfg = Config(seed=seed)
    assert discover(sc, codes, cfg)[0] == truth


@pytest.mark.parametrize("nodes", [1, 2, 4, 8])
def test_partition_count_does_not_change_result(sc, nodes):
    codes = toy(3)
    assert discover(sc, codes, Config(nodes=nodes))[0] == brute_force(codes, 3)


@pytest.mark.parametrize("flags", list(product([False, True], repeat=4)))
def test_flag_combinations(sc, flags):
    codes = toy(4)
    s, c, b, t = flags
    cfg = Config(sampling=s, cache=c, balance=b, two_way=t, sample_ratio=0.2, sample_min=10)
    assert discover(sc, codes, cfg)[0] == brute_force(codes, 3)


def test_balancing_splits_keys_and_stays_correct(sc):
    codes = toy(5, n_rows=400)
    cfg = Config(balance=True, skew_t=0.1)
    fds, stats = discover(sc, codes, cfg)
    assert stats["skewed_attributes"] >= 1
    assert fds == brute_force(codes, 3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_desbordante_matches_brute_force(seed):
    codes = toy(seed)
    df = pd.DataFrame(codes.astype(str), columns=[f"c{i}" for i in range(codes.shape[1])])
    assert tane(df, 3) == brute_force(codes, 3)
