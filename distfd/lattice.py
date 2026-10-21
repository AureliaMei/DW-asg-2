"""Attribute lattice: candidate generation, Lemma 2/3 pruning, two-way level order."""
from itertools import combinations


def level_order(max_lhs, two_way):
    """Bottom-up 1..L, or alternating 1, L, 2, L-1, ... for two-way traversal."""
    if not two_way:
        return list(range(1, max_lhs + 1))
    out, lo, hi = [], 1, max_lhs
    while lo <= hi:
        out.append(lo)
        if hi != lo:
            out.append(hi)
        lo, hi = lo + 1, hi - 1
    return out


class Lattice:
    """Known-valid and known-invalid FDs over candidates X -> A with |X| <= max_lhs.

    FDs are (frozenset X, A). A constant attribute A gives ∅ -> A. Constant
    attributes are left out of every LHS: X -> A with constant B in X is
    equivalent to X \\ {B} -> A, so it is never minimal.
    """

    def __init__(self, n_attrs, max_lhs, constants=()):
        self.n_attrs = n_attrs
        self.max_lhs = max_lhs
        self.valid = {(frozenset(), a) for a in constants}
        self.invalid = set()
        self.done = []
        self.free = [a for a in range(n_attrs) if a not in set(constants)]

    def _valid_subset(self, x, a):
        items = sorted(x)
        for k in range(len(items)):
            for y in combinations(items, k):
                if (frozenset(y), a) in self.valid:
                    return True
        return False

    def _invalid_superset(self, x, a):
        above = [lvl for lvl in self.done if lvl > len(x)]
        if not above:
            return False
        # The nearest *processed* level above is enough: processed levels above len(x)
        # are always contiguous (the top frontier descends one level at a time), and every
        # subset of an invalid FD at those levels is recorded as invalid (Lemma 2).
        lvl = min(above)
        extra = [b for b in self.free if b not in x and b != a]
        for e in combinations(extra, lvl - len(x)):
            if (x | frozenset(e), a) in self.invalid:
                return True
        return False

    def candidates(self, level):
        """Unresolved X -> A with |X| = level, as (sorted X tuple, A)."""
        out = []
        for xs in combinations(self.free, level):
            x = frozenset(xs)
            for a in range(self.n_attrs):
                if a in x or self._valid_subset(x, a):  # Lemma 3: implied by a smaller valid FD
                    continue
                if self._invalid_superset(x, a):  # Lemma 2: a superset already fails
                    self.invalid.add((x, a))
                    continue
                out.append((xs, a))
        return out

    def record(self, level, valid, invalid):
        self.valid.update((frozenset(x), a) for x, a in valid)
        self.invalid.update((frozenset(x), a) for x, a in invalid)
        self.done.append(level)

    def minimal(self):
        return {(x, a) for x, a in self.valid if not self._valid_subset(x, a)}
