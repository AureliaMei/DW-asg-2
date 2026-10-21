"""Partitions of the rows by attribute values.

Π_X groups the rows that agree on X into equivalence classes. We store a
partition as a label array (row -> class id). Theorem 1 of the paper:
X -> A holds iff |Π_X| = |Π_XA|.
"""
import numpy as np


class Partition:
    __slots__ = ("labels", "n_classes", "stripped_size")

    def __init__(self, labels, n_classes):
        self.labels = labels
        self.n_classes = int(n_classes)
        counts = np.bincount(labels, minlength=self.n_classes)
        # ||Π̂||: rows in non-singleton classes, i.e. the size of the stripped partition
        self.stripped_size = int(counts[counts > 1].sum())

    @classmethod
    def from_column(cls, col):
        uniq, inv = np.unique(col, return_inverse=True)
        return cls(inv.astype(np.int64).ravel(), len(uniq))

    @classmethod
    def whole(cls, n_rows):
        """Π_∅: all rows in one class."""
        return cls(np.zeros(n_rows, dtype=np.int64), 1 if n_rows else 0)


def product(p, q):
    """Π_X ∩ Π_Y = Π_XY."""
    key = p.labels * q.n_classes + q.labels
    uniq, inv = np.unique(key, return_inverse=True)
    return Partition(inv.astype(np.int64).ravel(), len(uniq))


def holds(px, pxa):
    return px.n_classes == pxa.n_classes
