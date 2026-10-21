"""Prefix-tree partition cache and RetrievePartition (paper Algorithm 3)."""
from collections import OrderedDict

from .partitions import Partition, product


class _Node:
    __slots__ = ("children", "part")

    def __init__(self):
        self.children = {}
        self.part = None


class PartitionCache:
    """Partitions of `data` keyed by sorted attribute tuples, stored in a prefix tree.

    Single-attribute partitions are always kept. Larger ones are evicted LRU
    once more than `capacity` are stored. With enabled=False nothing is cached
    and every partition is rebuilt from single columns.
    """

    def __init__(self, data, enabled=True, capacity=4096):
        self.data = data
        self.enabled = enabled
        self.capacity = capacity
        self.root = _Node()
        self.lru = OrderedDict()
        self.single = {}
        self.hits = 0
        self.misses = 0

    def _single(self, a):
        p = self.single.get(a)
        if p is None:
            p = self.single[a] = Partition.from_column(self.data[:, a])
        return p

    def get(self, attrs):
        attrs = tuple(sorted(attrs))
        if not attrs:
            return Partition.whole(len(self.data))
        if len(attrs) == 1:
            return self._single(attrs[0])
        if not self.enabled:
            p = self._single(attrs[0])
            for a in attrs[1:]:
                p = product(p, self._single(a))
            return p
        node = self._find(attrs)
        if node is not None and node.part is not None:
            self.hits += 1
            self.lru.move_to_end(attrs)
            return node.part
        self.misses += 1
        p = self._retrieve(attrs)
        self._store(attrs, p)
        return p

    def _find(self, attrs):
        node = self.root
        for a in attrs:
            node = node.children.get(a)
            if node is None:
                return None
        return node

    def _subsets(self, attrs):
        """All cached proper subsets of attrs: walk the tree following only attributes in attrs."""
        out = [((a,), self._single(a)) for a in attrs]
        stack = [(self.root, (), 0)]
        while stack:
            node, path, start = stack.pop()
            for i in range(start, len(attrs)):
                child = node.children.get(attrs[i])
                if child is None:
                    continue
                sub = path + (attrs[i],)
                if child.part is not None and 1 < len(sub) < len(attrs):
                    out.append((sub, child.part))
                stack.append((child, sub, i + 1))
        return out

    def _retrieve(self, attrs):
        subs = self._subsets(attrs)
        target = set(attrs)
        # Π_Y: the cached subset with the smallest stripped partition
        y_attrs, p = min(subs, key=lambda s: (s[1].stripped_size, -len(s[0])))
        covered = set(y_attrs)
        chosen = []
        while covered != target:
            # Z = max X / C: the cached subset adding the most uncovered attributes
            z = max(subs, key=lambda s: (len(set(s[0]) - covered), -s[1].stripped_size))
            chosen.append(z)
            covered |= set(z[0])
        chosen.sort(key=lambda s: s[1].stripped_size)
        for _, q in chosen:
            p = product(p, q)
        return p

    def _store(self, attrs, p):
        node = self.root
        for a in attrs:
            node = node.children.setdefault(a, _Node())
        node.part = p
        self.lru[attrs] = None
        if len(self.lru) > self.capacity:
            old, _ = self.lru.popitem(last=False)
            self._find(old).part = None
