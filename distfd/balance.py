"""Skew detection and the greedy key Assignment (paper Algorithm 4)."""
import math


def skewness(freq, n_rows):
    """Inc_A = frequency of the most common value / n."""
    return max(freq.values()) / n_rows if n_rows else 0.0


def assignment(freq, n_nodes):
    """Map each key to [(node, n_rows), ...]. Keys larger than Avg are split over several nodes.

    Step 1 cuts every key larger than Avg into Avg-sized chunks, each sent to the
    least-loaded node. Step 2 fills each node up to Avg with the remaining keys,
    largest first. Keys that fit nowhere (the paper leaves this case open) go to
    the least-loaded node.
    """
    avg = math.ceil(sum(freq.values()) / n_nodes)
    loads = [0] * n_nodes
    plan = {}
    remaining = {}
    for key, size in sorted(freq.items(), key=lambda kv: -kv[1]):
        while size > avg:
            node = min(range(n_nodes), key=loads.__getitem__)
            plan.setdefault(key, []).append((node, avg))
            loads[node] += avg
            size -= avg
        if size:
            remaining[key] = size
    rest = sorted(remaining.items(), key=lambda kv: -kv[1])
    for node in range(n_nodes):
        left = []
        for key, size in rest:
            if loads[node] + size <= avg:
                plan.setdefault(key, []).append((node, size))
                loads[node] += size
            else:
                left.append((key, size))
        rest = left
    for key, size in rest:
        node = min(range(n_nodes), key=loads.__getitem__)
        plan.setdefault(key, []).append((node, size))
        loads[node] += size
    return plan, loads


class Router:
    """Picks the destination node of a row when redistributing on one attribute."""

    def __init__(self, attr, n_nodes, plan=None):
        self.attr = attr
        self.n_nodes = n_nodes
        self.plan = plan or {}
        self.split = {k for k, chunks in self.plan.items() if len(chunks) > 1}

    def __call__(self, idx, row):
        key = row[self.attr]
        chunks = self.plan.get(key)
        if chunks is None:
            return hash(key) % self.n_nodes
        if len(chunks) == 1:
            return chunks[0][0]
        # Spread a split key's rows over its chunks in proportion to chunk size.
        pos = (idx * 2654435761) % (2 ** 32) % sum(c for _, c in chunks)
        for node, size in chunks:
            if pos < size:
                return node
            pos -= size
        return chunks[-1][0]
