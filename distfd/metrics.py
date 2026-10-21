"""Precision, recall and F1 of a found FD set against the truth. One FD (X, A) is one instance."""


def prf(found, truth):
    tp = len(found & truth)
    fp = len(found - truth)
    fn = len(truth - found)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
