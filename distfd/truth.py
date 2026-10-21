"""Ground truth: Desbordante's TANE (exact, error = 0), capped at |LHS| <= max_lhs."""
import desbordante


def tane(df, max_lhs):
    algo = desbordante.afd.algorithms.Tane()
    algo.load_data(table=df)
    algo.execute(error=0.0, max_lhs=max_lhs)
    fds = {(frozenset(fd.lhs_indices), fd.rhs_index) for fd in algo.get_fds()}
    # keep only minimal FDs (TANE already returns minimal ones; this is a safeguard)
    return {(x, a) for x, a in fds if not any(y < x and b == a for y, b in fds)}
