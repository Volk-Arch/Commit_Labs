"""Цепи как объекты: мировые линии и что между ними.

Цепь — последовательность событий, каждое следующее прямо связано
с предыдущим. Это ближайший аналог мировой линии частицы.
"""
import numpy as np


def links_of(R):
    A = np.asarray(R, dtype=np.float32)
    return np.asarray(R & ((A @ A) == 0), dtype=bool)


def random_chain(L, start, rs, maxlen=30):
    cur = start
    ch = [start]
    for _ in range(maxlen):
        nxt = np.flatnonzero(L[:, cur])
        if len(nxt) == 0:
            break
        cur = int(rs.choice(nxt))
        ch.append(cur)
    return ch


def longest_chains(L, D, rs, want=20, minlen=4):
    """Цепи, идущие от самых ранних событий."""
    starts = np.flatnonzero(D == 0)
    out = []
    for _ in range(want*20):
        if len(starts) == 0:
            break
        s = int(rs.choice(starts))
        ch = random_chain(L, s, rs)
        if len(ch) >= minlen:
            out.append(ch)
        if len(out) >= want:
            break
    return out


def chain_relation(R, c1, c2):
    """Как две цепи связаны: доля пар, где событие одной раньше другой."""
    a = np.array(c1); b = np.array(c2)
    fwd = int(R[np.ix_(b, a)].sum())
    bwd = int(R[np.ix_(a, b)].sum())
    tot = len(a)*len(b)
    return fwd/tot, bwd/tot, (fwd-bwd)/max(fwd+bwd, 1)
