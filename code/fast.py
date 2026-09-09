import numpy as np

class Order2D:
    """Двумерный порядок с инкрементальным пересчётом обилий интервалов.

    Обмен соседних рангов в u или v меняет ровно одно отношение (a,b).
    При добавлении a<b: b входит в интервалы (a,w) для всех w из будущего b,
    и a входит в интервалы (x,b) для всех x из прошлого a. Больше ничего.
    Стоимость шага O(N) вместо O(N^3)."""

    KMAX = 6

    def __init__(self, N, seed=0):
        r = np.random.default_rng(seed)
        self.N = N
        self.u = r.permutation(N).astype(np.int32)
        self.v = r.permutation(N).astype(np.int32)
        self.posu = np.empty(N, np.int32); self.posu[self.u] = np.arange(N)
        self.posv = np.empty(N, np.int32); self.posv[self.v] = np.arange(N)
        self._rebuild()

    def _rel(self):
        u, v = self.u, self.v
        return (u[None, :] < u[:, None]) & (v[None, :] < v[:, None])   # R[j,i]: i<j

    def _rebuild(self):
        R = self._rel()
        C = R.T.astype(np.float32)                 # C[i,j]=1 при i<j
        self.btw = (C @ C).astype(np.int32)
        self.rel = R.T.copy()                      # rel[i,j] = i<j
        self.cnt = np.zeros(self.KMAX + 2, np.int64)
        b = self.btw[self.rel]
        for k in range(self.KMAX + 1):
            self.cnt[k] = int((b == k).sum())
        self.cnt[self.KMAX + 1] = int((b > self.KMAX).sum())

    def _batch(self, rows, col, byrow, delta):
        """Векторное обновление: все затронутые интервалы разом."""
        if len(rows) == 0: return
        K = self.KMAX + 1
        if byrow:
            old = self.btw[rows, col]; self.btw[rows, col] = old + delta
        else:
            old = self.btw[col, rows]; self.btw[col, rows] = old + delta
        o = np.minimum(old, K); n = np.minimum(old + delta, K)
        np.subtract.at(self.cnt, o, 1)
        np.add.at(self.cnt, n, 1)

    def swap(self, axis, k):
        w = self.u if axis == 0 else self.v
        pos = self.posu if axis == 0 else self.posv
        a = int(pos[k]); b = int(pos[k + 1])
        adding = (self.v[a] < self.v[b]) if axis == 0 else (self.u[a] < self.u[b])
        if adding: lo, hi, d = a, b, -1
        else:      lo, hi, d = b, a, +1
        past = np.flatnonzero(self.rel[:, lo])
        fut = np.flatnonzero(self.rel[hi, :])
        if d > 0:
            self.rel[lo, hi] = True; self.btw[lo, hi] = 0; self.cnt[0] += 1
        self._batch(past, hi, True, d)
        self._batch(fut, lo, False, d)
        if d < 0:
            self.cnt[min(int(self.btw[lo, hi]), self.KMAX + 1)] -= 1
            self.rel[lo, hi] = False; self.btw[lo, hi] = 0
        w[a], w[b] = w[b], w[a]
        pos[k], pos[k + 1] = pos[k + 1], pos[k]
        return (axis, k)

    def action2d(self):
        n, c = self.N, self.cnt
        return float(n - 2 * (c[0] - 2 * c[1] + c[2]))

def run_mcmc(N, beta, steps, seed=0, every=None, burn=None):
    O = Order2D(N, seed=seed)
    r = np.random.default_rng(seed + 1000)
    if burn is None: burn = steps // 2
    if every is None: every = max(20, steps // 400)
    S = O.action2d(); rec = []; acc = 0
    for t in range(steps):
        ax = int(r.integers(0, 2)); k = int(r.integers(0, N - 1))
        cnt_before = O.cnt.copy(); O.swap(ax, k)
        S2 = O.action2d()
        if r.random() < np.exp(np.clip(-beta * (S2 - S), -700, 700)):
            S = S2; acc += 1
        else:
            O.swap(ax, k); O.cnt = cnt_before          # откат
        if t >= burn and t % every == 0:
            rec.append([S / N, O.cnt[0] / max(O.cnt.sum(), 1)])
    R = np.array(rec)
    return R.mean(axis=0), R.std(axis=0), acc / steps

def block_mcmc(N, beta, steps, block=1, seed=0, every=None, burn=None):
    """Крупный ход: block обменов подряд принимаются или отвергаются как одно целое."""
    O = Order2D(N, seed=seed); r = np.random.default_rng(seed + 7777)
    if burn is None: burn = steps // 2
    if every is None: every = max(10, steps // 400)
    S = O.action2d(); rec = []; acc = 0
    for t in range(steps):
        moves = [(int(r.integers(0, 2)), int(r.integers(0, N - 1))) for _ in range(block)]
        cnt_before = O.cnt.copy()
        for ax, k in moves: O.swap(ax, k)
        S2 = O.action2d()
        if r.random() < np.exp(np.clip(-beta * (S2 - S), -700, 700)):
            S = S2; acc += 1
        else:
            for ax, k in reversed(moves): O.swap(ax, k)
            O.cnt = cnt_before
        if t >= burn and t % every == 0:
            rec.append([S / N, O.cnt[0] / max(O.cnt.sum(), 1)])
    R = np.array(rec)
    return R.mean(axis=0), R.std(axis=0), acc / steps
