"""
КУСОК ПРОСТРАНСТВА-ВРЕМЕНИ
==========================
Всё из одного правила: событию даны k чисел, i раньше j если раньше по всем.

Собрано вместе: геометрия, время, поле, спин, заряд.
"""
import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from cliff import clifford, transposition_op


class Spacetime:
    def __init__(self, N=1500, k=4, seed=0):
        self.N, self.k = N, k
        rs = np.random.default_rng(seed)
        self.X = rs.random((N, k))
        self.R = self._order()
        self.L = self._links()
        self.D = self._depth(self.R)
        self.Dc = self._depth(np.ascontiguousarray(self.R.T))
        self.g = clifford(k)

    # ---------------- геометрия ----------------
    def _order(self):
        R = np.ones((self.N, self.N), dtype=bool)
        for m in range(self.k):
            R &= (self.X[None, :, m] < self.X[:, None, m])
        np.fill_diagonal(R, False)
        return np.asarray(R, dtype=bool)          # R[i,j]: j раньше i

    def _links(self):
        A = self.R.astype(np.float32)
        return self.R & ((A @ A) == 0)

    def _depth(self, R):
        Rb = np.ascontiguousarray(R, dtype=bool)
        ind = Rb.sum(axis=1).astype(np.int64)
        D = np.zeros(self.N, dtype=np.int64)
        rd = list(np.flatnonzero(ind == 0))
        while rd:
            j = rd.pop(); kd = np.flatnonzero(Rb[:, j])
            if kd.size:
                D[kd] = np.maximum(D[kd], D[j]+1); ind[kd] -= 1
                rd.extend(kd[ind[kd] == 0].tolist())
        return D

    def dimension(self):
        from math import gamma
        from scipy.optimize import brentq
        f = self.R.sum()/(self.N*(self.N-1)/2)
        g = lambda d: gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))
        try: return brentq(lambda d: g(d)-f, 1.0, 40)
        except ValueError: return float('nan')

    # ---------------- время ----------------
    def proper_time(self, i, j):
        """Собственное время между связанными событиями = длина макс. цепи."""
        if not self.R[j, i]: return None
        mid = np.flatnonzero(self.R[:, i] & self.R[j, :])
        if len(mid) == 0: return 1
        sub = self.R[np.ix_(mid, mid)]
        return 1 + int(self._depth(sub).max()) + 1

    def coordinate_time(self, i):
        """Координатное время = сумма фаз."""
        return float(self.X[i].sum())

    # ---------------- поле ----------------
    def propagate(self, sources, horizon=None):
        """Комплексная амплитуда по причинным связям."""
        Lb = self.L
        order = self._topo()
        V = np.zeros((self.N, self.k), dtype=complex)
        A = np.exp(2j*np.pi*self.X)
        for v in order:
            if v in sources: V[v] = sources[v]; continue
            par = np.flatnonzero(Lb[v])
            if len(par): V[v] = A[v]*(V[par].sum(0)/np.sqrt(len(par)))
        return V

    def _topo(self):
        Lb = np.ascontiguousarray(self.L, dtype=bool)
        ind = Lb.sum(axis=1).astype(np.int64)
        rd = list(np.flatnonzero(ind == 0)); out = []
        while rd:
            j = rd.pop(); out.append(j)
            kd = np.flatnonzero(Lb[:, j])
            if kd.size:
                ind[kd] -= 1; rd.extend(kd[ind[kd] == 0].tolist())
        return out

    # ---------------- заряд ----------------
    def charge(self, i):
        """Заряд события = разброс его шкал."""
        return float(self.X[i].var())

    # ---------------- спин ----------------
    def spinor(self, i, v0=None):
        """Спинор события из ранжировки его шкал."""
        d = self.g[0].shape[0]
        if v0 is None:
            v0 = np.zeros(d, dtype=complex); v0[0] = 1.0
        p = tuple(np.argsort(self.X[i]))
        M = np.eye(d, dtype=complex)
        pa = list(range(self.k))
        for a in range(self.k):
            if pa[a] != p[a]:
                b = pa.index(p[a])
                M = M @ transposition_op(self.g, a, b)
                pa[a], pa[b] = pa[b], pa[a]
        return M @ v0

    def spin_transport(self, chain):
        """Перенос спинора вдоль мировой линии."""
        d = self.g[0].shape[0]
        M = np.eye(d, dtype=complex)
        perms = [tuple(np.argsort(self.X[c])) for c in chain]
        for t in range(1, len(perms)):
            pa = list(perms[t-1]); b = perms[t]
            for a in range(self.k):
                if pa[a] != b[a]:
                    q = pa.index(b[a])
                    M = M @ transposition_op(self.g, a, q)
                    pa[a], pa[q] = pa[q], pa[a]
        return M

    # ---------------- траектории ----------------
    def worldline(self, start, maxlen=40, rs=None):
        r = rs if rs is not None else np.random.default_rng(0)
        cur = start; ch = [start]
        for _ in range(maxlen):
            nxt = np.flatnonzero(self.L[:, cur])
            if len(nxt) == 0: break
            cur = int(r.choice(nxt)); ch.append(cur)
        return ch
