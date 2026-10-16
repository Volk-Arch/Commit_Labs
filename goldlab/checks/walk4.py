"""ДИРАК НА КРИСТАЛЛЕ, шаг 4: нити для больших миров (разреженное паросочетание) и поле ядра.
Нити: событие → «раньше»-событие с Δx·сторона > 0, Δt < cap, цена — собственное время; несовпавшие — граница (заглушки).
Поле: на каждом шаге нити амплитуда умножается на e^{−i∫V dt} (векторный потенциал, как нить Вильсона), V = −α/√(x² + a²)."""
import sys, numpy as np, time
sys.path.insert(0, '/home/claude/goldlab')
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from goldlab import worlds as Wd
def threads_sparse(X, side, cap=1.6, big=1e3):
    n = len(X); t, x = X[:, 0], X[:, 1]; o = np.argsort(t); rows, cols, vals = [], [], []
    for i in range(n):                                            # кандидаты: в будущем, ближе cap по времени
        j0 = np.searchsorted(t, t[i] + 1e-9); j1 = np.searchsorted(t, t[i] + cap)
        js = np.arange(j0, j1); dx = x[js] - x[i]; dt = t[js] - t[i]
        ok = (dx * side > 1e-9) & (dt > np.abs(dx) + 1e-12); js = js[ok]
        rows += [i] * len(js); cols += list(js); vals += list(np.sqrt(dt[ok] ** 2 - dx[ok] ** 2) + 1e-3 * dt[ok] + 1.0)
    rows += list(range(n)); cols += list(range(n, 2 * n)); vals += [big] * n          # заглушка «уйти за край» для каждого события
    G = coo_matrix((vals, (rows, cols)), shape=(n, 2 * n)).tocsr()
    r, c = min_weight_full_bipartite_matching(G); nxt = -np.ones(n, int); nxt[r] = c; nxt[nxt >= n] = -1; return nxt
class Walk:
    def __init__(self, w, cap=1.6):
        X = w.X; self.w = w; t = X[:, 0]; n = w.n
        assert np.all(np.diff(t) >= 0)
        self.R = threads_sparse(X, +1, cap); self.L = threads_sparse(X, -1, cap)
        pR = -np.ones(n, int); pR[self.R[self.R >= 0]] = np.nonzero(self.R >= 0)[0]; pL = -np.ones(n, int); pL[self.L[self.L >= 0]] = np.nonzero(self.L >= 0)[0]
        dR = np.where(pR >= 0, t - t[np.maximum(pR, 0)], np.nan); dL = np.where(pL >= 0, t - t[np.maximum(pL, 0)], np.nan)
        both = np.c_[dR, dL]; cnt = np.sum(~np.isnan(both), 1); self.dtb = np.where(cnt > 0, np.nansum(both, 1) / np.maximum(cnt, 1), np.nanmedian(both))
        sR = X[self.R[self.R >= 0]] - X[self.R >= 0]; sL = X[self.L[self.L >= 0]] - X[self.L >= 0]
        self.c = 0.5 * (np.mean(sR[:, 1] / sR[:, 0]) - np.mean(sL[:, 1] / sL[:, 0]))
        inner = np.abs(t) + np.abs(X[:, 1]) < 0.8 * max(np.abs(t).max(), 1)
        self.full = (inner & (self.R >= 0) & (self.L >= 0) & (pR >= 0) & (pL >= 0)).sum() / inner.sum()
    def phases(self, Vint):
        """Vint(t0, x0, t1, x1) — ∫V dt по шагу; фазы для шагов вправо и влево."""
        X = self.w.X; ph = []
        for nx in (self.R, self.L):
            p = np.ones(self.w.n, complex); ok = nx >= 0
            p[ok] = np.exp(-1j * Vint(X[ok, 0], X[ok, 1], X[nx[ok], 0], X[nx[ok], 1])); ph.append(p)
        return ph
    def run(self, m, fR, fL, ph=None):
        n = self.w.n; t = self.w.t; aR = fR.astype(complex).copy(); aL = fL.astype(complex).copy(); leak = 0.0
        pR, pL = ph if ph is not None else (np.ones(n), np.ones(n)); c = np.cos(m * self.dtb); s = np.sin(m * self.dtb)
        for z in range(n):                                        # события уже по времени
            bR = c[z] * aR[z] - 1j * s[z] * aL[z]; bL = -1j * s[z] * aR[z] + c[z] * aL[z]
            if self.R[z] >= 0: aR[self.R[z]] += bR * pR[z]
            else: leak += abs(bR) ** 2
            if self.L[z] >= 0: aL[self.L[z]] += bL * pL[z]
            else: leak += abs(bL) ** 2
        return aR, aL, leak
def soft_coulomb_int(alpha, a, k=8):
    gx, gw = np.polynomial.legendre.leggauss(k)
    def Vint(t0, x0, t1, x1):
        tot = 0
        for g, wg in zip(gx, gw):
            s = 0.5 * (g + 1); xs = x0 + s * (x1 - x0); tot = tot + 0.5 * wg * (-alpha / np.sqrt(xs * xs + a * a))
        return tot * (t1 - t0)
    return Vint
if __name__ == '__main__':
    for L in (16, 30):
        T0 = time.time(); w = Wd.gold11(4.0, L); W = Walk(w)
        print(f'L = {L}: событий {w.n}; нити за {time.time() - T0:.0f} с; полных внутри {W.full:.1%}; c = {W.c:.3f}', flush=True)
