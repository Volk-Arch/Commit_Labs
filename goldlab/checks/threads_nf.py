"""НИТИ БЕЗ ВЫДЕЛЕННОЙ СИСТЕМЫ. Цена шага вправо — Δv = Δt − Δx (световая координата, вдоль которой правый бегун НЕ движется),
влево — Δu = Δt + Δx. При бусте все Δv умножаются на одно число ⇒ какое паросочетание наилучшее — от системы не зависит
(инвариантное правило). Предел по Δt нужен лишь как «окно поиска» (он один и выделяет масштаб). Смотрим: скорость нитей c,
покрытие, и работает ли блуждание (частота покоя ≈ m)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from goldlab import worlds as Wd, observe as Ob
import walk4
def threads_null(X, side, cap, big=1e3):
    n = len(X); t, x = X[:, 0], X[:, 1]; rows, cols, vals = [], [], []
    for i in range(n):
        j0 = np.searchsorted(t, t[i] + 1e-9); j1 = np.searchsorted(t, t[i] + cap); js = np.arange(j0, j1); dx = x[js] - x[i]; dt = t[js] - t[i]
        ok = (dx * side > 1e-9) & (dt > np.abs(dx) + 1e-12); js = js[ok]
        rows += [i] * len(js); cols += list(js); vals += list((dt[ok] - side * dx[ok]) / (dt[ok] + side * dx[ok]) + 1e-6)   # «насколько не свет»: Δv/Δu — тоже инвариантно к бусту
    rows += list(range(n)); cols += list(range(n, 2 * n)); vals += [big] * n
    r, c = min_weight_full_bipartite_matching(coo_matrix((vals, (rows, cols)), shape=(n, 2 * n)).tocsr()); nxt = -np.ones(n, int); nxt[r] = c; nxt[nxt >= n] = -1; return nxt
L = 24; w = Wd.gold11(4.0, L); t, x = w.t, w.x
for cap in (1.6, 3.0, 5.0):
    orig = walk4.threads_sparse; walk4.threads_sparse = lambda X, side, cap_=cap, big=1e3: threads_null(X, side, cap)
    W = walk4.Walk(w, cap=cap); walk4.threads_sparse = orig
    sR = w.X[W.R[W.R >= 0]] - w.X[W.R >= 0]
    lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9; g = np.where(src, np.exp(-x * x / 50), 0)
    aR, aL, leak = W.run(1.0, g, g)
    sel = (t > ts + 1.5) & (t < L - 4) & (np.abs(x) < 8); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
    s = np.zeros(len(tu), complex); np.add.at(s, inv, (np.exp(-x * x / 50) * (aR + aL))[sel]); om = np.linspace(-2, 0, 2001); S = Ob.spectrum(tu, s, om); S /= S.max()
    pk = Ob.peaks(om, S, 0.1)
    print(f'предел {cap}: c = {W.c:.3f}; полных внутри {W.full:.1%}; шаг вправо Δt медиана {np.median(sR[:, 0]):.2f}, макс {sR[:, 0].max():.2f}; норма {(g ** 2).sum() * 2:.3f} → {leak:.3f}; частота покоя ' + ' '.join(f'{-e:.3f}' for e, q in pk), flush=True)
