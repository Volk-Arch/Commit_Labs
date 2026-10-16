"""РАЗРЕЖЕННАЯ СВЕТОВАЯ СЕТЬ. Нити — только из почти световых шагов: шаг вправо допустим, если Δv/Δu < κ (то же — влево с Δu/Δv);
это условие инвариантно к бусту. Нить не обязана проходить через все события. Паросочетание: каждое событие — не более одного
шага вправо и одного предшественника справа (то же влево); «не продолжать нить» стоит D (D > κ — сеть стремится быть длиннее).
Узлы смешения — события, где продолжаются и правая, и левая нити: там монета; на остальных событиях нити — проход без смешения.
Проверяем: скорость нитей c, долю событий в сети, утечку нормы внутри (концы нитей в объёме), частоту покоя."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from goldlab import worlds as Wd, observe as Ob
def lines(X, side, kappa, cap, D=1.0):
    n = len(X); t, x = X[:, 0], X[:, 1]; rows, cols, vals = [], [], []
    for i in range(n):
        j0 = np.searchsorted(t, t[i] + 1e-9); j1 = np.searchsorted(t, t[i] + cap); js = np.arange(j0, j1); dx = x[js] - x[i]; dt = t[js] - t[i]
        r = (dt - side * dx) / np.maximum(dt + side * dx, 1e-12); ok = (dx * side > 1e-9) & (dt > np.abs(dx) + 1e-12) & (r < kappa)
        rows += [i] * ok.sum(); cols += list(js[ok]); vals += list(r[ok] + 1e-3 * dt[ok])
    rows += list(range(n)); cols += list(range(n, 2 * n)); vals += [D] * n
    rr, cc = min_weight_full_bipartite_matching(coo_matrix((vals, (rows, cols)), shape=(n, 2 * n)).tocsr()); nxt = -np.ones(n, int); nxt[rr] = cc; nxt[nxt >= n] = -1; return nxt
def walk(X, R, Lf, m, fR, fL):
    n = len(X); t = X[:, 0]; aR = fR.astype(complex).copy(); aL = fL.astype(complex).copy()
    pR = -np.ones(n, int); pR[R[R >= 0]] = np.nonzero(R >= 0)[0]; pL = -np.ones(n, int); pL[Lf[Lf >= 0]] = np.nonzero(Lf >= 0)[0]
    lastR = np.full(n, np.nan); lastL = np.full(n, np.nan)        # время последнего узла смешения на подходящей нити
    leak_in = 0.0; leak_edge = 0.0; Lmax = np.abs(t).max()
    for z in range(n):
        site = (R[z] >= 0) and (Lf[z] >= 0)
        bR, bL = aR[z], aL[z]
        if site:
            dR = t[z] - (lastR[z] if np.isfinite(lastR[z]) else t[z] - 0.79); dL = t[z] - (lastL[z] if np.isfinite(lastL[z]) else t[z] - 0.79)
            th = m * 0.5 * (dR + dL); c, s = np.cos(th), np.sin(th); bR, bL = c * aR[z] - 1j * s * aL[z], -1j * s * aR[z] + c * aL[z]
        for b, nx, last in ((bR, R[z], lastR), (bL, Lf[z], lastL)):
            if nx >= 0:
                if b is bR: aR[nx] += b
                else: aL[nx] += b
                last[nx] = t[z] if site else (lastR[z] if b is bR else lastL[z])
            elif abs(b) > 0:
                near_edge = abs(t[z]) + abs(X[z, 1]) > Lmax - 3
                if near_edge: leak_edge += abs(b) ** 2
                else: leak_in += abs(b) ** 2
    return aR, aL, leak_in, leak_edge
L = 24; w = Wd.gold11(4.0, L); X, t, x = w.X, w.t, w.x; n = w.n
for kappa, cap in ((1.0, 1.6), (0.3, 4.0), (0.1, 6.0), (0.03, 8.0)):
    R = lines(X, +1, kappa, cap); Lf = lines(X, -1, kappa, cap)
    sR = X[R[R >= 0]] - X[R >= 0]; sL = X[Lf[Lf >= 0]] - X[Lf >= 0]; c = 0.5 * (np.mean(sR[:, 1] / sR[:, 0]) - np.mean(sL[:, 1] / sL[:, 0]))
    inner = np.abs(t) + np.abs(x) < L - 4; onR = (R >= 0); onL = (Lf >= 0); site = onR & onL
    # норма: источник — события-узлы на одном слое, ветвь (1, 1)
    lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 5)))]; src = (np.abs(t - ts) < 1e-9) & site
    g = np.where(src, np.exp(-x * x / 50), 0); aR, aL, li, le = walk(X, R, Lf, 1.0, g, g); inj = 2 * (g ** 2).sum()
    sel = (t > ts + 1.5) & (t < L - 4) & (np.abs(x) < 8); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
    s_ = np.zeros(len(tu), complex); np.add.at(s_, inv, (np.exp(-x * x / 50) * (aR + aL))[sel]); om = np.linspace(-2, 0, 2001); S = Ob.spectrum(tu, s_, om); S /= max(S.max(), 1e-300)
    pk = Ob.peaks(om, S, 0.1)
    print(f'κ = {kappa}, окно {cap}: c = {c:.3f}; шаг Δt медиана {np.median(np.r_[sR[:, 0], sL[:, 0]]):.2f}; узлов смешения {site[inner].mean():.1%} событий; '
          f'источник {src.sum()} узлов; утечка внутри {li / inj:.1%}, на краю {le / inj:.1%}; частота покоя ' + ' '.join(f'{-e:.3f}' for e, q in pk), flush=True)
