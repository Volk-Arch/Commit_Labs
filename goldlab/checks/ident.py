"""МЕРА ВЕРШИНЫ ИЗ КРИСТАЛЛА: проверка тождества свёртки по одновременному слою с ТОЧНЫМ c = −i.
∫dx_z G(i → z)G(z → j) ≈ (−i/2m)·G(i → j). Веса события z на слое: (a) 1/(ρ·Δt_слоя) — прежняя «мера уровня»;
(b) Δx_z — длина ячейки Вороного события вдоль своего слоя (половина расстояний до соседей слева и справа) — из кристалла."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, core
L, rho = 16.0, 4.0; w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n
fwd = t[None, :] > t[:, None]; bwd = t[None, :] < t[:, None]
LV = list(core.layers(w.X)); tl = np.array([q for q, _ in LV]); gap = np.diff(tl); half = 0.5 * (np.r_[gap[0], gap] + np.r_[gap, gap[-1]])
dx = np.zeros(n); lay = np.zeros(n, int)
for k, (q, idx) in enumerate(LV):
    o = idx[np.argsort(x[idx])]; xs = x[o]; lay[o] = k
    if len(o) == 1: dx[o] = np.nan; continue
    d = np.diff(xs); dx[o] = 0.5 * (np.r_[d[0], d] + np.r_[d, d[-1]])
for m in (1.0, 0.6):
    Wp, Wm, _ = Pr.sj(Pr.johnston11(w.X, rho, m)); G = -1j * (Wp * fwd + Wm * bwd); del Wp, Wm
    I = np.nonzero((np.abs(t + 4) < 0.5) & (np.abs(x) < 2))[0]; J = np.nonzero((np.abs(t - 4) < 0.5) & (np.abs(x) < 2))[0]; den = G[np.ix_(I, J)]
    print(f'\nm = {m}: ожидание R/G = −i/(2m) = {-1 / (2 * m):+.3f}i')
    print('  слой t     событий  ρΔt     ⟨1/Δx⟩ |  (a) 1/(ρΔt): R/G·2m      |  (b) Вороной Δx: R/G·2m')
    for k in np.nonzero(np.abs(tl) < 1.6)[0]:
        Z = LV[k][1]; 
        for name, wz in (('a', np.full(len(Z), 1 / (rho * rho * half[k] / rho) / rho) if False else 1 / (rho * half[k]) * np.ones(len(Z))), ('b', dx[Z])):
            num = (G[np.ix_(I, Z)] * wz[None, :]) @ G[np.ix_(Z, J)]; r = (num.ravel() @ den.ravel().conj()) / (den.ravel() @ den.ravel().conj())
            res = np.linalg.norm(num - r * den) / np.linalg.norm(num)
            if name == 'a': ra, resa = r, res
            else: rb, resb = r, res
        print(f'  {tl[k]:+.3f}  {len(Z):6d}  {rho * half[k]:.3f}  {np.mean(1 / dx[Z]):6.3f} |  {abs(ra) * 2 * m:.3f} ∠{np.degrees(np.angle(ra)) :+6.1f}° ост {resa:.2f} |  {abs(rb) * 2 * m:.3f} ∠{np.degrees(np.angle(rb)):+6.1f}° ост {resb:.2f}', flush=True)
