"""Теорема: в золотом кристалле нет световых пар; наименьший |интервал²| ≥ 1/(4r²) (норма Галуа). Замер."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab.core import crystal, r_for_density, TAU, ST, SI
for rho, L in ((4.0, 10.0), (4.0, 14.0), (8.0, 10.0)):
    r = r_for_density(rho, 2); X = crystal([(-L, L), (-L, L)], r); X = X[np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L]
    d = X[None] - X[:, None]; q = d[..., 0] ** 2 - d[..., 1] ** 2; iu = np.triu_indices(len(X), 1); q = q[iu]
    tl = q[q > 1e-12]; sl = -q[q < -1e-12]; z = np.sum(np.abs(q) <= 1e-9)
    print(f'rho={rho} L={L} n={len(X)} пар={len(q)} нулевых={z}  min тайм q={tl.min():.4f} min простр -q={sl.min():.4f}  оценка 1/(4r²)={1/(4*r*r):.4f}  (τ_min={math.sqrt(tl.min()):.3f})')
    # значения интервала дискретны: сколько разных q < 1
    print('   разные значения q в (0, 1):', len(np.unique(np.round(tl[tl < 1], 7))), ' первые:', np.unique(np.round(tl[tl < 1], 5))[:6])
