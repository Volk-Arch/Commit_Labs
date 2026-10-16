#!/usr/bin/env python3
"""ФАЗА ИЗ ТЕНИ: сумма по путям (по связям) с амплитудой e^{iφ} на каждое звено.

След разности распадается на две части: наш интервал s² = ΔT² − Δx² и теневой σ² = ΔT'² + Δx'².
Кандидаты фазы звена:  act — φ = k·σ² (теневая часть следа как «действие»),
                       ang — φ = n·угол теневого вектора звена.
Z(B) = Σ по путям A→B Π e^{iφ}. Когерентность R = |Z| / (число путей): 1 — все пути в фазе,
~1/√(число путей) — фазы случайны. Сравниваем кристалл и засев, порядок по срезу и по следу.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe import golden_flat, poisson_flat, build_order

def links_of(R):
    Rf = R.astype(np.float32); return R & ((Rf @ Rf) == 0)

def pathsum(w, L, src, phase):
    """Z[j] по всем событиям из src; phase(i, j) -> массив фаз для звеньев."""
    n = w.n; Z = np.zeros(n, complex); N = np.zeros(n); Z[src] = 1; N[src] = 1
    I, J = np.nonzero(L); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    ph = phase(I, J); st = np.searchsorted(J, np.arange(n)); en = np.searchsorted(J, np.arange(n), side='right')
    for j in range(src + 1, n):
        a, b = st[j], en[j]
        if a == b: continue
        ii = I[a:b]; N[j] = N[ii].sum(); Z[j] = (Z[ii] * np.exp(1j * ph[a:b])).sum()
    return Z, N

if __name__ == '__main__':
    rs = np.random.default_rng(4)
    worlds = [('кристалл, по срезу', golden_flat, 'proj'), ('засев, по срезу', poisson_flat, 'proj'),
              ('кристалл, по следу', golden_flat, 'trace'), ('засев, по следу', poisson_flat, 'trace')]
    ks = [0.05, 0.2, 0.5, 1.0, 2.0]
    for name, make, rule in worlds:
        w = make(4.0, 16.0, np.random.default_rng(7)); R = build_order(w, rule); L = links_of(R)
        src = int(np.argmin(np.abs(w.X[:, 0] + 12) + np.abs(w.X[:, 1])))       # у низа ромба
        tgt = [j for j in range(w.n) if R[src, j] and 10 <= w.tau(w.X[src], w.X[j]) <= 16 and abs(w.X[j, 1]) < 6]
        dW = lambda I, J: w.W[J] - w.W[I]
        out = []
        for k in ks:
            Z, N = pathsum(w, L, src, lambda I, J: k * (dW(I, J) ** 2).sum(1))
            r = np.abs(Z[tgt]) / N[tgt]; out.append((np.median(r), np.median(1 / np.sqrt(N[tgt]))))
        Z1, N1 = pathsum(w, L, src, lambda I, J: np.arctan2(dW(I, J)[:, 1], dW(I, J)[:, 0]))
        r1 = np.median(np.abs(Z1[tgt]) / N1[tgt])
        print(f'{name}: событий {w.n}, целей {len(tgt)}, путей до цели ~10^{np.log10(np.median(N[tgt])):.0f}')
        print('   фаза k·σ²: ' + ', '.join(f'k={k}: R={o[0]:.3g}' for k, o in zip(ks, out)) + f'   (случайные фазы ≈ {out[0][1]:.1e})')
        print(f'   фаза = угол теневого звена: R = {r1:.3g}'); sys.stdout.flush()
