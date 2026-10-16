import sys, math, numpy as np
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of, pathsum
for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
    w = make(4.0, 16.0, np.random.default_rng(7)); R = build_order(w, 'proj'); L = links_of(R)
    I, J = np.nonzero(L); d = w.X[J] - w.X[I]; dW = w.W[J] - w.W[I]
    s2 = d[:, 0] ** 2 - d[:, 1] ** 2; sig2 = (dW ** 2).sum(1); tr = s2 - sig2
    print(f'{name}: след звена s² − σ² — отклонение от целого: max {np.abs(tr - np.round(tr)).max():.2e}; значения {np.unique(np.round(tr))[:8]}')
    src = int(np.argmin(np.abs(w.X[:, 0] + 12) + np.abs(w.X[:, 1])))
    tgt = [j for j in range(w.n) if R[src, j] and 10 <= w.tau(w.X[src], w.X[j]) <= 16 and abs(w.X[j, 1]) < 6]
    row = []
    for k in (math.pi / 2, math.pi, 2 * math.pi, 4 * math.pi, 2 * math.pi * 1.618034, 2 * math.pi + 0.3):
        Z, N = pathsum(w, L, src, lambda I_, J_: k * ((w.W[J_] - w.W[I_]) ** 2).sum(1))
        Zw, _ = pathsum(w, L, src, lambda I_, J_: k * ((w.X[J_, 0] - w.X[I_, 0]) ** 2 - (w.X[J_, 1] - w.X[I_, 1]) ** 2))
        r = np.median(np.abs(Z[tgt]) / N[tgt]); same = np.median(np.abs(np.abs(Z[tgt]) - np.abs(Zw[tgt])) / N[tgt])
        row.append(f'k={k:.3f}: R={r:.3g} (|Zтень|−|Zмир| ~ {same:.1e})')
    print('   ' + '\n   '.join(row)); sys.stdout.flush()
