import sys, math, numpy as np
from universe import golden_flat, build_order
from phasesum import links_of, pathsum
k = math.pi
for seed in (11, 12):
    w = golden_flat(4.0, 20.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    A = int(np.argmin(np.abs(X[:, 0] + 15) + np.abs(X[:, 1]))); ZA, _ = pathsum(w, L, A, ph)
    scr = sorted([j for j in range(w.n) if abs(X[j, 0] - 10) < 0.6 and abs(X[j, 1]) < 8], key=lambda j: X[j, 1])
    cache = {}
    for d in (2.0, 3.0, 4.0):
        Zc = {}
        for c in (-1, 1):
            sl = [j for j in range(w.n) if abs(X[j, 0] + 4) < 0.5 and abs(X[j, 1] - c * d) < 0.5 and R[A, j]]
            acc = np.zeros(w.n, complex)
            for s in sl:
                if s not in cache: cache[s] = pathsum(w, L, s, ph)[0]
                acc += ZA[s] * cache[s]
            Zc[c] = acc
        z1, z2 = Zc[-1][scr], Zc[1][scr]; x = X[scr, 1]; ok = (np.abs(z1) > 0) & (np.abs(z2) > 0)
        c = np.cos(np.angle(z1 * np.conj(z2)))[ok]; xx = x[ok]
        # первые нули справа и слева от центрального максимума
        zc = [0.5 * (xx[i] + xx[i + 1]) for i in range(len(c) - 1) if c[i] * c[i + 1] < 0]
        right = [z for z in zc if z > 0]; left = [z for z in zc if z < 0]
        print(f'зерно {seed}, щели на ±{d:g}: cos Δφ в центре {c[np.argmin(np.abs(xx))]:+.2f}; первые нули слева {max(left) if left else None}, справа {min(right) if right else None}; всего нулей {len(zc)}')
        sys.stdout.flush()
