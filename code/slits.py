#!/usr/bin/env python3
"""ДВЕ ЩЕЛИ в золотом кристалле с фазой из тени. Источник A, две «щели» — малые группы событий
слева и справа, экран — события в полосе по времени. Амплитуда через щель c:
Z_c(B) = Σ_{s ∈ щель c} Z(A→s)·Z(s→B), фаза звена e^{ik·σ²} (теневая часть следа).
Интерференционный член cos(arg Z1 − arg Z2) вдоль экрана: полосы, если фаза гладкая по x."""
import sys, math, numpy as np
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of, pathsum
k = float(sys.argv[1]) if len(sys.argv) > 1 else math.pi
for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
    w = make(4.0, 18.0, np.random.default_rng(11)); R = build_order(w, 'proj'); L = links_of(R)
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    X = w.X
    A = int(np.argmin(np.abs(X[:, 0] + 14) + np.abs(X[:, 1])))
    sl = {c: [j for j in range(w.n) if abs(X[j, 0] + 4) < 0.5 and abs(X[j, 1] - c * 3.0) < 0.5 and R[A, j]] for c in (-1, 1)}
    scr = [j for j in range(w.n) if abs(X[j, 0] - 9) < 0.6 and abs(X[j, 1]) < 7]
    scr = sorted(scr, key=lambda j: X[j, 1])
    ZA, NA = pathsum(w, L, A, ph)
    Zc = {}
    for c in (-1, 1):
        acc = np.zeros(w.n, complex)
        for s in sl[c]:
            Zs, Ns = pathsum(w, L, s, ph); acc += ZA[s] * Zs
        Zc[c] = acc
    z1, z2 = Zc[-1][scr], Zc[1][scr]; ok = (np.abs(z1) > 0) & (np.abs(z2) > 0)
    x = X[scr, 1][ok]; dphi = np.angle(z1[ok] * np.conj(z2[ok]))
    c = np.cos(dphi); I = np.abs(z1[ok] + z2[ok]) ** 2; I0 = np.abs(z1[ok]) ** 2 + np.abs(z2[ok]) ** 2
    # гладкость: связь соседних точек экрана
    smooth = np.corrcoef(c[:-1], c[1:])[0, 1] if len(c) > 3 else float('nan')
    print(f'{name}, k = {k:.3f}: щели {len(sl[-1])} и {len(sl[1])} событий, экран {ok.sum()} точек; '
          f'связь cos Δφ у соседей по экрану {smooth:+.3f}; средний |I/I0 − 1| {np.mean(np.abs(I / I0 - 1)):.3f}')
    order = np.argsort(x)
    print('   x:     ' + ' '.join(f'{v:+5.1f}' for v in x[order][::max(1, len(x) // 24)]))
    print('   cosΔφ: ' + ' '.join(f'{v:+5.2f}' for v in c[order][::max(1, len(x) // 24)]))
    sys.stdout.flush()
