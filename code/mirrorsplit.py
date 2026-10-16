#!/usr/bin/env python3
"""СВЕТОДЕЛИТЕЛЬ ИЗ СИММЕТРИИ КРИСТАЛЛА.

Кристалл (окно в центре, без сдвигов) точно симметричен относительно отражения x → −x (тень
x′ → −x′, круг окна на месте) и буста η₀ = 1.0613 (тень поворачивается на 128.17°, круг на месте).
Возле любой точки есть и почти точное зеркало: x → v − x, где v = a + bτ — число решётки с малой
тенью v′ (окно сдвигается на v′, меняются только события у самого края окна).

Делитель Алисы: пути A₁ и A₂ = зеркало(A₁), порты p и q = зеркало(p). По симметрии
T = [[a, b], [b, a]] — уравновешен сам. Без потерь (с точностью до общего множителя) он, когда
фазы a и b отличаются на 90°. Боб — точное отражение Алисы относительно x = 0.
Проверяем: насколько точна симметрия, есть ли делители без перекоса, и Белл с пластинкой.
"""
import sys, math, itertools, numpy as np
from scipy.optimize import minimize
from bentcut import zt_range, ST, SI, TAU
from universe import World, build_order
from phasesum import links_of
from phaseplate import amp_sum
from beamsplit import smax

def sym_world(rho, L):
    r = math.sqrt(5 * rho / math.pi)
    t, tp = zt_range(-L / ST, L / ST, -r / SI, r / SI); T, Tq = t * ST, tp * SI
    c, cp = zt_range(-L, L, -r, r)
    i, j = np.nonzero(Tq[:, None] ** 2 + cp[None, :] ** 2 <= r * r)
    X = np.stack([T[i], c[j]], 1); W = np.stack([Tq[i], cp[j]], 1)
    m = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L
    return World(X[m], W[m], 'flat', rho, r, region=L)

def find(w, X, Wsh, tol=1e-6):
    d = np.abs(w.X - X).sum(1) + np.abs(w.W - Wsh).sum(1); k = int(np.argmin(d)); return k if d[k] < tol else -1

def lattice_v(target, maxshadow=0.06):
    best = None
    for b in range(-40, 41):
        a = round(target - b * TAU); v, vs = a + b * TAU, a - b / TAU
        if abs(v - target) < 0.6 and abs(vs) < maxshadow and (best is None or abs(vs) < abs(best[1])): best = (v, vs)
    return best

def mirror(w, k, v, vs):
    """событие-зеркало k относительно x → v − x (если оно есть в мире)"""
    return find(w, np.array([w.X[k, 0], v - w.X[k, 1]]), np.array([w.W[k, 0], vs - w.W[k, 1]]))

BAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
if __name__ == '__main__':
    rho, L, k_ph = 4.0, 16.0, math.pi
    w = sym_world(rho, L); R = build_order(w, 'proj'); Lk = links_of(R); X = w.X
    ph = lambda I, J: k_ph * ((w.W[J] - w.W[I]) ** 2).sum(1)
    # проверка точной симметрии x → −x
    miss = sum(find(w, np.array([X[k, 0], -X[k, 1]]), np.array([w.W[k, 0], -w.W[k, 1]])) < 0 for k in range(w.n))
    print(f'мир: событий {w.n}; без зеркального двойника (x → −x): {miss}')
    v, vs = lattice_v(6.854, 0.2); print(f'местное зеркало: v = {v:.4f} (центр {v / 2:.3f}), тень v′ = {vs:+.4f}, окно r = {w.r:.3f}')
    # источник: полоса в момент −7 вокруг x = 0 (симметрична)
    src = [j for j in range(w.n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
    # пути Алисы: события около t ≈ 1.5, x ≈ 2, и их зеркала
    pairsA = []
    for k in range(w.n):
        if abs(X[k, 0] - 1.5) < 0.5 and 1.0 < X[k, 1] < 3.0:
            m = mirror(w, k, v, vs)
            if m >= 0 and all(R[s, k] and R[s, m] for s in src[:1]): pairsA.append((k, m))
    print(f'пар путей Алисы с точным местным зеркалом: {len(pairsA)}')
    results = []
    Zsrc = [amp_sum(w, Lk, s, ph, 'past') for s in src]
    for (A1, A2) in pairsA[:6]:
        B1 = find(w, np.array([X[A1, 0], -X[A1, 1]]), np.array([w.W[A1, 0], -w.W[A1, 1]]))
        B2 = find(w, np.array([X[A2, 0], -X[A2, 1]]), np.array([w.W[A2, 0], -w.W[A2, 1]]))
        if B1 < 0 or B2 < 0: continue
        za = np.array([[Z[A1], Z[A2]] for Z in Zsrc]); zb = np.array([[Z[B1], Z[B2]] for Z in Zsrc])
        M = np.einsum('ri,rj->ij', za, zb); psi = M.reshape(4) / np.linalg.norm(M); S_ideal = smax(np.outer(psi, psi.conj()))
        Z1, Z2 = amp_sum(w, Lk, A1, ph, 'past'), amp_sum(w, Lk, A2, ph, 'past')
        # порты: p около t ≈ 6, и его зеркало q
        best = []
        for p in range(w.n):
            if abs(X[p, 0] - 6) < 0.6 and R[A1, p] and R[A2, p]:
                q = mirror(w, p, v, vs)
                if q < 0 or q == p or not (R[A1, q] and R[A2, q]): continue
                T = np.array([[Z1[p], Z2[p]], [Z1[q], Z2[q]]]); nrm = (np.abs(T) ** 2).sum()
                if nrm == 0: continue
                symerr = (abs(T[0, 0] - T[1, 1]) + abs(T[0, 1] - T[1, 0])) / math.sqrt(nrm)
                bal = 2 * abs(T[0, 0]) * abs(T[0, 1]) / (abs(T[0, 0]) ** 2 + abs(T[0, 1]) ** 2)
                best.append(((2 * abs(np.linalg.det(T)) / nrm) * (bal >= BAL), symerr, T, p, q))
        best.sort(key=lambda t: -t[0])
        if not best: continue
        uA, se, TA, p, q = best[0]
        # Боб — отражение Алисы: та же матрица (точная симметрия x → −x)
        TB = TA
        def P(al, be):
            amp = (TA @ np.diag([1, np.exp(1j * al)])) @ M @ (TB @ np.diag([1, np.exp(1j * be)])).T
            pp = np.abs(amp) ** 2; return pp / pp.sum()
        E = lambda pp: pp[0, 0] - pp[0, 1] - pp[1, 0] + pp[1, 1]
        def score(x):
            Ps = [[P(x[0], x[2]), P(x[0], x[3])], [P(x[1], x[2]), P(x[1], x[3])]]
            sig = max(abs(Ps[0][0].sum(1) - Ps[0][1].sum(1)).max(), abs(Ps[1][0].sum(1) - Ps[1][1].sum(1)).max(),
                      abs(Ps[0][0].sum(0) - Ps[1][0].sum(0)).max(), abs(Ps[0][1].sum(0) - Ps[1][1].sum(0)).max())
            return abs(E(Ps[0][0]) + E(Ps[0][1]) + E(Ps[1][0]) - E(Ps[1][1])), sig
        rs = np.random.default_rng(1); bS = (0, 1)
        for _ in range(40):
            res = minimize(lambda x: -score(x)[0] + 60 * max(0, score(x)[1] - 0.005), rs.uniform(0, 2 * math.pi, 4), method='Nelder-Mead', options=dict(maxiter=3000))
            S, sig = score(res.x)
            if sig <= 0.005 and S > bS[0]: bS = (S, sig)
        print(f'пути A ({X[A1,1]:.2f}, {X[A2,1]:.2f}): S идеальный {S_ideal:.3f}; делитель: качество {uA:.4f}, ошибка симметрии {se:.1e}, '
              f'|a|/|b| = {abs(TA[0,0]) / max(abs(TA[0,1]), 1e-30):.2f}, разность фаз {math.degrees(abs(np.angle(TA[0,0] * np.conj(TA[0,1])))):.1f}°; '
              f'S с пластинкой {bS[0]:.3f} (сигнал {bS[1]:.4f})'); sys.stdout.flush()
