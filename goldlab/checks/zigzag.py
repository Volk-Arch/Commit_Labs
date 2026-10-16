"""ПАУЛИ КАК ПРАВИЛО ПЕРЕСЕЧЕНИЙ (1+1). Зигзаг-частица на кристалле (шахматка Фейнмана): из каждого события — кратчайший шаг
вправо и кратчайший шаг влево (связи; вертикальные не берём). Прямо — множитель 1, поворот — −i·m·Δt.
Две одинаковые частицы:  F  = K₁₁K₂₂ − K₁₂K₂₁ (фермионы, определитель);  H = сумма по парам историй, которые НЕ ПЕРЕСЕКАЮТСЯ и не
встречаются (без знаков). В непрерывной 1+1 F = H (Карлин–Макгрегор / Жирардо). На кристалле — проверяем.
Пересечение отрезков — геометрия на плоскости (t, x); в 1+1 она выражается через порядок (два световых упорядочения)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
def orient(a, b, c): return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
def zig_graph(X):
    la, lb = Cs.staircase11(X); n = len(X); dt = X[lb, 0] - X[la, 0]; dx = X[lb, 1] - X[la, 1]; R = -np.ones(n, int); Lf = -np.ones(n, int)
    for i in range(n):
        s = la == i
        for side, sel in ((R, dx > 1e-9), (Lf, dx < -1e-9)):
            c = np.nonzero(s & sel)[0]
            if len(c): side[i] = lb[c[np.argmin(dt[c])]]
    EA, EB, D = [], [], []
    for i in range(n):
        for d, side in ((+1, R), (-1, Lf)):
            if side[i] >= 0: EA.append(i); EB.append(side[i]); D.append(d)
    return np.array(EA), np.array(EB), np.array(D)
def run(L, m, s_off=1.2, rho=4.0, verbose=True, rule=True):
    w = Wd.gold11(rho, L); X, t, x, n = w.X, w.t, w.x, w.n; EA, EB, Dir = zig_graph(X); E = len(EA)
    out = [[] for _ in range(n)]; inn = [[] for _ in range(n)]
    for e in range(E): out[EA[e]].append(e); inn[EB[e]].append(e)
    def wt(e_in, e_out):                                  # прямо — 1, поворот — −i m Δt (Δt выходящего шага)
        return 1.0 if Dir[e_in] == Dir[e_out] else -1j * m * (t[EB[e_out]] - t[EA[e_out]])
    near = lambda tt, xx: int(np.argmin(np.abs(t - tt) * 4 + np.abs(x - xx)))
    s1, s2 = near(-L + 3, -s_off), near(-L + 3, s_off); tl = t[s1]
    assert abs(t[s2] - tl) < 1e-9, 'источники должны быть на одном слое'
    tgt = np.unique(np.round(t, 9)); tgt = tgt[np.argmin(np.abs(tgt - (-L + 3 + 0.45 * L)))]     # слой-мишень
    P, Q = X[EA], X[EB]
    def single(s):
        a = np.zeros(E, complex); a[out[s]] = 1.0 / np.sqrt(2); arr = np.zeros(n, complex)
        for z in np.argsort(t, kind='stable'):
            if t[z] <= t[s] + 1e-12 or not inn[z]: continue
            if abs(t[z] - tgt) < 1e-9: arr[z] = a[inn[z]].sum(); continue          # прибыл на слой-мишень — дальше не идёт
            if t[z] > tgt: continue
            for ei in inn[z]:
                for eo in out[z]: a[eo] += a[ei] * wt(ei, eo)
                a[ei] = 0
        return arr
    K1, K2 = single(s1), single(s2)
    # две частицы: A[e1, e2]; правило: запрет встречи в событии и пересечения отрезков (жёсткие стенки)
    A = np.zeros((E, E), complex)
    for e1 in out[s1]:
        for e2 in out[s2]: A[e1, e2] = 0.5
    arrive = {}
    order = [z for z in np.argsort(t, kind='stable') if t[z] > tl + 1e-12 and t[z] <= tgt + 1e-9]
    for z in order:
        ins = inn[z]
        if not ins: continue
        at_t = abs(t[z] - tgt) < 1e-9
        if rule and not at_t: A[np.ix_(ins, ins)] = 0                            # встреча обеих в одном событии — запрещена
        for p in (0, 1):                                                         # частица 1 (строки) или 2 (столбцы)
            M = A if p == 0 else A.T                                              # вид: «своя» ось первая
            if at_t:
                continue
            for ei in ins:
                row = M[ei].copy()
                if not row.any(): continue
                M[ei] = 0
                for eo in out[z]:
                    a, b = P[eo], Q[eo]
                    cross = (orient(a, b, P) * orient(a, b, Q) < 0) & (orient(P, Q, a[None]) * orient(P, Q, b[None]) < 0)   # собственное пересечение
                    M[eo] += np.where(cross & rule, 0, row * wt(ei, eo))
    # прибытие обеих на слой-мишень
    Z = np.nonzero(np.abs(t - tgt) < 1e-9)[0]; Z = Z[np.argsort(x[Z])]; H = np.zeros((len(Z), len(Z)), complex)
    for a_, z1 in enumerate(Z):
        for b_, z2 in enumerate(Z):
            if inn[z1] and inn[z2]: H[a_, b_] = A[np.ix_(inn[z1], inn[z2])].sum()
    k1, k2 = K1[Z], K2[Z]; F = np.outer(k1, k2) - np.outer(k2, k1); Bz = np.outer(k1, k2) + np.outer(k2, k1)
    if not rule:
        P12 = np.outer(k1, k2); print(f'   без правила: |H − K₁⊗K₂|/|K₁⊗K₂| = {np.linalg.norm(H - P12) / np.linalg.norm(P12):.2e}', flush=True)
    up = np.triu_indices(len(Z), 1); h, f, bz = H[up], F[up], Bz[up]
    mask = (np.abs(x[Z][up[0]]) < 5) & (np.abs(x[Z][up[1]]) < 5)
    h, f, bz = h[mask], f[mask], bz[mask]
    r = (np.vdot(f, h)) / np.vdot(f, f)
    if verbose:
        print(f'L = {L}, m = {m}: рёбер {E}, пар состояний {E * E}; слой-мишень t = {tgt:.2f}, событий на нём {len(Z)}\n'
              f'   |H − F|/|F| = {np.linalg.norm(h - f) / np.linalg.norm(f):.3f};  H ≈ ({r.real:+.3f}{r.imag:+.3f}i)·F, остаток {np.linalg.norm(h - r * f) / np.linalg.norm(h):.3f};'
              f'  для сравнения |B − F|/|F| = {np.linalg.norm(bz - f) / np.linalg.norm(f):.3f}', flush=True)
    return h, f
if __name__ == '__main__':
    h, f = run(8, 1.0, rule=False, verbose=False)
    print('проверка счёта: без правила H должно быть K₁₁K₂₂ — см. ниже отдельно')
    for L, m in ((8, 0.0), (8, 1.0), (10, 1.0), (12, 1.0)):
        run(L, m)
