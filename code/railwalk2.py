#!/usr/bin/env python3
"""ВОЛНОВОДЫ С ЯЧЕЙКАМИ-«КРЕСТАМИ».

Первая попытка (railwalk.py) показала: если разрешить перескоки на всех событиях рельсов, у части событий
два входа и один выход, и норма теряется (до 70 %). Световод так и не устроен: частица в нём перескакивает
не где попало, а только в элементе связи. Делаем так:
  * рельс — тонкая цепочка событий; без связи событие передаёт амплитуду только дальше по рельсу (1 → 1);
  * элемент связи — «крест» из звеньев кристалла: соседние события a → b′ на рельсе 1, зеркальные h(a) → h(b′)
    на рельсе 2, и перекрёстные звенья a → h(b′), h(a) → b′ (зеркало h их гарантирует парами).
    В a: (свой, перескок) = (cos θ, i sin θ)·вход; в b′: свой вход + перескок. Ячейка целиком —
    [[cos θ, i sin θ], [i sin θ, cos θ]], точно унитарна, хотя событие b′ «два входа — один выход».
Тогда вся установка унитарна точно, детекторы на концах рельсов ловят ВСЁ — отбора после подготовки нет.
Подготовка пары — как раньше (кристалл, правило «поворот = i»): это единственное место, где есть потери
(«попала ли пара в световоды»), и оно до настроек — как «оповещение» (heralding) в опыте."""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from beamsplit import smax
sys.argv = _a
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R); LK = CH.prep(w, Lk, 1); X = w.X; n = w.n
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)

def rail_of(A, length, delta=0.25):
    r = [k for k in range(n) if abs(Xa[k, 1] - Xa[A, 1]) < delta and Xa[A, 0] - 1e-9 <= Xa[k, 0] <= Xa[A, 0] + length
         and CH.img(w, k, H, Hs) >= 0 and np.abs(X[k]).sum() < 23]
    return sorted(r, key=lambda k: Xa[k, 0])

def crosses(r1):
    """ячейки: a на рельсе 1 с перескоком a → y на рельс 2; b′ = h(y) — где перескок h(a) → b′ приходит обратно.
    Между a и b′ амплитуда идёт по рельсу 1 (1 → 1), так что ячейка — точно [[cos, i sin], [i sin, cos]]"""
    r2 = [CH.img(w, k, H, Hs) for k in r1]; ev = r1 + r2
    Rs = R[np.ix_(ev, ev)]; Rf = Rs.astype(np.float32); L = Rs & ((Rf @ Rf) == 0)
    idx = {k: i for i, k in enumerate(ev)}
    cells = []
    r1set = set(r1)
    for a in r1:
        for y in r2:
            if L[idx[a], idx[y]]:                                  # перескок a → y на рельс 2
                b = CH.img(w, y, H, Hs)                             # его зеркальный партнёр b′ на рельсе 1
                if b in r1set and L[idx[CH.img(w, a, H, Hs)], idx[b]]: cells.append((a, b))
    return cells, L, idx

def device(cell_list, theta):
    """матрица установки (выход рельс 1, рельс 2) ← (вход A1, вход A2): цепочка ячеек"""
    T = np.eye(2, dtype=complex)
    for _ in cell_list: T = np.array([[math.cos(theta), 1j * math.sin(theta)], [1j * math.sin(theta), math.cos(theta)]]) @ T
    return T

if __name__ == '__main__':
    src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
    PS = {(s, m): CH.chiral(w, LK, R, s, m) for s in src for m in (0, 1)}
    for xa in (0.62, 1.00, 1.62):
        cand = [k for k in range(n) if 2.4 <= Xa[k, 0] <= 2.9 and abs(Xa[k, 1] - xa) < 0.01]
        if not cand: continue
        A1 = cand[0]; A2 = CH.img(w, A1, H, Hs); B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
        r1 = rail_of(A1, 4.5); cells, L, idx = crosses(r1)
        # проверка «вне ячеек 1 → 1»: у каждого события рельса ровно один свой выход (кроме последнего)
        own_ok = all(L[idx[a], idx[b]] for a, b in zip(r1[:-1], r1[1:]))
        print(f'\nрельсы x̃ = ±{xa}: событий на рельсе {len(r1)}; цепочка рельса без разрывов: {own_ok}; ячеек-«крестов» {len(cells)} '
              f'(на высоте T̃ = {[round(Xa[a, 0] - Xa[A1, 0], 2) for a, b in cells]})')
        if not cells: continue
        # пара: подготовка в кристалле
        for m0 in (0, 1):
            za = np.array([[PS[s, m][A1, m0], PS[s, m][A2, 1 - m0]] for s in src for m in (0, 1)])
            zb = np.array([[PS[s, m][B1, 1 - m0], PS[s, m][B2, m0]] for s in src for m in (0, 1)])
            M = za.T @ zb; Mn = M / np.linalg.norm(M)
            S_id = smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj())); C = 2 * abs(np.linalg.det(Mn))
            # установка: пластинка на рельсе 2 до ячейки, одна ячейка θ = 45° (у Боба — зеркальная копия)
            T = device(cells[:1], math.pi / 4)
            a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
            P = np.zeros((72, 72, 2, 2))
            for i, al in enumerate(a):
                TA = T @ np.diag([1, np.exp(1j * al)])
                for j, be in enumerate(a):
                    TB = T @ np.diag([1, np.exp(1j * be)])
                    P[i, j] = np.abs(TA @ Mn @ TB.T) ** 2       # БЕЗ нормировки на «сработал детектор»
            tot = P.sum((2, 3)); E = P[..., 0, 0] - P[..., 0, 1] - P[..., 1, 0] + P[..., 1, 1]
            mA = P[..., 0, :].sum(-1); sig = np.abs(mA - mA[:, :1]).max()
            bS = 0
            for i0 in range(72):
                S = np.abs(E[i0][:, None] + E[i0][None, :] + E[:, :, None] - E[:, None, :]); bS = max(bS, S.max())
            print(f'   вход {m0}: C = {C:.3f}, S ид. {S_id:.3f}; полная вероятность на детекторах {tot.min():.12f}…{tot.max():.12f}; '
                  f'сигнал {sig:.1e}; S без всякого отбора после подготовки = {bS:.3f}')
        sys.stdout.flush()

def two_cell_S(Mn, restarts=60, seed=0):
    """две ячейки θ = 45° (интерферометр Маха–Цендера) и две пластинки на рельсе 2 (до первой и между ячейками)"""
    from scipy.optimize import minimize
    C45 = device([0], math.pi / 4)
    def T(a, g): return C45 @ np.diag([1, np.exp(1j * g)]) @ C45 @ np.diag([1, np.exp(1j * a)])
    def S(x):
        TA = [T(x[0], x[1]), T(x[2], x[3])]; TB = [T(x[4], x[5]), T(x[6], x[7])]
        E = [[(lambda P: P[0, 0] - P[0, 1] - P[1, 0] + P[1, 1])(np.abs(ta @ Mn @ tb.T) ** 2) for tb in TB] for ta in TA]
        return abs(E[0][0] + E[0][1] + E[1][0] - E[1][1])
    rs = np.random.default_rng(seed); best = 0
    for _ in range(restarts):
        r = minimize(lambda x: -S(x), rs.uniform(0, 2 * math.pi, 8), method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        best = max(best, -r.fun)
    return best

if __name__ == '__main__':
    print('\nдве ячейки + две пластинки на сторону (любая ось измерения):')
    for xa in (0.62, 1.00, 1.62):
        cand = [k for k in range(n) if 2.4 <= Xa[k, 0] <= 2.9 and abs(Xa[k, 1] - xa) < 0.01]
        A1 = cand[0]; A2 = CH.img(w, A1, H, Hs); B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
        cells, _, _ = crosses(rail_of(A1, 4.5))
        if len(cells) < 2: print(f'   x̃ = ±{xa}: ячеек меньше двух'); continue
        for m0 in (0, 1):
            za = np.array([[PS[s, m][A1, m0], PS[s, m][A2, 1 - m0]] for s in src for m in (0, 1)])
            zb = np.array([[PS[s, m][B1, 1 - m0], PS[s, m][B2, m0]] for s in src for m in (0, 1)])
            M = za.T @ zb; Mn = M / np.linalg.norm(M)
            print(f'   x̃ = ±{xa}, вход {m0}: S ид. {smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj())):.3f}; S без отбора = {two_cell_S(Mn):.3f}')
