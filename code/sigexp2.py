#!/usr/bin/env python3
"""КОНТРОЛЬ и продолжение замера sigma(S).

Контроль 1. Счётчик `counts4` сверяется с НЕЗАВИСИМОЙ реализацией
`bdcontrol.bd_from_poset`, которая строит матрицу отношений, замыкает
её и считает N_m своим путём. Именно здесь у меня уже был баг (вес
w(0) применялся к несвязанным парам), и тогда его поймала ровно эта
сверка.

Контроль 2. Показатель самого числа связей: если <N0> растёт как
N^{1.70}, замер идёт в переходной области (асимптотика N^{1.5}), и
показатель sigma(S) может ещё поехать. Мерю <N_m> отдельно.

Контроль 3. Пол дробового шума: беру те же счёты, но РАЗРУШАЮ
корреляцию между ними — перемешиваю каждый столбец по засевам
независимо. Если наблюдаемый разброс суммы равен перемешанному,
корреляций нет и работает простой подсчёт; если меньше —
сокращение держится корреляциями.
"""
import math

import numpy as np

from qcarrier import sprinkle4
from sigexp import counts4, ens, fit, K4


W4 = {0: -1.0, 1: 9.0, 2: -16.0, 3: 8.0}


def bd_slow(P):
    """S/k_4 на множествах: |I(x,y)| прямым пересечением, без матриц.

    Независимый путь кода: отношение проверяется поэлементно из
    координат, промежуточные элементы считаются пересечением множеств
    будущего и прошлого. Ни одного matmul, ни float32.
    """
    n = len(P)
    pt = [tuple(float(v) for v in P[i]) for i in range(n)]

    def prec(i, j):
        dt = pt[j][0] - pt[i][0]
        if dt <= 0.0:
            return False
        s = sum((pt[j][k] - pt[i][k]) ** 2 for k in (1, 2, 3))
        return dt * dt > s

    fut = [set(j for j in range(n) if prec(i, j)) for i in range(n)]
    pas = [set(i for i in range(n) if prec(i, j)) for j in range(n)]
    tot = float(n)
    for i in range(n):
        for j in fut[i]:
            m = len(fut[i] & pas[j])
            tot += W4.get(m, 0.0)
    return K4 * tot


def ctrl_counter(rs, trials=12, n=18):
    print('КОНТРОЛЬ 1: счётчик против независимой реализации на множествах')
    bad = 0
    for _ in range(trials):
        P = sprinkle4(n, rs, 1.0)
        nn, N0, N1, N2, N3 = counts4(P)
        S_mine = K4 * (nn - N0 + 9 * N1 - 16 * N2 + 8 * N3)
        S_ctl = bd_slow(P)
        if abs(S_mine - S_ctl) > 1e-9:
            bad += 1
            print(f'  РАСХОЖДЕНИЕ: мой {S_mine:.6f} против '
                  f'контрольного {S_ctl:.6f}')
    print(f'  {trials - bad}/{trials} совпали точно'
          f'{"" if bad == 0 else "  <-- БАГ"}')
    big = sprinkle4(260, rs, 1.0)
    nn, N0, N1, N2, N3 = counts4(big)
    sm = K4 * (nn - N0 + 9 * N1 - 16 * N2 + 8 * N3)
    sc = bd_slow(big)
    print(f'  и на N=260 (проверка float32-свёртки F@F): '
          f'{sm:.6f} против {sc:.6f}, разность {abs(sm-sc):.2e}')
    return bad == 0 and abs(sm - sc) < 1e-9


def ctrl_mean_scaling(rs):
    print('\nКОНТРОЛЬ 2: показатели самих счётов (переходная область?)')
    plan = [(200, 60), (400, 40), (800, 25), (1600, 14), (3200, 6)]
    Ns, M = [], []
    for n, k in plan:
        rows = [counts4(sprinkle4(n, rs, 1.0)) for _ in range(k)]
        A = np.array(rows, dtype=float)
        Ns.append(n)
        M.append(A.mean(axis=0))
        print(f'  N={n:>5}  <N0>={M[-1][1]:>9.0f}  <N1>={M[-1][2]:>8.0f}'
              f'  <N2>={M[-1][3]:>8.0f}  <N3>={M[-1][4]:>8.0f}', flush=True)
    M = np.array(M)
    lx = np.log(Ns)
    for i, nm in zip((1, 2, 3, 4), ('N0', 'N1', 'N2', 'N3')):
        p = np.polyfit(lx, np.log(M[:, i]), 1)[0]
        loc = (math.log(M[-1, i] / M[-2, i])
               / math.log(Ns[-1] / Ns[-2]))
        print(f'  <{nm}> ~ N^{p:.3f}  (на последнем шаге N^{loc:.3f}, '
              f'асимптотика 1.5)')
    return float(np.polyfit(lx, np.log(M[:, 1]), 1)[0])


def ctrl_shuffle(rs, n=800, k=200):
    print('\nКОНТРОЛЬ 3: пол дробового шума (разрушить корреляции)')
    S, A = ens(n, k, rs, False)
    coef = np.array([1.0, -1.0, 9.0, -16.0, 8.0])
    B = A.copy()
    for j in range(B.shape[1]):
        B[:, j] = rs.permutation(B[:, j])
    S_sh = K4 * (B @ coef)
    print(f'  наблюдаемый  sigma(S) = {S.std(ddof=1):>8.1f}')
    print(f'  перемешанный sigma(S) = {S_sh.std(ddof=1):>8.1f}')
    r = S_sh.std(ddof=1) / S.std(ddof=1)
    print(f'  отношение = {r:.2f} — {"корреляции держат сокращение" if r > 1.15 else "корреляций нет, простой подсчёт применим"}')


def main():
    print(__doc__)
    rs = np.random.default_rng(271828)
    ok = ctrl_counter(rs)
    if not ok:
        print('\nсчётчик не прошёл — дальше не считаю')
        return
    ctrl_mean_scaling(rs)
    ctrl_shuffle(rs)
    print('\nПРОДОЛЖЕНИЕ: показатель sigma(S) с N=3200 (поедет ли вниз)')
    plan = [(200, 400), (400, 300), (800, 200), (1600, 90), (3200, 40)]
    Ns, sd, ns = [], [], []
    for n, k in plan:
        S, _ = ens(n, k, rs, False)
        Ns.append(n)
        sd.append(float(S.std(ddof=1)))
        ns.append(k)
        print(f'  N={n:>5} засевов={k:>4}  sigma(S)={sd[-1]:>9.1f}',
              flush=True)
    a, e, rr = fit(np.array(Ns), np.array(sd), ns)
    print(f'  полный ряд: sigma(S) ~ N^{a:.4f} ± {e:.4f}')
    print('  остатки: ' + ', '.join(f'{v:+.1f}' for v in rr))
    for i in range(len(Ns) - 1):
        loc = math.log(sd[i + 1] / sd[i]) / math.log(Ns[i + 1] / Ns[i])
        print(f'  местный показатель {Ns[i]:>5}->{Ns[i+1]:<5} = {loc:.3f}')


if __name__ == '__main__':
    main()
