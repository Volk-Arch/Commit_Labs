#!/usr/bin/env python3
"""Шаг роста вместо суммы по историям: считается ли фаза локально.

ПОВОД. Возражение: мы ищем крошечный остаток от сокращения гигантских
чисел, а это не нужно, потому что цепочка не в один момент — для
динамики достаточно нескольких шагов, а не всех историй сразу. Это та же
мысль, что про кадр. В физике она известна как переход от интеграла по
историям к пошаговой эволюции, и знаковая проблема при этом уходит
(тензорные сети, кластерные алгоритмы, дуальные переменные).

УПРОЩЕНИЕ, КОТОРОЕ НАДО ПРОВЕРИТЬ. Если новое событие добавляется как
МАКСИМАЛЬНОЕ (позже всех), оно не лежит ни в одном интервале, потому что
для этого нужен кто-то ещё позже. Значит приращение действия — только
пары нового события с его прошлым:

    dS_k = eps + sum_{b < x_k} W(n(x_k, b))

а это ровно «сумма на событие», которую считает weight.per_event. Если
так, то все N приращений получаются одним матричным умножением, и
проверять надо не S, а dS.

ЧТО РЕШАЕТ ДЕЛО. Не разброс dS, а СРЕДНЯЯ ФАЗА |<exp(i dS)>|. Для
гауссова dS со среднеквадратичным sigma она равна exp(-sigma^2/2). Если
на шаг она близка к единице, знакопеременная сумма разыгрывается; если
на весь мир она равна e^(-1700), то глобально нет. Это и есть цена
локальности, в правильных единицах.

  python3 step.go.py
  python3 step.go.py --ts 1.0 1.5 2.0 2.5 3.0
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel, between, w_of, per_event


def check_maximal(T, eps, rs):
    """Сверка: dS при добавлении максимального = сумма на событие."""
    X = diamond(T, rs)
    X = X[np.argsort(X[:, 0])]                 # по времени
    R = cone_rel(X)
    btw = between(R)
    s, _ = per_event(R, btw, eps)
    wt = w_of(np.arange(len(X) + 1), eps)
    worst = 0.0
    for k in (len(X) // 3, len(X) // 2, 2 * len(X) // 3, len(X) - 1):
        P, Q = X[:k], X[:k + 1]
        Sp, Sq = full_S(P, eps, wt), full_S(Q, eps, wt)
        worst = max(worst, abs((Sq - Sp) - s[k]))
    return worst, len(X)


def full_S(X, eps, wt):
    R = cone_rel(X)
    A = np.asarray(R, np.float32)
    b = np.rint(A @ A).astype(np.int32)
    return float(eps * len(X) + wt[np.clip(b, 0, len(wt) - 1)][R].sum())


def main():
    ap = argparse.ArgumentParser(description='Фаза шага роста')
    ap.add_argument('--ts', type=float, nargs='+',
                    default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument('--eps', type=float, nargs='+',
                    default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument('--seeds', type=int, default=4)
    a = ap.parse_args()

    t0 = time.time()
    print('ПРОВЕРКА УПРОЩЕНИЯ: dS максимального = сумма на событие')
    rs = np.random.default_rng(77)
    w, n = check_maximal(1.2, 0.2, rs)
    print(f'  N={n}, наибольшее расхождение по четырём шагам: {w:.3e}')
    print('  (ноль означает, что максимальное событие ни в одном '
          'интервале не лежит)')
    print()

    print('ФАЗА ШАГА. Мир растёт при постоянной плотности, N = 150*T^4.')
    print('Берутся события верхней половины по времени — у них есть прошлое.')
    print(f"{'eps':>6}{'T':>5}{'N':>7}{'<dS>':>9}{'разброс':>9}"
          f"{'фаза шага':>12}{'разброс S':>11}{'фаза мира':>12}")

    tab = {}
    for T in a.ts:
        for t in range(a.seeds):
            rs = np.random.default_rng(880 + 17 * t)
            X = diamond(T, rs)
            X = X[np.argsort(X[:, 0])]
            R = cone_rel(X)
            btw = between(R)
            N = len(X)
            for eps in a.eps:
                s, _ = per_event(R, btw, eps)
                up = s[N // 2:]                      # верхняя половина
                tab.setdefault((eps, T), []).append(
                    (up.mean(), up.std(), float(s.sum()), N))
            del R, btw, X
    for eps in a.eps:
        for T in a.ts:
            v = np.array([r[:3] for r in tab[(eps, T)]], float)
            N = tab[(eps, T)][0][3]
            m, sd = v[:, 0].mean(), v[:, 1].mean()
            ph = math.exp(-sd * sd / 2)
            sS = v[:, 2].std()
            phw = math.exp(-min(sS * sS / 2, 700))
            print(f'{eps:>6.2f}{T:>5.1f}{N:>7}{m:>9.4f}{sd:>9.4f}'
                  f'{ph:>12.4f}{sS:>11.2f}'
                  f'{phw:>12.1e}', flush=True)
        print()

    print('МАСШТАБИРОВАНИЕ РАЗБРОСА ШАГА ПО N  и  до какого N фаза > 0.1')
    print(f"{'eps':>6}{'разброс при наиб. N':>21}{'показатель по N':>17}"
          f"{'N, где фаза=0.1':>18}")
    ns = np.log([150.0 * T ** 4 for T in a.ts])
    for eps in a.eps:
        sds = np.array([np.mean([r[1] for r in tab[(eps, T)]])
                        for T in a.ts])
        p = np.polyfit(ns, np.log(sds), 1)[0]
        # фаза 0.1 при sigma = sqrt(2 ln 10) = 2.146
        A = sds[-1] / (150.0 * a.ts[-1] ** 4) ** p
        nstar = (2.1460 / A) ** (1.0 / p) if p > 0 else float('inf')
        print(f'{eps:>6.2f}{sds[-1]:>21.4f}{p:>17.3f}'
              f'{nstar:>18.3g}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
