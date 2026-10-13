#!/usr/bin/env python3
"""Носитель как величина: видит ли взвешенная сумма локальный объект.

ПОВОД. weight-doc.md: у соседства локальности нет и быть не может
(теорема о некомпактности орбит Лоренца), но у СУММЫ по соседству она
есть — дальнобойность конечна и от размера мира не зависит. Отсюда
заявка: носителем может быть не область с внутренностью, а величина с
конечной дальнобойностью. Это и проверяется.

inertia.py мерил число изменённых отношений и померил бухгалтерию
(рассыпанный контроль дал те же числа). Здесь мерится сглаженная
взвешенная сумма — величина мира, а не счёт.

ОПЫТ 1: ПРОФИЛЬ. Вставить в мир плотную мировую трубку и посмотреть на
средний взвешенный вклад фоновых событий по расстоянию до оси. Трубка
покоится, значит её система отсчёта выделена ею самой и расстояние в ней
законно. Вычитается далёкое поле (последняя корзина), чтобы убрать общий
сдвиг от изменения плотности.

  решающая проверка: мир растёт при ПОСТОЯННОЙ плотности. Если
  дальнобойность сигнатуры фиксирована, она не должна зависеть от
  размера мира. Если растёт — это снова артефакт конечного мира, как
  было с ситом.

ОПЫТ 2: ЦЕНА ИЗГИБА. Полная сумма для этого не годится: изгиб меняет
отношения по всему миру и дальнобойный шум съедает сигнал (померено:
ошибки 2.5–14 при сигнале того же порядка). Поэтому мерится ЛОКАЛЬНАЯ
сумма — только по событиям в окне вокруг трубки. Это законно именно
потому, что величина локальна.

  предсказание инерции: цена ~ M * delta^2, и у контроля её нет.

  python3 weightmass.py
  python3 weightmass.py --seeds 20
"""
import argparse
import math
import time

import numpy as np

from weight import cone_rel, between, per_event

T_MAX = 4.0
RHO = 25.0


def box_sprinkle(N, L, rs):
    X = np.empty((N, 4))
    X[:, 0] = rs.uniform(0, T_MAX, N)
    X[:, 1:] = rs.uniform(-L, L, (N, 3))
    return X


def n_for(L):
    return int(round(RHO * T_MAX * (2 * L) ** 3))


def tube(M, rad, rs):
    v = rs.normal(size=(M, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(0, T_MAX, M)
    X[:, 1:] = v * r[:, None]
    return X


def bend(X, delta):
    Y = X.copy()
    late = Y[:, 0] > T_MAX / 2
    Y[late, 1] += delta * (Y[late, 0] - T_MAX / 2)
    return Y


def per_ev(X, eps):
    R = cone_rel(X)
    btw = between(R)
    s, _ = per_event(R, btw, eps)
    del R, btw
    return s


def main():
    ap = argparse.ArgumentParser(description='Носитель как величина')
    ap.add_argument('--m', type=int, default=400)
    ap.add_argument('--rad', type=float, default=0.05)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--seeds', type=int, default=12)
    ap.add_argument('--win', type=float, default=0.5)
    a = ap.parse_args()

    t0 = time.time()
    reach = (5 * 24 / (RHO * math.pi)) ** 0.25
    print(f'плотность {RHO} постоянна, ящик 4.0 x (2L)^3')
    print(f'трубка {a.m} событий радиуса {a.rad}, сглаживание eps={a.eps}')
    print(f'дальнобойность веса (n<=5): tau ~ {reach:.2f}')
    print(f'засевов {a.seeds}')
    print()

    edges = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.8, 2.6])
    print('ОПЫТ 1: ПРОФИЛЬ (разница трубка минус контроль, '
          'далёкое поле вычтено)')
    print(f"{'L':>5}{'N':>7}" + ''.join(f'{e:>9.2f}' for e in edges[1:-1])
          + f"{'далёкое':>10}")

    for L in (1.0, 1.4, 1.8):
        N = n_for(L)
        prof = {}
        for mk in (True, False):
            acc = []
            for t in range(a.seeds):
                rs = np.random.default_rng(300 + 17 * t)
                bg = box_sprinkle(N, L, rs)
                ob = tube(a.m, a.rad, rs) if mk else box_sprinkle(a.m, L, rs)
                X = np.vstack([ob, bg])
                s = per_ev(X, a.eps)[a.m:]
                r = np.linalg.norm(bg[:, 1:], axis=1)
                k = np.digitize(r, edges) - 1
                row = np.full(len(edges) - 1, np.nan)
                for b in range(len(row)):
                    m = k == b
                    if m.sum() > 5:
                        row[b] = s[m].mean()
                acc.append(row)
                del X, s
            A = np.array(acc)
            prof[mk] = (np.nanmean(A, 0), np.nanstd(A, 0) / math.sqrt(a.seeds))
        d = prof[True][0] - prof[False][0]
        e = np.hypot(prof[True][1], prof[False][1])
        far = np.nanmean(d[-2:])                 # далёкое поле
        dd = d - far
        n_in = len(edges) - 2
        print(f'{L:>5.1f}{N:>7}'
              + ''.join(f'{v:>9.4f}' for v in dd[:n_in]) + f'{far:>10.4f}')
        print(f"{'значим.':>12}"
              + ''.join(f'{v:>9.1f}' for v in
                        np.abs(dd[:n_in]) / np.maximum(e[:n_in], 1e-12)))

    print()
    print(f'ОПЫТ 2: ЦЕНА ИЗГИБА локальной суммы, окно |x| < {a.win}, L=1.0')
    deltas = (0.1, 0.2, 0.4)
    masses = (100, 400)
    L, N = 1.0, n_for(1.0)
    print(f"{'мир':>10}{'M':>5}"
          + ''.join(f'{f"δ={d}":>19}' for d in deltas) + f"{'показ. по δ':>13}")
    for lbl, mk in (('трубка', True), ('контроль', False)):
        for M in masses:
            vals = {d: [] for d in deltas}
            for t in range(a.seeds):
                rs = np.random.default_rng(300 + 17 * t)
                bg = box_sprinkle(N, L, rs)
                ob = tube(M, a.rad, rs) if mk else box_sprinkle(M, L, rs)
                win = np.linalg.norm(bg[:, 1:], axis=1) < a.win
                base = per_ev(np.vstack([ob, bg]), a.eps)[M:][win].sum()
                for dl in deltas:
                    v = per_ev(np.vstack([bend(ob, dl), bg]),
                               a.eps)[M:][win].sum()
                    vals[dl].append(float(v - base))
            row = [float(np.mean(vals[d])) for d in deltas]
            err = [float(np.std(vals[d]) / math.sqrt(a.seeds)) for d in deltas]
            p = np.polyfit(np.log(deltas), np.log(np.abs(row) + 1e-12), 1)[0]
            print(f'{lbl:>10}{M:>5}'
                  + ''.join(f'{v:>12.3f} ±{s:<5.3f}'
                            for v, s in zip(row, err)) + f'{p:>13.3f}')
            print(f"{'значим.':>15}"
                  + ''.join(f'{abs(v)/max(s,1e-12):>19.1f}'
                            for v, s in zip(row, err)))
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
