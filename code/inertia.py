#!/usr/bin/env python3
"""Масса как сопротивление изменению.

У свободной мировой линии геометрической массы нет: все времениподобные
линии одинаковы с точностью до буста. В схеме Вигнера масса — метка от
СДВИГОВ, а спин, киральность и статистика — от вращений. Проект нашёл всё
вращательное и ничего сдвигового, и это объясняет, почему свойства есть,
а носителя нет.

Масса измеряется только сопротивлением изменению. Поэтому:

  берём мировую трубку, СГИБАЕМ её на скорость δ в середине времени,
  и считаем, сколько отношений в порядке от этого изменилось.

Предсказание, если это инерция: стоимость изгиба идёт как

    цена  ~  M · δ²

линейно по числу событий трубки, квадратично по изгибу — то есть это
кинетическая энергия, а M играет роль массы.

Контроль: то же число событий, рассыпанных по миру, сдвинутых так же.

  python3 inertia.py
  python3 inertia.py --n 4000
"""
import argparse
import math
import time

import numpy as np

T_MAX = 4.0
L_BOX = 1.0


def box_sprinkle(N, rs):
    X = np.empty((N, 4))
    X[:, 0] = rs.uniform(0, T_MAX, N)
    X[:, 1:] = rs.uniform(-L_BOX, L_BOX, (N, 3))
    return X


def tube(M, rho, rs):
    v = rs.normal(size=(M, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = rho * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = rs.uniform(0, T_MAX, M)
    X[:, 1:] = v * rad[:, None]
    return X


def bend(X, delta):
    """Поворот мировой линии в середине времени на скорость delta.

    События позже T/2 сдвигаются по x1 пропорционально прошедшему
    времени — линия ломается и уходит со скоростью delta.
    """
    Y = X.copy()
    late = Y[:, 0] > T_MAX / 2
    Y[late, 1] += delta * (Y[late, 0] - T_MAX / 2)
    return Y


def cone_rel(X):
    dt = X[:, None, 0] - X[None, :, 0]
    dx = np.linalg.norm(X[:, None, 1:] - X[None, :, 1:], axis=-1)
    R = dt > dx
    np.fill_diagonal(R, False)
    return R


def cost(bg, obj, delta):
    """Сколько отношений изменилось от изгиба объекта."""
    X0 = np.vstack([obj, bg])
    X1 = np.vstack([bend(obj, delta), bg])
    return int((cone_rel(X0) != cone_rel(X1)).sum())


def main():
    ap = argparse.ArgumentParser(description='Масса как сопротивление')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--rho', type=float, default=0.05)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    deltas = (0.02, 0.05, 0.10, 0.20)
    masses = (50, 100, 200, 400)

    print(f'фон {a.n}, радиус трубки {a.rho}, семян {a.seeds}')
    print('предсказание инерции: цена ~ M * delta^2')
    print()
    print('ЦЕНА ИЗГИБА')
    print(f"{'M':>6}" + ''.join(f'{f"δ={d}":>13}' for d in deltas)
          + f"{'показ. по δ':>14}")
    tab = {}
    for M in masses:
        row = []
        for d in deltas:
            acc = []
            for s in range(a.seeds):
                rs = np.random.default_rng(100 + 17 * s)
                bg = box_sprinkle(a.n, rs)
                ob = tube(M, a.rho, rs)
                acc.append(cost(bg, ob, d))
            row.append(float(np.mean(acc)))
        tab[M] = row
        p = np.polyfit(np.log(deltas), np.log(np.maximum(row, 1)), 1)[0]
        print(f'{M:>6}' + ''.join(f'{v:>13.0f}' for v in row)
              + f'{p:>14.3f}')

    print()
    print('ПОКАЗАТЕЛЬ ПО МАССЕ при каждом изгибе')
    print(f"{'δ':>8}{'показатель по M':>18}{'ожидалось':>12}")
    for i, d in enumerate(deltas):
        y = [tab[M][i] for M in masses]
        p = np.polyfit(np.log(masses), np.log(np.maximum(y, 1)), 1)[0]
        print(f'{d:>8.2f}{p:>18.3f}{1:>12}')

    print()
    print('КОНТРОЛЬ: те же события, рассыпанные по миру')
    print(f"{'M':>6}" + ''.join(f'{f"δ={d}":>13}' for d in deltas))
    for M in masses:
        row = []
        for d in deltas:
            acc = []
            for s in range(a.seeds):
                rs = np.random.default_rng(100 + 17 * s)
                bg = box_sprinkle(a.n, rs)
                ob = box_sprinkle(M, rs)
                acc.append(cost(bg, ob, d))
            row.append(float(np.mean(acc)))
        print(f'{M:>6}' + ''.join(f'{v:>13.0f}' for v in row))
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
