#!/usr/bin/env python3
"""Инвариантность порядка к монотонному перешкалированию.

Проверяет две вещи:
  1. порядок не меняется, если каждую шкалу преобразовать своей
     возрастающей функцией (Коши, экспонента, степень, tan);
  2. какие из наших величин выживают при таком преобразовании,
     а какие нет.

Вторая таблица — тест на физичность: величина, которая меняется
при перешкалировании, зависит от выбора координат, а не от мира.

  python3 rescale.py                # обе проверки
  python3 rescale.py --n 2000       # крупнее
"""
import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of, dimension, spectral, height, bd_action


# ---- монотонные преобразования шкалы ----------------------------------
# каждое строго возрастает на (0,1), поэтому порядок обязано сохранить

TRANSFORMS = {
    'тождество':      lambda u: u,
    'Коши (tan)':     lambda u: np.tan(np.pi * (u - 0.5)),
    'экспонента':     lambda u: -np.log1p(-u),
    'степень x^5':    lambda u: u ** 5,
    'логит':          lambda u: np.log(u / (1 - u)),
}


def check_order_invariance(N=800, k=4, seed=3):
    """Один и тот же U, разные преобразования шкал -> тот же порядок?"""
    rs = np.random.default_rng(seed)
    U = np.clip(rs.random((N, k)), 1e-12, 1 - 1e-12)
    R0 = rel_of(U)

    rows = []
    for name, f in TRANSFORMS.items():
        R = rel_of(f(U))
        diff = int((R != R0).sum())
        rows.append((name, diff, R0.size))

    # и самое сильное: СВОЁ преобразование на каждую шкалу
    V = U.copy()
    fs = list(TRANSFORMS.values())
    for m in range(k):
        V[:, m] = fs[(m + 1) % len(fs)](U[:, m])
    rows.append(('разные на каждой шкале', int((rel_of(V) != R0).sum()), R0.size))
    return rows


def local_density(X, radius):
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return (d < radius).sum(1) - 1


# величины: (имя, функция(X, R), порядковая ли)
QUANTITIES = [
    ('размерность',        lambda X, R: dimension(R=R),                       True),
    ('спектр',             lambda X, R: spectral(R),                          True),
    ('высота',             lambda X, R: float(height(R)[0]),                  True),
    ('действие BD / N',    lambda X, R: bd_action(R) / len(X),                True),
    ('доля связанных пар', lambda X, R: float(R.sum()) / (len(X) * (len(X) - 1)), True),
    ('плотность (медиана)', lambda X, R: float(np.median(local_density(X, 0.15))), False),
    ('разброс шкал',       lambda X, R: float(X.var(1).mean()),               False),
    ('расстояние 0-1',     lambda X, R: float(np.linalg.norm(X[0] - X[1])),   False),
]


def check_quantities(N=700, k=4, seed=3):
    rs = np.random.default_rng(seed)
    U = np.clip(rs.random((N, k)), 1e-12, 1 - 1e-12)
    V = U.copy()
    fs = list(TRANSFORMS.values())
    for m in range(k):
        V[:, m] = fs[(m + 1) % len(fs)](U[:, m])

    Ru, Rv = rel_of(U), rel_of(V)
    assert np.array_equal(Ru, Rv), 'порядок изменился — преобразование не монотонно'

    rows = []
    for name, f, is_order in QUANTITIES:
        a, b = f(U, Ru), f(V, Rv)
        rows.append((name, a, b, abs(a - b) < 1e-9, is_order))
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Инвариантность порядка к монотонному перешкалированию')
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()

    print('ПРОВЕРКА 1: меняется ли порядок')
    print(f"{'преобразование шкалы':>26}{'различий':>12}{'из':>12}")
    for name, diff, total in check_order_invariance(a.n, a.k, a.seed):
        print(f'{name:>26}{diff:>12}{total:>12}')

    print()
    print('ПРОВЕРКА 2: какие величины выживают')
    print(f"{'величина':>22}{'до':>12}{'после':>12}{'выжила':>9}{'тип':>13}")
    for name, before, after, same, is_order in check_quantities(700, a.k, a.seed):
        mark = 'да' if same else 'НЕТ'
        kind = 'порядковая' if is_order else 'координатная'
        print(f'{name:>22}{before:>12.4f}{after:>12.4f}{mark:>9}{kind:>13}')

    print()
    print('  Вывод: порядковые величины инвариантны, координатные — нет.')
    print('  Координатная величина не может быть физической.')


if __name__ == '__main__':
    main()
