#!/usr/bin/env python3
"""ЕСТЬ ЛИ ВЫДЕЛЕННЫЙ КОНУС: двадцать граней — константа или артефакт.

Мысль: что если существует объект, складывающий ровно 20-гранный конус,
и это минимальный объект, а конус - первая геометрия.

Сперва надо снять ореол с «двадцати», который я сам и создал. Мои же
числа в логарифмах: при k = 3,4,6,10,20 выходит χ² = 5891,1030,381,48,5.9
- это гладкий степенной закон k^{-3.6}, БЕЗ излома. «20» есть точка, где
прямая ушла под шумовой пол при N=500, а не особое число.

ДВА ВОПРОСА:
 (1) РАСТЁТ ЛИ требуемое k с N. Растёт => двадцать есть артефакт размера
     и минимального объекта нет; насыщается => есть.
 (2) ВЫДЕЛЕНЫ ЛИ правильные расстановки. Икосаэдр - наилучшее правильное
     приближение сферы, и если 12 его направлений работают заметно лучше
     12 равномерных по Фибоначчи, это настоящий эффект.

КОНТРОЛЬ: разброс настоящих засевов печатается при каждом N - иначе
«вошло в шум» ничего не значит.
"""
import numpy as np

from sectors import sprinkle_d
from orthant import prof, region, rf
from star import fib_sphere, pts

PHI = (1 + 5 ** 0.5) / 2


def faceted(P, U):
    """бегущий максимум по направлениям — без массива N^2 x k."""
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    X = P[:, 1:]
    best = None
    for m in range(U.shape[0]):
        pr = X @ U[m]
        d = pr[None, :] - pr[:, None]
        best = d if best is None else np.maximum(best, d)
    return (dt > 0) & (dt > best)


def norml(V):
    V = np.array(V, float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def icosa():
    v = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            v += [(0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s2 * PHI, 0, s1)]
    return norml(v)


def dodeca():
    v = [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]
    for s1 in (1, -1):
        for s2 in (1, -1):
            v += [(0, s1 / PHI, s2 * PHI), (s1 / PHI, s2 * PHI, 0),
                  (s2 * PHI, 0, s1 / PHI)]
    return norml(v)


def main():
    print(__doc__)
    rs = np.random.default_rng(1123581321)
    print('(1) требуемое k по N')
    for n in (200, 400, 800):
        chi, _ = region(lambda: sprinkle_d(n, 4, rs), 30, rs)
        hi = max(float(chi(prof(sprinkle_d(n, 4, rs)))) for _ in range(6))
        row = [(k, float(np.mean([float(chi(prof(faceted(pts(n, rs),
                                                         fib_sphere(k)))))
                                  for _ in range(2)])))
               for k in (6, 10, 14, 20, 28)]
        need = next((k for k, v in row if v <= hi), None)
        print(f'  N={n:>4}: контроль до {hi:.1f};  '
              + '  '.join(f'k={k}:{v:.1f}' for k, v in row)
              + f'   => нужно k = {need}', flush=True)

    n = 400
    print(f'\n(2) правильные расстановки против равномерных, N={n}')
    chi, _ = region(lambda: sprinkle_d(n, 4, rs), 30, rs)
    ctl = [float(chi(prof(sprinkle_d(n, 4, rs)))) for _ in range(6)]
    print('  контроль: ' + ' '.join(f'{v:.1f}' for v in ctl))
    for lab, U in (('икосаэдр-12', icosa()),
                   ('Фибоначчи-12', fib_sphere(12)),
                   ('додекаэдр-20', dodeca()),
                   ('Фибоначчи-20', fib_sphere(20))):
        v = [float(chi(prof(faceted(pts(n, rs), U)))) for _ in range(3)]
        r = rf(faceted(pts(n, rs), U))
        print(f'  {lab:>14} ({U.shape[0]:>2} напр.): '
              + ' '.join(f'{x:.1f}' for x in v)
              + f'   среднее {np.mean(v):.1f}, доля {r:.4f}', flush=True)


if __name__ == '__main__':
    main()
