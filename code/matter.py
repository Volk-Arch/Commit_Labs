#!/usr/bin/env python3
"""B·K ≈ delta: есть ли на порядке локальный волновой оператор.

Условие ИНОГО РОДА, чем все семь закрытых маршрутов: не значение
статистики, а СУЩЕСТВОВАНИЕ локального оператора, аннулирующего
пропагатор. Это решение обратной задачи, и трёхпараметрическое
семейство врага такое подогнать не может по построению — ему нечем
обеспечить согласованность N^2 чисел.

Оба объекта теперь есть и проверены в 1+1:
  * пропагатор Джонстона K = I + Phi (I - b Phi)^{-1}, Phi = a A,
    a = 1/2, b = -m^2/rho (`johnston-doc.md`, контроль против
    (1/2) J_0(mT) — точное тождество);
  * даламбертиан (B f)(x) = (1/l^2)[-1/2 f(x) + sum (1,-2,1) f(y)],
    собственный коэффициент -1/2 НАЙДЕН ИЗМЕРЕНИЕМ (waveid.py:
    <sum w n> = 0.5, не 1) и проверен на гладких функциях — все девять
    значений в пределах 1.7 сигмы от точных 2, -2, 4 (waveop5.py).

МЕРА «дельтовости», свободная от нормировки:
    D = ||диагональ B·K|| / ||внедиагональ B·K||
У засева локальный оператор есть, значит D велико. У врага собственного
времени нет, значит B·K не должно собираться на диагонали.

ОГОВОРКА ЗАРАНЕЕ: у слоистого врага нет геометрии, значит rho в b есть
СОГЛАШЕНИЕ. Поэтому берутся три значения массы, и вывод не должен от
них зависеть.

КОНТРОЛЬ: при m = 0 пропагатор обязан дать K = I + A/2 ровно.
"""
import math

import numpy as np

from waveop import causal, sprinkle_11
from lkdeg import layered

W11 = (1.0, -2.0, 1.0)


def Bmat(Rc, rho):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    B = np.zeros((n, n))
    for m, w in enumerate(W11):
        B += w * (Rc.T & (Iv.T == m))
    np.fill_diagonal(B, -0.5)
    return B * rho


def Kmat(Rc, rho, mass):
    n = Rc.shape[0]
    A = Rc.astype(float)
    Phi = 0.5 * A
    b = -mass * mass / rho
    return np.eye(n) + Phi @ np.linalg.inv(np.eye(n) - b * Phi)


def deltaness(Rc, rho, mass):
    B = Bmat(Rc, rho)
    K = Kmat(Rc, rho, mass)
    M = B @ K
    d = np.abs(np.diag(M))
    off = M.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.sqrt((d ** 2).mean())
                 / max(math.sqrt((off ** 2).mean()), 1e-300))


def main():
    print(__doc__)
    rs = np.random.default_rng(112358)
    n = 500
    print('КОНТРОЛЬ: при m = 0 пропагатор обязан быть I + A/2')
    P = sprinkle_11(n, rs)
    Rc = causal(P)
    K0 = Kmat(Rc, n / 0.5, 0.0)
    err = float(np.abs(K0 - (np.eye(n) + 0.5 * Rc)).max())
    print(f'  наибольшее отклонение {err:.2e} '
          f'=> {"ОК" if err < 1e-9 else "СБОЙ"}')

    print(f'\nДЕЛЬТОВОСТЬ D = ||диаг|| / ||внедиаг||, N = {n}')
    print(f"{'ансамбль':>22}" + ''.join(f'{f"m={m}":>11}'
                                        for m in (0.0, 2.0, 6.0)))
    rows = []
    for lab, mk in (('засев 1+1', lambda: causal(sprinkle_11(n, rs))),
                    ('слоистый k=3 q=0.2', lambda: layered(n, 3, 0.2, rs)),
                    ('слоистый k=5 q=0.1', lambda: layered(n, 5, 0.1, rs)),
                    ('слоистый k=12 q=0.4',
                     lambda: layered(n, 12, 0.4, rs))):
        vs = []
        for mass in (0.0, 2.0, 6.0):
            ds = [deltaness(mk(), n / 0.5, mass) for _ in range(3)]
            vs.append(float(np.mean(ds)))
        rows.append((lab, vs))
        print(f'{lab:>22}' + ''.join(f'{v:>11.3f}' for v in vs),
              flush=True)
    base = rows[0][1]
    print('\nотношение к засеву:')
    for lab, vs in rows[1:]:
        print(f'{lab:>22}' + ''.join(f'{v/b:>11.3f}'
                                     for v, b in zip(vs, base)))
    print('\nЧТЕНИЕ: если у засева D заметно выше при всех массах —')
    print('локальный волновой оператор есть у него и нет у врага, и это')
    print('условие, которое враг подогнать не может.')


if __name__ == '__main__':
    main()
