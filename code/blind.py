#!/usr/bin/env python3
"""СЛЕПОЕ НАПРАВЛЕНИЕ: при каком размере интервала оно проявляется.

Прошлая попытка (`order2.py`) мерила второй порядок подгонкой
`X_2 = A T² + C T⁴`. Не вышло: столбцы `T²` и `T⁴` на диапазоне
`T = 0.5..2.3` слишком коррелированы, все `C` легли в пределах одной-двух
сигм от нуля, а регрессия на квадратичные инварианты дала `chi²` = 43.7
на 6 степеней — то есть подгоняла шум. Её вывод отозван.

ЗДЕСЬ постановка без разложения вообще. Первый порядок при покое равен

    A = 0.09519 u - 0.05675 vv,    u = a''/a,  vv = a'²/a²

значит есть ЦЕЛАЯ ЛИНИЯ пар `(u, vv)` с одинаковым `A` — направление,
которого первый порядок не видит в принципе. Берётся пучок метрик вдоль
этой линии и считается ТОЧНЫЙ `X_2` при растущем `T`. Если кривые
расходятся — слепое направление проявилось, и видно, при каком `tau² R`
это происходит.

Семейство: `a = exp(b1 eta + b2 eta²)`, откуда при `eta = 0`
`vv = b1²`, `u = 2 b2 + b1²` — любая пара достижима.

КОНТРОЛЬ: две метрики с ОДИНАКОВЫМИ `(u, vv)`, но разными высшими
производными (`b3` член) — их расхождение показывает, сколько вносят
производные выше второй, то есть насколько «слепое направление» вообще
определено компонентами.

  python3 blind.py
"""
import math

import numpy as np

from pencil import c_k
from vcal import in_diamond, mink

A1, A2 = 0.09519, -0.05675
NB = 5


def x2_gen(b1, b2, b3, T, n, rs, d=4):
    """Точный <X_2> для `a = exp(b1 e + b2 e² + b3 e³)`, интервал в покое."""
    tx, sx = in_diamond(rs, n, T, d)
    x = np.column_stack([tx - 0.5 * T, sx])
    q4 = np.zeros((n, d)); q4[:, 0] = 0.5 * T
    dd = q4 - x
    t2 = mink(dd, dd)
    ok = t2 > 0
    x, dd, t2 = x[ok], dd[ok], t2[ok]
    tau = np.sqrt(t2); m = len(x)
    e0 = dd / tau[:, None]
    B = np.zeros((m, d - 1, d))
    seed = np.zeros((m, d - 1, d))
    for i in range(d - 1):
        seed[:, i, i + 1] = 1.0
    for i in range(d - 1):
        v = seed[:, i, :].copy()
        v -= mink(v, e0)[:, None] * e0
        for j in range(i):
            v -= (-mink(v, B[:, j, :]))[:, None] * B[:, j, :]
        v /= np.sqrt(np.maximum(-mink(v, v), 1e-300))[:, None]
        B[:, i, :] = v
    ty, sy = in_diamond(rs, m, 1.0, d)
    y = x + (ty * tau)[:, None] * e0
    for i in range(d - 1):
        y = y + (sy[:, i] * tau)[:, None] * B[:, i, :]

    def wf(e):
        return np.exp(4.0 * (b1 * e + b2 * e * e + b3 * e ** 3))

    wx, wy = wf(x[:, 0]), wf(y[:, 0])
    return (wx * tau ** 4 * wy).mean() / (c_k(2) * T ** 4 * wx.mean() ** 2) - 1


def pair(u, vv):
    b1 = math.sqrt(max(vv, 0.0))
    return b1, 0.5 * (u - vv)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(577215)
    N = 1600000
    Atgt = 0.008567
    PTS = [(0.0900, 0.0000), (0.1437, 0.0900), (0.1973, 0.1800)]
    print(f'целевой первый порядок A = {Atgt:+.6f}; точек {N}, '
          f'блоков {NB}')
    print()
    print('пучок вдоль СЛЕПОГО направления (одинаковый первый порядок):')
    for u, vv in PTS:
        b1, b2 = pair(u, vv)
        print(f'  u = {u:.4f}, vv = {vv:.4f}  ->  b1 = {b1:.4f}, '
              f'b2 = {b2:+.4f},  A = {A1*u + A2*vv:+.6f}')
    print()
    TS = (1.0, 1.8, 2.6, 3.4, 4.2)
    print(f"{'T':>5}{'tau² u':>9}", end='')
    for i in range(len(PTS)):
        print(f'{f"X_2 #{i+1}":>13}', end='')
    print(f'{"размах":>12}{"сигма размаха":>15}{"сигм":>7}')
    for T in TS:
        vals, errs = [], []
        for u, vv in PTS:
            b1, b2 = pair(u, vv)
            r = np.array([x2_gen(b1, b2, 0.0, T, N, rs) for _ in range(NB)])
            vals.append(r.mean()); errs.append(r.std(ddof=1) / math.sqrt(NB))
        vals, errs = np.array(vals), np.array(errs)
        sp = vals.max() - vals.min()
        esp = math.sqrt(errs[np.argmax(vals)] ** 2 + errs[np.argmin(vals)] ** 2)
        print(f'{T:>5.1f}{T*T*PTS[0][0]:>9.3f}', end='')
        for v_, e_ in zip(vals, errs):
            print(f'{f"{v_:+.4f}":>13}', end='')
        print(f'{sp:>12.5f}{esp:>15.5f}{sp/max(esp,1e-12):>7.1f}',
              flush=True)
    print()
    print('=' * 78)
    print('КОНТРОЛЬ: те же (u, vv), но ненулевой третий член b3')
    u, vv = PTS[1]
    b1, b2 = pair(u, vv)
    for b3 in (0.0, 0.02, 0.05):
        r = np.array([x2_gen(b1, b2, b3, 2.6, N, rs) for _ in range(NB)])
        print(f'  b3 = {b3:.2f}: X_2 = {r.mean():+.5f} +- '
              f'{r.std(ddof=1)/math.sqrt(NB):.5f}')
    print()
    print('''ЧТЕНИЕ. Столбец «сигм» — во сколько размах пучка превышает
свою ошибку. Если он растёт с `T`, слепое направление проявляется, и
`tau² u` в той строке говорит, НАСКОЛЬКО глубоко за защиту применимости
(0.8) надо выйти. Контроль по `b3` показывает, не подменяется ли эффект
вкладом производных выше второй: если он сравним с размахом пучка,
«слепое направление» компонентами не описывается.''')


if __name__ == '__main__':
    main()
