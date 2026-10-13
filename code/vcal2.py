#!/usr/bin/env python3
"""ТОЧНЫЙ ИНТЕГРАЛ ПРОТИВ ДОПУЩЕНИЯ КОНВЕЙЕРА.

Разложение (`vcal.py`) дало `X_2/T^2 = 0.0952 u - 0.0568 vv`, где
`u = a''/a`, `vv = a'^2/a^2`. Конвейер же предполагает
`X_2 = V2CAL * R_uu * tau^2`, что при покое даёт
`V2CAL * 3(vv - u) = 0.0603 u - 0.0603 vv` при замеренном
`V2CAL = -0.0201`.

При `vv` версии расходятся на 6% — внутри калибровочной погрешности
14%, не различить. При `u` — на 58%.

Разводит их ОДНО семейство. У линейного `u = 0` (при `a = 1 + h(eta-eta_0)`
вторая производная нуль), поэтому оно обе версии не различает вовсе. У
экспоненциального `a = e^{k eta}` выходит `u = vv = k^2`, и тогда моя
формула даёт `0.038 k^2`, а конвейер — РОВНО НУЛЬ, потому что у exp
`R_etaeta = 3(vv - u) = 0` точно.

Здесь считается ТОЧНЫЙ интеграл, без разложения и без сприклинга. Для
конформно-плоских метрик причинность плоская, поэтому

    1 + X_2 = < w_x tau_xq^4 w_y > / ( c_2 T^4 <w_x>^2 ),   w = a^4

где `y` — одна равномерная точка ромба `I(x,q)`. Сканируется `T`, и
коэффициент при `T^2` извлекается подгонкой `X_2 = A T^2 + B T^4`.

КОНТРОЛЬ: плоское `a = 1` обязано дать `X_2 = 0` при всех `T`.

  python3 vcal2.py
"""
import math

import numpy as np

from pencil import c_k
from vcal import in_diamond, mink

NB = 6


def a_fun(fam, par, e):
    if fam == 'flat':
        return np.ones_like(e)
    if fam == 'lin':
        return 1.0 + par * e
    if fam == 'exp':
        return np.exp(par * e)
    if fam == 'root':
        return np.sqrt(np.maximum(1.0 + par * e, 1e-9))
    if fam == 'quad':
        return 1.0 + 0.5 * par * e * e
    raise ValueError(fam)


def uvv(fam, par):
    """(u, vv) = (a''/a, a'^2/a^2) при eta = 0, где a(0) = 1."""
    if fam == 'flat':
        return 0.0, 0.0
    if fam == 'lin':
        return 0.0, par ** 2
    if fam == 'exp':
        return par ** 2, par ** 2
    if fam == 'root':
        return -0.25 * par ** 2, 0.25 * par ** 2
    if fam == 'quad':
        return par, 0.0
    raise ValueError(fam)


def x2_exact(fam, par, T, n, rs, d=4):
    """Точный <X_2> для интервала В ПОКОЕ, середина в eta = 0."""
    tx, sx = in_diamond(rs, n, T, d)
    x = np.column_stack([tx - 0.5 * T, sx])          # середина в 0
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
        v -= (mink(v, e0))[:, None] * e0
        for j in range(i):
            v -= (-mink(v, B[:, j, :]))[:, None] * B[:, j, :]
        v /= np.sqrt(np.maximum(-mink(v, v), 1e-300))[:, None]
        B[:, i, :] = v
    ty, sy = in_diamond(rs, m, 1.0, d)
    y = x + (ty * tau)[:, None] * e0
    for i in range(d - 1):
        y = y + (sy[:, i] * tau)[:, None] * B[:, i, :]
    wx = a_fun(fam, par, x[:, 0]) ** 4
    wy = a_fun(fam, par, y[:, 0]) ** 4
    num = (wx * tau ** 4 * wy).mean()
    den = c_k(2) * T ** 4 * (wx.mean() ** 2)
    return num / den - 1.0


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(31415)
    N = 2500000
    TS = np.array([0.6, 0.9, 1.2, 1.6, 2.0])
    print(f'точек на точку кривой {N}, блоков {NB}, T = {list(TS)}')
    print()
    print(f"{'семейство':>10}{'параметр':>10}{'u':>10}{'vv':>10}"
          f"{'A = X_2/T^2':>16}{'моё':>11}{'конвейер':>11}")
    MEAS = -0.0201
    for fam, par in (('flat', 0.0), ('lin', 0.25), ('exp', 0.25),
                     ('quad', 0.06), ('root', 0.5)):
        rows = []
        for b in range(NB):
            x2 = np.array([x2_exact(fam, par, T, N, rs) for T in TS])
            D = np.column_stack([TS ** 2, TS ** 4])
            rows.append(np.linalg.lstsq(D, x2, rcond=None)[0][0])
        rows = np.array(rows)
        A, eA = rows.mean(), rows.std(ddof=1) / math.sqrt(NB)
        u, vv = uvv(fam, par)
        mine = 0.09519 * u - 0.05675 * vv
        pipe = MEAS * 3.0 * (vv - u)
        print(f'{fam:>10}{par:>10.3f}{u:>10.4f}{vv:>10.4f}'
              f'{f"{A:+.5f}+-{eA:.5f}":>16}{mine:>+11.5f}{pipe:>+11.5f}',
              flush=True)
    print()
    print('''ЧТЕНИЕ. `flat` обязан дать нуль — это контроль. У `lin` обе
версии почти совпадают, он ничего не решает. Решает `exp`: если A
окажется около +0.038 k^2, прав точный интеграл и допущение конвейера
`X_2 ~ R_uu` при покое НЕВЕРНО. Если около нуля — неверно моё
разложение.''')


if __name__ == '__main__':
    main()
