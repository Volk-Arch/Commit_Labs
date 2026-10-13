#!/usr/bin/env python3
"""`w` НАПРЯМУЮ, одним параметром — обход всего вырождения.

ЧТО ВЫЯСНИЛОСЬ. Слепое направление первого порядка (линия пар
`(u, vv)` с одинаковым `A = 0.09519 u - 0.05675 vv`) на больших
интервалах расходится сильно: 5 сигм при `tau² u = 0.29`, 30 сигм при
1.59, размах 25% от сигнала (`blind.py`). Но контроль там показал, что
добавление третьей производной меняет `X_2` на 19...52%, то есть
БОЛЬШЕ размаха пучка: на больших интервалах наблюдаемая перестаёт быть
функцией кривизны в точке и становится функционалом формы `a(eta)`.
Для модельно-независимого извлечения компонент это убивает канал.

НО для ФИЗИЧЕСКОГО семейства высшие производные не свободны. При
постоянном `w` масштабный фактор равен точно

    a(eta) = (eta/eta_0)^p,    p = 2/(1 + 3w)

то есть вся форма задана ОДНИМ числом. Тогда при `eta = eta_0`:

    vv = a'²/a² = p²/eta_0²,   u = a''/a = p(p-1)/eta_0²

и отношение `vv/u = p/(p-1)` ФИКСИРОВАНО каждым `w`. Значит каждое
уравнение состояния — это ЛУЧ в плоскости `(u, vv)`, а слепое
направление первого порядка — это и есть движение по лучам. Поэтому `w`
и не выходил.

Проверка соответствия: `w = 1/3` даёт `p = 1`, то есть `a ~ eta` —
это линейное семейство (`u = 0`); `w = +1` даёт `p = 1/2`, `a ~ sqrt(eta)` —
корневое; `w = -1/3` — предел `p -> oo`, экспоненциальное. Мои четыре
«семейства» были четырьмя значениями `w`.

ЗДЕСЬ считается точный `X_2(w, tau)` и чувствительность `dX_2/dw` на
больших интервалах. Если она велика, `w` мерится ОДНОПАРАМЕТРИЧЕСКОЙ
подгонкой с сильным сигналом вместо двухпараметрической вырожденной со
слабым.

КОНТРОЛЬ: при `w = 1/3` выходит `p = 1`, `a ~ eta`, и тогда скаляр
кривизны `R = 6 a''/a³ = 0` точно — излучение. Это проверяемый ноль.

  python3 weos.py
"""
import math

import numpy as np

from pencil import c_k
from vcal import in_diamond, mink

NB = 5


def x2_w(w, tau, n, rs, eta0=1.0, d=4):
    """Точный <X_2> для a = (eta/eta_0)^p, p = 2/(1+3w); интервал в покое
    с серединой в eta_0."""
    p = 2.0 / (1.0 + 3.0 * w)
    tx, sx = in_diamond(rs, n, tau, d)
    x = np.column_stack([tx - 0.5 * tau + eta0, sx])
    q4 = np.zeros((n, d)); q4[:, 0] = eta0 + 0.5 * tau
    dd = q4 - x
    t2 = mink(dd, dd)
    ok = (t2 > 0) & (x[:, 0] > 1e-6)
    x, dd, t2 = x[ok], dd[ok], t2[ok]
    tt = np.sqrt(t2); m = len(x)
    e0 = dd / tt[:, None]
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
    y = x + (ty * tt)[:, None] * e0
    for i in range(d - 1):
        y = y + (sy[:, i] * tt)[:, None] * B[:, i, :]
    good = y[:, 0] > 1e-6
    x, y, tt = x[good], y[good], tt[good]
    wx = (x[:, 0] / eta0) ** (4.0 * p)
    wy = (y[:, 0] / eta0) ** (4.0 * p)
    return ((wx * tt ** 4 * wy).mean()
            / (c_k(2) * tau ** 4 * wx.mean() ** 2) - 1)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(1414213)
    N = 1600000
    WS = (-0.20, 0.0, 1.0 / 3.0, 0.60, 1.00)
    print(f'точек {N}, блоков {NB}, eta_0 = 1')
    print()
    print(f"{'tau':>5}", end='')
    for w in WS:
        print(f'{f"w={w:+.2f}":>12}', end='')
    print(f'{"dX2/dw у 1/3":>14}{"ошибка":>10}{"сигм на 0.1":>13}')
    for tau in (0.4, 0.8, 1.2, 1.5, 1.8):
        vals, errs = [], []
        for w in WS:
            r = np.array([x2_w(w, tau, N, rs) for _ in range(NB)])
            vals.append(r.mean()); errs.append(r.std(ddof=1) / math.sqrt(NB))
        vals, errs = np.array(vals), np.array(errs)
        # производная у w = 1/3 центральной разностью по соседям
        i = 2
        der = (vals[i + 1] - vals[i - 1]) / (WS[i + 1] - WS[i - 1])
        eder = math.sqrt(errs[i + 1] ** 2 + errs[i - 1] ** 2) \
            / (WS[i + 1] - WS[i - 1])
        print(f'{tau:>5.1f}', end='')
        for v_ in vals:
            print(f'{f"{v_:+.5f}":>12}', end='')
        print(f'{der:>14.5f}{eder:>10.5f}'
              f'{abs(der)*0.1/max(errs[i],1e-12):>13.1f}', flush=True)
    print()
    print('=' * 86)
    print('КОНТРОЛЬ: при w = 1/3 получается p = 1, a ~ eta, и скаляр')
    print('кривизны R = 6 a\'\'/a³ равен нулю ТОЧНО. Проверяется, что')
    print('X_2 при w = 1/3 совпадает с линейным семейством того же наклона.')
    from vcal2 import x2_exact
    for tau in (0.8, 1.5):
        a = np.array([x2_w(1.0 / 3.0, tau, N, rs) for _ in range(NB)])
        b = np.array([x2_exact('lin', 1.0, tau, N, rs) for _ in range(NB)])
        ea = a.std(ddof=1) / math.sqrt(NB); eb = b.std(ddof=1) / math.sqrt(NB)
        print(f'  tau = {tau}: через w  {a.mean():+.5f} +- {ea:.5f};  '
              f'через lin  {b.mean():+.5f} +- {eb:.5f};  '
              f'{abs(a.mean()-b.mean())/math.sqrt(ea**2+eb**2):.1f} сигм')
    print()
    print('''ЧТЕНИЕ. Столбец «сигм на 0.1» — во сколько сигм отличаются
`w = 1/3` и `w = 0.43` при шуме ОДНОЙ оценки. Если он растёт с `tau`,
то `w` мерится однопараметрической подгонкой, и вся двухкомпонентная
вырожденная задача обходится стороной.''')


if __name__ == '__main__':
    main()
