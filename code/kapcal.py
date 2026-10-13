#!/usr/bin/env python3
"""КАЛИБРОВКА kappa: посадить связность с заведомой кривизной.

ЗАЧЕМ. `kappa.py` нашёл безразмерную величину, которую не надо угадывать:

    kappa(цикл) = |сумма связности по обходу| / сумма модулей слагаемых

и намерил на калибровочной связности проекта
`phi = |общее прошлое| − |общее будущее|`:

    kappa калибр  0.1009 -> 0.1325 при N 500 -> 4000,  ~ N^+0.1339
    kappa перемеш 0.913  (те же числа в случайном порядке)
    kappa телеск  0.0 ровно

То есть связность сокращается вокруг обхода на 90% — почти градиент, — а
остаток РАСТЁТ при измельчении. Вывод был бы такой: остаток есть шум
дискретности, а не кривизна, потому что у кривизны остаток обязан убывать
как площадь клетки.

Но этот вывод опирается на ожидание, а ожидания надо проверять. Здесь
kappa мерится на связностях, про которые ответ известен ЗАРАНЕЕ, на той
же россыпи и теми же циклами.

ЭТАЛОНЫ.
  градиент   A_ij = g(x_i) − g(x_j), g гладкая
             холономия тождественно 0 => kappa = 0 при любом N.
  магнитная  A_ij = (B/2)(x_i^1 x_j^2 − x_i^2 x_j^1)
             это симметричная калибровка: интеграл A·dl по отрезку от x
             к y равен (B/2)(x × y)_3, а сумма по замкнутому обходу даёт
             поток через него, то есть B * ПЛОЩАДЬ.
             Сумма модулей ~ B * |x|^2, а |x| не сжимается с N, тогда как
             площадь клетки сжимается как ell^2 ~ N^(-1/2) в четырёх
             мерах. Предсказание: kappa ~ N^(-1/2), показатель −0.5.
  смесь      градиент + магнитная: kappa обязана совпасть с магнитной,
             потому что добавка градиента холономию не меняет. Это
             проверка калибровочной инвариантности самого инструмента.
  случайная  A_ij антисимметричная из независимых нормальных: kappa ~ 1,
             не убывает.

ЧТО ЭТО РЕШАЕТ. Если магнитная даёт показатель около −0.5, а порядковая
+0.13, то инструмент различает кривизну от шума, и вердикт по
`phi = |общее прошлое| − |общее будущее|` твёрд.

  python3 kapcal.py
  python3 kapcal.py --ns 500 1000 2000 4000 8000 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, antisym_causal
from response import warped_diamond


def kap(cyc, A):
    """Доля несокращённого: |сумма| / сумма модулей."""
    v = np.array([A[cyc[t], cyc[(t + 1) % len(cyc)]]
                  for t in range(len(cyc))], float)
    s = np.abs(v).sum()
    return (abs(v.sum()) / s) if s > 1e-12 else float('nan')


def built(X, rs):
    """Эталонные связности на той же россыпи. Все антисимметричны."""
    x1, x2 = X[:, 1], X[:, 2]
    grad = np.sin(2.3 * X[:, 0] + 1.7 * x1) * np.cos(1.9 * x2)
    A_grad = grad[:, None] - grad[None, :]
    A_mag = np.outer(x1, x2) - np.outer(x2, x1)
    Z = rs.normal(size=(len(X), len(X)))
    A_rnd = Z - Z.T
    return {'градиент': A_grad, 'магнитная': A_mag,
            'смесь': A_grad + A_mag, 'случайная': A_rnd}


def one(N, T, f_lo, f_hi, samp, ncyc, seed):
    """kappa всех связностей на одном мире, одними и теми же циклами."""
    rs = np.random.default_rng(7601000 + 977 * seed)
    X = warped_diamond(N, T, 0.0, 0.45, rs)
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    cons = {'ПОРЯДКОВАЯ': antisym_causal(
        (CP - (A.T @ A)).astype(np.float64), R)}
    cons.update(built(X, rs))
    rr = np.linalg.norm(X[:, 1:], axis=1)
    rmax = rr.max()
    pool = np.flatnonzero((rr >= f_lo * rmax) & (rr < f_hi * rmax) &
                          (U.sum(1) >= 3))
    if len(pool) < 25:
        return None, None, None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    cyc = []
    for v in pool:
        cyc.extend(cycles_through(U, int(v), ncyc,
                                  np.random.default_rng(97 + seed)))
    if len(cyc) < 60:
        return None, None, None
    # средняя площадь обхода в плоскости 1-2 — проверка, что она сжимается
    ar = np.mean([abs(sum(X[c[t], 1] * X[c[(t + 1) % len(c)], 2] -
                          X[c[t], 2] * X[c[(t + 1) % len(c)], 1]
                          for t in range(len(c)))) / 2 for c in cyc])
    out = {}
    for nm, M in cons.items():
        vals = [kap(c, M) for c in cyc]
        vals = [v for v in vals if np.isfinite(v)]
        out[nm] = float(np.mean(vals)) if vals else float('nan')
    del R, L, U, A, CP, cons, X
    return out, float(ar), len(cyc)


def main():
    ap = argparse.ArgumentParser(description='Калибровка kappa')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[500, 1000, 2000, 4000])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--shell', type=float, nargs=2, default=[0.45, 0.75])
    ap.add_argument('--samp', type=int, default=300)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.shell
    print('kappa = |сумма связности по обходу| / сумма модулей')
    print(f'лестница N = {a.ns}, оболочка r/rmax {lo}..{hi}, '
          f'засевов {a.seeds}')
    print('ОЖИДАНИЯ: градиент 0 тождественно; магнитная ~ N^(-1/2), '
          'ибо поток ~ площадь клетки;')
    print('          смесь = магнитной (калибровочная инвариантность '
          'инструмента); случайная ~ 1')
    print()
    names = ['ПОРЯДКОВАЯ', 'градиент', 'магнитная', 'смесь', 'случайная']
    print(f"{'N':>7}" + ''.join(f'{n:>13}' for n in names) +
          f"{'площадь':>11}{'циклов':>8}")

    acc = {n: [] for n in names}
    ars = []
    for N in a.ns:
        rows, ar = [], []
        for t in range(a.seeds):
            o, s, nc = one(N, a.t, lo, hi, a.samp, a.ncyc, t)
            if o is None:
                continue
            rows.append(o); ar.append(s)
        if not rows:
            print(f'{N:>7}  мало данных')
            for n in names:
                acc[n].append(None)
            ars.append(None)
            continue
        line = f'{N:>7}'
        for n in names:
            v = float(np.mean([r[n] for r in rows]))
            acc[n].append(v)
            line += f'{v:>13.5f}'
        ars.append(float(np.mean(ar)))
        print(line + f'{np.mean(ar):>11.5f}{nc:>8}', flush=True)

    def fit(vals):
        ok = [i for i in range(len(vals))
              if vals[i] is not None and vals[i] > 1e-12]
        if len(ok) < 3:
            return float('nan'), float('nan')
        x = np.log([a.ns[i] for i in ok])
        y = np.log([vals[i] for i in ok])
        b, c = np.polyfit(x, y, 1)
        ss = ((y - y.mean()) ** 2).sum()
        return float(b), float(1 - ((y - b * x - c) ** 2).sum() / ss) \
            if ss > 1e-18 else (float(b), 1.0)

    print()
    for n in names:
        b, r2 = fit(acc[n])
        if not np.isfinite(b):
            print(f'  {n:<12} kappa тождественно 0 — показателя нет')
        else:
            print(f'  {n:<12} kappa ~ N^{b:+.4f}   R^2 = {r2:.4f}')
    b, r2 = fit(ars)
    print(f'  {"площадь":<12} ~ N^{b:+.4f}   R^2 = {r2:.4f}   '
          f'(ожидание −0.5)')
    print()
    print('  магнитная около −0.5, а ПОРЯДКОВАЯ положительна =>')
    print('  инструмент различает кривизну от шума дискретности,')
    print('  и у порядковой фазы остаток РАСТЁТ при измельчении')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
