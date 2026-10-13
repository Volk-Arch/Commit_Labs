#!/usr/bin/env python3
"""СКОЛЬКО ИНТЕРВАЛОВ БРАТЬ С МИРА — оптимум дизайна набора.

Два факта уже измерены. Первый (`neff.py`): из 420 интервалов на мир
независимо работают ~230, доля 0.55 (контроль на плоском плече даёт ту
же долю, значит мера читает шум, а не сигнал). Второй: стоимость
СТРОГО линейна по интервалам — 59, 56, 55 мс на интервал при per = 150,
420, 1000, то есть постоянной цены мира практически нет.

Вместе это значит, что мир стоит ровно столько, сколько его интервалы, а
интервалы внутри мира теряют половину в корреляции. Тогда выгодно брать
МЕНЬШЕ интервалов с БОЛЬШЕГО числа миров — но только если доля
независимости при малом `per` действительно выше. Это и мерится.

Величина решения — независимых интервалов в секунду:

    скорость = доля(per) x per / время(per)

  python3 perscan.py
"""
import time

import numpy as np

from expfam import harvest, CFG
from pencil import IC, IX2


def run(per, worlds):
    t0 = time.time()
    mu, vin = [], []
    for s in range(worlds):
        rs = np.random.default_rng(770000 + 131 * s + per)
        r, _ = harvest('lin', 0.035, CFG, rs, per)
        A = np.array(r)
        if len(A) < per // 2:
            continue
        w = A[:, IC]
        mm = (A[:, IX2] * w).sum() / w.sum()
        mu.append(mm)
        vin.append(((A[:, IX2] - mm) ** 2 * w).sum() / w.sum())
    dt = (time.time() - t0) / max(len(mu), 1)
    mu = np.array(mu)
    frac = np.mean(vin) / mu.var(ddof=1) / per
    return dt, frac, len(mu)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print(f"{'per':>6}{'миров':>7}{'с/мир':>8}{'доля незав.':>13}"
          f"{'независ./мир':>14}{'независ./с':>12}")
    best = None
    for per, wd in ((120, 26), (420, 14), (900, 8)):
        dt, frac, nw = run(per, wd)
        rate = frac * per / dt
        print(f'{per:>6}{nw:>7}{dt:>8.1f}{frac:>13.2f}'
              f'{frac*per:>14.0f}{rate:>12.2f}', flush=True)
        if best is None or rate > best[0]:
            best = (rate, per)
    print()
    print(f'оптимум per = {best[1]}, {best[0]:.2f} независимых интервалов в '
          f'секунду')
    print('''
ЧТЕНИЕ. Доля независимости при малом `per` обязана быть выше: интервалы
меньше перекрываются. Если выигрыш в доле перебивает рост цены за
интервал — набор надо перестраивать на малый `per`, и выигрыш идёт прямо
в ошибку как корень.''')


if __name__ == '__main__':
    main()
