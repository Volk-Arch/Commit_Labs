#!/usr/bin/env python3
"""ПОЧЕМУ ЦЕПЬ НЕ СХОДИТСЯ: ЛЕСТНИЦА ИЛИ ХОДЫ.

crithalf.py: при N=24 два прогона по 1800 развёрток дали H = 4.507+-0.087
и 3.870+-0.069 - 5.7 сигмы врозь, половины выборки расходятся на 2.6-3.1
сигмы. Равновесия нет, блочные ошибки занижены.

Ни разу не мерялась доля принятых ОБМЕНОВ между ступенями - только
ходов. А темперирование держится на обменах: если они не идут, каждая
цепь сидит в своей яме и лестница бесполезна.

Прикидка из теплоёмкости: C = beta^2 Var(S) = 70 при beta=0.04 даёт
sigma(S) ~ 209, значит Delta_beta * sigma(S) ~ 0.01 * 209 ~ 2.1 и обмен
принимается с вероятностью ~0.12. Нужно Delta_beta ~ 1/209 ~ 0.005, то
есть лестница вдвое грубее необходимого.

Проверка: густая лестница (шаг 0.005 в окрестности перехода) против
прежней, с печатью доли принятых обменов ПО КАЖДОЙ ПАРЕ. Если на
прежней лестнице есть пары с долей много ниже 0.2, причина найдена.
"""
import math
import sys

import numpy as np

from crit9 import Ch
from critphase import obs, summarize
from critgeom3 import stats, best, box, KEYS
from crithalf import get_model

OLD = [0.0, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.25]
NEW = [0.0, 0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
       0.06, 0.07, 0.10, 0.15, 0.25]


def ladder(n, bets, nsweep, rs, every=6):
    ch = [Ch(n, b) for b in bets]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = nsweep // 3
    recs = [[] for _ in ch]
    sw_ok = np.zeros(len(ch) - 1)
    sw_try = np.zeros(len(ch) - 1)
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            sw_try[k] += 1
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                sw_ok[k] += 1
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn and (it - burn) % every == 0:
            for k, c in enumerate(ch):
                recs[k].append(obs(c.R.copy()))
    return recs, sw_ok / np.maximum(sw_try, 1), \
        [c.acc / max(c.tries, 1) for c in ch]


def halves(rows, mod, dias):
    h = len(rows) // 2
    res = []
    for part in (rows[:h], rows[h:], rows):
        tm, te = {}, {}
        for k in KEYS:
            mu, er = summarize(part, k)
            tm[k], te[k] = float(mu), float(er)
        res.append((tm, te) + best(tm, te, mod, dias))
    worst = max(abs(res[0][0][k] - res[1][0][k])
                / math.hypot(res[0][1][k], res[1][1][k]) for k in KEYS)
    return res[2], worst


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    nsweep = int(sys.argv[2]) if len(sys.argv) > 2 else 1100
    print(__doc__)
    rs = np.random.default_rng(20000 + n)
    mod, dias, cached = get_model(n, rs)
    m, e = stats(lambda: box(n, 2, 2.0, rs), 120, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')

    for lab, bets in (('прежняя (9 ступеней)', OLD),
                      ('густая (14 ступеней)', NEW)):
        recs, swr, mvr = ladder(n, bets, nsweep, rs)
        k = bets.index(0.04)
        (tm, te, chi, geo, sig), worst = halves(recs[k], mod, dias)
        print(f'\n{lab}, N={n}, {nsweep} развёрток')
        print('  доля принятых ОБМЕНОВ по парам: '
              + ' '.join(f'{v:.2f}' for v in swr))
        print(f'  наихудшая пара {swr.min():.3f}, '
              f'ходы на beta=0.04 {mvr[k]:.3f}')
        print(f'  beta=0.04: ' + '  '.join(f'{x}={tm[x]:.3f}±{te[x]:.3f}'
                                           for x in KEYS))
        print(f'  хи2={chi:.1f}  {geo}  H={sig["H"]:+.1f}σ')
        print(f'  РАВНОВЕСИЕ: наихудшее расхождение половин {worst:.1f}σ'
              f' => {"есть" if worst < 3 else "НЕТ"}', flush=True)


if __name__ == '__main__':
    main()
