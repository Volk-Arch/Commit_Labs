#!/usr/bin/env python3
"""ПЕРЕМЕР КРИТИЧНОСТИ С ВЕРНЫМ ЗНАКОМ.

`retraction-doc.md`: у слоистых S ~ -0.18 n^2, у засевов S/n^2 -> 0,
энтропия +0.173 n^2. Значит подавляет слоистые вес exp(+beta S), то есть
в коде - ОТРИЦАТЕЛЬНОЕ beta (вес exp(-beta S) при beta<0 и есть
exp(+|beta| S)). Порог из отзыва: |beta| > 0.96.

Весь день считалось beta > 0, то есть знаком, который слоистые УСИЛИВАЕТ.
Отсюда и выходило, что рост beta уводит от геометрии.

ПРЕДСКАЗАНИЕ до замера: при beta < 0 ансамбль обязан идти К геометрии -
доля связанных к значению Майрхейма-Майера, совместная подгонка (d,a) к
малым хи2. Если вместо этого он уходит в цепь (r -> 1) - значит знак
перегибает, и надо смотреть промежуточные |beta|.

КОНТРОЛИ: доля принятых обменов по парам; при beta=0 обязано выйти
прежнее значение; совместная подгонка проверяется известным засевом.
"""
import math

import numpy as np

from crit9 import Ch, blockerr
from critphase import obs, summarize
from critgeom3 import best, stats, box, KEYS
from crithalf import get_model

N = 24
# тонкая лестница: весь переход сидит между 0 и -0.1 (при beta=0
# действие 40, при -0.1 уже 868, значит шаг 0.1 даёт Delta_beta*sigma
# в сотни, и обмены не идут - та же ошибка, что утром)
BET = [0.0, -0.004, -0.008, -0.013, -0.018, -0.024, -0.030, -0.038,
       -0.047, -0.057, -0.070, -0.085, -0.100]


def ladder(n, bets, nsweep, rs, every=6):
    ch = [Ch(n, b) for b in bets]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = nsweep // 3
    recs = [[] for _ in ch]
    acts = [[] for _ in ch]
    sok = np.zeros(len(ch) - 1)
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                sok[k] += 1
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn and (it - burn) % every == 0:
            for k, c in enumerate(ch):
                recs[k].append(obs(c.R.copy()))
                acts[k].append(c.S)
    return recs, acts, sok / (nsweep + burn), \
        [c.acc / max(c.tries, 1) for c in ch]


def main():
    print(__doc__)
    rs = np.random.default_rng(606070)
    mod, dias, _ = get_model(N, rs)
    m, e = stats(lambda: box(N, 2, 2.0, rs), 120, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ подгонки: засев d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')

    recs, acts, sw, mv = ladder(N, BET, 900, rs)
    print(f'обмены по парам: ' + ' '.join(f'{v:.2f}' for v in sw))
    print(f'обмены мин {sw.min():.2f}, образцов {len(recs[0])} '
          f'=> {"ОК" if sw.min() > 0.1 else "ЛЕСТНИЦА СЛОМАНА"}')
    S0 = np.array(acts[0], float)
    print(f'разброс действия при beta=0: {S0.std():.1f} '
          f'(шаг лестницы обязан быть ~1/{S0.std():.0f})')
    print(f"\n{'beta':>7}{'приn.':>7}{'<S>':>9}{'r':>14}{'H':>12}"
          f"{'L':>14}{'хи2':>8}  геометрия")
    for k, b in enumerate(BET):
        tm, te = {}, {}
        for key in KEYS:
            mu, er = summarize(recs[k], key)
            tm[key], te[key] = float(mu), float(er)
        chi, geo, _ = best(tm, te, mod, dias)
        print(f'{b:>7.2f}{mv[k]:>7.3f}{np.mean(acts[k]):>9.1f}'
              f"{tm['r']:>9.3f}±{te['r']:.3f}"
              f"{tm['H']:>7.2f}±{te['H']:.2f}"
              f"{tm['L']:>9.1f}±{te['L']:.1f}{chi:>8.1f}  {geo}",
              flush=True)
    print('\nЧТЕНИЕ: идёт ли хи2 ВНИЗ с ростом |beta| — тогда верный знак')
    print('ведёт к геометрии, и отзыв подтверждён замером.')


if __name__ == '__main__':
    main()
