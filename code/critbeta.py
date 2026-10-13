#!/usr/bin/env python3
"""ЗАВИСИТ ЛИ ВЫВОД ОТ ВЫБОРА beta.

critgeom3.py: хи2 фазы против семейства засевов (d, a) растёт монотонно
8.9 -> 30.0 -> 46.2 -> 111.2 при N = 12,16,20,24, пол контроля 0-3.4.
Подпись настоящего несовпадения. Но beta всё время держалось на 0.04, а
критическая точка с N ползёт (0.070, 0.060, 0.040, 0.040). Если при
N=24 она уехала ниже 0.04, рост хи2 есть отъезд от критичности.

Один прогон лестницы даёт ВСЁ сразу: наблюдаемые на каждой ступени и
теплоёмкость C = beta^2 Var(S) для локализации пика. Если хи2 велико на
всей критической окрестности, вывод от выбора beta не зависит.

ОШИБКИ: блочные (наивные занижают втрое). Доля принятых печатается по
каждой ступени: где она мала, ступени не верить.
"""
import math

import numpy as np

from crit9 import Ch, blockerr
from critphase import obs, summarize, BET
from critgeom3 import model, stats, best, KEYS
from sectors import sprinkle_d

DS = (2, 3, 4)


def ladder(n, nsweep, rs, every=6):
    ch = [Ch(n, b) for b in BET]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    npr = len(pairs)
    burn = nsweep // 3
    recs = [[] for _ in ch]
    acts = [[] for _ in ch]
    for it in range(nsweep + burn):
        for c in ch:
            for _ in range(npr):
                i, j = pairs[rs.integers(npr)]
                c.step(i, j, rs)
        for k in range(len(ch) - 1):
            a, b = ch[k], ch[k + 1]
            if rs.random() < math.exp(min(
                    0.0, (a.beta - b.beta) * (a.S - b.S))):
                a.R, b.R = b.R, a.R
                a.Iv, b.Iv = b.Iv, a.Iv
                a.c, b.c = b.c, a.c
                a.S, b.S = b.S, a.S
                a.nrel, b.nrel = b.nrel, a.nrel
        if it >= burn and (it - burn) % every == 0:
            for k, c in enumerate(ch):
                recs[k].append(obs(c.R.copy()))
                acts[k].append(c.S)
    rates = [c.acc / max(c.tries, 1) for c in ch]
    return recs, acts, rates


def main():
    print(__doc__)
    n, nsweep = 24, 900
    rs = np.random.default_rng(606162)
    mod = model(n, 150, rs)
    dias = {d: stats(lambda: sprinkle_d(n, d, rs), 150, rs) for d in DS}

    m, e = stats(lambda: __import__('critgeom3').box(n, 2, 2.0, rs),
                 150, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')

    recs, acts, rates = ladder(n, nsweep, rs)
    print(f'\nN={n}, {len(recs[0])} образцов на ступень')
    print(f"{'beta':>6}{'принято':>9}{'C=b²Var(S)':>12}"
          f"{'r':>14}{'H':>13}{'хи2':>8}  геометрия")
    for k, b in enumerate(BET):
        tm, te = {}, {}
        for key in KEYS:
            mu, er = summarize(recs[k], key)
            tm[key], te[key] = float(mu), float(er)
        S = np.array(acts[k], float)
        C = b * b * float(S.var())
        chi, geo, _ = best(tm, te, mod, dias)
        print(f'{b:>6.2f}{rates[k]:>9.3f}{C:>12.3f}'
              f"{tm['r']:>9.3f}±{te['r']:.3f}"
              f"{tm['H']:>8.2f}±{te['H']:.2f}{chi:>8.1f}  {geo}",
              flush=True)
    print('\nЧТЕНИЕ: если хи2 велико на всей окрестности пика C,')
    print('вывод от выбора beta не зависит.')


if __name__ == '__main__':
    main()
