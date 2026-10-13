#!/usr/bin/env python3
"""ТРЕНД С КОНТРОЛЕМ РАВНОВЕСИЯ ВНУТРИ ПРОГОНА.

criteq.py: при N=24 удвоение прогона сдвинуло r на 3.1 сигмы и H на 4.0,
а хи2 упал 113.4 -> 41.6. Значит на 900 развёртках равновесия нет, и
прежний «монотонный рост» 8.9/30/46/111 недействителен: при фиксированном
числе развёрток равновесие портится с ростом N.

Починка - контроль РАВНОВЕСИЯ ВНУТРИ прогона: первая половина выборки
против второй. Бесплатно и строго. Точка принимается, только если
половины совпадают.

Модель (сетка засевов) кешируется на диск, чтобы весь бюджет вызова
уходил на развёртки.

ЧТО СЧИТАЕТСЯ РЕЗУЛЬТАТОМ: не величина хи2, а отклонение по ВЫСОТЕ,
которое устояло при удвоении и обострилось (+5.5σ). Если половины
совпадают и H-отклонение держится - несовпадение настоящее.
"""
import json
import math
import os
import sys

import numpy as np

from critphase import BET, summarize
from critbeta import ladder
from critgeom3 import model, stats, best, box, KEYS, DENSE
from sectors import sprinkle_d

DS = (2, 3, 4)
CACHE = '/tmp/claude-0/critmodel-{}.json'


def get_model(n, rs):
    p = CACHE.format(n)
    if os.path.exists(p):
        with open(p) as f:
            raw = json.load(f)
        mod = {int(d): ({k: np.array(v[0][k]) for k in KEYS}, v[1])
               for d, v in raw['mod'].items()}
        dias = {int(d): (v[0], v[1]) for d, v in raw['dias'].items()}
        return mod, dias, True
    mod = model(n, 120, rs)
    dias = {d: stats(lambda: sprinkle_d(n, d, rs), 120, rs) for d in DS}
    with open(p, 'w') as f:
        json.dump(dict(mod={str(d): [{k: v[0][k].tolist() for k in KEYS},
                                     v[1]] for d, v in mod.items()},
                       dias={str(d): [v[0], v[1]]
                             for d, v in dias.items()}), f)
    return mod, dias, False


def show(lab, rows, mod, dias):
    tm, te = {}, {}
    for k in KEYS:
        mu, er = summarize(rows, k)
        tm[k], te[k] = float(mu), float(er)
    chi, geo, sig = best(tm, te, mod, dias)
    print(f'  {lab:>22}: ' + '  '.join(f'{k}={tm[k]:.3f}±{te[k]:.3f}'
                                       for k in KEYS))
    print(f'  {"":>22}  хи2={chi:.1f}  {geo}  '
          f'H={sig["H"]:+.1f}σ', flush=True)
    return tm, te, chi, sig


def main():
    n = int(sys.argv[1])
    nsweep = int(sys.argv[2])
    print(__doc__)
    rs = np.random.default_rng(10000 + n)
    mod, dias, cached = get_model(n, rs)
    m, e = stats(lambda: box(n, 2, 2.0, rs), 120, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}  (модель из кеша: {cached})')

    k = BET.index(0.04)
    recs, acts, rates = ladder(n, nsweep, rs)
    rows = recs[k]
    h = len(rows) // 2
    print(f'\nN={n}, {nsweep} развёрток, принято {rates[k]:.3f}, '
          f'образцов {len(rows)}')
    a1 = show('первая половина', rows[:h], mod, dias)
    a2 = show('вторая половина', rows[h:], mod, dias)
    print('  РАВНОВЕСИЕ, половина против половины:')
    okall = True
    for key in KEYS:
        dd = abs(a1[0][key] - a2[0][key]) / math.hypot(a1[1][key],
                                                       a2[1][key])
        okall &= dd < 3.0
        print(f'    {key}: расхождение {dd:.1f}σ'
              f'{"" if dd < 3 else "   <-- НЕ СОШЛОСЬ"}')
    print(f'  => {"равновесие есть" if okall else "РАВНОВЕСИЯ НЕТ"}')
    full = show('весь прогон', rows, mod, dias)
    with open(f'/tmp/claude-0/crithalf-{n}.json', 'w') as f:
        json.dump(dict(n=n, nsweep=nsweep, eq=bool(okall),
                       chi=full[2], H=full[3]['H'], rate=rates[k],
                       tm=full[0], te=full[1]), f)


if __name__ == '__main__':
    main()
