#!/usr/bin/env python3
"""ПЕРЕМЕР N=24 ГУСТОЙ ЛЕСТНИЦЕЙ, по одному зерну за прогон.

Главный висящий пункт после critseeds.py: при N=24 прежняя (сломанная)
лестница давала разброс по зёрнам вчетверо больше блочной ошибки, и
точка была недоступна. Густая лестница чинит обмены (минимум 0.19
против 0.07), и вопрос - хватит ли развёрток.

Протокол сам ответит: накапливаем независимые зёрна, ошибку берём как
максимум из блочной и разброса. Если разброс не превышает блочную -
развёрток хватило, и точка N=24 появляется; если превышает - не хватило,
и это записывается так.

Результаты дописываются в /tmp/claude-0/n24.jsonl, сводка считается при
вызове с аргументом 'сводка'.
"""
import json
import math
import os
import sys

import numpy as np

from critphase import summarize
from critmix import ladder, NEW
from critgeom3 import stats, best, box, KEYS
from crithalf import get_model

OUT = '/tmp/claude-0/n24.jsonl'
N = 24


def summary():
    rows = [json.loads(l) for l in open(OUT)]
    print(f'зёрен: {len(rows)}')
    rs = np.random.default_rng(31337)
    mod, dias, _ = get_model(N, rs)
    m, e = stats(lambda: box(N, 2, 2.0, rs), 120, rs)
    c0, g0, _ = best(m, e, mod, dias)
    print(f'КОНТРОЛЬ d=2 a=2.0 -> {g0}, хи2={c0:.1f} '
          f'=> {"ОК" if c0 < 8 else "СБОЙ"}')
    tm, te = {}, {}
    print(f"\n{'набл':>5}{'среднее':>10}{'блочная':>10}"
          f"{'по зёрнам':>11}{'отнош':>8}")
    for k in KEYS:
        vs = np.array([r['tm'][k] for r in rows])
        blk = float(np.mean([r['te'][k] for r in rows]))
        sct = float(vs.std(ddof=1) / math.sqrt(len(vs))) if len(vs) > 1 \
            else blk
        tm[k] = float(vs.mean())
        te[k] = max(blk, sct)
        print(f'{k:>5}{vs.mean():>10.3f}{blk:>10.3f}{sct:>11.3f}'
              f'{sct / blk:>8.2f}')
    ok = all(max(1e-9, np.array([r['te'][k] for r in rows]).mean()) >=
             (np.array([r['tm'][k] for r in rows]).std(ddof=1)
              / math.sqrt(len(rows))) for k in KEYS) if len(rows) > 1 \
        else False
    print(f'\nразвёрток {"ХВАТИЛО" if ok else "НЕ ХВАТИЛО"}')
    chi, geo, sig = best(tm, te, mod, dias)
    print(f'вердикт N=24 с честной ошибкой: хи2={chi:.1f}, {geo}')
    print('  отклонения: '
          + '  '.join(f'{k}={sig[k]:+.1f}σ' for k in KEYS))
    print('  (пол контроля ~2; при N=16 было 13.8, при N=20 35.3)')


def main():
    if sys.argv[1:] and sys.argv[1] == 'сводка':
        summary()
        return
    sd = int(sys.argv[1])
    nsweep = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
    print(__doc__)
    rs0 = np.random.default_rng(777000 + N)
    mod, dias, cached = get_model(N, rs0)
    rs = np.random.default_rng(sd)
    recs, swr, mvr = ladder(N, NEW, nsweep, rs)
    k = NEW.index(0.04)
    rows = recs[k]
    h = len(rows) // 2
    parts = []
    for part in (rows[:h], rows[h:], rows):
        tm, te = {}, {}
        for key in KEYS:
            mu, er = summarize(part, key)
            tm[key], te[key] = float(mu), float(er)
        parts.append((tm, te))
    worst = max(abs(parts[0][0][k2] - parts[1][0][k2])
                / math.hypot(parts[0][1][k2], parts[1][1][k2])
                for k2 in KEYS)
    tm, te = parts[2]
    chi, geo, sig = best(tm, te, mod, dias)
    print(f'зерно {sd}, {nsweep} развёрток, обмены мин {swr.min():.2f}, '
          f'ходы {mvr[k]:.3f}, образцов {len(rows)}')
    print('  ' + '  '.join(f'{x}={tm[x]:.3f}±{te[x]:.3f}' for x in KEYS))
    print(f'  половины расходятся на {worst:.1f}σ, '
          f'хи2={chi:.1f}, {geo}')
    with open(OUT, 'a') as f:
        f.write(json.dumps(dict(seed=sd, nsweep=nsweep, tm=tm, te=te,
                                worst=worst, chi=chi,
                                swmin=float(swr.min()))) + '\n')


if __name__ == '__main__':
    main()
