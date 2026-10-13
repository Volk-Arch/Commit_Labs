#!/usr/bin/env python3
"""СХОДИТСЯ ЛИ РЕЗКИЙ ОПЕРАТОР ПРИ БОЛЬШИХ N.

`fastwave.py` сверен с матричным счётом до 1e-14. По закону
`matter-doc.md` множитель ослабления зависит только от eps*N, значит
резкий оператор (eps=1) обязан сойтись при N ~ 1.6e5 - то, что утром
было объявлено недостижимым.

Цели: <B t^2> = 2, <B x^2> = -2, <B tau^2> = 4 (знак есть соглашение
сигнатуры; сегодня измерено, что оператор сходится к обратному знаку,
поэтому печатаются и значения, и отношения к целям).

ЧТО РАЗЛИЧАЕТ ИСХОДЫ: |отношение| идёт к 1 с ростом N => оператор
сходится, материя в 1+1 достижима; стоит на 0.2-0.3 => закон eps*N к
eps=1 не продолжается, и это тоже надо знать.
"""
import math
import sys
import time

import numpy as np

from fastwave import run

TG = {'t2': 2.0, 'x2': -2.0, 'tau2': 4.0}


def main():
    print(__doc__)
    print(f"{'N':>8}{'заc.':>5}" + ''.join(f'{f"<B {k}>":>20}' for k in TG)
          + f"{'время':>8}")
    for n, reps, npts in ((1000, 4, 250), (3000, 4, 250), (10000, 3, 250),
                          (30000, 3, 200), (100000, 2, 150),
                          (300000, 1, 120)):
        t0 = time.time()
        acc = {k: [] for k in TG}
        for r in range(reps):
            rs = np.random.default_rng(900 + 17 * r + n)
            out = run(n, rs, npts=npts)
            if out is None:
                continue
            for k in TG:
                acc[k].append(out[k][0])
        row = f'{n:>8}{reps:>5}'
        for k in TG:
            v = np.array(acc[k])
            m = v.mean()
            e = v.std() / math.sqrt(len(v)) if len(v) > 1 else float('nan')
            row += f'{m:>11.2f}±{e:<8.2f}'
        print(row + f'{time.time() - t0:>7.0f}с', flush=True)
        print('         отношения к целям: '
              + '  '.join(f'{k}: {np.mean(acc[k]) / TG[k]:+.3f}'
                          for k in TG), flush=True)


if __name__ == '__main__':
    main()
