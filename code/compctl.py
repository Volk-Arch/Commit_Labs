#!/usr/bin/env python3
"""КОНТРОЛЬ КОНВЕЙЕРА для замера компонент порознь.

БЕЗ ЭТОГО числа из `comp.py` силы не имеют. Половина плоского плеча
объявляется «семейством» — то экспоненциальным, то корневым, — и все
физические параметры обязаны выйти НУЛЯМИ. У плоского плеча кривизны
нет по построению, поэтому любой ненулевой ответ есть изъян конвейера:
помехи, разбиения на ячейки или самого складного ножа.

ВТОРАЯ ПОЛОВИНА контроля: засевы делятся ПО МИPАМ, а не по строкам —
единица независимости есть сприклинг, и деление по строкам создало бы
общие миры в обеих половинах.

  python3 compctl.py
"""
import glob
import math
import pickle

import numpy as np

from pencil import SCRATCH, VB, truth
from comp import cells, design, solve, KF, V2CAL, V3CAL, IX2, IX3


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    FA, FS = res[[k for k in res if k[0] == 'flat'][0]]
    seeds = np.array(sorted(set(FS.tolist())))
    h = len(seeds) // 2
    s1, s2 = set(seeds[:h].tolist()), set(seeds[h:].tolist())
    m1 = np.array([s in s1 for s in FS])
    m2 = ~m1
    print(f'плоское плечо: {len(FA)} строк, {len(seeds)} миров'
          f' -> половины по {h} и {len(seeds)-h} МИPОВ')
    print(f'  (деление по миpам, не по строкам: '
          f'{m1.sum()} и {m2.sum()} интервалов)')
    print()
    A1, S1 = FA[m1], FS[m1]
    A2, S2 = FA[m2], FS[m2]

    names = ('R_ηη lin', 'R_xx lin', 'R_ηη quad', 'R_xx quad',
             'R_xx exp', 'R_ηη root')

    for fake in ('exp', 'root'):
        print('=' * 78)
        print(f'ПОЛОВИНА ПЛОСКОГО объявлена семейством «{fake}»')

        def run(col, drop=None):
            blocks = []
            for fam, (A, sd) in (('flat', (A1, S1)), (fake, (A2, S2))):
                cv = cells(A, sd, col, drop)
                blocks.append((design(cv, fam), cv[:, 0]))
            return solve(blocks)

        for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
            p, cond, nrow = run(col)
            jk = []
            for s in sorted(s1 | s2):
                try:
                    q, _, _ = run(col, s)
                    jk.append(q)
                except Exception:
                    pass
            jk = np.array(jk); n = len(jk)
            se = np.sqrt((n - 1) / n
                         * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
            idx = 4 if fake == 'exp' else 5
            v, e = p[idx] / (vcal * KF), se[idx] / abs(vcal * KF)
            print(f'  {cname}:  {names[idx]} = {v:+.5f} ± {e:.5f}'
                  f'   = {abs(v)/max(e,1e-12):.1f} сигм от НУЛЯ'
                  f'   (ячеек {nrow})')
        print()
    print('=' * 78)
    print('''ЧТЕНИЕ. Меньше 2 сигм от нуля — конвейер чист, и числа
`comp.py` (R_xx = +0.00169 ± 0.00044 при истине +0.00180, 0.2 сигм;
R_ηη = +0.00228 ± 0.00042 при истине +0.00184, 1.1 сигм) имеют силу.
Больше — изъян в конвейере, и замер надо снимать.''')


if __name__ == '__main__':
    main()
