#!/usr/bin/env python3
"""ЧЕСТНОСТЬ К ЗАСЕВУ и попытка починить условие.

Первый прогон: у засева d=4 медианное отклонение 0.47, а у
иерархического слоистого при q=0.6 — 0.0041, то есть в СТО раз лучше.
Условие не пусто, а ВЫВЕРНУТО: оно предпочитает соперника.

Но прежде чем это записывать — два пропущенных контроля.

1. УСТОЙЧИВОСТЬ ПО nmin. У засева прошлые разного размера, и малое
   прошлое шумит само по себе: для m точек с долей r ожидаемое
   относительное отклонение ~ 1/sqrt(r*C(m,2)). При m=20, r=0.085 это
   25%. Значит часть провала засева — статистика, а не структура.

2. ПОЧИНКА. Правильная форма «часть похожа на целое» — сравнить
   НАБЛЮДЁННЫЙ разброс с ОЖИДАЕМЫМ. Отношение observed/expected:
   у самоподобного порядка оно около единицы, у несамоподобного
   больше. Ожидаемый разброс считается биномиально по размеру прошлого.

  python3 selfsim2.py
"""
import math

import numpy as np

from sectors import sprinkle_d, layered
from selfsim import dominance, percolation, hier_layered, chain


def past_table(Rc, nmin):
    n = Rc.shape[0]
    rg = Rc.sum() / (n * (n - 1) / 2)
    rows = []
    for x in range(n):
        idx = np.nonzero(Rc[:, x])[0]
        m = len(idx)
        if m < nmin:
            continue
        sub = Rc[np.ix_(idx, idx)]
        npair = m * (m - 1) / 2
        rows.append((m, sub.sum() / npair, npair))
    return rg, rows


def stats(Rc, nmin):
    rg, rows = past_table(Rc, nmin)
    if not rows:
        return rg, None, None, 0
    dev = np.array([abs(r - rg) / max(rg, 1e-12) for _, r, _ in rows])
    # ожидаемое относительное отклонение (биномиальное)
    exp = np.array([math.sqrt(max(rg * (1 - rg), 1e-12) / np_)
                    / max(rg, 1e-12) for _, _, np_ in rows])
    ratio = dev / exp
    return rg, float(np.median(dev)), float(np.median(ratio)), len(rows)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(555777)
    N = 600
    cases = [
        ('цепь', lambda: chain(N)),
        ('засев d=2', lambda: sprinkle_d(N, 2, rs)),
        ('засев d=3', lambda: sprinkle_d(N, 3, rs)),
        ('засев d=4', lambda: sprinkle_d(N, 4, rs)),
        ('доминирование k=4', lambda: dominance(N, 4, rs)),
        ('иерарх. b=3 d=6 q=0.6', lambda: hier_layered(3, 6, 0.6, rs)),
        ('иерарх. b=3 d=6 q=0.3', lambda: hier_layered(3, 6, 0.3, rs)),
        ('иерарх. b=5 d=4 q=0.6', lambda: hier_layered(5, 4, 0.6, rs)),
        ('слоистый k=50 q=0.7', lambda: layered(N, 50, 0.7, rs)[0]),
        ('слоистый k=3 q=0.3', lambda: layered(N, 3, 0.3, rs)[0]),
        ('перколяция p=0.019', lambda: percolation(N, 0.019, rs)),
    ]
    mats = {nm: f() for nm, f in cases}
    print('ОТКЛОНЕНИЕ (медиана |r(x)-r|/r) по порогу nmin')
    print(f"{'сектор':>24}{'nmin=20':>10}{'50':>10}{'100':>10}{'150':>10}")
    for nm, Rc in mats.items():
        line = f'{nm:>24}'
        for nmin in (20, 50, 100, 150):
            rg, md, _, cnt = stats(Rc, nmin)
            line += f'{(f"{md:.4f}" if md is not None else "—"):>10}'
        print(line, flush=True)
    print()
    print('ПОЧИНКА: наблюдённый разброс / ОЖИДАЕМЫЙ (самоподобный ~ 1)')
    print(f"{'сектор':>24}{'nmin=20':>10}{'50':>10}{'100':>10}{'150':>10}"
          f"{'r':>9}")
    for nm, Rc in mats.items():
        line = f'{nm:>24}'
        rgv = None
        for nmin in (20, 50, 100, 150):
            rg, _, rt, cnt = stats(Rc, nmin)
            rgv = rg
            line += f'{(f"{rt:.2f}" if rt is not None else "—"):>10}'
        print(line + f'{rgv:>9.4f}', flush=True)
    print()
    print('''ЧТЕНИЕ. Если у засева отношение около единицы, а у соперников
заметно больше — условие в починенном виде работает. Если у соперников
оно тоже около единицы или МЕНЬШЕ — условие не отбирает:近 к единице
его загоняет жёсткость (у почти полного порядка и наблюдённый, и
ожидаемый разброс стремятся к нулю), а не похожесть на мир.''')


if __name__ == '__main__':
    main()
