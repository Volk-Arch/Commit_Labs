#!/usr/bin/env python3
"""ПРОЩУПАТЬ ЛАЗЕЙКУ: тонкий скан у q -> 1 и зависимость запаса от N.

Предыдущий прогон: из 58 секторов с энтропией выше мировой не проходит
ни один, но ближайший — иерархический b=2, q=0.9, r=0.9996 — стоит на
отношении 3.90 против порога 3.69. Запас 6 процентов, и это та же
ЖЁСТКАЯ лазейка: почти полный порядок проходит самоподобие даром.

Видна и развилка: при q -> 1 отношение падает (жёстче), но энтропия
тоже падает (slots * H(q) -> 0). Вопрос в том, пересекаются ли кривые.
Здесь q сканируется плотно у единицы, и смотрится зависимость от N.

Энтропия иерархического взята ЩЕДРО: slots*H(q) + ln(N!) целиком, без
деления на автоморфизмы — в пользу соперника.

  python3 selfsim5.py
"""
import math

import numpy as np

from sectors import sprinkle_d, H2
from selfsim2 import stats
from selfsim4 import hier_n


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(161803)
    for N in (300, 600, 1000):
        S_world = 4.0 * N * math.log(N)
        sp = []
        for d in (2, 3, 4, 5, 6):
            rg, md, rt, cnt = stats(sprinkle_d(N, d, rs), 50)
            if rt is not None and cnt >= 10:
                sp.append(rt)
        thr = max(sp)
        print('=' * 72)
        print(f'N = {N}: порог по засевам {thr:.2f}, '
              f'энтропия мира {S_world:.0f}')
        print(f"{'b':>4}{'q':>7}{'r':>9}{'отношение':>11}{'S':>10}"
              f"{'S>мир':>7}{'проходит':>10}{'ОПАСНО':>8}")
        bad = []
        for b in (2, 3, 4):
            for q in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99,
                      0.995, 0.999):
                Rc, slots = hier_n(N, b, q, rs)
                S = slots * H2(q) + math.lgamma(N + 1)
                rg, md, rt, cnt = stats(Rc, 50)
                if rt is None or cnt < 10:
                    continue
                hi = S > S_world
                ok = rt <= thr
                danger = hi and ok
                if danger:
                    bad.append((b, q, rt, S, rg))
                print(f'{b:>4}{q:>7.3f}{rg:>9.5f}{rt:>11.2f}{S:>10.0f}'
                      f'{("да" if hi else "нет"):>7}'
                      f'{("ДА" if ok else "нет"):>10}'
                      f'{("<<<" if danger else ""):>8}', flush=True)
        print()
        if bad:
            print(f'  ЛАЗЕЙКА НАЙДЕНА: {len(bad)} конфигураций и богаче '
                  f'энтропией, и самоподобны не хуже мира:')
            for b, q, rt, S, rg in bad:
                print(f'    b={b}, q={q}: отношение {rt:.2f} <= {thr:.2f}, '
                      f'S = {S:.0f} > {S_world:.0f}, r = {rg:.5f}')
            print('  Условие самоподобия отбора НЕ даёт.')
        else:
            print(f'  лазейки нет: всюду либо S < {S_world:.0f}, '
                  f'либо отношение > {thr:.2f}')
        print()


if __name__ == '__main__':
    main()
