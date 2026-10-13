#!/usr/bin/env python3
"""ВЕРСИЯ 3: монотонность по d вместо поиска, и семь элементов.

Версия 2 выдала «корона нереализуема при d = 4», что НЕВОЗМОЖНО:
реализуемость МОНОТОННА по d — конфигурация в d измерениях вкладывается
в d+1 добавлением нулевой пространственной координаты, причинные
отношения сохраняются точно. Значит виноват поиск (в четырёх измерениях
пространство поиска больше), а не ответ. Здесь монотонность используется
как ЗАКОН, а не проверяется: найдено при d — реализуемо при всех
больших, и искать там нечего.

Порядки на семи элементах строятся расширением шестиэлементных новым
МАКСИМАЛЬНЫМ элементом по всем нижним множествам — это полно, потому
что у любого порядка есть максимальный элемент, и его удаление даёт
порядок на шести.

  python3 realiz3.py
"""
import itertools
import math

import numpy as np

from realiz2 import (perms_of, canon_batch, closure_batch, all_posets,
                     crown, causal_one, causal_batch, found_keys, search)

COUNTS = {4: 16, 5: 63, 6: 318, 7: 2045}


def extend(pos6, n=7):
    """Порядки на n элементах: добавить максимальный элемент."""
    P, W = perms_of(n), 2.0 ** np.arange(n * n)
    cand = []
    for R6 in pos6.values():
        m = R6.shape[0]
        # нижние множества (down-sets) порядка R6
        for mask in range(1 << m):
            S = [i for i in range(m) if (mask >> i) & 1]
            ok = True
            for i in S:                      # замкнутость вниз
                for j in range(m):
                    if R6[j, i] and j not in S:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            R = np.zeros((n, n), dtype=bool)
            R[:m, :m] = R6
            for i in S:
                R[i, m] = True
            cand.append(R)
    out = {}
    for s in range(0, len(cand), 64):
        B = np.array(cand[s:s + 64])
        k = canon_batch(B, P, W)
        for i in range(len(B)):
            out.setdefault(int(k[i]), B[i].copy())
    return out


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(777001)
    pos = {}
    for n in (4, 5, 6):
        pos[n] = all_posets(n)
    pos[7] = extend(pos[6], 7)
    print('КОНТРОЛЬ: число неизоморфных порядков')
    for n in (4, 5, 6, 7):
        print(f'  n = {n}: {len(pos[n]):>5} (должно {COUNTS[n]:>5})  '
              f'{"ВЕРНО" if len(pos[n])==COUNTS[n] else "НЕВЕРНО"}',
              flush=True)
    P6, W6 = perms_of(6), 2.0 ** np.arange(36)
    ckey = int(canon_batch(crown()[None], P6, W6)[0])
    print()
    print('МИНИМАЛЬНАЯ РАЗМЕРНОСТЬ РЕАЛИЗАЦИИ (монотонность по d — закон)')
    print(f"{'n':>3}{'всего':>7}{'d=2':>8}{'d=3':>8}"
          f"{'нереализуемо при d>=3':>24}")
    unreal2 = {}
    for n in (4, 5, 6, 7):
        Pn, Wn = perms_of(n), 2.0 ** np.arange(n * n)
        keys = set(pos[n])
        # d = 2
        got2 = found_keys(n, 2, {4: 60000, 5: 40000, 6: 30000,
                                 7: 20000}[n], rs, Pn, Wn) & keys
        miss2 = sorted(keys - got2)
        for key in list(miss2):
            if search(pos[n][key], 2, rs, iters=6000, restarts=10) is not None:
                got2.add(key)
        u2 = sorted(keys - got2)
        unreal2[n] = u2
        # d = 3: ищем только то, что не вышло при d = 2 (монотонность)
        got3 = set(got2)
        got3 |= found_keys(n, 3, {4: 40000, 5: 30000, 6: 25000,
                                  7: 15000}[n], rs, Pn, Wn) & keys
        miss3 = sorted(keys - got3)
        for key in list(miss3):
            if search(pos[n][key], 3, rs, iters=8000, restarts=14) is not None:
                got3.add(key)
        u3 = sorted(keys - got3)
        print(f'{n:>3}{len(keys):>7}{len(got2):>8}{len(got3):>8}'
              f'{len(u3):>24}', flush=True)
    print()
    print('КОНТРОЛЬ КОРОНЫ')
    print(f'  d = 2: корона нереализуема — '
          f'{"ВЕРНО" if ckey in unreal2[6] else "ПРОВАЛ"}')
    print()
    print('ЧТО ИМЕННО ЗАПРЕЩЕНО ПРИ d = 2 (на 6 элементах)')
    for key in unreal2[6]:
        R = pos[6][key]
        mins = [i for i in range(6) if not R[:, i].any()]
        maxs = [i for i in range(6) if not R[i, :].any()]
        print(f'  класс: отношений {int(R.sum())}, минимальных '
              f'{len(mins)}, максимальных {len(maxs)}'
              f'{"   <- КОРОНА" if key == ckey else ""}')


if __name__ == '__main__':
    main()
