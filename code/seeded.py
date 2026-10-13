#!/usr/bin/env python3
"""РАЗГОН: виновата воронка или начальное условие?

ЧТО НАШЁЛ `locanti.py`. Локальная антицепь НЕ насыщает границу по
звеньям (3.8…17.5 вместо `m` = 20), и доля упорядоченных пар уходит к
`r` = 0.87…0.996, то есть почти полный порядок, `d` = 1.0…2.4.

ДИАГНОЗ. Локально не находится `m` попарно несравнимых: близкие по
общему прошлому оказываются сравнимыми МЕЖДУ СОБОЙ. Но в настоящем
сприклинге близкие события как раз несравнимы, значит запрет не
фундаментален. Похоже на САМОПОДДЕРЖИВАЮЩУЮСЯ воронку: ранний порядок
цепочечен, локальность в цепочечном порядке находит сравнимых, это
держит его цепочечным.

ПРОВЕРКА. Начать не с одного элемента, а с ШИРОКОЙ АНТИЦЕПИ размера
`W`: тогда на раннем этапе несравнимых много. Если правило удержит
размерность — виноват был разгон. Если всё равно съедет к цепи —
воронка настоящая, и это свойство правила, а не начала.

КОНТРОЛЬ, отличающий одно от другого: печатается `r` и число звеньев
при РАЗНЫХ `W`. Если ответ от `W` не зависит — начальное условие
забывается, то есть воронка; если зависит — помнит.

  python3 seeded.py
"""
import math
import time

import numpy as np

from grow import gen_sprinkle
from anti import stat, HDR, show


def gen_seeded(n, m, F, W, rs):
    """Старт: антицепь из W элементов (нет отношений). Дальше — локальная
    антицепь, как в `locanti.py`."""
    A = np.zeros((n, n), dtype=bool)
    nfut = np.zeros(n, dtype=np.int64)
    W = min(int(W), n - 1)
    for j in range(W, n):
        cand = np.flatnonzero(nfut[:j] <= F)
        if len(cand) == 0:
            cand = np.flatnonzero(nfut[:j] == nfut[:j].min())
        a = int(cand[rs.integers(len(cand))])
        S = [a]
        k = min(int(m) - 1, j - 1)
        if k > 0:
            inc = np.flatnonzero(~A[:j, a] & ~A[a, :j])
            inc = inc[inc != a]
            if len(inc) > 0:
                ov = (A[:, inc] & A[:, a][:, None]).sum(axis=0)
                order = inc[np.argsort(-ov)]
                blocked = A[:j, a] | A[a, :j]
                for x in order:
                    if len(S) - 1 >= k:
                        break
                    if blocked[x]:
                        continue
                    S.append(int(x))
                    blocked |= A[:j, x] | A[x, :j]
        S = np.array(S, dtype=np.int64)
        col = np.zeros(n, dtype=bool)
        col[S] = True
        col[:j] |= A[:j, S].any(axis=1)
        A[:, j] = col
        nfut[:j] += col[:j]
    return A


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 110)
    print('ЭТАЛОН')
    print(HDR)
    show('сприклинг 4D', 1200, stat(gen_sprinkle, 1200, 6, 300, d=4))
    print()
    print('=' * 110)
    print('ШИРИНА РАЗГОНА W при F = 200, m = 20, N = 1200')
    print('(если ответ от W не зависит — начало забывается, воронка)')
    print(HDR)
    for W in (1, 20, 100, 300, 600):
        show(f'разгон W={W}', 1200,
             stat(gen_seeded, 1200, 5, 1100 + W, m=20, F=200, W=W))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ТО ЖЕ при F = 10⁹ (ширина не ограничена)')
    print(HDR)
    for W in (1, 100, 600):
        show(f'разгон W={W}', 1200,
             stat(gen_seeded, 1200, 5, 1300 + W, m=20, F=10 ** 9, W=W))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ПАМЯТЬ О РАЗГОНЕ: держится ли при росте N (W = N/2)?')
    print(HDR)
    for n in (600, 1200, 2400):
        show(f'W=N/2, N={n}', n,
             stat(gen_seeded, n, 4, 1500 + n, m=20, F=200, W=n // 2))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
