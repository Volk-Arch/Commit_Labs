#!/usr/bin/env python3
"""ЛОКАЛЬНАЯ АНТИЦЕПЬ: два диагноза сходятся в одно правило.

ДВА ИЗМЕРЕНИЯ, которые сюда ведут.
  1. `growth-doc.md`: звеньев не хватает, потому что `N₀/N ≤ ⟨m⟩` и при
     равномерном случайном `S` почти все его элементы под другими.
  2. `anti.py`: присоединение к АНТИЦЕПИ насыщает границу точно
     (19.22 звена при `m = 20`), но многообразность рушится и портится с
     размером (`C₄/пред` = 2.76 → 8.85 → 30.35 при N = 600, 1200, 2400).
  3. `local2.py`: локальность (близость по ОБЩЕМУ ПРОШЛОМУ) исправляет
     знак наклона `N_m/N₀`, но без неё размерность упирается в 2.13.

Антицепь в `anti.py` набиралась жадно и РАВНОМЕРНО среди всех
несравнимых, то есть размазана по всему порядку — снова нелокально. Оба
диагноза указывают в одно место: антицепь должна быть ЛОКАЛЬНОЙ.

ПРАВИЛО. Опора берётся среди элементов с малым будущим (`|future| ≤ F`
— порядковое выражение «недавности», оно же регулятор ширины). Затем
антицепь размера `m` набирается среди несравнимых с опорой, но не
равномерно, а по убыванию ПЕРЕКРЫТИЯ ПРОШЛОГО с опорой. Прошлое нового
элемента — замыкание вниз.

    F — ширина поверхности,  m — число звеньев (ровно, по построению)

ОГОВОРКА та же: это не КПР, общая ковариация нарушена, и на вывод про
КПР отсюда ничего не следует.

ПРОВЕРКА ПОСТРОЕНИЯ: звеньев на элемент обязано выйти ≈ `m`.

  python3 locanti.py
"""
import math
import time

import numpy as np

from grow import gen_sprinkle
from anti import stat, HDR, show


def gen_locanti(n, m, F, rs):
    A = np.zeros((n, n), dtype=bool)
    nfut = np.zeros(n, dtype=np.int64)
    for j in range(1, n):
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
                # жадно, но В ПОРЯДКЕ БЛИЗОСТИ: антицепь из ближайших
                taken = np.zeros(j, dtype=bool)
                taken[a] = True
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
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 110)
    print('ШИРИНА F при m = 20 (звеньев обязано быть ≈ 20)')
    print(HDR)
    for F in (5, 20, 60, 200, 10 ** 9):
        show(f'лок.антиц. F={F:g} m=20', 1200,
             stat(gen_locanti, 1200, 5, 700 + int(math.log1p(F) * 31),
                  m=20, F=F))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ЧИСЛО ЗВЕНЬЕВ m при лучшей ширине')
    print(HDR)
    for m in (8, 20, 40):
        show(f'лок.антиц. F=60 m={m}', 1200,
             stat(gen_locanti, 1200, 5, 800 + m, m=m, F=60))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('ЗАВИСИМОСТЬ ОТ РАЗМЕРА (у anti.py C₄ рос 2.8 → 8.8 → 30.4)')
    print(HDR)
    for n in (600, 1200, 2400):
        show('лок.антиц. F=60 m=20', n,
             stat(gen_locanti, n, 4, 900 + n, m=20, F=60))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
