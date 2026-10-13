#!/usr/bin/env python3
"""ЛОКАЛЬНЫЙ РОСТ, ИСПРАВЛЕННЫЙ: поверхности нужен регулятор ширины.

ЧТО БЫЛО НЕ ТАК в `newgrow.py`. Правило «новый элемент берёт предков
среди МАКСИМАЛЬНЫХ и сам становится максимальным» начинает с одного
максимального элемента и не может расширить поверхность: |max| = 1
навсегда, выходит чистая цепь. Все двенадцать строк дали ровно 1.000
независимо от параметров — подпись вырождения, а не измерения.

ПОЧЕМУ ЭТО ВАЖНО, А НЕ ПРОСТО ОПЕЧАТКА. В сприклинге ширина
поверхности растёт как `N^((d−1)/d)`, а длина как `N^(1/d)`. Значит
ОТНОШЕНИЕ ширины к длине и ЕСТЬ размерность. Правило роста, у которого
нет регулятора ширины, размерность задать не может в принципе — оно
даёт либо цепь (ширина 1), либо антицепь. Нужен ровно один знобик, и он
обязан быть порядковым.

РЕГУЛЯТОР. «Недавность» выражается порядком: у недавнего элемента МАЛОЕ
БУДУЩЕЕ. Опора выбирается равномерно среди элементов с `|future| ≤ F`;
при `F = 0` это максимальные (цепь), при `F → ∞` — все (широко).
Остальные предки — ближайшие к опоре по ОБЩЕМУ ПРОШЛОМУ, среди
несравнимых с ней. Прошлое нового элемента — замыкание вниз.

    F — ширина поверхности,   m — локальность присоединения.

Два знобика, и вопрос: есть ли точка, где оба прибора читают
многообразие сразу — `d ≈ 4`, обилия в единицах, `S⁴/N ≈ 0`?

ОГОВОРКА. Это НЕ КПР: выбор зависит не только от числа предков, значит
общая ковариация Райдаута–Соркина нарушена, и на вывод про КПР отсюда
ничего не следует. Проверяется другой вопрос — может ли локальное
правило роста давать многообразие вообще.

  python3 local2.py
"""
import math
import time

import numpy as np

from grow import mm_dim, c_k, gen_sprinkle, intervals
from bd import action
from density import iv_rows
from newgrow import both, HDR, line


def gen_loc2(n, m, F, rs, soft=0.0):
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
            # несравнимые с опорой
            inc = np.flatnonzero(~A[:j, a] & ~A[a, :j])
            inc = inc[inc != a]
            if len(inc) > 0:
                ov = (A[:, inc] & A[:, a][:, None]).sum(axis=0).astype(float)
                kk = min(k, len(inc))
                if soft <= 0:
                    S += list(inc[np.argsort(-ov)[:kk]])
                else:
                    w = (ov + 1.0) ** soft
                    w /= w.sum()
                    S += list(rs.choice(inc, size=kk, replace=False, p=w))
        S = np.array(S, dtype=np.int64)
        col = np.zeros(n, dtype=bool)
        col[S] = True
        col[:j] |= A[:j, S].any(axis=1)
        A[:, j] = col
        nfut[:j] += col[:j]
        A[:, j] = col
    return A


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N = 1000
    print('=' * 112)
    print('ЭТАЛОН и ПРОВЕРКА РЕГУЛЯТОРА (меняется ли что-нибудь от F?)')
    print(HDR)
    for dd in (2, 4):
        line(f'сприклинг {dd}D', both(gen_sprinkle, N, 8, 200 + dd, d=dd))
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 112)
    print('ШИРИНА: F при постоянной локальности m = 6')
    print(HDR)
    for F in (0, 2, 5, 12, 30, 80, 200, 10 ** 9):
        line(f'F={F:<10g} m=6',
             both(gen_loc2, N, 5, 6000 + int(math.log1p(F) * 97),
                  m=6, F=F))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 112)
    print('ЛОКАЛЬНОСТЬ: m при ширине F, настроенной на середину')
    print(HDR)
    for m in (2, 3, 6, 12, 24):
        line(f'F=30 m={m}',
             both(gen_loc2, N, 5, 6500 + m, m=m, F=30))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('МЯГКАЯ локальность: вероятность ∝ (перекрытие)^b')
    print(HDR)
    for b in (0.5, 2.0, 8.0):
        line(f'F=30 m=6 b={b}',
             both(gen_loc2, N, 5, 6700 + int(10 * b), m=6, F=30, soft=b))
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
