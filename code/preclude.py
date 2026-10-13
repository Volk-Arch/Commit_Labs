#!/usr/bin/env python3
"""ЗАПРЕТ вместо выбора: есть ли множества историй нулевой меры.

ПОВОД (его). Орбиты электрона — не орбиты, а облака вероятностей.
Значит мы искали не то: искали объект с границей, а нужна структура без
границ. Может, дело в вероятностях.

ЧТО ЭТО ДАЁТ ТОЧНО. Теорема о некомпактности запрещает инвариантное
соседство (множество с границей), но НЕ запрещает спадающий вес:
гауссиан имеет бесконечный носитель и конечную норму. И теорема о
перешкалировании запрещает размерные величины, но вероятность —
безразмерное отношение, то есть единственный вид величины, который эта
конструкция имеет право содержать.

Отсюда переформулировка задачи выбора:
    не «пусть мир ВЫБЕРЕТ расстановку», а «пусть мир ЗАПРЕТИТ часть».

Квантовая мера Соркина ровно такая: mu(A) = |сумма амплитуд|^2, она не
аддитивна, и множества с mu = 0 ПРЕДОТВРАЩЕНЫ — мир их не может.
Это отбор без выбора, без порога и без параметра.

ЧТО МЕРИМ. Перечисляем все истории роста до N событий (история =
причинное множество с порядком рождения), считаем действие
Бенинказы-Доукера

    B = N - n0 + 9*n1 - 16*n2 + 8*n3      (ЦЕЛОЕ число)

и спрашиваем: существуют ли нетривиальные множества историй с нулевой
мерой?

КЛЮЧЕВОЕ НАБЛЮДЕНИЕ, КОТОРОЕ ЭТО РЕШАЕТ. B целое, значит при фазе
exp(i*B) углы суть целые числа радиан. Числа exp(i*s) при разных целых s
линейно независимы над рациональными (1 радиан несоизмерим с 2*pi),
поэтому целочисленная сумма обращается в нуль ТОЛЬКО тривиально.
Запрета не возникает вовсе.

А при фазе exp(2*pi*i*B/n) это корни из единицы, и погашения есть:
сумма по полной смежной подгруппе простого порядка p | n равна нулю.
Тогда запрет существует, и его устройство — арифметическое по модулю n.

Значит проверяется не «бывает ли запрет», а ЧТО ОН ОТСЕКАЕТ: имеет ли
предотвращённое геометрический характер. Нужно промежуточное — отсечь
часть, не всё и не ничего.

  python3 preclude.py
  python3 preclude.py --n 7 --mods 2 3 4 6 8 12
"""
import argparse
import math
import time
from itertools import combinations

import numpy as np


def downsets(past, n):
    """Все вниз-замкнутые подмножества (битовые маски) текущего порядка."""
    out = []
    for s in range(1 << n):
        ok = True
        m = s
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            if past[i] & ~s:
                ok = False
                break
        if ok:
            out.append(s)
    return out


def grow(N):
    """Все истории роста до N событий.

    История = последовательность рождений; возвращаем конечные порядки
    в виде кортежа масок прошлого. Новое событие максимально, его
    прошлое — любое вниз-замкнутое подмножество.
    """
    cur = [((0,), 1)]                      # (маски прошлого, число историй)
    for n in range(1, N):
        nxt = {}
        for past, cnt in cur:
            for s in downsets(past, n):
                np_ = past + (s,)
                nxt[np_] = nxt.get(np_, 0) + cnt
        cur = list(nxt.items())
    return cur


def bd_action(past, n):
    """B = N - n0 + 9*n1 - 16*n2 + 8*n3, целое."""
    nk = [0, 0, 0, 0]
    for j in range(n):
        pj = past[j]
        m = pj
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            # элементы строго между i и j: прошлое j, содержащее i
            btw = pj & ~past[i] & ~(1 << i)
            c = 0
            b = btw
            while b:
                z = (b & -b).bit_length() - 1
                b &= b - 1
                if past[z] >> i & 1:
                    c += 1
                if c > 3:
                    break
            if c < 4:
                nk[c] += 1
    return n - nk[0] + 9 * nk[1] - 16 * nk[2] + 8 * nk[3]


def frac_pairs(past, n):
    rel = sum(bin(p).count('1') for p in past)
    return rel / (n * (n - 1) / 2)


def main():
    ap = argparse.ArgumentParser(description='Запрет вместо выбора')
    ap.add_argument('--n', type=int, default=7)
    ap.add_argument('--mods', type=int, nargs='+',
                    default=[2, 3, 4, 5, 6, 8, 10, 12, 16, 24])
    a = ap.parse_args()

    t0 = time.time()
    for N in range(4, a.n + 1):
        hist = grow(N)
        tot = sum(c for _, c in hist)
        # группируем по действию
        byB = {}
        for past, cnt in hist:
            B = bd_action(past, N)
            f = frac_pairs(past, N)
            d = byB.setdefault(B, [0, 0.0, 0])
            d[0] += cnt
            d[1] += f * cnt
            d[2] += 1
        Bs = sorted(byB)
        print(f'\nN = {N}:  {len(hist)} помеченных порядков, '
              f'{tot} историй, {len(Bs)} значений действия')
        print(f'   действие B от {min(Bs)} до {max(Bs)}')

        # 1. фаза exp(i*B): возможен ли запрет вообще
        amp = np.array([byB[B][0] * np.exp(1j * B) for B in Bs])
        full = abs(amp.sum()) ** 2
        # минимум |сумма| по НЕТРИВИАЛЬНЫМ подмножествам значений
        best = None
        idx = list(range(len(Bs)))
        for r in range(1, min(4, len(Bs)) + 1):
            for cmb in combinations(idx, r):
                if r == len(Bs):
                    continue
                v = abs(sum(amp[list(cmb)]))
                if best is None or v < best[0]:
                    best = (v, cmb)
        print(f'   фаза exp(i*B):  мера всего {full:.4g};  '
              f'минимум |сумма| по подмножествам {best[0]:.6g}')
        print('      нуля нет и быть не может: exp(i*s) при целых s '
              'линейно независимы над Q')

        # 2. фаза exp(2*pi*i*B/n): запрет есть, если есть полная
        #    смежная подгруппа простого порядка p | n среди
        #    ПРИСУТСТВУЮЩИХ значений B mod n
        print(f'   фаза exp(2*pi*i*B/n):')
        print(f"{'n':>8}{'запрет':>9}{'размер':>8}{'историй в нём':>15}"
              f"{'доля пар в нём':>16}{'доля пар всего':>16}")
        allf = sum(byB[B][1] for B in Bs) / tot
        for n in a.mods:
            res = {}
            for B in Bs:
                res.setdefault(B % n, []).append(B)
            found = None
            for p in (2, 3, 5, 7, 11, 13):
                if n % p or p > n:
                    continue
                step = n // p
                for r in range(step):
                    cls = [(r + j * step) % n for j in range(p)]
                    if all(c in res for c in cls):
                        found = (p, cls)
                        break
                if found:
                    break
            if not found:
                print(f'{n:>8}{"нет":>9}')
                continue
            p, cls = found
            # предотвращённое множество: по одной истории каждого класса
            Bsel = [res[c][0] for c in cls]
            cnt = sum(byB[B][0] for B in Bsel)
            fr = sum(byB[B][1] for B in Bsel) / max(cnt, 1)
            chk = abs(sum(np.exp(2j * np.pi * B / n) for B in Bsel))
            print(f'{n:>8}{"ДА":>9}{p:>8}{cnt:>15}{fr:>16.4f}'
                  f'{allf:>16.4f}   |сумма|={chk:.2e}  B={Bsel}')

    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
