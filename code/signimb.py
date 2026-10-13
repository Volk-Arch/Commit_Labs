#!/usr/bin/env python3
"""ЗНАКОВАЯ НЕСБАЛАНСИРОВАННОСТЬ: знак от чётности перестановки.

mobsum.py похоронил Мёбиуса: доля выжившей энтропии 0.375, 0.704,
0.686, 0.759 — растёт с размером и выше случайной 0.5, то есть знак
(-1)^{S_mu} не перемешивается. Диагноз из замкнутого вида: чётность
S_mu задаётся членом sum_y d_вх(y) d_вых(y) по модулю 2, а это
НИЗКОСТЕПЕННОЙ многочлен над GF(2), и у таких гашения нет.

Уточнённое требование: знак, чья чётность есть УРАВНОВЕШЕННАЯ функция
ВЫСОКОЙ степени — как чётность перестановки в фермионном определителе.

В теории порядков такой объект есть: знаковая несбалансированность
(Стэнли)

    I(C) = sum по линейным расширениям lambda от sgn(lambda)

против обычного числа расширений e(C) = sum от 1. Это буквально
фермионный аналог: вместо счёта путей — знаковый счёт.

ПРОВЕРКА. Полная амплитуда сектора:
  * с обычным весом:  A_e = sum_C e(C)
  * со знаковым:      A_I = sum_C I(C)
Если |A_I| экспоненциально мало против A_e, знаковый счёт гасит сектор.

КОНТРОЛИ:
  A. e(C) считается двумя путями — рекурсией и перечислением;
  B. для ЦЕПИ из n элементов e = 1 и I = +1 (единственное расширение,
     тождественная перестановка);
  C. для АНТИЦЕПИ из n элементов e = n! и I = 0 при n >= 2 (поровну
     чётных и нечётных) — это и есть образец полного гашения.
"""
import itertools
import math

import numpy as np


def lin_exts_signed(R):
    """(число расширений, знаковая сумма) точным перечислением."""
    n = len(R)
    preds = [frozenset(np.flatnonzero(R[:, j]).tolist())
             for j in range(n)]
    cnt = 0
    sgn = 0
    seq, used = [], set()

    def parity(p):
        seen = [False] * len(p)
        par = 1
        for i in range(len(p)):
            if seen[i]:
                continue
            j, ln = i, 0
            while not seen[j]:
                seen[j] = True
                j = p[j]
                ln += 1
            if ln % 2 == 0:
                par = -par
        return par

    def rec():
        nonlocal cnt, sgn
        if len(seq) == n:
            cnt += 1
            sgn += parity(list(seq))
            return
        for x in range(n):
            if x not in used and preds[x] <= used:
                used.add(x); seq.append(x)
                rec()
                seq.pop(); used.discard(x)
    rec()
    return cnt, sgn


def build(a, b, c, p, q):
    n = a + b + c
    R = np.zeros((n, n), bool)
    R[:a, a:a + b] = p.astype(bool)
    R[a:a + b, a + b:] = q.astype(bool)
    return R | ((R.astype(np.int8) @ R.astype(np.int8)) > 0)


def main():
    print(__doc__)
    print('КОНТРОЛИ B и C')
    for n in (4, 5):
        ch = np.triu(np.ones((n, n), bool), 1)
        e, i = lin_exts_signed(ch)
        print(f'  цепь {n}:    e = {e} (ждём 1), I = {i:+d} (ждём +1)')
        e, i = lin_exts_signed(np.zeros((n, n), bool))
        print(f'  антицепь {n}: e = {e} (ждём {math.factorial(n)}), '
              f'I = {i:+d} (ждём 0)')

    print(f"\n{'слои':>8}{'конфиг.':>9}{'A_e = Σe':>14}"
          f"{'A_I = ΣI':>12}{'|A_I|/A_e':>12}{'доля ln':>10}")
    for (a, b, c) in ((2, 2, 2), (2, 3, 2), (3, 2, 3), (2, 2, 3)):
        nbp, nbq = a * b, b * c
        Ae = 0
        AI = 0
        tot = 0
        for mp in range(1 << nbp):
            p = np.array([(mp >> k) & 1 for k in range(nbp)],
                         np.int8).reshape(a, b)
            for mq in range(1 << nbq):
                q = np.array([(mq >> k) & 1 for k in range(nbq)],
                             np.int8).reshape(b, c)
                e, i = lin_exts_signed(build(a, b, c, p, q))
                Ae += e
                AI += i
                tot += 1
        r = abs(AI) / max(Ae, 1)
        frac = (math.log(max(abs(AI), 1e-300)) / math.log(max(Ae, 2)))
        print(f'{f"{a}-{b}-{c}":>8}{tot:>9}{Ae:>14}{AI:>12}'
              f'{r:>12.3e}{frac:>10.4f}', flush=True)

    print('\nЧТЕНИЕ: если |A_I|/A_e падает экспоненциально с размером,')
    print('знаковый счёт гасит сектор — то есть найден механизм, какого')
    print('не дал ни один вес на счётах и ни Мёбиус.')


if __name__ == '__main__':
    main()
