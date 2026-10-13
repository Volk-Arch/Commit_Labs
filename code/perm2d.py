#!/usr/bin/env python3
"""ДОСТАТОЧНО ЛИ ОДНОГО ОГРАНИЧЕНИЯ: засев против равномерной меры.

В 1+1 кинематика решена точно (`tri-doc.md`): световые координаты дают
x < y <=> u_x < u_y и v_x < v_y, то есть причинный порядок РОВНО двумерен,
а засев в диамант есть случайные точки в квадрате с покоординатным
порядком. Плюс пара (sigma, tau) даёт порядок, изоморфный тому, что даёт
(id, sigma^{-1} tau), значит засев в 1+1 есть порядок ОДНОЙ случайной
перестановки:
    i < j  <=>  i < j (как числа)  и  tau(i) < tau(j)
Пространство перечислимо целиком, и проблема выборки, бившая весь день
(сэмплер на N~24, оператор на N~10^6), здесь исчезает.

ДВЕ МЕРЫ различаются ровно КРАТНОСТЬЮ - числом реализаторов:
  мера A: равномерно по перестановкам  = ЗАСЕВ (геометрическая мера);
  мера B: равномерно по РАЗЛИЧНЫМ порядкам, кратность убрана.
Если они совпадают - кинематического ограничения ДОСТАТОЧНО, вес не
нужен. Если различаются - ограничения мало, и недостающий вес есть ровно
ЧИСЛО РЕАЛИЗАТОРОВ, то есть число способов вложить.

КОНТРОЛЬ, точный: под мерой A доля связанных обязана быть ровно 1/2
(P(tau(i)<tau(j)) = 1/2), и это же значение Майрхейма-Майера для d=2.
Если не 1/2 - сломан код.

Попутно считается ЭНТРОПИЯ внутри ограничения: число различных двумерных
порядков, и проверка, что это exp(O(n ln n)), с коэффициентом.
"""
import math
from itertools import permutations

import numpy as np

KEYS = ('r', 'H', 'n0', 'n1', 'n2', 'n3')


def stats(t, n):
    T = np.array(t)
    R = np.triu(np.ones((n, n), bool), 1) & (T[:, None] < T[None, :])
    npairs = n * (n - 1) // 2
    Iv = (R.astype(np.int16) @ R.astype(np.int16))
    sz = Iv[R]
    # наибольшая возрастающая подпоследовательность = высота
    best = np.zeros(n, int)
    for i in range(n):
        best[i] = 1 + max([best[j] for j in range(i) if T[j] < T[i]],
                          default=0)
    return (R.tobytes(),
            dict(r=R.sum() / npairs, H=int(best.max()),
                 n0=int((sz == 0).sum()), n1=int((sz == 1).sum()),
                 n2=int((sz == 2).sum()), n3=int((sz == 3).sum())))


def main():
    print(__doc__)
    print(f"{'n':>3}{'перестановок':>14}{'различных':>11}{'доля':>8}"
          f"{'ln(разл)/(n ln n)':>19}{'макс кратность':>16}")
    rows = {}
    for n in range(3, 10):
        tally = {}
        acc = {}
        for t in permutations(range(n)):
            k, s = stats(t, n)
            tally[k] = tally.get(k, 0) + 1
            if k not in acc:
                acc[k] = s
        tot = math.factorial(n)
        d = len(tally)
        rows[n] = (tally, acc, tot, d)
        print(f'{n:>3}{tot:>14}{d:>11}{d / tot:>8.4f}'
              f'{math.log(d) / (n * math.log(n)):>19.3f}'
              f'{max(tally.values()):>16}', flush=True)

    print('\nсравнение мер (A = засев, B = равномерно по порядкам):')
    print(f"{'n':>3}" + ''.join(f'{f"A:{k}":>11}{f"B:{k}":>11}'
                                for k in ('r', 'H')))
    for n in (6, 7, 8, 9):
        tally, acc, tot, d = rows[n]
        a = {k: sum(tally[x] * acc[x][k] for x in tally) / tot
             for k in KEYS}
        b = {k: sum(acc[x][k] for x in tally) / d for k in KEYS}
        print(f'{n:>3}' + ''.join(f'{a[k]:>11.4f}{b[k]:>11.4f}'
                                  for k in ('r', 'H')))
        if n == 9:
            print('\n  подробно при n=9:')
            print(f"    {'величина':>10}{'A (засев)':>12}{'B (равном.)':>13}"
                  f"{'расхождение':>13}")
            for k in KEYS:
                dd = (b[k] - a[k]) / max(abs(a[k]), 1e-9) * 100
                print(f'    {k:>10}{a[k]:>12.4f}{b[k]:>13.4f}'
                      f'{dd:>12.1f}%')
    print('\nКОНТРОЛЬ: A:r обязано быть ровно 0.5000 при всех n')


if __name__ == '__main__':
    main()
