#!/usr/bin/env python3
"""КВАНТОВАЯ МЕРА на историях роста: сдвигает ли интерференция вес.

ФОРМУЛЫ СВЕРЕНЫ (Rideout-Sorkin, через обзор Imperial MSc 2013):
  вероятность выбрать множество предков мощности k ~ t_k,
  нормировка lambda(N) = sum_i C(N,i) t_i,
  транзитивная перколяция: t_n = t^n, то есть независимое включение
  каждого предка с p = t/(1+t).
Перколяцию архив уже закрыл (разрыв 12 сигм по действию BD), поэтому
здесь строится КВАНТОВАЯ версия.

КВАНТОВАЯ МЕРА (Соркин). Вместо вероятности — амплитуда на историю, и
    mu(A) = |сумма амплитуд по A|^2
Она НЕ аддитивна: интерференция
    I2(A,B) = mu(A u B) - mu(A) - mu(B)
а третьего порядка нет:
    I3(A,B,C) = mu(AuBuC) - mu(AuB) - mu(AuC) - mu(BuC)
                + mu(A) + mu(B) + mu(C) = 0
Последнее — тождество для любой меры вида |сумма|^2, проверяется как
контроль арифметики.

АМПЛИТУДА БЕЗ ПАРАМЕТРА. Приращение действия BD по шагам роста
аддитивно (проверено до 4.6e-15), значит произведение шаговых амплитуд
exp(i*dS) равно exp(i*S) готового порядка. Для порядка C с n(C)
историями полная амплитуда = n(C) * exp(i*S(C)).

ЧТО МЕРИМ. Три сектора по числу слоёв:
    блины  3 слоя  — сектор Клейтмана-Ротшильда, не многообразный
    среднее 4 слоя — промежуточное, где должен сидеть конус
    цепи   >=5     — вырождение в другую сторону
Доля веса каждого сектора под тремя мерами: равновероятной, от роста,
квантовой. И интерференция между секторами.
"""
import math
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/code')
from preclude import downsets, bd_action

def enumerate_growth(N):
    """Все помеченные порядки до N: число историй и вес от роста."""
    cur = {(0,): (1, 1.0)}
    for n in range(1, N):
        nxt = defaultdict(lambda: [0, 0.0])
        for past, (c, p) in cur.items():
            ds = downsets(past, n)
            w = 1.0 / len(ds)
            for s in ds:
                k = past + (s,)
                nxt[k][0] += c
                nxt[k][1] += p * w
        cur = {k: (v[0], v[1]) for k, v in nxt.items()}
    return cur

def nlayers(past, n):
    h = np.zeros(n, int)
    for j in range(n):
        m, best = past[j], 0
        while m:
            i = (m & -m).bit_length() - 1
            m &= m - 1
            best = max(best, h[i] + 1)
        h[j] = best
    return int(h.max()) + 1

for N in (6, 7):
    cur = enumerate_growth(N)
    rows = []
    for past, (c, p) in cur.items():
        L = nlayers(past, N)
        B = bd_action(past, N)
        rel = sum(bin(x).count('1') for x in past)
        rows.append((L, B, c, p, rel / (N * (N - 1) / 2)))
    L = np.array([r[0] for r in rows]); Bv = np.array([r[1] for r in rows])
    C = np.array([r[2] for r in rows], float)
    P = np.array([r[3] for r in rows]); F = np.array([r[4] for r in rows])
    amp = C * np.exp(1j * Bv)

    # ПОПРАВКА. Делить сектора числом слоёв при N=7 нельзя: «три слоя»
    # это и сектор Клейтмана-Ротшильда, и конус (сприклинг семи событий
    # в четырёхмерии имеет высоту 3). Разделяет ДОЛЯ ПАР: у конуса около
    # 0.15, у блина 0.4..0.5. Это и основной различитель проекта.
    sec = {'разрежен. <0.20': F < 0.20,
           'конусный 0.20-0.35': (F >= 0.20) & (F < 0.35),
           'Клейтман 0.35-0.55': (F >= 0.35) & (F < 0.55),
           'плотный >=0.55': F >= 0.55}
    def mu(m):
        return abs(amp[m].sum()) ** 2
    tot = mu(np.ones(len(rows), bool))
    print(f'N={N}: {len(rows)} помеченных порядков, '
          f'действие B от {Bv.min()} до {Bv.max()}')
    print(f"{'сектор по доле пар':>20}{'равновер.':>11}{'от роста':>11}"
          f"{'квантовая':>12}{'<доля пар>':>12}{'слоёв':>9}")
    for nm, m in sec.items():
        if m.sum() == 0:
            continue
        print(f'{nm:>20}{m.mean():>11.4f}{P[m].sum()/P.sum():>11.4f}'
              f'{mu(m)/tot:>12.4f}{F[m].mean():>12.4f}{L[m].mean():>9.2f}')
    # интерференция между секторами
    ks = [k for k in sec if sec[k].sum() > 0]
    print('   интерференция I2 между секторами (в долях от полной меры):')
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = sec[ks[i]], sec[ks[j]]
            i2 = (mu(a | b) - mu(a) - mu(b)) / tot
            print(f'      {ks[i]:>16} / {ks[j]:<16} {i2:>+9.4f}')
    # контроль арифметики: I3 = 0
    if len(ks) >= 3:
        a, b, c3 = sec[ks[0]], sec[ks[1]], sec[ks[2]]
        i3 = (mu(a | b | c3) - mu(a | b) - mu(a | c3) - mu(b | c3)
              + mu(a) + mu(b) + mu(c3)) / tot
        print(f'   контроль I3 (обязан быть 0): {i3:+.2e}')
    print()

print('  если квантовая мера кладёт на «среднее» заметно больше, чем')
print('  равновероятная и от роста — интерференция работает в нужную сторону')
