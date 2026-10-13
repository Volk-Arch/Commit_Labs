#!/usr/bin/env python3
"""Есть ли множитель фазы, при котором конусный сектор выходит вперёд.

qmeasure.py при амплитуде exp(i*B) (без множителя) дал при N=7:
    разрежен. <0.20     0.311
    конусный 0.20-0.35  0.330
    Клейтман 0.35-0.55  1.481     <- главный
    плотный >=0.55      0.022     <- задушен в 17 раз, и это достижение

Интерференция лечит вырождение в цепь, но не проблему энтропии.
У Соркина в комплексной перколяции связь t комплексная, то есть у фазы
есть множитель. Сканируем его: amp = n(C) * exp(i*beta*B(C)).

ВОПРОС: есть ли beta, при котором конусный сектор обгоняет Клейтмана.
Если нет ни одного — путь закрыт; если есть — это условие на связь.
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/claude/code')
from preclude import downsets, bd_action
sys.path.insert(0, '/tmp/claude-0/-home-claude/8f20417e-eecf-5054-8071-09c440210416/scratchpad')
from qmeasure import enumerate_growth

N = 7
cur = enumerate_growth(N)
rows = []
for past, (c, p) in cur.items():
    B = bd_action(past, N)
    rel = sum(bin(x).count('1') for x in past)
    rows.append((B, c, rel / (N * (N - 1) / 2)))
Bv = np.array([r[0] for r in rows], float)
C = np.array([r[1] for r in rows], float)
F = np.array([r[2] for r in rows])
sec = {'разр': F < 0.20, 'конус': (F >= 0.20) & (F < 0.35),
       'Клейт': (F >= 0.35) & (F < 0.55), 'плотн': F >= 0.55}

print(f'N={N}, {len(rows)} порядков; amp = n(C)*exp(i*beta*B)')
print(f"{'beta':>8}{'разр':>10}{'конус':>10}{'Клейт':>10}{'плотн':>10}"
      f"{'конус/Клейт':>13}{'|полная|':>12}")
best = None
for beta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0,
             np.pi/2, np.pi, 2*np.pi/7, 2*np.pi/13):
    amp = C * np.exp(1j * beta * Bv)
    tot = abs(amp.sum()) ** 2
    if tot < 1e-30:
        print(f'{beta:>8.3f}   полная мера обнулилась')
        continue
    vals = {k: abs(amp[m].sum()) ** 2 / tot for k, m in sec.items()}
    r = vals['конус'] / max(vals['Клейт'], 1e-30)
    print(f'{beta:>8.3f}{vals["разр"]:>10.4f}{vals["конус"]:>10.4f}'
          f'{vals["Клейт"]:>10.4f}{vals["плотн"]:>10.4f}{r:>13.4f}'
          f'{np.sqrt(tot):>12.3e}')
    if best is None or r > best[1]:
        best = (beta, r)

print(f'\n  лучшее отношение конус/Клейтман: {best[1]:.4f} при beta={best[0]:.3f}')
print('  равновероятная мера даёт 0.278/0.526 = 0.528')
print('  мера от роста      даёт 0.127/0.497 = 0.256')
print('  больше 1 означало бы, что конусный сектор ГЛАВНЫЙ')
