#!/usr/bin/env python3
"""Почему сборка провалилась: фазы из порядка КОГЕРЕНТНЫ.

assemble.py: квантовое правило |A|^2 дало долю пар 0.27 против 0.075 у
равновероятного, то есть ПЕРЕСВЯЗАЛО. Предсказание было обратное: у
множества из |D| предков со случайными фазами |A|^2 ~ |D|, что должно
штрафовать плотное прицепление.

ПОДОЗРЕНИЕ. Фазы построены ИЗ ПОРЯДКА (доли общего прошлого с опорами),
значит они гладкие функции события: у близких событий почти равны. Сумма
почти равных фаз даёт |A| ~ |D|, то есть |A|^2 ~ |D|^2 — плотное
прицепление ПООЩРЯЕТСЯ.

Архив это знал: «Порядок не хранит относительных фаз вовсе, а
интерференция держится именно на них» (ORDER-final.md:1866).

МЕРИМ показатель: |A(D)|^2 ~ |D|^alpha.
    alpha = 1  случайные фазы, интерференция работает
    alpha = 2  когерентные фазы, интерференции нет
Контроль: те же множества, но фазы заменены случайными.
"""
import sys
import numpy as np

sys.path.insert(0, '/home/claude/code')
from weight import diamond, cone_rel

T, K = 2.0, 4
rs = np.random.default_rng(5201)
X = diamond(T, rs)
N = len(X)
R = cone_rel(X)
A = R.astype(np.float32)
sz = A.sum(1)
refs = np.argsort(-sz)[:K]                       # опоры с большим прошлым
ph = np.zeros(N)
for m in range(K):
    inter = (R & R[int(refs[m])][None, :]).sum(1)
    ph += inter / np.maximum(sz, 1)
amp = np.exp(2j * np.pi * ph)
amp_rnd = np.exp(2j * np.pi * rs.random(N))

print(f'N={N}, опор {K}; фаза = sum_m |past&past_r|/|past|')
print(f'разброс суммарной фазы: {ph.std():.4f} (у случайных '
      f'{rs.random(N).std()*K:.4f} на k шкал)')
# когерентность: корреляция фазы у связанных пар
ii, jj = np.where(R)
samp = rs.choice(len(ii), min(200000, len(ii)), replace=False)
print(f'корреляция фазы у СВЯЗАННЫХ пар: '
      f'{np.corrcoef(ph[ii[samp]], ph[jj[samp]])[0,1]:+.4f}')
print()
print(f"{'|D|':>6}{'|A|^2 из порядка':>20}{'|A|^2 случайные':>18}")
sizes, v_ord, v_rnd = [], [], []
for target in (2, 4, 8, 16, 32, 64, 128):
    vo, vr = [], []
    for _ in range(300):
        i = int(rs.integers(0, N))
        p = np.flatnonzero(R[i])
        if len(p) < target:
            continue
        D = rs.choice(p, target, replace=False)
        vo.append(abs(amp[D].sum()) ** 2)
        vr.append(abs(amp_rnd[D].sum()) ** 2)
    if not vo:
        continue
    sizes.append(target); v_ord.append(np.mean(vo)); v_rnd.append(np.mean(vr))
    print(f'{target:>6}{np.mean(vo):>20.1f}{np.mean(vr):>18.1f}')
s = np.log(sizes)
print()
print(f'  показатель из порядка:  {np.polyfit(s, np.log(v_ord), 1)[0]:.3f}')
print(f'  показатель случайных:   {np.polyfit(s, np.log(v_rnd), 1)[0]:.3f}')
print('  1 = интерференция работает, 2 = фазы когерентны и её нет')
