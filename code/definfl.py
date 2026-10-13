#!/usr/bin/env python3
"""Влияет ли дефект: сравнение двух выборок, а не корреляция.

ПОЧЕМУ ПЕРВАЯ ПОПЫТКА (topoinfl.py) НЕ СОСТОЯЛАСЬ.
  1. Ненулевых индексов 9..14 из ~200 событий: корреляция переменной,
     которая на 95% нули, даёт нуль по построению (вышло 0.058).
  2. Моя базовая линия не воспроизвела архивную: корреляция
     сохранение-влияние +0.215 вместо архивных -0.657, значит
     определения не совпадают и сравнивать с «линией гладких» нельзя.
  3. Архивный тест на сохранение (передача вдоль связи) топологическому
     заряду НЕ ПОДХОДИТ: заряд сохраняется суммой по области, и этот
     тест индекс уже прошёл (масштабная инвариантность 0.888, дисперсия
     0.452 от пуассоновской).

Значит вопрос один: ВЛИЯЕТ ЛИ. И мерить надо сравнением двух выборок.

ЧТО МЕРИМ. В точках дефектов (|индекс| >= 0.5) против остальных:
    доля прямых связей у события   — архивный признак кривизны
    число связей
    высота
Плюс КОНТРОЛЬ: то же сравнение для случайно выбранных «псевдодефектов»
того же числа, чтобы отделить эффект от размера выборки.
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from topo import diamond, cone_rel
from topoinfl import links_of, loop_index

N, T, RHO, NST, MINS, PMIN, SEEDS, SAMP = 2500, 2.0, 0.10, 24, 25, 0.25, 12, 300

dv, di, dh, dl = [], [], [], []
nv, ni, nh, nl = [], [], [], []
tot = 0
for t in range(SEEDS):
    rs = np.random.default_rng(5501000 + 977 * t)
    X = diamond(N, T, rs)
    R = cone_rel(X)
    L = links_of(R)
    rel_n = R.sum(1) + R.sum(0)
    lnk_n = L.sum(1) + L.sum(0)
    infl = lnk_n / np.maximum(rel_n, 1)
    h = np.zeros(len(X), np.int64)
    for i in np.argsort(R.sum(1), kind='stable'):
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    i0 = int(np.argmax(R.sum(0) * (X[:, 0] < 0.25 * T)))
    cand = np.flatnonzero(R[:, i0])
    if len(cand) < 300:
        del R, L, X; continue
    pool = rs.choice(cand, min(SAMP, len(cand)), replace=False)
    vals, evs = [], []
    for j0 in pool:
        v = loop_index(X, R, i0, cand, int(j0), RHO, NST, MINS, PMIN)
        if v is not None:
            vals.append(v); evs.append(int(j0))
    vals = np.array(vals); evs = np.array(evs, int)
    if len(vals) < 50:
        del R, L, X; continue
    q = np.abs(np.round(vals * 2) / 2)
    isd = q >= 0.5
    tot += len(vals)
    for arr_i, arr_l, arr_h, m in ((di, dl, dh, isd), (ni, nl, nh, ~isd)):
        if m.sum() == 0:
            continue
        arr_i.append(infl[evs[m]]); arr_l.append(lnk_n[evs[m]].astype(float))
        arr_h.append(h[evs[m]].astype(float))
    del R, L, X

def cat(a):
    return np.concatenate(a) if a else np.array([])
DI, NI = cat(di), cat(ni)
DL, NL = cat(dl), cat(nl)
DH, NH = cat(dh), cat(nh)
print(f'засевов {SEEDS}, индексов всего {tot}; '
      f'дефектов {len(DI)}, остальных {len(NI)}')
print(f'доля дефектов {len(DI)/max(tot,1):.4f}')
print()
print(f"{'величина':>22}{'в дефектах':>13}{'в остальных':>13}"
      f"{'разница':>11}{'сигм':>8}")
for nm, A, B in (('доля прямых связей', DI, NI),
                 ('число связей', DL, NL),
                 ('высота', DH, NH)):
    if len(A) < 5 or len(B) < 5:
        print(f'{nm:>22}{"мало":>13}'); continue
    d = A.mean() - B.mean()
    se = math.hypot(A.std() / math.sqrt(len(A)), B.std() / math.sqrt(len(B)))
    print(f'{nm:>22}{A.mean():>13.4f}{B.mean():>13.4f}'
          f'{d:>+11.4f}{abs(d)/max(se,1e-12):>8.1f}')

# контроль: псевдодефекты того же числа из тех же выборок
rs = np.random.default_rng(777)
zs = []
for _ in range(400):
    k = len(DI)
    allv = np.concatenate([DI, NI])
    pick = rs.choice(len(allv), k, replace=False)
    rest = np.setdiff1d(np.arange(len(allv)), pick)
    d = allv[pick].mean() - allv[rest].mean()
    se = math.hypot(allv[pick].std() / math.sqrt(k),
                    allv[rest].std() / math.sqrt(len(rest)))
    zs.append(abs(d) / max(se, 1e-12))
print(f'\n  контроль (псевдодефекты того же числа): '
      f'медиана {np.median(zs):.1f} сигм, 95-й процентиль '
      f'{np.percentile(zs, 95):.1f}')
print('  если настоящие дефекты дают заметно больше контроля — ВЛИЯЮТ')
