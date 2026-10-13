#!/usr/bin/env python3
"""Сбой в одной точке: как он расходится по причинным путям.

Источник — неподвижная «трубка» событий радиуса 0.25 вдоль оси времени
(это и есть ячейка, которой запрещено делиться, повторённая во времени).
Вопрос: сколько СВЯЗЕЙ (ближайших «раньше») идёт от источника к событию на
расстоянии R. По Джонстону (запаздывающий пропагатор причинного множества
в 4d ∝ матрице связей) это и есть поле источника, и оно обязано идти как 1/R.
Сравниваем квазикристалл и засев той же плотности. Контроль: события источника
выбраны случайно по объёму (тот же счёт) — 1/R быть не должно.
"""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import points_in_box, r_for_density
RAD, TL, RC = 0.25, 2.0, 1.3
def region(P):
    return P[(np.abs(P[:, 0]) <= TL) & (np.linalg.norm(P[:, 1:], axis=1) <= RC)]
def links_from(P, src):
    o = np.argsort(P[:, 0]); P = P[o]; src = src[o]
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    R = (dt > dx); np.fill_diagonal(R, False)
    A = R.astype(np.float32)
    between = (A @ A) > 0.5                        # [x,y]: есть z, y ≺ z ≺ x
    Lk = R & ~between
    return P, Lk[:, src].sum(1), src
def profile(P, cnt, src):
    keep = (~src) & (P[:, 0] > 0.6) & (P[:, 0] < 1.9)
    Rr = np.linalg.norm(P[keep, 1:], axis=1); c = cnt[keep]
    bins = np.array([0.35, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2])
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (Rr >= lo) & (Rr < hi)
        if m.sum() > 5: out.append(((lo + hi) / 2, c[m].mean(), c[m].std() / math.sqrt(m.sum())))
    return np.array(out)
rho = float(sys.argv[1]) if len(sys.argv) > 1 else 160.0
r = r_for_density(rho); t0 = time.time()
res = {}
for lab in ('квазикристалл', 'засев', 'контроль: источник рассыпан'):
    acc = []
    for s in range(4):
        rs = np.random.default_rng(50 + s)
        if lab == 'квазикристалл':
            c = rs.uniform(0, 80, 4); P = points_in_box([(c[k] - (TL if k == 0 else RC), c[k] + (TL if k == 0 else RC)) for k in range(4)], r) - c
        else:
            n = rs.poisson(rho * 2 * TL * (2 * RC) ** 3)
            P = np.column_stack([rs.uniform(-TL, TL, n), rs.uniform(-RC, RC, (n, 3))])
        P = region(P)
        src = np.linalg.norm(P[:, 1:], axis=1) < RAD
        if lab.startswith('контроль'):
            k = src.sum(); src = np.zeros(len(P), bool); src[rs.choice(len(P), k, replace=False)] = True
        P2, cnt, src2 = links_from(P, src)
        acc.append(profile(P2, cnt, src2))
    pr = np.mean(acc, 0); er = np.sqrt((np.array(acc)[:, :, 2] ** 2).mean(0)) / math.sqrt(len(acc))
    k = np.polyfit(np.log(pr[:, 0]), np.log(pr[:, 1]), 1)[0]
    res[lab] = k
    print(f'{lab}: событий {len(P)}, в источнике {src.sum()}   [{time.time()-t0:.0f} с]')
    print('   R:      ' + '  '.join(f'{x:6.3f}' for x in pr[:, 0]))
    print('   связей: ' + '  '.join(f'{x:6.3f}' for x in pr[:, 1]))
    print('   R·связей' + '  '.join(f'{x:6.3f}' for x in pr[:, 0] * pr[:, 1]))
    print(f'   показатель спадания: {k:+.3f}  (Кулон/Ньютон −1)'); sys.stdout.flush()
