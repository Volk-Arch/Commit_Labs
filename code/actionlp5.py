#!/usr/bin/env python3
"""ТРИ ПРОВЕРКИ НЕВЫПОЛНИМОСТИ И ПОИСК СПАСАЮЩЕГО СЧЁТА.

1. устоит ли при ПЕРЕСЕВЕ секторов (счёты меряны по 1-3 выборкам);
2. дело в N или в ПЛОТНОСТИ сетки соперников;
3. какой ДОПОЛНИТЕЛЬНЫЙ счёт возвращает разделимость — это главный
   вопрос, потому что невыполнимость доказана лишь ОТНОСИТЕЛЬНО набора
   счётов, и добавление счёта её снимает.

Кандидаты сверх БД: N4..N8 (БД обрезает распределение размеров
интервала на трёх), C5 (пятицепи), число максимальных и минимальных
элементов.
"""
import math

import numpy as np
from scipy.optimize import linprog

from sectors import sprinkle_d, layered, s_layered

BASE = ('R', 'N0', 'N1', 'N2', 'N3', 'C3', 'C4', 'H')
EXTRA = ('N4', 'N5', 'N6', 'N7', 'N8', 'C5', 'MAXMIN')
ALL = BASE + EXTRA


def counts2(Rc):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    FF = F @ F
    Iv = np.rint(FF).astype(np.int64)
    o = {'R': float(Rc.sum())}
    for m in range(0, 9):
        o[f'N{m}'] = float((Rc & (Iv == m)).sum())
    o['C3'] = float(Iv.sum())
    F3 = FF @ F
    o['C4'] = float(np.rint(F3).sum())
    o['C5'] = float(np.rint(F3 @ F).sum())
    hgt = np.zeros(n, dtype=np.int64)
    order = np.argsort(Rc.sum(1))[::-1]
    for _ in range(2):
        for x in order:
            pre = np.nonzero(Rc[:, x])[0]
            hgt[x] = (hgt[pre].max() + 1) if len(pre) else 0
    o['H'] = float(hgt.max() + 1)
    o['MAXMIN'] = float((Rc.sum(1) == 0).sum() + (Rc.sum(0) == 0).sum())
    return o


def build2(N, rs, ks, qs, nsamp=2):
    secs = []
    for d in (2, 3, 4, 5, 6):
        acc = [counts2(sprinkle_d(N, d, rs)) for _ in range(nsamp)]
        f = {k: float(np.mean([a[k] for a in acc])) for k in ALL}
        secs.append({'name': f'засев d={d}', 'f': f,
                     'S': 4.0 * N * math.log(N)})
    for k in ks:
        if N // k < 2:
            continue
        for q in qs:
            Rc, sizes = layered(N, k, q, rs)
            secs.append({'name': f'слоистый k={k} q={q:.3g}',
                         'f': counts2(Rc), 'S': s_layered(N, sizes, q)})
    M = N * (N - 1) / 2
    fch = {'R': M, 'C3': float(math.comb(N, 3)),
           'C4': float(math.comb(N, 4)), 'C5': float(math.comb(N, 5)),
           'H': float(N), 'MAXMIN': 2.0}
    for m in range(0, 9):
        fch[f'N{m}'] = float(max(N - 1 - m, 0))
    secs.append({'name': 'цепь', 'f': fch, 'S': math.lgamma(N + 1)})
    f0 = {k: 0.0 for k in ALL}
    f0['H'] = 1.0
    f0['MAXMIN'] = 2.0 * N
    secs.append({'name': 'антицепь', 'f': f0, 'S': 0.0})
    return secs


def margin(secs, tgtname, feats):
    scale = {k: max(abs(s['f'][k]) for s in secs) or 1.0 for k in feats}
    tgt = [s for s in secs if s['name'] == tgtname][0]
    ft = np.array([tgt['f'][k] / scale[k] for k in feats])
    G, C, nm = [], [], []
    for s in secs:
        if s is tgt:
            continue
        G.append(ft - np.array([s['f'][k] / scale[k] for k in feats]))
        C.append(s['S'] - tgt['S'])
        nm.append(s['name'])
    G, C = np.array(G), np.array(C)
    nf = G.shape[1]
    A = np.hstack([-G, np.ones((len(C), 1))])
    c = np.zeros(nf + 1); c[-1] = -1.0
    r = linprog(c, A_ub=A, b_ub=-C,
                bounds=[(None, None)] * (nf + 1), method='highs')
    if r.status == 3:
        return math.inf, None, nm, G, C
    if r.status != 0:
        return float('nan'), None, nm, G, C
    return float(r.x[-1]), r.x[:-1], nm, G, C


def cert(G, C):
    nj, nf = G.shape
    Aeq = np.vstack([G.T, np.ones((1, nj))])
    beq = np.concatenate([np.zeros(nf), [1.0]])
    r = linprog(-C, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nj,
                method='highs')
    if r.status != 0:
        return None, None, None
    return r.x, float(C @ r.x), float(np.abs(G.T @ r.x).max())


KS = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50, 75]
QS = [0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]
KSD = KS + [7, 9, 12, 17, 25, 40, 60, 100, 150]
QSD = sorted(QS + [0.03, 0.07, 0.12, 0.2, 0.3, 0.42, 0.58, 0.72, 0.85])


def main():
    print(__doc__.strip())
    print()
    print('ПРОВЕРКА 1-2: плотность сетки и N (цель = засев d=4, счёты BASE)')
    print(f"{'N':>6}{'соперников':>12}{'запас m':>14}{'вердикт':>15}")
    for N in (300, 600):
        for ks, qs, lab in ((KS, QS, 'плотная'), (KSD, QSD, 'очень плотная')):
            rs = np.random.default_rng(1234 + N)
            secs = build2(N, rs, ks, qs)
            m, _, nm, G, C = margin(secs, 'засев d=4', BASE)
            v = 'ВЫПОЛНИМО' if m > 1e-9 else 'НЕВЫПОЛНИМО'
            print(f'{N:>6}{len(secs)-1:>12}{m:>14.4g}{v:>15}  ({lab})',
                  flush=True)
    print()
    print('ПРОВЕРКА 1: устойчивость к пересеву (очень плотная, N=600)')
    for seed in (11, 22, 33):
        rs = np.random.default_rng(seed)
        secs = build2(600, rs, KSD, QSD)
        m, _, nm, G, C = margin(secs, 'засев d=4', BASE)
        lam, val, res = cert(G, C)
        print(f'  зерно {seed}: m = {m:>10.4g}, '
              f'сертификат sum lambda*C = {val:+.5g}, остаток {res:.1e}')
    print()
    print('ПРОВЕРКА 3: какой счёт возвращает разделимость (N=600, '
          'очень плотная)')
    rs = np.random.default_rng(77)
    secs = build2(600, rs, KSD, QSD)
    m0, _, _, _, _ = margin(secs, 'засев d=4', BASE)
    print(f'  базовый набор (8 счётов): m = {m0:.4g}')
    for e in EXTRA:
        m, _, _, _, _ = margin(secs, 'засев d=4', BASE + (e,))
        flag = ' <- ВОЗВРАЩАЕТ' if m > 1e-9 else ''
        print(f'  + {e:>7}: m = {m:>12.4g}{flag}')
    m, _, _, _, _ = margin(secs, 'засев d=4', ALL)
    print(f'  + все семь сразу: m = {m:.4g}')


if __name__ == '__main__':
    main()
