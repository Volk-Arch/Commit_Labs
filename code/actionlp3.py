#!/usr/bin/env python3
"""ЧТО ИМЕННО НУЖНО, ЧТОБЫ ВЫДЕЛИТЬ ПРОСТРАНСТВО-ВРЕМЯ.

Разделяющее действие существует (`actionlp2.py`), но «выполнимо» —
ХРУПКОЕ направление: добавление сектора оболочку только расширяет.
Поэтому здесь (1) соперников становится в разы больше, (2) проверяется,
какие ПОДНАБОРЫ счётов достаточны, (3) ищется L1-минимальное
разделяющее действие — то есть САМОЕ РАЗРЕЖЕННОЕ, что отвечает на
вопрос «какими счётами вообще можно выделить пространство-время».

L1-минимум: min sum|theta_a| при theta.g_j >= b_j + 1. Линейная задача
через расщепление theta на положительную и отрицательную части.

  python3 actionlp3.py
"""
import itertools
import json
import math

import numpy as np
from scipy.optimize import linprog

from sectors import counts, sprinkle_d, layered, s_layered, FEAT

SUBSETS = {
    'БД (N0,N1,N2,N3)': ('N0', 'N1', 'N2', 'N3'),
    'перколяция (R,N0)': ('R', 'N0'),
    'БД + R': ('R', 'N0', 'N1', 'N2', 'N3'),
    'цепи (C3,C4)': ('C3', 'C4'),
    'высота (H)': ('H',),
    'R + цепи': ('R', 'C3', 'C4'),
    'всё': FEAT,
}


def build(N, rs, dense=True):
    secs = []
    for d in (2, 3, 4, 5, 6):
        acc = [counts(sprinkle_d(N, d, rs)) for _ in range(3)]
        f = {k: float(np.mean([a[k] for a in acc])) for k in FEAT}
        secs.append({'name': f'засев d={d}', 'f': f,
                     'S': 4.0 * N * math.log(N)})
    ks = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50, 75] if dense else [2, 3, 6, 20]
    qs = ([0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]
          if dense else [0.05, 0.35, 0.5, 0.95])
    for k in ks:
        if N // k < 2:
            continue
        for q in qs:
            Rc, sizes = layered(N, k, q, rs)
            f = counts(Rc)
            secs.append({'name': f'слоистый k={k} q={q}',
                         'f': {kk: float(f[kk]) for kk in FEAT},
                         'S': s_layered(N, sizes, q)})
    M = N * (N - 1) / 2
    secs.append({'name': 'цепь',
                 'f': {'R': M, 'N0': N - 1, 'N1': N - 2, 'N2': N - 3,
                       'N3': N - 4, 'C3': float(math.comb(N, 3)),
                       'C4': float(math.comb(N, 4)), 'H': float(N)},
                 'S': math.lgamma(N + 1)})
    f0 = {k: 0.0 for k in FEAT}
    f0['H'] = 1.0
    secs.append({'name': 'антицепь', 'f': f0, 'S': 0.0})
    return secs


def rows(secs, tgt, feats, scale):
    ft, St = np.array([tgt['f'][k] / scale[k] for k in feats]), tgt['S']
    A, b, nm = [], [], []
    for s in secs:
        if s is tgt:
            continue
        g = ft - np.array([s['f'][k] / scale[k] for k in feats])
        A.append(g)
        b.append(St - s['S'])
        nm.append(s['name'])
    return np.array(A), np.array(b), nm


def l1_min(secs, tgt, feats, scale):
    """min sum|theta| при theta.g >= b + 1."""
    A, b, nm = rows(secs, tgt, feats, scale)
    nf = len(feats)
    # переменные: theta+ , theta-
    Aub = np.hstack([-A, A])
    bub = -(b + 1.0)
    c = np.ones(2 * nf)
    r = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(0, None)] * (2 * nf),
                method='highs')
    if r.status != 0:
        return None, None, nm, A, b
    th = r.x[:nf] - r.x[nf:]
    return th, float(np.abs(th).sum()), nm, A, b


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(555)
    for N in (300, 600):
        secs = build(N, rs, dense=True)
        scale = {k: max(abs(s['f'][k]) for s in secs) or 1.0 for k in FEAT}
        tgt = [s for s in secs if s['name'] == 'засев d=4'][0]
        print('=' * 72)
        print(f'N = {N}, соперников {len(secs)-1} (плотная сетка)')
        print(f"{'поднабор счётов':>22}{'выполнимо?':>14}"
              f"{'||theta||_1':>14}{'связывает':>24}")
        for lab, feats in SUBSETS.items():
            th, nrm, nm, A, b = l1_min(secs, tgt, feats, scale)
            if th is None:
                print(f'{lab:>22}{"НЕТ":>14}{"—":>14}{"":>24}')
                continue
            res = A @ th - b
            i = int(np.argmin(res))
            print(f'{lab:>22}{"да":>14}{nrm:>14.4g}{nm[i]:>24}')
        th, nrm, nm, A, b = l1_min(secs, tgt, FEAT, scale)
        if th is not None:
            thr = th / np.array([scale[k] for k in FEAT])
            print()
            print('  L1-минимальное действие (исходные единицы), '
                  'ненулевые компоненты:')
            for k, v in zip(FEAT, thr):
                if abs(v) > 1e-12:
                    print(f'      {k:>3} : {v:+.6g}')
            print(f'  использовано счётов: '
                  f'{sum(1 for v in th if abs(v) > 1e-12)} из {len(FEAT)}')
        print()


if __name__ == '__main__':
    main()
