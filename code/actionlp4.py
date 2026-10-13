#!/usr/bin/env python3
"""ЧИСТАЯ ПОСТАНОВКА + СЕРТИФИКАТ. Исправлен знак энтропийного дефицита.

В `actionlp3.py` требование было записано как theta.g >= b + 1, где
b = S_цель - S_соперник, тогда как нужно theta.g >= -b + 1. Из-за этого
«невыполнимо» вышло по ошибке знака, а не по геометрии. Здесь всё
определено явно:

    g_j = f_цель - f_j        (счёты, нормированные)
    c_j = S_j - S_цель        (энтропийный дефицит цели)
    требование:  theta.g_j > c_j   для всех j

ВЫПОЛНИМОСТЬ: max m при theta.g_j - m >= c_j, theta без границ;
неограниченность = выполнимо.

СЕРТИФИКАТ ФАРКАША: система несовместна тогда и только тогда, когда
существуют lambda_j >= 0 с
    sum_j lambda_j g_j = 0    и    sum_j lambda_j c_j >= 0
Это ДОКАЗАТЕЛЬСТВО несовместности, а не неудача решателя: оно прямо
говорит, что счёты цели суть выпуклая смесь счётов соперников, у
которых энтропия не меньше. Сертификат проверяется подстановкой.

КОНТРОЛИ:
  1. цель = цепь обязана быть ВЫПОЛНИМОЙ (крайняя точка);
  2. цель = антицепь обязана быть выполнимой (другая крайняя точка);
  3. если найден сертификат — подстановка обязана дать нули;
  4. плотность сетки соперников: выполнимость может только падать.

  python3 actionlp4.py
"""
import math

import numpy as np
from scipy.optimize import linprog

from actionlp3 import build, SUBSETS
from sectors import FEAT


def prep(secs, tgt, feats, scale):
    ft = np.array([tgt['f'][k] / scale[k] for k in feats])
    G, C, nm = [], [], []
    for s in secs:
        if s is tgt:
            continue
        G.append(ft - np.array([s['f'][k] / scale[k] for k in feats]))
        C.append(s['S'] - tgt['S'])
        nm.append(s['name'])
    return np.array(G), np.array(C), nm


def feas(G, C):
    """max m s.t. G theta - m >= C."""
    nf = G.shape[1]
    A = np.hstack([-G, np.ones((len(C), 1))])
    c = np.zeros(nf + 1)
    c[-1] = -1.0
    r = linprog(c, A_ub=A, b_ub=-C,
                bounds=[(None, None)] * (nf + 1), method='highs')
    if r.status == 3:
        return 'ВЫПОЛНИМО', None, math.inf
    if r.status == 0:
        return ('ВЫПОЛНИМО' if r.x[-1] > 1e-9 else 'НЕВЫПОЛНИМО'), \
            r.x[:-1], float(r.x[-1])
    return f'статус {r.status}', None, float('nan')


def farkas(G, C):
    """lambda >= 0, G^T lambda = 0, sum lambda_j C_j >= 0, sum lambda = 1."""
    nj, nf = G.shape
    Aeq = np.vstack([G.T, np.ones((1, nj))])
    beq = np.concatenate([np.zeros(nf), [1.0]])
    # максимизируем sum lambda C
    r = linprog(-C, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nj,
                method='highs')
    if r.status != 0:
        return None, None
    return r.x, float(C @ r.x)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(808)
    for N in (300, 600):
        for dense in (False, True):
            secs = build(N, rs, dense=dense)
            scale = {k: max(abs(s['f'][k]) for s in secs) or 1.0
                     for k in FEAT}
            tgt = [s for s in secs if s['name'] == 'засев d=4'][0]
            print('=' * 72)
            print(f'N = {N}, соперников {len(secs)-1}, '
                  f'сетка {"плотная" if dense else "редкая"}')
            for cname in ('цепь', 'антицепь'):
                cs = [s for s in secs if s['name'] == cname][0]
                G, C, _ = prep(secs, cs, FEAT, scale)
                v, _, m = feas(G, C)
                print(f'  КОНТРОЛЬ (цель = {cname}): {v}, m = {m:.4g}')
            G, C, nm = prep(secs, tgt, FEAT, scale)
            v, th, m = feas(G, C)
            print(f'  ЦЕЛЬ = засев d=4, все счёты: {v}, m = {m:.4g}')
            if v == 'НЕВЫПОЛНИМО':
                lam, val = farkas(G, C)
                if lam is not None:
                    resid = np.abs(G.T @ lam).max()
                    print(f'  СЕРТИФИКАТ ФАРКАША: sum lambda*C = {val:+.5g} '
                          f'(нужно >= 0), остаток |G^T lambda| = '
                          f'{resid:.2e}')
                    idx = np.argsort(-lam)[:5]
                    print('     смесь: ' + ', '.join(
                        f'{nm[i]} {lam[i]:.3f}' for i in idx if lam[i] > 1e-6))
            print('  по поднаборам счётов:')
            for lab, feats in SUBSETS.items():
                G2, C2, _ = prep(secs, tgt, feats, scale)
                v2, _, m2 = feas(G2, C2)
                print(f'     {lab:>22}: {v2:>13}  m = {m2:.4g}')
            print()


if __name__ == '__main__':
    main()
