#!/usr/bin/env python3
"""КОРОНЫ: локальная конфигурация, кодирующая размерность.

Правило на ПАРАХ - всё, что делалось весь день. Правило на малых
КОНФИГУРАЦИЯХ кодирует размерность: dim <= 2 равносильно отсутствию
короны S_3. Минимальный размер - ШЕСТЬ элементов, не пять: при n=5
порядков с dim>2 ноль, при n=6 их три (`dimcount.py`).

И главное: орта размерности k НЕ СОДЕРЖИТ S_{k+1} вовсе. Значит
конфигурации - единственный тип теста, способный поймать орту, которая
прошла все локальные проверки (`facecontrol-doc.md`: при опросе пяти
событий 0.910).

S_k есть КЛИКА размера k в «графе совместимости»: вершины - несравнимые
пары (a,b), а две пары совместимы, если a1<b2 и a2<b1. Тогда:
  * число корон S_3 = число треугольников = trace(A^3)/6, одно умножение
    матриц;
  * максимальное S_k = максимальная клика, и её ищем ОДИНАКОВЫМ усилием
    у всех ансамблей, иначе сравнение нечестно.

КОНТРОЛЬ: засев 1+1 обязан дать НОЛЬ корон - у него dim = 2 ровно
(световое тождество, `tri-doc.md`). Если короны найдутся, сломан код.
"""
import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from orthant import scales, thresh_order
from lkdeg import layered

N = 500
MPAIR = 1500


def compat(Rc, rs, m=MPAIR):
    inc = ~(Rc | Rc.T)
    np.fill_diagonal(inc, False)
    ii, jj = np.nonzero(np.triu(inc, 1))
    if len(ii) < 3:
        return None
    sel = rs.choice(len(ii), size=min(m, len(ii)), replace=False)
    A, B = ii[sel], jj[sel]
    C = Rc[np.ix_(A, B)]
    G = C & C.T
    np.fill_diagonal(G, False)
    return G


def triangles(G):
    F = G.astype(np.float32)
    return float(np.trace(F @ F @ F) / 6.0), G.shape[0]


def maxclique(G, rs, tries=300):
    m = G.shape[0]
    deg = G.sum(axis=1)
    best = 1
    for t in range(tries):
        order = np.argsort(-deg) if t == 0 else rs.permutation(m)
        mask = np.ones(m, bool)
        cnt = 0
        for v in order:
            if not mask[v]:
                continue
            cnt += 1
            mask &= G[v]
            mask[v] = False
            if not mask.any():
                break
        best = max(best, cnt)
    return best


def main():
    print(__doc__)
    rs = np.random.default_rng(1414)
    ens = (('засев 1+1 (контроль)', lambda: causal(sprinkle_11(N, rs))),
           ('засев 4d', lambda: sprinkle_d(N, 4, rs)),
           ('орта k=4', lambda: thresh_order(scales(N, 4, rs), 4)),
           ('орта k=6', lambda: thresh_order(scales(N, 6, rs), 6)),
           ('слоистый k=3 q=0.2', lambda: layered(N, 3, 0.2, rs)))
    print(f"{'ансамбль':>22}{'вершин':>8}{'корон S_3':>12}"
          f"{'плотность':>12}{'макс S_k':>10}")
    for lab, mk in ens:
        tr, mc, sk = [], [], []
        for _ in range(3):
            G = compat(mk(), rs)
            if G is None:
                continue
            t, m = triangles(G)
            tr.append(t)
            mc.append(m)
            sk.append(maxclique(G, rs))
        if not tr:
            print(f'{lab:>22}{"—":>8}{"нет несравнимых пар":>24}')
            continue
        m = np.mean(mc)
        print(f'{lab:>22}{m:>8.0f}{np.mean(tr):>12.1f}'
              f'{np.mean(tr) / m ** 3:>12.3e}{max(sk):>10}', flush=True)
    print('\nЧТЕНИЕ: у засева 1+1 корон обязан быть НОЛЬ (dim=2 ровно).')
    print('У орты k максимальная S_k обязана не превышать k.')


if __name__ == '__main__':
    main()
