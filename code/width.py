#!/usr/bin/env python3
"""ШИРИНА как пространство: растёт ли максимальная антицепь как N^{(d-1)/d}.

Картина склейки (`compose-doc.md`) говорит: блок передаёт будущему свою
ГРАНИЦУ, и граница есть максимальная антицепь — дискретный
пространственный срез. Тогда у засева граница должна быть срезом
толщиной в одну длину дискретности, то есть

    ширина ~ N^{(d-1)/d}      (3/4 при d=4)

а у слоистого врага граница есть целый слой, то есть ~N. Это и объяснит,
почему ГРАНИЧНЫЙ счёт был единственным, что хоть что-то разделяло
(`actionexist-doc.md`): он мерит отношение состояния к содержимому.

ТОЧНО, а не оценкой. По теореме Дилворта максимальная антицепь равна
минимальному числу цепей, покрывающих порядок, а по Кёнигу–Фулкерсону
минимальное покрытие цепями равно N минус максимальное паросочетание в
двудольном графе отношения. Паросочетание считается точно
(`scipy.sparse.csgraph.maximum_bipartite_matching`).

ПРЕДСКАЗАНИЯ, записанные до прогона:
  d=2 -> 0.500,  d=3 -> 0.667,  d=4 -> 0.750,  d=5 -> 0.800
  слоистый с фиксированным k -> 1.000

КОНТРОЛИ:
  1. цепь: ширина 1 при любом N;
  2. антицепь: ширина N;
  3. слоистый с k слоями: ширина равна размеру слоя, то есть N/k —
     известный ответ, проверяет и сам алгоритм, и то, что показатель 1
     действительно достижим.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from sectors import sprinkle_d


def width(Rc):
    """Максимальная антицепь точно: Дилворт + Кёниг."""
    n = Rc.shape[0]
    M = csr_matrix(Rc)
    mt = maximum_bipartite_matching(M, perm_type='column')
    return n - int((mt >= 0).sum())


def rc_sprinkle(n, d, rs):
    P = sprinkle_d(n, d, rs).astype(np.float64)
    dt = P[None, :, 0] - P[:, None, 0]
    dd = P[None, :, 1:] - P[:, None, 1:]
    return (dt > 0) & (dt * dt > (dd * dd).sum(-1))


def layered_rc(n, k, q, rs):
    a = n // k
    sz = [a] * k
    sz[-1] += n - a * k
    off = np.cumsum([0] + sz)
    R = np.zeros((n, n), bool)
    for i in range(k - 1):
        R[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sz[i], sz[i + 1])) < q)
    C = R.copy()
    for _ in range(k):
        nx = C | ((C.astype(np.float32) @ C.astype(np.float32)) > .5)
        if (nx == C).all():
            break
        C = nx
    return C, max(sz)


def fit(Ns, W):
    lx, ly = np.log(np.array(Ns, float)), np.log(np.array(W, float))
    A = np.vstack([lx, np.ones(len(lx))]).T
    be, *_ = np.linalg.lstsq(A, ly, rcond=None)
    res = ly - A @ be
    se = (math.sqrt(float((res ** 2).sum() / max(len(lx) - 2, 1))
                    / float(((lx - lx.mean()) ** 2).sum()))
          if len(lx) > 2 else float('nan'))
    return float(be[0]), se


def main():
    print(__doc__)
    rs = np.random.default_rng(1123581321)

    print('КОНТРОЛИ')
    n = 200
    ch = np.triu(np.ones((n, n), bool), 1)
    print(f'  цепь {n}:      ширина {width(ch)} (ожидалось 1)')
    print(f'  антицепь {n}:  ширина {width(np.zeros((n, n), bool))} '
          f'(ожидалось {n})')
    for k in (4, 10):
        C, mx = layered_rc(n, k, 0.5, rs)
        print(f'  слоистый k={k}: ширина {width(C)} '
          f'(ожидалось ~{mx} = размер слоя)')

    print('\nЗАСЕВЫ: показатель ширины по N')
    Ns = [200, 400, 800, 1600, 3200]
    print(f"{'d':>3}" + ''.join(f'{f"N={x}":>9}' for x in Ns)
          + f"{'показатель':>13}{'предсказ.':>11}{'отклон.':>10}")
    for d in (2, 3, 4, 5):
        W, ok = [], True
        for nn in Ns:
            vals = [width(rc_sprinkle(nn, d, rs))
                    for _ in range(3 if nn <= 800 else 2)]
            W.append(float(np.mean(vals)))
        a, se = fit(Ns, W)
        pred = (d - 1) / d
        print(f'{d:>3}' + ''.join(f'{w:>9.1f}' for w in W)
              + f'{a:>8.4f}±{se:<4.3f}{pred:>11.3f}'
              f'{abs(a-pred)/max(se,1e-9):>9.1f}с', flush=True)

    print('\nСЛОИСТЫЙ ВРАГ (k фиксировано): показатель обязан быть 1')
    for k, q in ((3, 0.2), (5, 0.1)):
        W = []
        for nn in Ns:
            C, _ = layered_rc(nn, k, q, rs)
            W.append(float(width(C)))
        a, se = fit(Ns, W)
        print(f'  k={k}, q={q}: ' + ', '.join(f'{w:.0f}' for w in W)
              + f'  показатель {a:.4f}±{se:.3f} (предсказано 1.000)')

    print('\nОТНОШЕНИЕ состояние/содержимое (ширина / N)')
    print(f"{'':>14}" + ''.join(f'{f"N={x}":>9}' for x in Ns))
    for lab, gen in (('засев d=4', lambda nn: width(rc_sprinkle(nn, 4, rs))),
                     ('слоистый k=3', lambda nn: width(layered_rc(nn, 3, 0.2, rs)[0]))):
        print(f'  {lab:>12}' + ''.join(
            f'{gen(nn)/nn:>9.3f}' for nn in Ns), flush=True)


if __name__ == '__main__':
    main()
