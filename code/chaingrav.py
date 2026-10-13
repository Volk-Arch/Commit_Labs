#!/usr/bin/env python3
"""Гравитация на порядковых величинах.

Прежние «гравитационные» измерения шли на евклидовом расстоянии и
градиенте плотности — величинах, которые не выживают при монотонном
перешкалировании (см. docs/rescale-doc.md). Значит они описывали запись,
а не мир. Здесь всё меряется из порядка:

  собственное время  tau = длина самой длинной цепи между событиями
  объём              V   = число ФОНОВЫХ событий в интервале
  кривизна           расхождение между V и tau^d

Три мира на одном и том же фоне (порядок фоновых событий между собой от
добавления новых не меняется, меняются только цепи сквозь них):

  фон        N событий
  тело       + M событий в шаре (сосредоточенная масса)
  контроль   + M событий равномерно по всему миру

Контроль обязателен: любые добавленные события удлиняют цепи. Вопрос в
том, удлиняет ли СОСРЕДОТОЧЕННАЯ масса сильнее при том же числе
добавленных событий внутри интервала.

  python3 chaingrav.py
  python3 chaingrav.py --k 5 --n 2000
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of


def chain_len(Rsub):
    """Число событий в самой длинной цепи подмножества.

    Сортировка по числу предшественников — законный топологический
    порядок для транзитивного отношения: если j раньше i, то прошлое j
    строго меньше прошлого i.
    """
    m = len(Rsub)
    if m == 0:
        return 0
    npred = Rsub.sum(1)
    d = np.zeros(m, dtype=np.int64)
    for i in np.argsort(npred, kind='stable'):
        pr = np.flatnonzero(Rsub[i])
        if pr.size:
            d[i] = d[pr].max() + 1
    return int(d.max()) + 1


def build(N, M, k, r, seed):
    """Фон и два способа добавить M событий."""
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, k))
    # шар радиуса r в центре: равномерно по объёму шара
    v = rs.normal(size=(M, k))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = r * rs.random(M) ** (1.0 / k)
    Xbody = 0.5 + v * rad[:, None]
    Xunif = rs.random((M, k))
    return Xb, np.vstack([Xb, Xbody]), np.vstack([Xb, Xunif])


def measure(X, N, pairs):
    """Для каждой пары фоновых событий четыре порядковые величины:

      0  tau      длина самой длинной цепи от a к b, в звеньях
      1  V_bg     число ФОНОВЫХ событий в интервале
      2  n_add    число ДОБАВЛЕННЫХ событий в интервале
      3  h_add    высота подмножества добавленных внутри интервала

    h_add — потолок прироста tau: цепь, идущая сквозь добавленные
    события, не может выиграть больше, чем их собственная высота.
    """
    R = rel_of(X)
    out = np.zeros((len(pairs), 4), dtype=np.int64)
    for t, (a, b) in enumerate(pairs):
        # a раньше b:  R[b,a] == True
        inside = np.flatnonzero(R[:, a] & R[b])      # a раньше l раньше b
        idx = np.concatenate(([a], inside, [b]))
        out[t, 0] = chain_len(R[np.ix_(idx, idx)]) - 1
        out[t, 1] = int((inside < N).sum())
        add = inside[inside >= N]
        out[t, 2] = len(add)
        out[t, 3] = chain_len(R[np.ix_(add, add)]) if len(add) else 0
    return out


def sample_pairs(X, N, n_pairs, min_vol, rs):
    """Связанные пары фоновых событий с интервалом не меньше min_vol."""
    R = rel_of(X[:N])
    i, j = np.nonzero(R)                             # j раньше i
    keep = (i < N) & (j < N)
    i, j = i[keep], j[keep]
    sel = rs.choice(len(i), min(20 * n_pairs, len(i)), replace=False)
    out = []
    for t in sel:
        a, b = int(j[t]), int(i[t])
        vol = int((R[:, a] & R[b]).sum())
        if vol >= min_vol:
            out.append((a, b))
        if len(out) >= n_pairs:
            break
    return out


def run(N, M, k, r, seed, n_pairs, min_vol):
    Xb, Xbody, Xunif = build(N, M, k, r, seed)
    rs = np.random.default_rng(seed + 1)
    pairs = sample_pairs(Xb, N, n_pairs, min_vol, rs)
    res = {}
    for name, X in (('фон', Xb), ('тело', Xbody), ('контроль', Xunif)):
        res[name] = measure(X, N, pairs)
    return res, pairs


def main():
    ap = argparse.ArgumentParser(description='Гравитация на порядковых величинах')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--m', type=int, default=300)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--r', type=float, default=0.12)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--pairs', type=int, default=700)
    ap.add_argument('--minvol', type=int, default=8)
    a = ap.parse_args()

    res, pairs = run(a.n, a.m, a.k, a.r, a.seed, a.pairs, a.minvol)
    print(f'N={a.n}, добавлено M={a.m}, k={a.k}, радиус тела {a.r}, '
          f'пар {len(pairs)}')
    print()
    tau0 = res['фон'][:, 0]
    print(f"{'мир':>12}{'tau средн.':>13}{'объём фона':>13}"
          f"{'добавл. внутри':>17}{'прирост tau':>14}")
    for name in ('фон', 'тело', 'контроль'):
        A = res[name]
        dt = float(np.mean(A[:, 0] - tau0))
        print(f'{name:>12}{float(A[:, 0].mean()):>13.3f}'
              f'{float(A[:, 1].mean()):>13.2f}{float(A[:, 2].mean()):>17.2f}'
              f'{dt:>14.4f}')

    print()
    print('ОБЪЯСНЯЕТСЯ ЛИ ПРИРОСТ ВЫСОТОЙ ДОБАВЛЕННЫХ')
    print('  прирост/потолок = 1  ->  цепь просто идёт сквозь них,')
    print('                           это счёт событий, а не гравитация')
    print(f"{'мир':>12}{'потолок h_add':>15}{'прирост tau':>14}"
          f"{'прирост/потолок':>18}")
    for name in ('тело', 'контроль'):
        A = res[name]
        m = A[:, 3] > 0
        dt = (A[m, 0] - tau0[m]).astype(float)
        hc = A[m, 3].astype(float)
        print(f'{name:>12}{hc.mean():>15.3f}{dt.mean():>14.3f}'
              f'{dt.sum() / hc.sum():>18.4f}')

    print()
    print('ПО КОРЗИНАМ ОДИНАКОВОГО ПОТОЛКА')
    print(f"{'потолок h_add':>15}{'тело: прирост':>16}{'пар':>7}"
          f"{'контроль: прирост':>20}{'пар':>7}{'разность, σ':>14}")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 6), (7, 10), (11, 10**6)):
        cells = []
        for name in ('тело', 'контроль'):
            A = res[name]
            m = (A[:, 3] >= lo) & (A[:, 3] <= hi)
            v = (A[m, 0] - tau0[m]).astype(float)
            cells.append((v.mean() if len(v) else float('nan'),
                          v.std() / max(np.sqrt(len(v)), 1) if len(v) else float('nan'),
                          len(v)))
        (m1, s1, n1), (m2, s2, n2) = cells
        sig = np.hypot(s1, s2)
        d = m1 - m2
        lbl = f'{lo}-{hi}' if hi < 10**5 else f'{lo}+'
        shown = abs(d) / sig if sig > 0 and n1 and n2 else float('nan')
        print(f'{lbl:>15}{m1:>16.3f}{n1:>7}{m2:>20.3f}{n2:>7}{shown:>14.2f}')


if __name__ == '__main__':
    main()
