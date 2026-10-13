#!/usr/bin/env python3
"""Петля на шкалах вида «время минус поперечное», в одних единицах.

ДВЕ ПРЕДЫДУЩИЕ ПОПЫТКИ И ПОЧЕМУ ОБЕ УПАЛИ В ТОТАЛЬНЫЙ ПОРЯДОК.

  fixpoint.py:  шкала = длина цепи от опоры. Монотонна по времени, и
                покомпонентное доминирование на монотонных шкалах есть
                почти тотальный порядок. 0.1140 -> 0.7862 -> 1.0000.
  transloop.py: шкала = поперечное q для несравнимых, но «1 + длина
                цепи» для событий после опоры. Опора берётся с большим
                будущим, значит почти все события попадают в монотонную
                ветвь. 0.1033 -> 0.7632 -> 0.9998. Смена знака шла по
                одной поверхности — световому конусу опоры.

ЧЕГО НЕ ХВАТАЛО. У s_u = t - u*x значение меняется с НАПРАВЛЕНИЕМ при
фиксированном времени: разные u по-разному упорядочивают одни и те же
пространственные точки. Именно это несогласие и делает пересечение
конусом, а не цепью.

ЗДЕСЬ. Шкала строится как дискретная запись t - |x|, причём ОБА
слагаемых в одних единицах — звеньях опорной цепи:

    v_i(b) = h(b) - W_i(b)/2

    h(b)    длина максимальной цепи до b          (время, в звеньях)
    W_i(b)  число событий опорной ЦЕПИ i, несравнимых с b
            partial-doc.md: W = 2*lam*r, проверено на трёх
            плотностях (68 -> 69.7+-2.3, 136 -> 136.6+-4.0)
            значит W/2 есть поперечное расстояние в звеньях той же цепи

Свободного параметра нет: единицы совпадают по доказанной формуле, а не
по подгонке. Опора — не событие, а МИРОВАЯ ЛИНИЯ (максимальная цепь
через выбранное событие), иначе поперечного расстояния не определить.

ПРОВЕРЯЕТСЯ ТО ЖЕ:
  1. сходится ли, и к чему;
  2. является ли конус (приближённой) неподвижной точкой;
  3. СХОДЯТСЯ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО — только это и был бы отбор.

  python3 coneloop.py
  python3 coneloop.py --n 700 --krefs 6 --iters 8
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral
from core import transitivity_violations
from fixpoint import diamond, cone_order, box_order, rel_of_int


def heights_of(R):
    """h[i] = длина максимальной цепи, кончающейся в i."""
    N = len(R)
    order = np.argsort(R.sum(1), kind='stable')
    h = np.zeros(N, np.int64)
    for i in order:
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return h


def chain_through(R, v, h):
    """Максимальная цепь через событие v: назад по прошлому, вперёд."""
    out = [int(v)]
    cur = int(v)
    while True:
        pre = np.flatnonzero(R[cur])
        if len(pre) == 0:
            break
        cur = int(pre[np.argmax(h[pre])])
        out.append(cur)
    cur = int(v)
    while True:
        nxt = np.flatnonzero(R[:, cur])
        if len(nxt) == 0:
            break
        cur = int(nxt[np.argmin(h[nxt])])
        out.append(cur)
    return np.array(sorted(set(out)), np.int64)


def cone_scales(R, srcs):
    """v[i, s] = h(i) - W_s(i)/2, обе части в звеньях опорной цепи."""
    N = len(R)
    h = heights_of(R).astype(np.float64)
    V = np.zeros((N, len(srcs)), np.float64)
    for s, v0 in enumerate(srcs):
        ch = chain_through(R, int(v0), h)
        if len(ch) < 3:
            V[:, s] = h
            continue
        rel = R[:, ch] | R[ch, :].T          # сравнимо с событием цепи
        W = (~rel).sum(1).astype(np.float64)
        V[:, s] = h - W / 2.0
    return V


def step(R, kref, rs):
    """Один проход: порядок -> шкалы «время минус поперечное» -> порядок."""
    N = len(R)
    h = heights_of(R)
    # опоры — события из разных мест: разные высоты и далёкие друг от друга
    cand = np.flatnonzero((h > np.percentile(h, 25))
                          & (h < np.percentile(h, 75)))
    if len(cand) < kref:
        cand = np.arange(N)
    srcs = rs.choice(cand, kref, replace=False)
    V = cone_scales(R, srcs)
    V = V + 1e-9 * np.arange(N)[:, None] / N
    return rel_of_int(V)


def report(R, lbl):
    f = frac_pairs(R)
    print(f'{lbl:>14}{f:>10.4f}{mm_dim(f):>10.2f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>+10.4f}{transitivity_violations(R):>10}',
          flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Петля на конусных шкалах')
    ap.add_argument('--n', type=int, default=700)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--krefs', type=int, default=5)
    ap.add_argument('--iters', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, опор {a.krefs}, итераций {a.iters}')
    print('шкала v = h - W/2, обе части в звеньях опорной цепи, '
          'параметра нет')
    print('ЭТАЛОН конуса: доля ~0.10, разм. 4.00, спектр > 0')
    print('на длинах цепей и на одном q петля шла в ТОТАЛЬНЫЙ порядок')
    print()

    rs = np.random.default_rng(3801000 + a.seed)
    X = diamond(a.n, a.t, rs)
    starts = {
        'конус': cone_order(X),
        'ящик': box_order(a.n, rs),
        'случайный': rel_of_int(rs.random((a.n, 2))),
    }
    finals = {}
    for name, R0 in starts.items():
        print(f'НАЧАЛО: {name}')
        print(f"{'итерация':>14}{'доля пар':>10}{'разм.':>10}{'высота':>8}"
              f"{'спектр':>10}{'наруш':>10}")
        R = R0
        report(R, '0 (начало)')
        prev = None
        for it in range(1, a.iters + 1):
            Rn = step(R, a.krefs, rs)
            same = float((Rn == R).mean())
            R = Rn
            f = report(R, f'{it}')
            if prev is not None and abs(f - prev) < 1e-4 and same > 0.9999:
                print(f'   неподвижная точка на итерации {it}')
                break
            prev = f
        finals[name] = R
        print()

    print('СОШЛИСЬ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО')
    ns = list(finals)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            A, B = finals[ns[i]], finals[ns[j]]
            print(f'  {ns[i]:>10} против {ns[j]:<10} совпадение '
                  f'{float((A == B).mean()):.4f}   доли '
                  f'{frac_pairs(A):.4f} / {frac_pairs(B):.4f}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
