#!/usr/bin/env python3
"""Петля на ПОПЕРЕЧНЫХ шкалах: то, чего не было при первом счёте.

ТОЧНАЯ ФОРМУЛИРОВКА СТЕНЫ, к которой пришли:

  чтобы быть в секторе k шкал, нужны k чисел на событие. Взять их
  извне и независимо — это координаты, и конус получается даром.
  Породить из самого мира — и они выходят МОНОТОННЫМИ по времени, а
  конусу нужны шкалы, МЕНЯЮЩИЕ ЗНАК по направлениям: у s_u = t - u*x
  часть -u*x разная по знаку. Длины цепей так не умеют, поэтому
  неподвижная точка петли «порядок -> числа -> порядок» оказалась
  тотальным порядком (fixpoint.py: 0.1140 -> 0.7862 -> 1.0000).

ЧТО ИЗМЕНИЛОСЬ. Когда петлю считали, все порядковые величины были
монотонны. Теперь есть немонотонные, найденные за последние два дня:

    q(a,b) = |past(a) & past(b)| / min(|past(a)|, |past(b)|)
             ранговая корреляция с разносом -0.801, разброс 0.60
             устойчив по всем корзинам            (PICTURE.md)
    W(b)   = число событий опоры, несравнимых с b; W = 2*lam*r,
             проверено на трёх плотностях         (partial-doc.md)

Обе смотрят ПОПЕРЁК, а не вперёд. Значит петлю можно замкнуть на них.

ПЕТЛЯ. По порядку R выбираются k опор. Для события b:

    v_i(b) = q(r_i, b)      если b несравнимо с опорой r_i
           = -(глубина)     если опора в прошлом b  (знак меняется!)
           = +(высота)      если b в прошлом опоры

Знак и есть то, чего не хватало: у события по одну сторону от опоры
величина положительна, по другую отрицательна, а поперёк — промежуточна.
Это ровно устройство s_u = t - u*x.

Новый порядок собирается нашим же правилом: R'[i,j] = v(j) < v(i)
покомпонентно.

ПРОВЕРЯЕТСЯ ТО ЖЕ, ЧТО В fixpoint.py:
  1. сходится ли, и к чему;
  2. является ли конус (приближённой) неподвижной точкой;
  3. СХОДЯТСЯ ЛИ РАЗНЫЕ НАЧАЛА В ОДНО — только это и был бы отбор.

  python3 transloop.py
  python3 transloop.py --n 700 --krefs 6 --iters 6
"""
import argparse
import math
import time

import numpy as np

from core import frac_pairs, mm_dim, height, spectral
from core import transitivity_violations
from fixpoint import diamond, cone_order, box_order, rel_of_int, longest_all


def trans_scales(R, srcs):
    """v[i, s] — поперечная величина события i относительно опоры srcs[s].

    Знак несёт причинное положение, модуль — поперечное перекрытие.
    """
    N = len(R)
    A = R.astype(np.float32)
    CP = A @ A.T                                   # |past(i) & past(j)|
    sz = A.sum(1)
    mn = np.minimum(sz[:, None], sz[None, :])
    q = CP / np.maximum(mn, 1.0)
    V = np.zeros((N, len(srcs)), np.float64)
    L = longest_all(R, srcs)                       # длины цепей от опор
    for s, r in enumerate(srcs):
        r = int(r)
        after = R[:, r]                            # опора раньше i
        before = R[r, :]                           # i раньше опоры
        inc = (~after) & (~before)
        V[inc, s] = q[inc, r]                      # поперёк: от 0 до 1
        V[after, s] = 1.0 + np.maximum(L[after, s], 0)
        V[before, s] = -1.0 - np.maximum(L[r, s] - L[before, s], 0)
    return V


def step(R, kref, rs):
    """Один проход: порядок -> поперечные числа -> порядок."""
    N = len(R)
    fut = R.sum(0)
    cand = np.argsort(-fut)[:max(4 * kref, 20)]
    srcs = np.array(sorted(int(c) for c in rs.choice(cand, kref,
                                                     replace=False)))
    V = trans_scales(R, srcs)
    V = V + 1e-9 * np.arange(N)[:, None] / N       # разрыв ничьих
    return rel_of_int(V)


def report(R, lbl):
    f = frac_pairs(R)
    print(f'{lbl:>14}{f:>10.4f}{mm_dim(f):>10.2f}{height(R)[0]:>8.0f}'
          f'{spectral(R):>+10.4f}{transitivity_violations(R):>10}',
          flush=True)
    return f


def main():
    ap = argparse.ArgumentParser(description='Петля на поперечных шкалах')
    ap.add_argument('--n', type=int, default=700)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--krefs', type=int, default=5)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, опор {a.krefs}, итераций {a.iters}')
    print('шкала: ЗНАК от причинного положения, МОДУЛЬ от поперечного q')
    print('ЭТАЛОН конуса: доля ~0.10, разм. 4.00, спектр > 0')
    print('на длинах цепей петля шла в ТОТАЛЬНЫЙ порядок (доля 1.0000)')
    print()

    rs = np.random.default_rng(3701000 + a.seed)
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
