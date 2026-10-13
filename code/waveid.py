#!/usr/bin/env python3
"""ТОЖДЕСТВО НА ОДИН ЭЛЕМЕНТ: верны ли веса оператора.

Оператор (B f)(x) = (1/l^2)[ -f(x) + sum_{y<x} w_{|I|} f(y) ] даёт
величину O(1) только если СКОБКА почти обращается в ноль — иначе
получается O(1/l^2) = O(rho), то есть расходимость. А скобка обращается
в ноль на постоянной функции f = 1 тогда и только тогда, когда на один
элемент выполняется

    sum_m w_m n_m = 1,    n_m = число y < x с |I(y,x)| = m

Это проверяется прямо, и оно же объясняет замеренное: в 1+1 среднее
уходило как -0.4n, то есть ровно как собственный член, значит скобка НЕ
обнулялась.

Проверяются оба набора:
  1+1: w = (1, -2, 1)   — мои, непроверенные
  4D:  w = (-1, 9, -16, 8) — весами БД, тождество
       sum w_m Г(m+1/2)/m! = 0 подтверждено машинно (`nm-doc.md`)

КОНТРОЛЬ: считается с ошибкой среднего, и по трём N — если тождество
верно, величина обязана идти к 1 и не зависеть от N.
"""
import math

import numpy as np

from qcarrier import sprinkle4
from waveop import causal, sprinkle_11

W11 = (1.0, -2.0, 1.0)
W4 = (-1.0, 9.0, -16.0, 8.0)


def per_elem(Rc, ws):
    """<sum_m w_m n_m> по элементам, с ошибкой среднего."""
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    tot = np.zeros(Rc.shape[0])
    for m, w in enumerate(ws):
        tot += w * (Rc.T & (Iv.T == m)).sum(axis=1)
    return float(tot.mean()), float(tot.std() / math.sqrt(len(tot)))


def main():
    print(__doc__)
    rs = np.random.default_rng(161803)
    print('1+1, веса (1, -2, 1) — обязано идти к 1')
    print(f"{'N':>7}{'<Σ w n>':>12}{'±':>9}{'от 1':>9}")
    for n in (400, 800, 1600, 3200):
        vs = [per_elem(causal(sprinkle_11(n, rs)), W11) for _ in range(4)]
        m = float(np.mean([v[0] for v in vs]))
        e = float(np.std([v[0] for v in vs])) / 2 + np.mean(
            [v[1] for v in vs])
        print(f'{n:>7}{m:>12.3f}{e:>9.3f}{abs(m-1)/max(e,1e-9):>8.1f}с')

    print('\n4D, веса БД (-1, 9, -16, 8) — обязано идти к 1')
    print(f"{'N':>7}{'<Σ w n>':>12}{'±':>9}{'от 1':>9}")
    for n in (400, 800, 1600, 3200):
        vs = []
        for _ in range(3):
            P = sprinkle4(n, rs, 1.0).astype(np.float64)
            t, X = P[:, 0], P[:, 1:]
            dt = t[None, :] - t[:, None]
            d = X[None, :, :] - X[:, None, :]
            Rc = (dt > 0) & (dt * dt > (d * d).sum(-1))
            vs.append(per_elem(Rc, W4))
        m = float(np.mean([v[0] for v in vs]))
        e = float(np.std([v[0] for v in vs])) / math.sqrt(3) + np.mean(
            [v[1] for v in vs])
        print(f'{n:>7}{m:>12.3f}{e:>9.3f}{abs(m-1)/max(e,1e-9):>8.1f}с',
              flush=True)

    print('\nЧТЕНИЕ: идёт к 1 и не зависит от N — веса верны, и тогда')
    print('расходимость оператора есть чистый шум. Растёт с N — веса')
    print('или само построение (какие пары включать) неверны.')


if __name__ == '__main__':
    main()
