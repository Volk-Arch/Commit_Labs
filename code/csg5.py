#!/usr/bin/env python3
"""ИНВАРИАНТ L - H = 1: заперто ли ВСЁ семейство, а не только срез.

Два точных тождества.

(1) L = сумма по рождениям от m, где m — число предшественников
    (максимальные элементы прошлого): элемент получает ровно m прямых
    связей. Значит показатель L = 1 + beta, где beta — показатель
    среднего числа предшественников.

(2) Высота нового элемента = 1 + max{height(s) : s in S}, потому что
    максимум высоты в замыкании ↓S достигается на максимальном
    элементе, а тот лежит в S.

И ключевое ограничение семейства: вероятность зависит ТОЛЬКО от |S|,
значит при данном |S| = k подмножество S РАВНОМЕРНО среди
k-подмножеств. Тогда вероятность задеть верхний слой есть ~k·c/n,
значит dH/dn ~ n^{beta-1} и показатель H = beta. Отсюда

    L - H = 1   тождественно, при ЛЮБОЙ последовательности t_k

А у d-мерного засева L - H = (2 - 2/d) - 1/d = 2 - 3/d, и равенство
единице даёт РОВНО d = 3.

ЗДЕСЬ это проверяется на последовательностях РАЗНОЙ ФОРМЫ, а не только
факториальных — потому что именно в форме последовательности и была бы
свобода, если бы она была:
  A  x^k/k!              факториал a=1 (узкая, пуассоновская)
  B  x^k/(k!)^2          факториал a=2
  C  x^k/(k!)^1.5        дробный показатель
  D  смесь двух масштабов
  E  обрезанная сверху    (очень узкая)
  F  с утяжелённым хвостом exp(c*sqrt(k))

ПРЕДСКАЗАНИЕ до прогона: L - H ≈ 1 у ВСЕХ шести, с разбросом от
конечного размера, тогда как у засева d=4 это 1.38 (замер) при теории
1.25. Если так — заперто всё семейство, а не срез.

КОНТРОЛЬ: тождество (2) проверяется ПОЭЛЕМЕНТНО на выращенном
порядке — высота каждого элемента сверяется с 1 + max по его S.
"""
import math

import numpy as np
from scipy.special import gammaln, logsumexp

from csg import fit
from sectors import sprinkle_d
from width3 import width


def seqs():
    lg = gammaln
    return {
        'A  x^k/k!': lambda k: -lg(k + 1),
        'B  x^k/(k!)^2': lambda k: -2 * lg(k + 1),
        'C  x^k/(k!)^1.5': lambda k: -1.5 * lg(k + 1),
        'D  смесь масштабов': lambda k: np.logaddexp(
            -2 * lg(k + 1), -3 * lg(k + 1) + 4.0),
        'E  обрезанная k<=6': lambda k: np.where(
            k <= 6, -lg(k + 1), -1e9),
        'F  тяжёлый хвост': lambda k: -lg(k + 1) + 1.5 * np.sqrt(k),
    }


def grow_full(nmax, lt, lx, rs):
    """Рост + поэлементная проверка тождества высоты."""
    Rc = np.zeros((nmax, nmax), bool)
    hgt = np.zeros(nmax, np.int64)
    Ltot, bad = 0, 0
    for n in range(1, nmax):
        k = np.arange(n + 1)
        lw = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
              + k * lx + lt(k))
        p = np.exp(lw - logsumexp(lw))
        kk = int(rs.choice(n + 1, p=p))
        if kk:
            S = rs.choice(n, size=kk, replace=False)
            sel = np.zeros(n, bool)
            sel[S] = True
            past = sel | Rc[:n, :n][:, sel].any(axis=1)
            Rc[:n, n] = past
            hgt[n] = int(hgt[S].max()) + 1
            # m = максимальные элементы прошлого
            idx = np.flatnonzero(past)
            sub = Rc[np.ix_(idx, idx)]
            m = int((~sub.any(axis=1)).sum())
            Ltot += m
            if hgt[n] != int(hgt[idx].max()) + 1:
                bad += 1
        else:
            hgt[n] = 0
    return Rc, hgt, Ltot, bad


def mean_S(n, lt, lx):
    k = np.arange(n + 1)
    lw = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
          + k * lx + lt(k))
    p = np.exp(lw - logsumexp(lw))
    return float((k * p).sum())


def tune(lt, target=3.0, n=200):
    lo, hi = -60.0, 30.0
    for _ in range(90):
        mid = .5 * (lo + hi)
        if mean_S(n, lt, mid) < target:
            lo = mid
        else:
            hi = mid
    return .5 * (lo + hi)


def main():
    print(__doc__)
    rs = np.random.default_rng(161718)
    ns = [125, 250, 500, 1000]

    print('ЭТАЛОНЫ')
    for d in (3, 4):
        o = {}
        for n in ns:
            Rc = sprinkle_d(n, d, rs)
            F = Rc.astype(np.float32)
            L = int((Rc & ~((F @ F) > .5)).sum())
            h = np.zeros(n, np.int64)
            for x in np.argsort(Rc.sum(1))[::-1]:
                pre = np.nonzero(Rc[:, x])[0]
                h[x] = (h[pre].max() + 1) if len(pre) else 0
            o[n] = (L, int(h.max() + 1), width(Rc))
        aL, eL = fit(ns, [o[n][0] for n in ns])
        aH, eH = fit(ns, [o[n][1] for n in ns])
        print(f'  засев d={d}: L={aL:.4f}±{eL:.3f}, H={aH:.4f}±{eH:.3f}'
              f'  =>  L-H = {aL-aH:.3f}  (теория {2-3/d:.3f})')

    print('\nПОСЛЕДОВАТЕЛЬНОСТИ РАЗНОЙ ФОРМЫ')
    print(f"{'последовательность':>22}{'E|S| пок.':>11}{'L':>9}{'H':>9}"
          f"{'L-H':>8}{'тождество H':>13}")
    for nm, lt in seqs().items():
        lx = tune(lt)
        es = [mean_S(n, lt, lx) for n in ns]
        ea, _ = fit(ns, es)
        LL, HH, badtot = [], [], 0
        for n in ns:
            Rc, hgt, Lt, bad = grow_full(n, lt, lx, rs)
            LL.append(Lt)
            HH.append(int(hgt.max()) + 1)
            badtot += bad
        aL, eL = fit(ns, LL)
        aH, eH = fit(ns, HH)
        print(f'{nm:>22}{ea:>11.3f}{aL:>9.3f}{aH:>9.3f}{aL-aH:>8.3f}'
              f'{("ОК" if badtot == 0 else f"{badtot} сбоев"):>13}',
              flush=True)

    print('\nЧТЕНИЕ: если L-H ≈ 1 у всех форм, инвариант есть свойство')
    print('СЕМЕЙСТВА, а не среза, и d>=4 закрыто арифметикой: засеву')
    print('нужно L-H = 2-3/d, что равно 1 только при d=3.')


if __name__ == '__main__':
    main()
