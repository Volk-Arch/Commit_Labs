#!/usr/bin/env python3
"""ПЕРЕСЧЁТ ПУНКТА 4 с верным действием. Явная формула для a1 = 2.

Из `cancelclosed.py` (проверено против перечисления порядков до 3e-15):

  u = e^{-i th},  psi_m = e^{i th w(m)} - 1  (m = 1,2,3; иначе 0)
  A(M) = sum_{m=1..3} psi_m u^m C(M,m)

  G = (1+u)^{a2}
    + (1+u)^{n00+n01} A(n10+n11)
    + (1+u)^{n00+n10} A(n01+n11)
    + (1+u)^{n00} B,
  B = sum_{t11=0..3} C(n11,t11) u^{t11} P(t11) Q(t11)
  P(t) = sum_{t10=0..3-t} C(n10,t10) u^{t10} psi(t10+t)
  Q(t) = sum_{t01=0..3-t} C(n01,t01) u^{t01} psi(t01+t)

Всё векторизовано по композициям a_2 на четыре класса, поэтому
max_{M12} |G|/2^{a_2} берётся точно при a_2 в сотни, а не до 7.

ЧТО РЕШАЕТСЯ. Вчера: s = -ln(max q)/a_2 = 0.0717 против нужного
ln 2 = 0.693, «не хватает 9.7», но s росло с a_2 (+0.0053), и линейная
экстраполяция дотягивала до порога при a_2 ~ 121. Здесь s(a_2) берётся
ТОЧНО до a_2 ~ 400, и экстраполяция больше не нужна.

  python3 cancelfast.py
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
W = {1: 9.0, 2: -16.0, 3: 8.0}


def psis(th):
    return {m: np.exp(1j * th * W[m]) - 1.0 for m in (1, 2, 3)}


def comps(a2):
    """Все (n00,n10,n01,n11) с суммой a_2, как массивы."""
    out = []
    for n11 in range(a2 + 1):
        for n10 in range(a2 - n11 + 1):
            n01 = np.arange(a2 - n11 - n10 + 1)
            n00 = a2 - n11 - n10 - n01
            out.append(np.stack([n00,
                                 np.full_like(n01, n10),
                                 n01,
                                 np.full_like(n01, n11)]))
    return np.concatenate(out, axis=1)


def binom(n, k):
    """C(n,k) поэлементно, 0 при k > n."""
    n = np.asarray(n, dtype=float)
    if k == 0:
        return np.ones_like(n)
    if k == 1:
        return np.where(n >= 1, n, 0.0)
    if k == 2:
        return np.where(n >= 2, n * (n - 1) / 2, 0.0)
    if k == 3:
        return np.where(n >= 3, n * (n - 1) * (n - 2) / 6, 0.0)
    raise ValueError


def g_vec(nc, th):
    """G для набора композиций nc = [n00,n10,n01,n11] (массивы)."""
    u = np.exp(-1j * th)
    ps = psis(th)
    n00, n10, n01, n11 = nc
    a2 = n00 + n10 + n01 + n11

    def A(M):
        return sum(ps[m] * u ** m * binom(M, m) for m in (1, 2, 3))

    def pq(nx, t):
        acc = 0.0 + 0.0j
        for tx in range(0, 4 - t):
            if tx + t < 1 or tx + t > 3:
                continue
            acc = acc + binom(nx, tx) * u ** tx * ps[tx + t]
        return acc

    one = (1.0 + u)
    G = one ** a2 + one ** (n00 + n01) * A(n10 + n11) \
        + one ** (n00 + n10) * A(n01 + n11)
    B = 0.0 + 0.0j
    for t11 in range(0, 4):
        B = B + binom(n11, t11) * u ** t11 * pq(n10, t11) * pq(n01, t11)
    G = G + one ** n00 * B
    return G


def maxq(a2, th):
    nc = comps(a2)
    g = np.abs(g_vec(nc, th))
    lq = np.log(np.maximum(g, 1e-300)) - a2 * math.log(2.0)
    i = int(np.argmax(lq))
    return lq[i], nc[:, i]


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    # контроль: явная формула против общей из cancelclosed
    from cancelclosed import g_closed
    th = K4 / 0.55
    print('КОНТРОЛЬ: явная формула a1=2 против общей')
    rs = np.random.default_rng(3)
    worst = 0.0
    for _ in range(8):
        v = rs.integers(0, 5, size=4)
        nc = np.array(v, dtype=int).reshape(4, 1)
        g1 = g_vec(nc, th)[0]
        g2 = g_closed(2, {0: int(v[0]), 1: int(v[1]), 2: int(v[2]),
                          3: int(v[3])}, th)
        worst = max(worst, abs(g1 - g2) / max(abs(g2), 1e-12))
    print(f'  худшее относительное расхождение {worst:.2e}')
    print()
    for hb in (0.55, 0.62):
        th = K4 / hb
        print('=' * 72)
        print(f'ТОЧНОЕ s(a_2) = -ln(max q)/a_2 ПРИ a1 = 2, hbar = {hb}')
        print(f"{'a_2':>5}{'max q':>14}{'s = -ln q/a_2':>16}"
              f"{'аргмакс (n00,n10,n01,n11)':>30}")
        for a2 in (2, 3, 4, 5, 6, 7, 10, 15, 20, 30, 50, 80, 120, 200, 300):
            lq, arg = maxq(a2, th)
            s = -lq / a2
            qs = (f'{math.exp(lq):.4e}' if lq > -700 else '~0')
            print(f'{a2:>5}{qs:>14}{s:>16.5f}'
                  f'{str(tuple(int(x) for x in arg)):>30}', flush=True)
        print(f'  нужно для гибели: ln 2 = {math.log(2):.5f}')
        print()


if __name__ == '__main__':
    main()
