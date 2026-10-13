#!/usr/bin/env python3
"""ТОЧНАЯ ПОВЕРХНОСТЬ ОТКЛИКА — без разложения вовсе.

Плотность набора задана как `n = Poisson(L^3 * int a^4 d eta)`, то есть
`rho = 1` и `l = 1`. Тогда `tau² = KFAC sqrt(C_1)`, и при `C_1 = 500`
выходит `tau² = 62`, `tau ~ 7.9` при `eta_0 = 10`. **Отношение
`tau/eta_0` у набора всегда было около 0.8**, то есть параметр
разложения порядка единицы, а не мал. Защита `tau² R_uu <= 0.8` звучала
охранительно, но 0.8 и есть единица.

Поэтому поправки первого и второго порядка — неверный путь. Для
конформно-плоских метрик причинность ТОЧНО плоская, вес `w = a^4`, и

    1 + X_2 = < w_x tau_xq^4 w_y > / ( c_2 tau^4 <w_x>^2 )
    1 + X_3 = < w_x tau_xq^4 w_y tau_yq^4 w_z > / ( c_3 tau^8 <w_x>^3 )

верны при ЛЮБОМ `tau`. Значит отклик считается точно, и разложение не
нужно ни в каком порядке.

ЗДЕСЬ предсказание считается для каждой ячейки `v² x C_1` каждого плеча
из ИСТИННОЙ метрики — без подгонки, без компонент, без разложения — и
сравнивается с замером.

КОНТРОЛЬ: плоское плечо обязано дать нуль при всех `tau` и `v`.

  python3 exact.py [--n 700000]
"""
import argparse
import math

import numpy as np

from pencil import KFAC, a_of, c_k, IC, IX2, IX3, IV, IM
from refit import load_all
from x3prof import rest_diamond, sub_sample

ETA0 = 10.0
VBB = (0.0, 0.45, 0.65, 0.85001)
CBB = (200.0, 400.0, 650.0, 900.0001)


def exact_x(fam, par, v, tau, mid, n, rs, d=4):
    """Точные <X_2>, <X_3> для интервала (v, tau) с серединой mid."""
    gam = 1.0 / math.sqrt(1.0 - v * v)

    def eta(pt):
        return mid + ((pt[:, 0] - 0.5 * tau) + v * pt[:, 1]) * gam

    def wgt(pt):
        a = a_of(eta(pt), fam, par, ETA0)
        return np.maximum(a, 1e-9) ** 4

    s0, w0 = rest_diamond(rs, n, tau, d)
    x = np.column_stack([s0, w0])
    q = np.zeros((n, d)); q[:, 0] = tau
    wx = wgt(x)
    y, txq, ok1 = sub_sample(x, q, rs, d)
    wy = wgt(y)
    num2 = (wx[ok1] * txq ** d * wy).mean()
    x2 = num2 / (c_k(2) * tau ** d * wx.mean() ** 2) - 1.0
    z, tyq, ok2 = sub_sample(y, q[ok1], rs, d)
    wz = wgt(z)
    num3 = (wx[ok1][ok2] * txq[ok2] ** d * wy[ok2]
            * tyq ** d * wz).mean()
    x3 = num3 / (c_k(3) * tau ** (2 * d) * wx.mean() ** 3) - 1.0
    return x2, x3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=700000)
    ap.add_argument('--blocks', type=int, default=3)
    g = ap.parse_args()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(271828)
    arms = load_all()
    print(f'точек {g.n} x {g.blocks} блоков на ячейку; l = 1, '
          f'tau² = KFAC sqrt(C_1)')
    print()
    for fam in ('flat', 'lin', 'quad', 'exp', 'root'):
        A, sd, par = arms[fam]
        print(f'--- {fam} (par = {par}) ---')
        print(f"{'v²':>12}{'C_1':>12}{'tau':>7}{'tau/eta_0':>11}"
              f"{'замер X_2':>20}{'точно X_2':>12}{'сигм':>7}"
              f"{'замер X_3':>20}{'точно X_3':>12}{'сигм':>7}")
        for i in range(len(VBB) - 1):
            for j in range(len(CBB) - 1):
                m = ((A[:, IV] >= VBB[i]) & (A[:, IV] < VBB[i + 1])
                     & (A[:, IC] >= CBB[j]) & (A[:, IC] < CBB[j + 1]))
                if m.sum() < 150:
                    continue
                v2 = A[m, IV].mean(); c1 = A[m, IC].mean()
                mid = A[m, IM].mean()
                tau = math.sqrt(KFAC * math.sqrt(c1))
                us = np.unique(sd[m])
                out = []
                for col in (IX2, IX3):
                    jk = np.array([A[m & (sd != u), col].mean() for u in us])
                    se = math.sqrt((len(us) - 1) / len(us)
                                   * ((jk - jk.mean()) ** 2).sum())
                    out.append((A[m, col].mean(), se))
                rr = np.array([exact_x(fam, par, math.sqrt(v2), tau, mid,
                                       g.n, rs) for _ in range(g.blocks)])
                ex2, ex3 = rr[:, 0].mean(), rr[:, 1].mean()
                ee2 = rr[:, 0].std(ddof=1) / math.sqrt(g.blocks)
                ee3 = rr[:, 1].std(ddof=1) / math.sqrt(g.blocks)
                s2 = abs(out[0][0] - ex2) / math.sqrt(out[0][1] ** 2 + ee2 ** 2)
                s3 = abs(out[1][0] - ex3) / math.sqrt(out[1][1] ** 2 + ee3 ** 2)
                print(f'{v2:>12.3f}{c1:>12.0f}{tau:>7.2f}{tau/ETA0:>11.2f}'
                      f'{f"{out[0][0]:+.5f}+-{out[0][1]:.5f}":>20}'
                      f'{ex2:>+12.5f}{s2:>7.1f}'
                      f'{f"{out[1][0]:+.5f}+-{out[1][1]:.5f}":>20}'
                      f'{ex3:>+12.5f}{s3:>7.1f}', flush=True)
        print()


if __name__ == '__main__':
    main()
