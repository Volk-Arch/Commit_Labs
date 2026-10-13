#!/usr/bin/env python3
"""ПРАВИЛЬНЫЙ ПЛАН: переподгонка на выведенных профилях A(v), B(v).

Допущение старого плана `X_2 = V2CAL (R_etaeta + R_xx v^2)/(1-v^2) tau^2`
опровергнуто точным интегралом (`vcal2.py`, промах 9.5 сигм на
экспоненциальном плече, где `R_etaeta = 0` точно). Верное выражение
выведено из моментов ромба (`vcal.py`, `vcal3.py`) и проверено двумя
независимыми сэмплерами, совпадение 0.3%:

    X_2 = [ A(v²) R_etaeta + B(v²) R_xx ] tau²,   tau² = KFAC sqrt(C_1) l²

Истинное отношение `B/A` МЕНЯЕТ ЗНАК около `v² = 0.5` (от -0.751 при
покое до +0.681 при 0.85), тогда как старый план полагал его равным
`v²`, то есть монотонно растущим от нуля. Размах истинного 1.43 против
0.85 — рычага в правильном плане БОЛЬШЕ. И при `v² ~ 0.45` коэффициент
при `R_xx` обращается в нуль: там интервал мерит чистую `R_etaeta`.

Проверка здесь двойная и жёсткая:
  КОНТРОЛЬ 1 — половина плоского плеча как семейство: нули.
  КОНТРОЛЬ 2 — все четыре плеча обязаны попасть в СВОИ истины. Старый
    план проходил только на линейном (которое к его ошибке слепо).

  python3 pwfix.py
"""
import math

import numpy as np

from pencil import KFAC, truth, IC, IX2, IV
from pw import load, nui_cols

# выведено в `vcal3.py`: v², A (при R_etaeta), B (при R_xx)
VTAB = np.array([0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85])
ATAB = np.array([-0.02539, -0.02877, -0.03341, -0.04102,
                 -0.05382, -0.08255, -0.13395])
BTAB = np.array([+0.01908, +0.01566, +0.01129, +0.00335,
                 -0.00941, -0.03828, -0.09121])


def plan(arms, order=1):
    A = np.vstack([a for a, _, _ in arms])
    sd = np.concatenate([s + 100000 * i
                         for i, (_, s, _) in enumerate(arms)])
    t2 = KFAC * np.sqrt(A[:, IC])          # tau²/l²
    av = np.interp(A[:, IV], VTAB, ATAB)
    bv = np.interp(A[:, IV], VTAB, BTAB)
    g1, g2 = av * t2, bv * t2
    cols, off = [], 0
    for a, _, curved in arms:
        if curved:
            m = np.zeros(len(A)); m[off:off + len(a)] = 1.0
            cols += [g1 * m, g2 * m]
        off += len(a)
    return A, np.column_stack(cols + nui_cols(A, order)), A[:, IC], sd


def do(arms, names, truths, label):
    A, D, w, sd = plan(arms)
    y = A[:, IX2]

    def sol(mask=None):
        DD, yy, ww = (D, y, w) if mask is None else (D[mask], y[mask], w[mask])
        Dw = DD * ww[:, None]
        return np.linalg.solve(DD.T @ Dw, Dw.T @ yy)

    full = sol()
    us = np.unique(sd)
    jk = np.array([sol(sd != u) for u in us])
    k = len(us)
    M = D.T @ (D * w[:, None])
    print(f'  {label}: обусловленность {np.linalg.cond(M):.1f}, '
          f'параметров {D.shape[1]}')

    def se(x):
        return math.sqrt((k - 1) / k * ((x - x.mean()) ** 2).sum())

    uu = np.array([1.0, 3.0])
    for i, (nm, tr) in enumerate(zip(names, truths)):
        al, be = full[2 * i], full[2 * i + 1]
        ja, jb = jk[:, 2 * i], jk[:, 2 * i + 1]
        C = np.array([[se(ja) ** 2, 0.0], [0.0, se(jb) ** 2]])
        pr = np.column_stack([ja, jb])
        C = (k - 1) / k * (pr - pr.mean(axis=0)).T @ (pr - pr.mean(axis=0))
        rho, erho = al + 3 * be, math.sqrt(uu @ C @ uu)
        dt = tr[0] + 3 * tr[1]
        print(f'    {nm}:')
        print(f'      R_etaeta l² = {al:+.5f} +- {math.sqrt(C[0,0]):.5f}'
              f'   (истина {tr[0]:+.5f})  '
              f'{abs(al-tr[0])/max(math.sqrt(C[0,0]),1e-12):.1f} сигм')
        print(f'      R_xx l²     = {be:+.5f} +- {math.sqrt(C[1,1]):.5f}'
              f'   (истина {tr[1]:+.5f})  '
              f'{abs(be-tr[1])/max(math.sqrt(C[1,1]),1e-12):.1f} сигм')
        print(f'      rho         = {rho:+.5f} +- {erho:.5f}'
              f'   (истина {dt:+.5f})  '
              f'от нуля {abs(rho)/max(erho,1e-12):.1f} сигм')
        if abs(rho) > 3 * erho and abs(dt) > 1e-9:
            wv = (al - be) / rho
            gw = np.array([1.0 / rho - (al - be) / rho ** 2,
                           -1.0 / rho - 3 * (al - be) / rho ** 2])
            ew = math.sqrt(gw @ C @ gw)
            wt = (tr[0] - tr[1]) / dt
            print(f'      w           = {wv:+.4f} +- {ew:.4f}'
                  f'   (истина {wt:+.4f})  '
                  f'{abs(wv-wt)/max(ew,1e-12):.1f} сигм')


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load()
    fk = [k for k in arms if k[0] == 'flat'][0]
    curved = [k for k in arms if k[0] != 'flat']
    Af, sdf = arms[fk]
    us = np.unique(sdf); h1 = np.isin(sdf, us[::2])
    print('КОНТРОЛЬ 1: половина плоского объявлена семейством')
    do([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
       ['половина плоского'], [(0.0, 0.0, 0.0)], 'контроль')
    print()
    print('КОНТРОЛЬ 2: все четыре плеча против СВОИХ истин')
    do([(arms[k][0], arms[k][1], True) for k in curved]
       + [(Af, sdf, False)],
       [f'{k[0]} {k[1]}' for k in curved],
       [truth(*k) for k in curved], 'совместно')


if __name__ == '__main__':
    main()
