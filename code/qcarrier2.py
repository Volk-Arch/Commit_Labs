#!/usr/bin/env python3
"""ВЕРСИЯ 2: пары прямо в окне, калибровка гладкая.

`qcarrier.py` упёрся в статистику: при больших N пар в окне x не
хватило, калибровка по 300 корзинам стала шумной, и бюджет пошёл ВНИЗ
с ростом N (9.93 -> 8.91) — что невозможно и выдало проблему.

Правки: (1) собирать пары сразу в окне x; (2) калибровка — гладкая
подгонка x по ln(наблюдаемой) полиномом, а не корзинами, чтобы шум
калибровки не подмешивался; (3) пар в разы больше.

Контроль 1 прошлого прогона в силе: V_cp ~ N^0.984, N_both ~ N^1.001 —
оба носителя ЛИНЕЙНЫ по N, значит «кромка» относится к ФОРМЕ
зависимости от разноса, а не к размеру носителя.

  python3 qcarrier2.py
"""
import math

import numpy as np

from qcarrier import sprinkle4


def harvest(n, rs, npair, xlo, xhi, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    t, X = P[:, 0], P[:, 1:]
    tau = np.sqrt(np.maximum(t.astype(np.float64) ** 2
                             - (X.astype(np.float64) ** 2).sum(1), 0.0))
    mid = (tau > 0.35 * T) & (tau < 0.65 * T)
    order = np.nonzero(mid)[0]
    order = order[np.argsort(tau[order])]
    ta = tau[order]
    rows = []
    tries, lim = 0, npair * 400
    while len(rows) < npair and tries < lim:
        tries += 1
        a = int(order[int(rs.integers(len(order)))])
        lo = np.searchsorted(ta, tau[a] * (1 - agew))
        hi = np.searchsorted(ta, tau[a] * (1 + agew))
        if hi - lo < 2:
            continue
        b = int(order[int(rs.integers(lo, hi))])
        if a == b:
            continue
        dt = float(t[b]) - float(t[a])
        dx2 = float(((X[b] - X[a]) ** 2).sum())
        if dt * dt > dx2:
            continue
        xv = math.sqrt(dx2 - dt * dt) / (0.5 * (tau[a] + tau[b]))
        if not (xlo <= xv < xhi):
            continue
        dta, dtb = t[a] - t, t[b] - t
        da = ((X - X[a]) ** 2).sum(1)
        db = ((X - X[b]) ** 2).sum(1)
        pa = (dta > 0) & (dta * dta > da)
        pb = (dtb > 0) & (dtb * dtb > db)
        fa = (dta < 0) & (dta * dta > da)
        fb = (dtb < 0) & (dtb * dtb > db)
        rows.append((xv, int((pa & pb).sum()),
                     int((~pa & ~fa & ~pb & ~fb).sum()),
                     int((pa | pb).sum())))
    return np.array(rows, dtype=np.float64)


def budget_smooth(D, col, deg=3):
    half = len(D) // 2
    cal, ev = D[:half], D[half:]
    lc = np.log(np.maximum(cal[:, col], 1.0))
    le = np.log(np.maximum(ev[:, col], 1.0))
    c = np.polyfit(lc - lc.mean(), cal[:, 0], deg)
    xh = np.polyval(c, le - lc.mean())
    err = float(np.std(xh - ev[:, 0], ddof=1))
    xm = float(ev[:, 0].mean())
    return xm / err, len(ev), err


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(1618033)
    XLO, XHI = 0.50, 0.65
    plan = [(20000, 9000), (70000, 8000), (250000, 6000), (700000, 4000)]
    data = {}
    for n, npr in plan:
        D = harvest(n, rs, npr, XLO, XHI)
        data[n] = D
        print(f'  N = {n:>7}: пар {len(D):>5}, <V_cp> = {D[:,1].mean():>9.0f}, '
              f'<N_both> = {D[:,2].mean():>9.0f}', flush=True)
    print()
    print(f'БЮДЖЕТЫ (окно x = {XLO}-{XHI}, калибровка гладкая)')
    print(f"{'N':>8}{'<V_cp>':>10}{'V_cp':>9}{'N_both':>9}{'V_un':>9}")
    res = {1: [], 2: [], 3: []}
    for n in sorted(data):
        D = data[n]
        line = f'{n:>8}{D[:,1].mean():>10.0f}'
        for col in (1, 2, 3):
            b, m, e = budget_smooth(D, col)
            res[col].append((D[:, col].mean(), b, m))
            line += f'{b:>9.2f}'
        print(line, flush=True)
    print()
    for col, nm in ((1, 'V_cp'), (2, 'N_both'), (3, 'V_un')):
        R = np.array(res[col])
        err = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
        A = np.vstack([np.log(R[:, 0]), np.ones(len(R))]).T
        W = np.diag(1.0 / (err / R[:, 1]) ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        be = cov @ (A.T @ W @ np.log(R[:, 1]))
        se = np.sqrt(np.diag(cov))
        rr = (np.log(R[:, 1]) - A @ be) / (err / R[:, 1])
        print(f'{nm:>8}: показатель {be[0]:.4f} ± {se[0]:.4f}  | '
              f'1/4: {abs(be[0]-0.25)/se[0]:.1f}с, '
              f'3/8: {abs(be[0]-0.375)/se[0]:.1f}с, '
              f'1/2: {abs(be[0]-0.5)/se[0]:.1f}с | остатки '
              + ', '.join(f'{v:+.1f}' for v in rr))
    print()
    print('КОНТРОЛЬ: перемешать x — бюджет обязан упасть к ~1')
    D = data[sorted(data)[1]].copy()
    D[:, 0] = rs.permutation(D[:, 0])
    for col, nm in ((1, 'V_cp'), (2, 'N_both')):
        b, m, e = budget_smooth(D, col)
        print(f'  {nm}: {b:.3f}')


if __name__ == '__main__':
    main()
