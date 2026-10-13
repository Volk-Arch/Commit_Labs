#!/usr/bin/env python3
"""ВЕРСИЯ 3: остаток по ПОЛНОМУ размаху. Пол незнания равен единице.

Корень трёх ошибок найден: я нормировал на величину, выбранную САМИМ
СОБОЙ — сначала ширину квантильной корзины, потом ширину корзины по W,
потом ширину окна оценки. В последнем случае пол равен (средняя x)/
(разброс x в окне) = 0.575/0.0433 = 13.3, и именно это «измерилось»;
перемешивание дало 13.527 против 13.52, то есть информации в замере не
было вовсе.

Задним числом это хоронит и числа `qruler5.py`: окно 0.45-0.55 даёт пол
17.3, а отчитано 9.47...17.97 — на уровне незнания или ниже.

ПРАВИЛЬНАЯ МЕТРИКА. Калибровка и остаток берутся по ПОЛНОМУ размаху x.
Тогда
    SNR = std(x) / std(x_оценка - x)
и при полном незнании SNR = 1 РОВНО, независимо от любых моих выборов.
Число различимых значений (радиан) есть mean(x)/std(остатка).

КОНТРОЛЬ: перемешать x — SNR обязан дать 1.00.

  python3 qcarrier3.py
"""
import math

import numpy as np

from qcarrier2 import harvest


def snr(D, col, deg=4):
    half = len(D) // 2
    cal, ev = D[:half], D[half:]
    mu = np.log(np.maximum(cal[:, col], 1.0)).mean()
    lc = np.log(np.maximum(cal[:, col], 1.0)) - mu
    le = np.log(np.maximum(ev[:, col], 1.0)) - mu
    c = np.polyfit(lc, cal[:, 0], deg)
    r = np.polyval(c, le) - ev[:, 0]
    sd = float(np.std(r, ddof=1))
    return (float(np.std(ev[:, 0], ddof=1)) / sd,
            float(ev[:, 0].mean()) / sd, len(ev))


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(2236068)
    XLO, XHI = 0.10, 1.60
    plan = [(20000, 12000), (70000, 10000), (250000, 7000), (700000, 4500)]
    data = {}
    for n, npr in plan:
        D = harvest(n, rs, npr, XLO, XHI)
        data[n] = D
        print(f'  N = {n:>7}: пар {len(D):>6}, <V_cp> = {D[:,1].mean():>9.0f}, '
              f'<N_both> = {D[:,2].mean():>9.0f}', flush=True)
    print()
    print('КОНТРОЛЬ: перемешать x — SNR обязан дать 1.00')
    D = data[sorted(data)[1]].copy()
    D[:, 0] = rs.permutation(D[:, 0])
    for col, nm in ((1, 'V_cp'), (2, 'N_both')):
        s, rad, m = snr(D, col)
        print(f'  {nm:>7}: SNR = {s:.4f}, радиан = {rad:.3f}')
    print()
    print(f'ПОЛНЫЙ РАЗМАХ x = {XLO}-{XHI}')
    print(f"{'N':>8}{'<V_cp>':>10}" + ''.join(f'{t:>18}' for t in
          ('V_cp SNR/рад', 'N_both SNR/рад', 'V_un SNR/рад')))
    res = {1: [], 2: [], 3: []}
    for n in sorted(data):
        D = data[n]
        line = f'{n:>8}{D[:,1].mean():>10.0f}'
        for col in (1, 2, 3):
            s, rad, m = snr(D, col)
            res[col].append((D[:, col].mean(), rad, m))
            line += f'{s:>9.2f}/{rad:<8.2f}'
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
              f'0 (плато): {abs(be[0])/se[0]:.1f}с, '
              f'1/4: {abs(be[0]-0.25)/se[0]:.1f}с, '
              f'3/8: {abs(be[0]-0.375)/se[0]:.1f}с, '
              f'1/2: {abs(be[0]-0.5)/se[0]:.1f}с')
        print(f'{"":>8}  остатки: ' + ', '.join(f'{v:+.1f}' for v in rr))


if __name__ == '__main__':
    main()
