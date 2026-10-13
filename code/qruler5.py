#!/usr/bin/env python3
"""БЕЗ КОРЗИН: разброс ОЦЕНКИ при известной истине.

Две предыдущие версии мерили ШИРИНУ СВОИХ КОРЗИН, а не разрешение:
  * `qruler3/4` брали «условный разброс x внутри квантильной корзины по
    u». Восемь корзин задают ПОЛ, который не улучшается с плотностью —
    отсюда ложное «плато»;
  * вторая линейка: r бралась логарифмически по размаху в 100 раз,
    корзина по W шириной 2.4 раза, и «разброс» 0.29 есть логарифм
    ширины корзины. Отсюда плато 3.5 вместо прогноза sqrt(W) = 2.7...24.

Остатки и контроль показателя это выдали (+6.8, −11.1, −4.8, +10.2 и
W^0.001 против прогноза 0.5).

ПРАВИЛЬНО: фиксировать ИСТИНУ и мерить разброс ОЦЕНКИ. Для фазы k*d
ошибка есть k*delta_d, где delta_d — ошибка оценщика при известном d.
Корзины не нужны.

  * вторая линейка: оценка r = W/(2 lambda) при точно заданном r;
  * первая линейка: оценка x = xbar(u) по калибровке, построенной на
    ОТДЕЛЬНОЙ половине данных, затем разброс оценки в узком окне
    истинной x.

  python3 qruler5.py
"""
import math

import numpy as np

from qruler4 import harvest_tag


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(271828)
    print('=== ВТОРАЯ ЛИНЕЙКА: разброс оценки при известном r ===')
    T, lam = 1.0, 20000.0
    print(f"{'r':>10}{'<W>':>10}{'2 lambda r':>12}{'отн. ошибка':>14}"
          f"{'бюджет':>9}{'прогноз sqrt(W)':>17}")
    rows = []
    for r in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        est = []
        for _ in range(3000):
            nch = rs.poisson(lam * T)
            tc = rs.random(nch) * T
            tx = 0.3 + 0.4 * rs.random()
            w = int(np.sum(np.abs(tc - tx) < r))
            est.append(w / (2.0 * lam))
        e = np.array(est)
        rel = float(e.std(ddof=1) / r)
        wm = 2 * lam * r
        rows.append((wm, 1.0 / rel))
        print(f'{r:>10.4f}{e.mean()*2*lam:>10.1f}{wm:>12.1f}'
              f'{rel:>14.5f}{1/rel:>9.2f}{math.sqrt(wm):>17.2f}')
    O = np.array(rows)
    A = np.vstack([np.log(O[:, 0]), np.ones(len(O))]).T
    be = np.linalg.lstsq(A, np.log(O[:, 1]), rcond=None)[0]
    resid = np.log(O[:, 1]) - A @ be
    print(f'  ЗАКОН: бюджет ~ W^{be[0]:.4f}, коэффициент '
          f'{math.exp(be[1]):.4f}  (прогноз W^0.5, коэффициент 1)')
    print('  остатки (лог):', ', '.join(f'{v:+.4f}' for v in resid))
    print()
    print('ПЕРЕСЧЁТ НА ПОЛНЫЙ РЕСУРС. W = 2 lambda r = 2 (r/tau) * L,')
    print('где L — длина цепи. Замер цепи (контроль 4 прошлого прогона):')
    print('  L/N^0.25 = 1.69, 1.96, 1.94, 1.89 при N = 500...4000,')
    print('  то есть L ~ 1.9 N^{1/4}. Значит W ~ 3.8 x N^{1/4} и')
    print(f'  бюджет второй линейки ~ sqrt(W) ~ N^{{1/8}} = N^0.125')
    print()
    print('=== ПЕРВАЯ ЛИНЕЙКА: разброс оценки при известной x ===')
    parts = {}
    for n, k in ((1000, 6), (2000, 5), (4000, 3), (8000, 2)):
        ps = [harvest_tag(n, rs, 40000) for _ in range(k)]
        parts[n] = np.vstack(ps)
        print(f'  N = {n}: пар {len(parts[n])}', flush=True)
    print()
    print(f"{'N':>7}{'<V_общ>':>10}{'окно x':>12}{'пар':>8}"
          f"{'отн. ошибка':>14}{'бюджет':>9}")
    for xlo, xhi in ((0.45, 0.55), (0.6, 0.7)):
        rows = []
        for n in sorted(parts):
            D = parts[n]
            half = len(D) // 2
            cal, ev = D[:half], D[half:]
            # калибровка: средняя x по тонким корзинам u, на ОТДЕЛЬНОЙ половине
            nb = 400
            qs = np.quantile(cal[:, 1], np.linspace(0, 1, nb + 1))
            idx = np.clip(np.searchsorted(qs, cal[:, 1]) - 1, 0, nb - 1)
            xbar = np.zeros(nb)
            cnt = np.zeros(nb)
            np.add.at(xbar, idx, cal[:, 0])
            np.add.at(cnt, idx, 1.0)
            good = cnt >= 10
            xbar[good] /= cnt[good]
            xbar[~good] = np.nan
            # оценка на второй половине
            ie = np.clip(np.searchsorted(qs, ev[:, 1]) - 1, 0, nb - 1)
            xh = xbar[ie]
            s = (ev[:, 0] >= xlo) & (ev[:, 0] < xhi) & np.isfinite(xh)
            if s.sum() < 500:
                continue
            xm = float(ev[s, 0].mean())
            err = float(np.std(xh[s] - ev[s, 0], ddof=1))
            rows.append((float(ev[s, 2].mean()), xm / err, int(s.sum())))
            print(f'{n:>7}{ev[s,2].mean():>10.0f}{f"{xlo}-{xhi}":>12}'
                  f'{int(s.sum()):>8}{err/xm:>14.5f}{xm/err:>9.2f}',
                  flush=True)
        if len(rows) >= 3:
            R = np.array(rows)
            errb = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
            A = np.vstack([np.log(R[:, 0]), np.ones(len(R))]).T
            W = np.diag(1.0 / (errb / R[:, 1]) ** 2)
            cov = np.linalg.inv(A.T @ W @ A)
            be = cov @ (A.T @ W @ np.log(R[:, 1]))
            se = np.sqrt(np.diag(cov))
            rr = (np.log(R[:, 1]) - A @ be) / (errb / R[:, 1])
            print(f'      -> бюджет ~ V^{be[0]:.4f} ± {se[0]:.4f}, '
                  f'остатки: ' + ', '.join(f'{v:+.1f}' for v in rr))
            print()


if __name__ == '__main__':
    main()
