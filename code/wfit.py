#!/usr/bin/env python3
"""ИЗМЕРЕНИЕ `w` ОТ НАЧАЛА ДО КОНЦА, из чистого порядка.

Всё подтверждено тремя контролями (`wsprink.py`): плоское даёт нуль,
шум идёт как `1/sqrt(C_1)`, а прямой сприклинг совпал с квадратурой на
шести конфигурациях (0.2...1.9 сигмы). Достижимая точность
`sigma(w) = 0.0095` при `tau/eta_0 = 1.5`.

Разворачивать набор не нужно: точки сыплются ПРЯМО в интервал с
физической плотностью. При фиксированных концах внутренность остаётся
пуассоновской, поэтому это то же распределение, что поиск интервала
внутри большого сприклинга, — но на два порядка дешевле. Машинерия
набора нужна, когда интервал не контролируешь; здесь контролируем.

ЧТО ДЕЛАЕТСЯ. Для метрики с известным `w` сыплются реализации при трёх
`tau`, и `w` подгоняется сопоставлением вектора `(X_2(tau_1), X_2(tau_2),
X_2(tau_3))` с кривой отклика, посчитанной квадратурой. Три `tau` берутся
потому, что отклик не монотонен: стационарная точка ползёт с `tau`
(0.22, 0.30, 0.43 при 0.6, 1.0, 1.5), и несколько `tau` снимают
двузначность.

КОНТРОЛИ:
  1. три РАЗНЫХ истинных `w` обязаны дать три разных подогнанных,
     каждое в пределах своей ошибки от истины;
  2. плоская метрика (`p = 0`) не принадлежит семейству ни при каком
     `w`, и подгонка обязана показать это плохим `chi²`;
  3. ошибка, полученная складным ножом по реализациям, обязана
     совпасть с предсказанной `шум/|dX/dw|/sqrt(n)`.

  python3 wfit.py [--reps 1500]
"""
import argparse
import math
import time

import numpy as np

from pencil import c_k
from vcal import ball

TAUS = (0.6, 1.0, 1.5)
C1 = 500
WGRID = np.array([-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.80, 1.00, 1.30])


def make_cdf(p, tau, eta0=1.0, d=4, M=3000):
    e = np.linspace(eta0 - 0.5 * tau, eta0 + 0.5 * tau, M)
    R = np.minimum(e - (eta0 - 0.5 * tau), (eta0 + 0.5 * tau) - e)
    dens = (e / eta0) ** (4.0 * p) * R ** (d - 1)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                                           * np.diff(e))])
    return e, cdf, R


def one(e, cdf, tau, n, rs, eta0=1.0, d=4):
    t = np.interp(rs.uniform(0, cdf[-1], n), cdf, e)
    t.sort()
    Rt = np.minimum(t - (eta0 - 0.5 * tau), (eta0 + 0.5 * tau) - t)
    X = ball(rs, n, d - 1, Rt)
    sq = np.einsum('ij,ij->i', X, X)
    dt = t[None, :] - t[:, None]
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    A = (dt > 0) & (dt * dt > d2)
    return A.sum() / (c_k(2) * n * (n - 1.0)) - 1.0


def batch(w, tau, reps, rs, flat=False):
    p = 0.0 if flat else 2.0 / (1.0 + 3.0 * w)
    e, cdf, _ = make_cdf(p, tau)
    return np.array([one(e, cdf, tau, C1, rs) for _ in range(reps)])


def response(rs, reps):
    """Кривая отклика по квадратуре-эквиваленту: много реализаций."""
    R = np.zeros((len(WGRID), len(TAUS)))
    for i, w in enumerate(WGRID):
        for j, t in enumerate(TAUS):
            R[i, j] = batch(w, t, reps, rs).mean()
    return R


def fit_w(obs, R):
    """Наименьшее chi² по сетке с параболическим уточнением."""
    d = ((R - obs[None, :]) ** 2).sum(axis=1)
    k = int(np.argmin(d))
    if 0 < k < len(WGRID) - 1:
        y0, y1, y2 = d[k - 1], d[k], d[k + 1]
        den = y0 - 2 * y1 + y2
        sh = 0.5 * (y0 - y2) / den if abs(den) > 1e-30 else 0.0
        sh = max(-1.0, min(1.0, sh))
        return WGRID[k] + sh * (WGRID[k + 1] - WGRID[k]), d[k]
    return WGRID[k], d[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=1500)
    ap.add_argument('--rref', type=int, default=3000)
    g = ap.parse_args()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    t0 = time.time()
    rs = np.random.default_rng(99991)
    print(f'C_1 = {C1}, tau = {list(TAUS)}, реализаций на точку '
          f'{g.reps}, на опорную кривую {g.rref}')
    print()
    print('опорная кривая отклика...', flush=True)
    R = response(np.random.default_rng(555), g.rref)
    print(f'  готова за {time.time()-t0:.0f} с')
    print()
    print('=' * 78)
    print('КОНТРОЛЬ 1: три разных истинных `w`')
    print(f"{'истина':>8}{'подогнано':>22}{'сигм':>7}"
          f"{'предсказанная ошибка':>22}")
    for wt in (0.00, 1.0 / 3.0, 1.00):
        V = np.column_stack([batch(wt, t, g.reps, rs) for t in TAUS])
        obs = V.mean(axis=0)
        wh, _ = fit_w(obs, R)
        # складной нож по блокам реализаций
        nb = 12
        sz = len(V) // nb
        jk = []
        for b in range(nb):
            m = np.ones(len(V), bool); m[b * sz:(b + 1) * sz] = False
            jk.append(fit_w(V[m].mean(axis=0), R)[0])
        jk = np.array(jk)
        se = math.sqrt((nb - 1) / nb * ((jk - jk.mean()) ** 2).sum())
        # предсказание: шум/|dX/dw|/sqrt(n) по самому чувствительному tau
        der = abs(np.gradient(R[:, 2], WGRID))
        d13 = np.interp(wt, WGRID, der)
        pred = V[:, 2].std(ddof=1) / max(d13, 1e-9) / math.sqrt(len(V))
        print(f'{wt:>8.3f}{f"{wh:+.4f}+-{se:.4f}":>22}'
              f'{abs(wh-wt)/max(se,1e-9):>7.1f}{pred:>22.4f}', flush=True)
    print()
    print('=' * 78)
    print('КОНТРОЛЬ 2: плоская метрика — вне семейства при любом `w`')
    V = np.column_stack([batch(0.0, t, g.reps, rs, flat=True)
                         for t in TAUS])
    obs = V.mean(axis=0)
    wh, dmin = fit_w(obs, R)
    er = V.std(axis=0, ddof=1) / math.sqrt(len(V))
    chi = (((R[int(np.argmin(((R - obs[None, :]) ** 2).sum(axis=1)))]
             - obs) / er) ** 2).sum()
    print(f'  наблюдено {np.round(obs, 5)}, ошибки {np.round(er, 5)}')
    print(f'  ближайшее `w` = {wh:+.3f}, chi² = {chi:.1f} на '
          f'{len(TAUS)-1} степени'
          f' -> {"ОТВЕРГНУТО, как и должно" if chi > 15 else "НЕ отвергнуто"}')
    print(f'\nвсего {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
