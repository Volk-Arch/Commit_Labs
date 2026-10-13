#!/usr/bin/env python3
"""РАЗБОР РАННЕЙ ВСЕЛЕННОЙ: плотность и возраст из счёта.

МОДЕЛЬ. Для излучения `R_uu = (3+v²)/(eta0⁴(1-v²))`, поэтому

    X_2 = lambda * g(v², C_1),     g = sqrt(C_1) (3+v²)/(1-v²)
    lambda = v_2 * k / eta0⁴,      k = sqrt(24/pi)

Форма `g` задана ЖЁСТКО физикой излучения: ни наклон, ни размер не
подгоняются, свободна только амплитуда `lambda` — по одной на эпоху.

ЧТО ИЗ НЕЁ ДОСТАЁТСЯ, и всё это сравнивается с истиной:

    R_uu l² (сопутствующий) = 3 lambda / (v_2 k)     против  3/eta0⁴
    плотность rho = R_uu/(8 pi)                      против  3/(32 pi t²)
    ВОЗРАСТ ВСЕЛЕННОЙ t = sqrt(3/(4 R_uu))           против  eta0²/2

Коэффициент `v_2 = -0.0201` взят из ДРУГОГО прогона (`invert.py`: другие
семейства, другой ящик, другой диапазон размеров), поэтому абсолютные
величины — предсказание на отложенных данных, а не подгонка.

ОТНОШЕНИЕ ЭПОХ КАЛИБРОВКИ НЕ ТРЕБУЕТ ВОВСЕ:

    lambda(eta_2)/lambda(eta_1) обязано дать (eta_1/eta_2)⁴ = (t_1/t_2)²

то есть закон `1/t²` проверяется без единого внешнего числа.

ПОМЕХА не подгоняется вместе с сигналом в одной амплитуде (эта ошибка
стоила прогона в `pencil.py`): своя `lambda` каждой эпохе, а мешающие
члены `1, 1/sqrt(C_1), v², v²/sqrt(C_1)` общие и закреплены ПЛОСКИМ
плечом, где `lambda = 0` по построению.

КОНТРОЛЬ КОНВЕЙЕРА: половина плоского объявляется «эпохой», её
`lambda` обязана выйти нулём.

  python3 frwfit.py
"""
import math
import pickle

import numpy as np

from frw import SCRATCH, KFAC, V2CAL, V3CAL, IC, IX2, IX3, IV

RHO_P = 5.155e96        # кг/м³


def plan(arms):
    """arms: список (A, sd, своя ли амплитуда)."""
    A = np.vstack([a for a, _, _ in arms])
    sd = np.concatenate([s + 100000 * i for i, (_, s, _) in enumerate(arms)])
    s = np.sqrt(A[:, IC])
    g = s * (3.0 + A[:, IV]) / (1.0 - A[:, IV])
    cols, off = [], 0
    for a, _, own in arms:
        if own:
            m = np.zeros(len(A))
            m[off:off + len(a)] = 1.0
            cols.append(g * m)
        off += len(a)
    cols += [np.ones(len(A)), 1.0 / s, A[:, IV], A[:, IV] / s]
    return A, np.column_stack(cols), A[:, IC], sd


def solve(D, y, w, mask=None):
    if mask is not None:
        D, y, w = D[mask], y[mask], w[mask]
    Dw = D * w[:, None]
    return np.linalg.solve(D.T @ Dw, Dw.T @ y)


def run(arms, names, etas, col, vcal, title):
    A, D, w, sd = plan(arms)
    y = A[:, col]
    full = solve(D, y, w)
    us = np.unique(sd)
    jk = np.array([solve(D, y, w, sd != u) for u in us])
    k = len(us)
    M = D.T @ (D * w[:, None])
    # обусловленность по нормированным столбцам: масштаб столбцов сам
    # даёт огромное число и маскирует настоящую коллинеарность
    nrm = np.sqrt(np.diag(M))
    print(f'  {title}: обусловленность плана '
          f'{np.linalg.cond(M / np.outer(nrm, nrm)):.1f}, строк {len(A)}')

    def se(x):
        return math.sqrt((k - 1) / k * ((x - x.mean()) ** 2).sum())

    f = vcal * KFAC
    out = {}
    for i, (nm, e0) in enumerate(zip(names, etas)):
        lam, jlam = full[i], jk[:, i]
        Ruu, jRuu = 3 * lam / f, 3 * jlam / f
        eR = se(jRuu)
        tru = 3.0 / e0 ** 4 if e0 else 0.0
        t_tru = e0 ** 2 / 2.0 if e0 else float('nan')
        print(f'    {nm}:')
        print(f'      lambda        = {lam:+.3e} +- {se(jlam):.3e}'
              f'   ({abs(lam)/max(se(jlam),1e-30):.1f} сигм от нуля)')
        print(f'      R_uu l²       = {Ruu:+.3e} +- {eR:.3e}'
              f'   (истина {tru:+.3e})'
              f'   {abs(Ruu-tru)/max(eR,1e-30):.1f} сигм')
        if e0 and Ruu > 0:
            jt = np.sqrt(3.0 / (4.0 * np.maximum(jRuu, 1e-300)))
            t = math.sqrt(3.0 / (4.0 * Ruu))
            rho = Ruu / (8 * math.pi)
            jrho = jRuu / (8 * math.pi)
            print(f'      ВОЗРАСТ t     = {t:.1f} +- {se(jt):.1f}'
                  f' планк. времён   (истина {t_tru:.1f})')
            print(f'      плотность rho = {rho:.3e} +- {se(jrho):.3e}'
                  f' планк.  = {rho*RHO_P:.2e} кг/м³')
            print(f'                      истина {3/(32*math.pi*t_tru**2):.3e}'
                  f' планк.')
        out[nm] = (lam, jlam)
    return out


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/frw.pkl', 'rb') as fh:
        res, cfg, VB = pickle.load(fh)
    fk = [k for k in res if k[0] == 'flat'][0]
    eps = sorted([k for k in res if k[0] != 'flat'], key=lambda k: k[1])
    print('набор:', {f'{k[0]} {k[1]}': len(res[k][0]) for k in res})
    Af, sdf = res[fk]
    us = np.unique(sdf)
    h1 = np.isin(sdf, us[::2])
    print()

    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print('=' * 74)
        print(cname)
        run([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
            ['половина плоского'], [0.0], col, vcal,
            'КОНТРОЛЬ КОНВЕЙЕРА')
        out = run([(res[k][0], res[k][1], True) for k in eps]
                  + [(Af, sdf, False)],
                  [f'eta0 = {k[1]}' for k in eps], [k[1] for k in eps],
                  col, vcal, 'ЭПОХИ СОВМЕСТНО')
        if len(eps) == 2:
            n1, n2 = f'eta0 = {eps[0][1]}', f'eta0 = {eps[1][1]}'
            l1, j1 = out[n1]
            l2, j2 = out[n2]
            r = l2 / l1
            jr = j2 / j1
            k = len(j1)
            er = math.sqrt((k - 1) / k * ((jr - jr.mean()) ** 2).sum())
            pred = (eps[0][1] / eps[1][1]) ** 4
            print(f'    ЗАКОН 1/t², БЕЗ КАЛИБРОВКИ:')
            print(f'      отношение амплитуд = {r:+.3f} +- {er:.3f}'
                  f'   (предсказание {pred:.3f})'
                  f'   {abs(r-pred)/max(er,1e-30):.1f} сигм')
        print()


if __name__ == '__main__':
    main()
