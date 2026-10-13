#!/usr/bin/env python3
"""ВТОРОЙ ПОРЯДОК КАК ПРИЧИНА ДЕФЕКТА: один параметр на все плечи.

Карта остатков (`resid.py`) показала, что отношение замер/предсказание
ПАДАЕТ с `C_1` внутри каждой полосы `v²`: у `lin` 1.49, 1.03, 0.89 и
0.91, 0.81, 0.74. Падает всегда в одну сторону.

Масштаб объясняет почему. При `C_1 = 700`, `v² = 0.7` выходит

    tau² R_uu = KFAC sqrt(C_1) (R_ee + R_xx v²)/(1 - v²) ~ 1.1

то есть набор сидит НЕ в области малой кривизны. Защита применимости
стояла на 0.8, но 0.8 — не «мало», и второй порядок там даёт десятки
процентов. Первый порядок систематически ПЕРЕОЦЕНИВАЕТ.

ПРОВЕРКА ОДНИМ ПАРАМЕТРОМ. Если причина именно в этом, то один и тот же
`eps` в

    замер = предсказание * (1 - eps * tau² R_uu)

должен выровнять ВСЕ ЧЕТЫРЕ плеча. Если `eps` выйдет разным по плечам —
причина другая.

Предсказание берётся от истинной кривизны В СОБСТВЕННОЙ СЕРЕДИНЕ каждого
интервала, а не в `eta_0`: интервалы сидят на серединах 8.4...11.6, где
у `root` кривизна идёт как `1/a^4`, у `lin` как `1/a^2`.

  python3 order2fit.py
"""
import math

import numpy as np

from pencil import KFAC, a_of, truth, IC, IX2, IX3, IV, IM
from refit import load_all, VT, A2, B2, A3, B3

ETA0 = 10.0


def comps_at_mid(fam, par, mid):
    """Истинные (R_ee, R_xx) l² в собственной середине интервала."""
    a = a_of(mid, fam, par, ETA0)
    if fam == 'lin':
        return 3 * par ** 2 / a ** 2, par ** 2 / a ** 2
    if fam == 'exp':
        return np.zeros_like(a), np.full_like(a, 2 * par ** 2)
    if fam == 'root':
        return 1.5 * par ** 2 / a ** 4, np.zeros_like(a)
    if fam == 'quad':
        sq = par * (mid - ETA0) ** 2
        return (6 * par * (sq - 2) / (sq + 2) ** 2,
                2 * par * (3 * sq + 2) / (sq + 2) ** 2)
    raise ValueError(fam)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load_all()
    print('ЭТАП 1 — предсказание от кривизны В СВОЕЙ СЕРЕДИНЕ')
    print(f"{'плечо':>7}{'от eta_0':>12}{'от своей середины':>20}"
          f"{'<tau²R_uu>':>13}")
    store = {}
    for fam in ('lin', 'quad', 'exp', 'root'):
        A, sd, par = arms[fam]
        tr0 = truth(fam, par)
        ree, rxx = comps_at_mid(fam, par, A[:, IM])
        t2 = KFAC * np.sqrt(A[:, IC])
        p0 = (np.interp(A[:, IV], VT, A2) * tr0[0]
              + np.interp(A[:, IV], VT, B2) * tr0[1]) * t2
        p1 = (np.interp(A[:, IV], VT, A2) * ree
              + np.interp(A[:, IV], VT, B2) * rxx) * t2
        truu = t2 * (ree + rxx * A[:, IV]) / (1.0 - A[:, IV])
        store[fam] = (A, sd, p1, truu, ree, rxx, t2)
        print(f'{fam:>7}{A[:, IX2].mean()/p0.mean():>12.3f}'
              f'{A[:, IX2].mean()/p1.mean():>20.3f}'
              f'{truu.mean():>13.3f}', flush=True)
    print()
    print('ЭТАП 2 — ОДИН `eps` на плечо: выравнивается ли')
    print(f"{'плечо':>7}{'eps':>12}{'ошибка':>10}"
          f"{'отношение после':>18}{'chi²/ячеек':>14}")
    VBB = (0.0, 0.3, 0.5, 0.65, 0.78, 0.85001)
    CBB = (200.0, 330.0, 470.0, 650.0, 900.0001)
    allE = []
    for fam in ('lin', 'quad', 'exp', 'root'):
        A, sd, p1, truu, *_ = store[fam]
        rows = []
        for i in range(len(VBB) - 1):
            for j in range(len(CBB) - 1):
                m = ((A[:, IV] >= VBB[i]) & (A[:, IV] < VBB[i + 1])
                     & (A[:, IC] >= CBB[j]) & (A[:, IC] < CBB[j + 1]))
                if m.sum() < 120:
                    continue
                us = np.unique(sd[m])
                jk = np.array([A[m & (sd != u), IX2].mean() for u in us])
                se = math.sqrt((len(us) - 1) / len(us)
                               * ((jk - jk.mean()) ** 2).sum())
                rows.append((A[m, IX2].mean(), p1[m].mean(),
                             (p1 * truu)[m].mean(), se))
        R = np.array(rows)
        # замер = p - eps * p*truu  ->  (p - замер) = eps * (p*truu)
        yv, xv, wv = R[:, 1] - R[:, 0], R[:, 2], 1.0 / R[:, 3] ** 2
        eps = (wv * xv * yv).sum() / (wv * xv * xv).sum()
        ee = 1.0 / math.sqrt((wv * xv * xv).sum())
        res = R[:, 0] - (R[:, 1] - eps * R[:, 2])
        chi = ((res / R[:, 3]) ** 2).sum()
        after = R[:, 0].sum() / (R[:, 1] - eps * R[:, 2]).sum()
        allE.append((eps, ee))
        print(f'{fam:>7}{eps:>12.4f}{ee:>10.4f}{after:>18.3f}'
              f'{f"{chi:.0f}/{len(R)}":>14}', flush=True)
    E = np.array(allE)
    wm = (E[:, 0] / E[:, 1] ** 2).sum() / (1.0 / E[:, 1] ** 2).sum()
    sp = E[:, 0].std(ddof=1)
    print()
    print(f'  средний eps = {wm:.4f}, разброс по плечам {sp:.4f}')
    print('  ->', 'ОДИН eps на все плечи: диагноз подтверждён'
          if sp < 3 * E[:, 1].mean() else
          'eps РАЗНЫЙ по плечам: причина не только во втором порядке')


if __name__ == '__main__':
    main()
