#!/usr/bin/env python3
"""ГДЕ ЖИВЁТ ВЫРОЖДЕНИЕ: какая комбинация компонент измерима и чем.

Урезание помехи до одной константы (`pw.py`) рычага НЕ дало:
обусловленность осталась 4500, а ошибки компонент вышли +-0.0009...0.0013
против записанных +-0.00042. Значит мой диагноз был неверен — вырождение
живёт не в столбцах помехи, а в самой паре столбцов кривизны:

    g_1 = sqrt(C_1)/(1-v²),   g_2 = sqrt(C_1) v²/(1-v²),   g_2/g_1 = v²

над нашим диапазоном `v²` они почти коллинеарны.

ЧТО МЕРЯЕТСЯ ЗДЕСЬ. Собственное разложение ковариации пары
`(alpha, beta)` по складному ножу: какая комбинация измерена хорошо,
какая плохо, и совпадает ли хорошая с `alpha + <v²> beta`, как
предсказывает коллинеарность. Затем — как ошибка ПЛОХОГО направления
(а `rho ~ alpha + 3 beta` сидит именно там) зависит от размаха `v²`:
подгонка по подвыборкам `v² <= vmax`. Это говорит, достижимо ли `w`
вообще и какой ценой — набором или рычагом.

  python3 pw2.py
"""
import math

import numpy as np

from pencil import KFAC, V2CAL, truth, IC, IX2, IV
from pw import load, plan, solve, se


def arm_cov(arms, names, col, order, vmax=None):
    """Ковариация (alpha, beta) по складному ножу, на v² <= vmax."""
    sel = []
    for a, s, c in arms:
        m = np.ones(len(a), bool) if vmax is None else (a[:, IV] <= vmax)
        sel.append((a[m], s[m], c))
    A, D, w, sd = plan(sel, order)
    y = A[:, col]
    full = solve(D, y, w)
    us = np.unique(sd)
    jk = np.array([solve(D, y, w, sd != u) for u in us])
    k = len(us)
    out = {}
    for i, nm in enumerate(names):
        p = np.column_stack([jk[:, 2 * i], jk[:, 2 * i + 1]])
        C = (k - 1) / k * (p - p.mean(axis=0)).T @ (p - p.mean(axis=0))
        out[nm] = (full[2 * i], full[2 * i + 1], C)
    return out, np.mean(A[:, IV][A[:, IV] > 0]) if vmax else A[:, IV].mean()


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load()
    fk = [k for k in arms if k[0] == 'flat'][0]
    curved = [k for k in arms if k[0] != 'flat']
    Af, sdf = arms[fk]
    pack = ([(arms[k][0], arms[k][1], True) for k in curved]
            + [(Af, sdf, False)])
    names = [f'{k[0]} {k[1]}' for k in curved]
    f = V2CAL * KFAC

    print('=' * 76)
    print('СОБСТВЕННЫЕ НАПРАВЛЕНИЯ ковариации (alpha, beta), помеха = 1')
    print(f"{'плечо':>14}{'хорошее напр.':>22}{'его ошибка':>13}"
          f"{'плохое напр.':>20}{'его ошибка':>13}")
    res, vbar = arm_cov(pack, names, IX2, 1)
    for nm in names:
        al, be, C = res[nm]
        ev, V = np.linalg.eigh(C)
        lo, hi = V[:, 0], V[:, -1]
        if lo[0] < 0:
            lo = -lo
        if hi[0] < 0:
            hi = -hi
        print(f'{nm:>14}'
              f'{f"a{lo[0]:+.2f} b{lo[1]:+.2f}":>22}'
              f'{math.sqrt(ev[0])/abs(f):>13.5f}'
              f'{f"a{hi[0]:+.2f} b{hi[1]:+.2f}":>20}'
              f'{math.sqrt(ev[-1])/abs(f):>13.5f}')
    print()
    print(f'  средний v² выборки = {vbar:.3f}; коллинеарность предсказывает'
          f' хорошее направление ~ a + {vbar:.2f} b')
    print()
    print('=' * 76)
    print('КАК ОШИБКА `rho` ЗАВИСИТ ОТ РАЗМАХА v² (rho ~ alpha + 3 beta)')
    print(f"{'vmax':>6}{'строк':>9}", end='')
    for nm in names:
        print(f'{nm.split()[0]:>11}', end='')
    print()
    hist = {nm: [] for nm in names}
    vms = (0.30, 0.45, 0.60, 0.72, 0.85)
    for vm in vms:
        res, _ = arm_cov(pack, names, IX2, 1, vmax=vm)
        nrow = sum(int((a[:, IV] <= vm).sum()) for a, _, _ in pack)
        print(f'{vm:>6.2f}{nrow:>9}', end='')
        for nm in names:
            al, be, C = res[nm]
            u = np.array([1.0, 3.0])
            er = math.sqrt(u @ C @ u) / abs(f)
            hist[nm].append(er)
            print(f'{er:>11.5f}', end='')
        print(flush=True)
    print()
    print('=' * 76)
    print('ВО СКОЛЬКО ПАДАЕТ ОШИБКА `rho` ПРИ РАСШИРЕНИИ ОКНА')
    for nm in names:
        h = hist[nm]
        tr = truth(*[k for k in curved if f'{k[0]} {k[1]}' == nm][0])
        den_t = tr[0] + 3 * tr[1]
        need = abs(den_t) / 5.0 if abs(den_t) > 1e-9 else float('nan')
        print(f'  {nm:>12}: {h[0]:.5f} -> {h[-1]:.5f} '
              f'(в {h[0]/h[-1]:.1f} раз), истина rho = {den_t:+.5f}, '
              f'для 5 сигм нужно {need:.5f}'
              + ('  ДОСТИГНУТО' if need == need and h[-1] <= need else ''))


if __name__ == '__main__':
    main()
