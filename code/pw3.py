#!/usr/bin/env python3
"""ДВЕ НАБЛЮДАЕМЫЕ ПРОТИВ ВЫРОЖДЕНИЯ: ломает ли `X_3` то, что не может `X_2`.

Установлено (`pw2.py`): хорошо измеряемое направление пары компонент —
`alpha + 0.71 beta` с ошибкой +-0.00018, плохое `alpha - 1.41 beta` — с
+-0.00160, в девять раз хуже. А `rho ~ alpha + 3 beta` сидит ровно в
плохом, поэтому `w` и не выходит. Ошибка `rho` падает с размахом окна
`v²` в 12 раз (0.30 -> 0.85) — втрое быстрее роста статистики, то есть
рычаг именно в окне. До пяти сигм на линейном плече не хватает 2.1.

ЗДЕСЬ пробуется рычаг, который ещё не использован: `X_2` и `X_3` — две
РАЗНЫЕ наблюдаемые на ОДНИХ И ТЕХ ЖЕ мирах. Подгоняются совместно, с
общими физическими `R_etaeta l²`, `R_xx l²` и своей константой помехи
на каждую наблюдаемую:

    X_o = v_o k [ R_etaeta g_1 + R_xx g_2 ] + n_o,   o = 2, 3

Складной нож берётся по МИРАМ, общим для обеих наблюдаемых, иначе их
корреляция недосчитается.

ЧТЕНИЕ. Если ошибка `rho` выйдет около `0.00303/sqrt(2) = 0.00214` —
наблюдаемые несут одно и то же направление, и выигрыш только
статистический. Если заметно лучше — направления разные, вырождение
сломано, и `w` достижимо на УЖЕ набранных данных.

  python3 pw3.py
"""
import math

import numpy as np

from pencil import KFAC, V2CAL, V3CAL, truth, IC, IX2, IX3, IV
from pw import load


def joint(arms, names, vmax=None):
    """Общие физические компоненты, две наблюдаемые, помеха на каждую."""
    rows, ys, ws, sds, obs, who = [], [], [], [], [], []
    for ai, (a, s, curved) in enumerate(arms):
        m = np.ones(len(a), bool) if vmax is None else (a[:, IV] <= vmax)
        A = a[m]; sd = s[m]
        for o, (col, vcal) in enumerate(((IX2, V2CAL), (IX3, V3CAL))):
            rows.append(A); ys.append(A[:, col]); ws.append(A[:, IC])
            sds.append(sd)                       # ТЕ ЖЕ миры, без сдвига
            obs.append(np.full(len(A), o))
            who.append(np.full(len(A), ai if curved else -1))
    A = np.vstack(rows)
    y = np.concatenate(ys); w = np.concatenate(ws)
    sd = np.concatenate(sds); ob = np.concatenate(obs)
    wh = np.concatenate(who)
    s = np.sqrt(A[:, IC]); g = 1.0 / (1.0 - A[:, IV])
    fac = np.where(ob == 0, V2CAL * KFAC, V3CAL * KFAC)
    g1, g2 = fac * s * g, fac * s * A[:, IV] * g
    cols = []
    for ai in range(len(arms)):
        if arms[ai][2]:
            m = (wh == ai).astype(float)
            cols += [g1 * m, g2 * m]
    cols += [(ob == 0).astype(float), (ob == 1).astype(float)]
    D = np.column_stack(cols)

    def sol(mask=None):
        DD, yy, ww = (D, y, w) if mask is None else (D[mask], y[mask], w[mask])
        Dw = DD * ww[:, None]
        return np.linalg.solve(DD.T @ Dw, Dw.T @ yy)

    full = sol()
    us = np.unique(sd)
    jk = np.array([sol(sd != u) for u in us])
    k = len(us)
    out = {}
    for i, nm in enumerate(names):
        p = np.column_stack([jk[:, 2 * i], jk[:, 2 * i + 1]])
        C = (k - 1) / k * (p - p.mean(axis=0)).T @ (p - p.mean(axis=0))
        out[nm] = (full[2 * i], full[2 * i + 1], C)
    return out, np.linalg.cond(D.T @ (D * w[:, None])), len(A)


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

    print('=' * 78)
    print('КОНТРОЛЬ КОНВЕЙЕРА: половина плоского объявлена семейством')
    us = np.unique(sdf); h1 = np.isin(sdf, us[::2])
    r, cd, nr = joint([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
                      ['половина плоского'])
    al, be, C = r['половина плоского']
    u = np.array([1.0, 3.0])
    print(f'  R_etaeta l² = {al:+.5f} +- {math.sqrt(C[0,0]):.5f}   '
          f'({abs(al)/math.sqrt(C[0,0]):.1f} сигм от нуля)')
    print(f'  R_xx l²     = {be:+.5f} +- {math.sqrt(C[1,1]):.5f}   '
          f'({abs(be)/math.sqrt(C[1,1]):.1f} сигм от нуля)')
    print(f'  rho         = {al+3*be:+.5f} +- {math.sqrt(u@C@u):.5f}   '
          f'({abs(al+3*be)/math.sqrt(u@C@u):.1f} сигм от нуля)')
    print()
    print('=' * 78)
    print('СЕМЕЙСТВА СОВМЕСТНО, две наблюдаемые, общие компоненты')
    res, cd, nr = joint(pack, names)
    print(f'  обусловленность {cd:.1f}, строк {nr}')
    print(f"{'плечо':>13}{'R_etaeta l²':>22}{'R_xx l²':>22}"
          f"{'rho':>20}{'w':>20}")
    for nm in names:
        al, be, C = res[nm]
        tr = truth(*[k for k in curved if f'{k[0]} {k[1]}' == nm][0])
        er_a, er_b = math.sqrt(C[0, 0]), math.sqrt(C[1, 1])
        rho, er_r = al + 3 * be, math.sqrt(u @ C @ u)
        den_t = tr[0] + 3 * tr[1]
        if abs(rho) > 2 * er_r:
            wv = (al - be) / rho
            gw = np.array([1.0 / rho - (al - be) / rho ** 2,
                           -1.0 / rho - 3 * (al - be) / rho ** 2])
            er_w = math.sqrt(gw @ C @ gw)
            wt = (tr[0] - tr[1]) / den_t if abs(den_t) > 1e-12 else float('nan')
            ws = (f'{wv:+.3f}+-{er_w:.3f}'
                  + (f' ({wt:+.2f})' if wt == wt else ' (н/о)'))
        else:
            ws = 'rho не разрешено'
        print(f'{nm:>13}'
              f'{f"{al:+.5f}+-{er_a:.5f}":>22}'
              f'{f"{be:+.5f}+-{er_b:.5f}":>22}'
              f'{f"{rho:+.5f}+-{er_r:.5f}":>20}{ws:>20}')
        print(f'{"истина":>13}{tr[0]:>+22.5f}{tr[1]:>+22.5f}'
              f'{den_t:>+20.5f}')
    print()
    print('=' * 78)
    print('ВЫИГРЫШ ПРОТИВ ОДНОЙ НАБЛЮДАЕМОЙ (было 0.00303 на lin)')
    al, be, C = res['lin 0.035']
    e = math.sqrt(u @ C @ u)
    print(f'  ошибка rho на lin: 0.00303 -> {e:.5f} (в {0.00303/e:.2f} раз); '
          f'чисто статистический предел был бы 0.00214 (в 1.41)')
    print('  ->', 'НАПРАВЛЕНИЯ РАЗНЫЕ, вырождение сломано'
          if e < 0.0018 else 'направление ОДНО, выигрыш только статистический')


if __name__ == '__main__':
    main()
