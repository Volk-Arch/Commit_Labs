#!/usr/bin/env python3
"""ПЕРЕВЗВЕШИВАНИЕ ПОД ПЛОХОЕ НАПРАВЛЕНИЕ — бесплатный выстрел.

Положение дел. `rho ~ alpha + 3 beta` (а с ним и `w`) сидит в
почти-нулевом направлении плана: ошибка 0.00160 против 0.00018 у
хорошего. Вторая наблюдаемая не помогает — `X_3` оказался почти
идеальным повтором `X_2` (`pw3.py`: 0.00303 -> 0.00324). Расширять окно
`v²` тоже нельзя: защита применимости требует
`tau² |R_etaeta + R_xx v²|/(1-v²) <= 0.8`, значит на границе окна
`sqrt(C_1) ~ (1-v²)`, и тогда `g_2 = sqrt(C_1) v²/(1-v²) ~ v²` упирается
в потолок. Два рычага противоречат друг другу.

Остаётся бесплатное. Веса `w = C_1` оптимальны для ХОРОШЕГО направления
(сигнал там идёт как `sqrt(C_1)`), а нужно ПЛОХОЕ. Взвешенный МНК
несмещён при ЛЮБЫХ весах, если модель верна, — меняется только
эффективность. Значит можно искать вес, который минимизирует ошибку
`rho`, ничего не набирая.

Сканируются `w = C_1^p (1-v²)^(-q)`: показатель `p` перераспределяет
между большими и малыми интервалами, `q` — к высоким `v²`, где и живёт
разделение `alpha` и `beta`.

КОНТРОЛЬ обязателен при лучшем весе: половина плоского объявляется
семейством. Перевзвешивание не может сместить верную модель, но МОЖЕТ
вытащить наружу её неверность — именно это контроль и проверяет.

  python3 pw4.py
"""
import math

import numpy as np

from pencil import KFAC, V2CAL, truth, IC, IX2, IV
from pw import load, nui_cols


def fit(arms, names, p, q, col=IX2, order=1):
    A = np.vstack([a for a, _, _ in arms])
    sd = np.concatenate([s + 100000 * i
                         for i, (_, s, _) in enumerate(arms)])
    s = np.sqrt(A[:, IC]); g = 1.0 / (1.0 - A[:, IV])
    g1, g2 = s * g, s * A[:, IV] * g
    cols, off = [], 0
    for a, _, curved in arms:
        if curved:
            m = np.zeros(len(A)); m[off:off + len(a)] = 1.0
            cols += [g1 * m, g2 * m]
        off += len(a)
    D = np.column_stack(cols + nui_cols(A, order))
    y = A[:, col]
    w = A[:, IC] ** p * (1.0 - A[:, IV]) ** (-q)

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
        pr = np.column_stack([jk[:, 2 * i], jk[:, 2 * i + 1]])
        C = (k - 1) / k * (pr - pr.mean(axis=0)).T @ (pr - pr.mean(axis=0))
        out[nm] = (full[2 * i], full[2 * i + 1], C)
    return out


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
    u = np.array([1.0, 3.0])

    print('=' * 76)
    print('СКАН ВЕСА: ошибка `rho` (в единицах R l²) при w = C_1^p (1-v²)^-q')
    print(f"{'p':>5}{'q':>5}", end='')
    for nm in names:
        print(f'{nm.split()[0]:>11}', end='')
    print('   лучшее')
    best = None
    for p in (0.0, 0.5, 1.0, 1.5):
        for q in (0.0, 1.0, 2.0):
            res = fit(pack, names, p, q)
            ers = []
            for nm in names:
                _, _, C = res[nm]
                ers.append(math.sqrt(u @ C @ u) / abs(f))
            print(f'{p:>5.1f}{q:>5.1f}', end='')
            for e in ers:
                print(f'{e:>11.5f}', end='')
            sc = ers[0]
            if best is None or sc < best[0]:
                best = (sc, p, q)
            print(f'   {"<-" if best[1:]==(p,q) else "":>7}', flush=True)
    sc, p, q = best
    print()
    print(f'лучший вес: p = {p}, q = {q}; ошибка rho на lin {sc:.5f} '
          f'против 0.00303 при p=1,q=0 (в {0.00303/sc:.2f} раз)')
    print()
    print('=' * 76)
    print(f'КОНТРОЛЬ КОНВЕЙЕРА при лучшем весе (p={p}, q={q})')
    us = np.unique(sdf); h1 = np.isin(sdf, us[::2])
    r = fit([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
            ['половина'], p, q)
    al, be, C = r['половина']
    print(f'  R_etaeta l² = {al/f:+.5f} +- {math.sqrt(C[0,0])/abs(f):.5f}  '
          f'({abs(al)/math.sqrt(C[0,0]):.1f} сигм от нуля)')
    print(f'  rho         = {(al+3*be)/f:+.5f} +- '
          f'{math.sqrt(u@C@u)/abs(f):.5f}  '
          f'({abs(al+3*be)/math.sqrt(u@C@u):.1f} сигм от нуля)')
    print()
    print('=' * 76)
    print(f'ИТОГ при лучшем весе')
    res = fit(pack, names, p, q)
    for nm in names:
        al, be, C = res[nm]
        tr = truth(*[k for k in curved if f'{k[0]} {k[1]}' == nm][0])
        rho, er = (al + 3 * be) / f, math.sqrt(u @ C @ u) / abs(f)
        dt = tr[0] + 3 * tr[1]
        line = (f'  {nm:>12}: rho = {rho:+.5f} +- {er:.5f}  '
                f'(истина {dt:+.5f}, {abs(rho-dt)/max(er,1e-12):.1f} сигм, '
                f'от нуля {abs(rho)/max(er,1e-12):.1f})')
        if abs(rho) > 3 * er and abs(dt) > 1e-9:
            wv = (al - be) / (al + 3 * be)
            gw = np.array([1.0 / (al + 3 * be)
                           - (al - be) / (al + 3 * be) ** 2,
                           -1.0 / (al + 3 * be)
                           - 3 * (al - be) / (al + 3 * be) ** 2])
            ew = math.sqrt(gw @ C @ gw)
            wt = (tr[0] - tr[1]) / dt
            line += (f'\n                w = {wv:+.3f} +- {ew:.3f}  '
                     f'(истина {wt:+.3f}, {abs(wv-wt)/max(ew,1e-12):.1f} сигм)')
        print(line)


if __name__ == '__main__':
    main()
