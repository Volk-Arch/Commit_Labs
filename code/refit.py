#!/usr/bin/env python3
"""ПЕРЕПОДГОНКА ВСЕГО НАБОРА на выведенных профилях — пункт 1 плана.

Старый план (`X_2 ~ R_uu`) опровергнут. Верные профили выведены для
`X_2` (`design-doc.md`) и теперь для `X_3` (`x3prof.py`), оба с
контролями на замкнутых формах `c_2`, `c_3`:

    X_o = [ A_o(v²) R_etaeta + B_o(v²) R_xx ] tau²,  tau² = KFAC sqrt(C_1) l²

Отношения `B_o/A_o` у двух наблюдаемых БЛИЗКИ, но не совпадают (разница
0.05...0.08), значит совместная подгонка даёт небольшой дополнительный
рычаг помимо статистики. `X_3` при этом в 2.3 раза чувствительнее к
`R_etaeta`, но и шумнее в 2.48 раза — отношение сигнала к шуму у них
одинаковое, и это количественно объясняет, почему вчера `X_3` не дал
ничего.

Подгоняются ОБЩИЕ физические компоненты на плечо, своя константа
помехи на наблюдаемую, складной нож по МИРАМ (общим для обеих
наблюдаемых, без сдвига — иначе их корреляция недосчитается).

Включены ночные чанки при `per = 120` (доля независимости 0.71 против
0.33 при 420).

КОНТРОЛЬ: половина плоского плеча объявляется семейством, её компоненты
обязаны выйти нулями.

  python3 refit.py
"""
import math
import pickle

import numpy as np

from pencil import SCRATCH, KFAC, truth, IC, IX2, IX3, IV

VT = np.array([0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.85])
A2 = np.array([-0.02539, -0.02877, -0.03341, -0.04102,
               -0.05382, -0.08255, -0.13395])
B2 = np.array([+0.01908, +0.01566, +0.01129, +0.00335,
               -0.00941, -0.03828, -0.09121])
A3 = np.array([-0.05708, -0.06471, -0.07598, -0.09376,
               -0.12170, -0.19109, -0.30993])
B3 = np.array([+0.03850, +0.03131, +0.02023, +0.00116,
               -0.02209, -0.09571, -0.22391])

CHUNKS = {'exp': (0, 30, 60, 500, 1000, 1100, 1200),
          'root': (0, 30),
          'lin': (500, 530, 1000, 1100)}


def load_all():
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    arms = {}
    for k, v in res.items():
        arms[k[0]] = {'A': [v[0]], 'sd': [v[1]], 'par': k[1]}
    for fam, chs in CHUNKS.items():
        for s0 in chs:
            try:
                with open(f'{SCRATCH}/{fam}fam_{s0}.pkl', 'rb') as fh:
                    r = pickle.load(fh)
            except FileNotFoundError:
                continue
            arms.setdefault(fam, {'A': [], 'sd': [], 'par': r['k']})
            arms[fam]['A'].append(r['A'])
            arms[fam]['sd'].append(r['sd'])
            arms[fam]['par'] = r['k']
    out = {}
    for fam, d in arms.items():
        A = np.vstack(d['A']); sd = np.concatenate(d['sd'])
        out[fam] = (A, sd, d['par'])
    return out


def build(arms):
    """arms: список (имя, A, sd, кривое ли)."""
    rows, ys, ws, sds, obs, who = [], [], [], [], [], []
    for ai, (nm, A, sd, curved) in enumerate(arms):
        for o, col in enumerate((IX2, IX3)):
            rows.append(A); ys.append(A[:, col]); ws.append(A[:, IC])
            sds.append(sd)
            obs.append(np.full(len(A), o))
            who.append(np.full(len(A), ai if curved else -1))
    M = np.vstack(rows)
    y = np.concatenate(ys); w = np.concatenate(ws)
    sd = np.concatenate(sds); ob = np.concatenate(obs)
    wh = np.concatenate(who)
    t2 = KFAC * np.sqrt(M[:, IC])
    av = np.where(ob == 0, np.interp(M[:, IV], VT, A2),
                  np.interp(M[:, IV], VT, A3)) * t2
    bv = np.where(ob == 0, np.interp(M[:, IV], VT, B2),
                  np.interp(M[:, IV], VT, B3)) * t2
    cols = []
    for ai, (_, _, _, curved) in enumerate(arms):
        if curved:
            m = (wh == ai).astype(float)
            cols += [av * m, bv * m]
    cols += [(ob == 0).astype(float), (ob == 1).astype(float)]
    return np.column_stack(cols), y, w, sd


def do(arms, truths, label):
    D, y, w, sd = build(arms)

    def sol(mask=None):
        DD, yy, ww = (D, y, w) if mask is None else (D[mask], y[mask], w[mask])
        Dw = DD * ww[:, None]
        return np.linalg.solve(DD.T @ Dw, Dw.T @ yy)

    full = sol()
    us = np.unique(sd)
    jk = np.array([sol(sd != u) for u in us])
    k = len(us)
    Mx = D.T @ (D * w[:, None])
    print(f'  {label}: обусловленность {np.linalg.cond(Mx):.1f}, '
          f'параметров {D.shape[1]}, строк {len(y)}, миров {k}')
    uu = np.array([1.0, 3.0])
    i = 0
    for (nm, _, _, curved), tr in zip(arms, truths):
        if not curved:
            continue
        al, be = full[2 * i], full[2 * i + 1]
        pr = np.column_stack([jk[:, 2 * i], jk[:, 2 * i + 1]])
        C = (k - 1) / k * (pr - pr.mean(axis=0)).T @ (pr - pr.mean(axis=0))
        ea, eb = math.sqrt(C[0, 0]), math.sqrt(C[1, 1])
        rho, er = al + 3 * be, math.sqrt(uu @ C @ uu)
        dt = tr[0] + 3 * tr[1]
        print(f'    {nm}:')
        print(f'      R_etaeta l² = {al:+.5f} +- {ea:.5f}  '
              f'(истина {tr[0]:+.5f})  {abs(al-tr[0])/max(ea,1e-12):.1f} сигм')
        print(f'      R_xx l²     = {be:+.5f} +- {eb:.5f}  '
              f'(истина {tr[1]:+.5f})  {abs(be-tr[1])/max(eb,1e-12):.1f} сигм')
        print(f'      rho         = {rho:+.5f} +- {er:.5f}  '
              f'(истина {dt:+.5f})  от нуля {abs(rho)/max(er,1e-12):.1f} сигм')
        if abs(rho) > 3 * er and abs(dt) > 1e-9:
            wv = (al - be) / rho
            gw = np.array([1.0 / rho - (al - be) / rho ** 2,
                           -1.0 / rho - 3 * (al - be) / rho ** 2])
            ew = math.sqrt(gw @ C @ gw)
            wt = (tr[0] - tr[1]) / dt
            print(f'      w           = {wv:+.4f} +- {ew:.4f}  '
                  f'(истина {wt:+.4f})  '
                  f'{abs(wv-wt)/max(ew,1e-12):.1f} сигм')
        i += 1


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load_all()
    print('набор:', {f'{k} {v[2]}': (len(v[0]), len(np.unique(v[1])))
                     for k, v in arms.items()})
    print('  (строк, миров)')
    print()
    Af, sdf, _ = arms['flat']
    us = np.unique(sdf); h1 = np.isin(sdf, us[::2])
    print('КОНТРОЛЬ: половина плоского объявлена семейством')
    do([('половина плоского', Af[h1], sdf[h1], True),
        ('остаток', Af[~h1], sdf[~h1], False)],
       [(0.0, 0.0, 0.0), None], 'контроль')
    print()
    print('ВСЕ ПЛЕЧИ СОВМЕСТНО, две наблюдаемые, верные профили')
    cur = [f for f in arms if f != 'flat']
    pack = [(f'{f} {arms[f][2]}', arms[f][0], arms[f][1], True) for f in cur]
    pack.append(('flat', Af, sdf, False))
    do(pack, [truth(f, arms[f][2]) for f in cur] + [None], 'совместно')


if __name__ == '__main__':
    main()
