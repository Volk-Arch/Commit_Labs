#!/usr/bin/env python3
"""УРАВНЕНИЕ СОСТОЯНИЯ `w` из чистого порядка.

ЧТО БЫЛО НЕ ТАК. Отпускание связей (`eos.py`) схлопывало измерение:
моя свободная подгонка расходилась с проверенным `pjoint` на тех же
данных. Диагноз найден: вырождение идёт ЦЕЛИКОМ от четырёх столбцов
помехи (обусловленность 2.4e9 против 62 без них), а плоское плечо
поддерживает всего ОДНУ константу.

Отсюда рычаг: урезать помеху. Но урезать законно лишь при двух
условиях, и оба проверяются здесь, а не назначаются.

  УСЛОВИЕ 1. Порядок помехи определяется ПО ДАННЫМ плоского плеча:
  ячейки (полоса `v²` x полоса `C_1`), средние со складным ножом по
  засевам, и `chi²` моделей порядка 1, 2, 4. Лишние столбцы подгоняют
  шум и раздувают ошибку кривизны.

  УСЛОВИЕ 2. Конвейерный контроль при выбранном порядке: половина
  плоского объявляется семейством, её `alpha, beta` обязаны выйти
  нулями. Урезание помехи может пустить гладкий тренд в кривизну и
  нарисовать сигнал — контроль ловит именно это.

ЧТО ДОСТАЁТСЯ. Уравнение состояния БЕЗ калибровки: при
`alpha = f R_etaeta l²`, `beta = f R_xx l²` множитель `f` сокращается в

    w = p/rho = (R_etaeta - R_xx)/(R_etaeta + 3 R_xx)
      = (alpha - beta)/(alpha + 3 beta)

потому что `R_00 = 4 pi G (rho + 3p)`, `R_11 = 4 pi G (rho - p)`.
Истины: `lin` даёт `+1/3` (излучение), `exp` даёт `-1/3`, `root` даёт
`+1` (жёсткая материя). У `quad` знаменатель равен нулю по построению
(`rho = 0`), поэтому `w` там не определено — и это сам по себе
контроль: подгонка обязана показать знаменатель, не отличимый от нуля.

Складной нож берётся по `w` НАПРЯМУЮ, а не по компонентам: отношение
нелинейно, и перенос ошибок через него врёт.

  python3 pw.py [--nui 0]
"""
import argparse
import math
import pickle

import numpy as np

from pencil import SCRATCH, KFAC, V2CAL, V3CAL, truth, IC, IX2, IX3, IV, VB

C1B = (200.0, 380.0, 560.0, 900.0001)
EXTRA = {'exp': (0, 30, 60), 'root': (0, 30)}


def load():
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    arms = {k: (v[0], v[1]) for k, v in res.items()}
    for nm, chunks in EXTRA.items():
        As, sds, par = [], [], None
        for s0 in chunks:
            with open(f'{SCRATCH}/{nm}fam_{s0}.pkl', 'rb') as fh:
                r = pickle.load(fh)
            As.append(r['A']); sds.append(r['sd']); par = r['k']
        arms[(nm, par)] = (np.vstack(As), np.concatenate(sds))
    return arms


def nui_cols(A, order):
    """Столбцы помехи: 1 -> константа; 2 -> +1/sqrt(C_1); 4 -> +v² члены."""
    s = np.sqrt(A[:, IC])
    one = np.ones(len(A))
    if order == 1:
        return [one]
    if order == 2:
        return [one, 1.0 / s]
    return [one, 1.0 / s, A[:, IV], A[:, IV] / s]


def plan(arms, order):
    A = np.vstack([a for a, _, _ in arms])
    sd = np.concatenate([s + 100000 * i
                         for i, (_, s, _) in enumerate(arms)])
    s = np.sqrt(A[:, IC])
    g = 1.0 / (1.0 - A[:, IV])
    g1, g2 = s * g, s * A[:, IV] * g
    cols, off = [], 0
    for a, _, curved in arms:
        if curved:
            m = np.zeros(len(A)); m[off:off + len(a)] = 1.0
            cols += [g1 * m, g2 * m]
        off += len(a)
    return A, np.column_stack(cols + nui_cols(A, order)), A[:, IC], sd


def solve(D, y, w, mask=None):
    if mask is not None:
        D, y, w = D[mask], y[mask], w[mask]
    Dw = D * w[:, None]
    return np.linalg.solve(D.T @ Dw, Dw.T @ y)


def se(x):
    k = len(x)
    return math.sqrt((k - 1) / k * ((x - x.mean()) ** 2).sum())


def order_test(Af, sdf, col):
    """УСЛОВИЕ 1: какой порядок помехи нужен плоскому плечу."""
    us = np.unique(sdf)
    cv, cm, cer = [], [], []
    for i in range(len(VB) - 1):
        for j in range(len(C1B) - 1):
            m = ((Af[:, IV] >= VB[i]) & (Af[:, IV] < VB[i + 1])
                 & (Af[:, IC] >= C1B[j]) & (Af[:, IC] < C1B[j + 1]))
            if m.sum() < 60:
                continue
            B, s_ = Af[m], sdf[m]
            wt = B[:, IC]
            mu = (B[:, col] * wt).sum() / wt.sum()
            jk = np.array([(B[s_ != u][:, col] * B[s_ != u][:, IC]).sum()
                           / B[s_ != u][:, IC].sum() for u in us])
            cv.append((B[:, IV].mean(), B[:, IC].mean()))
            cm.append(mu); cer.append(se(jk))
    cv, cm, cer = np.array(cv), np.array(cm), np.array(cer)
    print(f'    ячеек плоского {len(cm)}, '
          f'типичная ошибка ячейки {np.median(cer):.5f}')
    out = {}
    for order in (1, 2, 4):
        Afake = np.zeros((len(cv), 5))
        Afake[:, IV], Afake[:, IC] = cv[:, 0], cv[:, 1]
        D = np.column_stack(nui_cols(Afake, order))
        wt = 1.0 / cer ** 2
        p = np.linalg.solve(D.T @ (D * wt[:, None]), (D * wt[:, None]).T @ cm)
        chi = (((cm - D @ p) / cer) ** 2).sum()
        dof = len(cm) - order
        out[order] = chi / dof
        print(f'    порядок {order}: chi²/dof = {chi:.1f}/{dof} '
              f'= {chi/dof:.2f}')
    return out


def do(arms, names, truths, col, vcal, order, quiet=False):
    A, D, w, sd = plan(arms, order)
    y = A[:, col]
    full = solve(D, y, w)
    us = np.unique(sd)
    jk = np.array([solve(D, y, w, sd != u) for u in us])
    M = D.T @ (D * w[:, None])
    if not quiet:
        print(f'    обусловленность {np.linalg.cond(M):.1f}, '
              f'параметров {D.shape[1]}, строк {len(A)}')
    f = vcal * KFAC
    rows = []
    for i, (name, tr) in enumerate(zip(names, truths)):
        al, be = full[2 * i], full[2 * i + 1]
        jal, jbe = jk[:, 2 * i], jk[:, 2 * i + 1]
        Ree_t, Rxx_t, _ = tr
        den_t = Ree_t + 3 * Rxx_t
        wt_t = (Ree_t - Rxx_t) / den_t if abs(den_t) > 1e-12 else float('nan')
        den, eden = (al + 3 * be) / f, se((jal + 3 * jbe) / f)
        ww, eww = (al - be) / (al + 3 * be), se((jal - jbe) / (jal + 3 * jbe))
        print(f'    {name}:')
        print(f'      R_etaeta l²  = {al/f:+.5f} +- {se(jal)/abs(f):.5f}'
              f'    (истина {Ree_t:+.5f})')
        print(f'      R_xx l²      = {be/f:+.5f} +- {se(jbe)/abs(f):.5f}'
              f'    (истина {Rxx_t:+.5f})')
        print(f'      rho ~ знамен. = {den:+.5f} +- {eden:.5f}'
              f'    ({abs(den)/max(eden,1e-12):.1f} сигм от нуля)')
        if abs(den) > 2 * eden:
            d = (abs(ww - wt_t) / max(eww, 1e-12)
                 if wt_t == wt_t else float('nan'))
            print(f'      w            = {ww:+.4f} +- {eww:.4f}'
                  f'    (истина {wt_t:+.4f})   {d:.1f} сигм')
        else:
            print(f'      w            не определено: знаменатель '
                  f'не отличим от нуля (как и должно быть при rho = 0)')
        rows.append((name, al / f, be / f, ww, eww, wt_t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nui', type=int, default=0,
                    help='0 = выбрать по данным')
    g = ap.parse_args()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    arms = load()
    fk = [k for k in arms if k[0] == 'flat'][0]
    curved = [k for k in arms if k[0] != 'flat']
    print('набор:', {f'{k[0]} {k[1]}': len(arms[k][0]) for k in arms})
    Af, sdf = arms[fk]
    us = np.unique(sdf)
    h1 = np.isin(sdf, us[::2])

    for col, cname, vcal in ((IX2, 'X_2', V2CAL), (IX3, 'X_3', V3CAL)):
        print()
        print('=' * 76)
        print(f'{cname}')
        print('  УСЛОВИЕ 1: порядок помехи по данным плоского плеча')
        ot = order_test(Af, sdf, col)
        order = g.nui or (1 if ot[1] < 1.6 else (2 if ot[2] < 1.6 else 4))
        print(f'    -> взят порядок {order}')
        print(f'  УСЛОВИЕ 2: конвейерный контроль при порядке {order}')
        do([(Af[h1], sdf[h1], True), (Af[~h1], sdf[~h1], False)],
           ['половина плоского'], [(1.0, 0.0, 0.0)], col, vcal, order)
        print(f'  СЕМЕЙСТВА СОВМЕСТНО, помеха порядка {order}')
        do([(arms[k][0], arms[k][1], True) for k in curved]
           + [(Af, sdf, False)],
           [f'{k[0]} {k[1]}' for k in curved],
           [truth(*k) for k in curved], col, vcal, order)


if __name__ == '__main__':
    main()
