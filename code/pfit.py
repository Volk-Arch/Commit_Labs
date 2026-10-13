#!/usr/bin/env python3
"""РАЗБОР КАРАНДАША: R_etaeta и R_xx порознь, затем скаляр R свёрткой.

МОДЕЛЬ. Для интервала с наклоном `v` и числом событий `C_1`

    X_2 = v_2 * sqrt(24 C_1/pi) * R_uu l²,
    R_uu = [ R_etaeta + R_xx v² ] / (a² (1 - v²))

то есть, при `a ~ 1` в узкой полосе вокруг центра,

    X_2 = alpha * g_1 + beta * g_2,
    g_1 = sqrt(C_1)/(1-v²),   g_2 = sqrt(C_1) v²/(1-v²)

где `alpha = v_2 k R_etaeta l²`, `beta = v_2 k R_xx l²`, `k = sqrt(24/pi)`.

МЕШАЮЩИЕ ЧЛЕНЫ НЕ ПОДГОНЯЮТСЯ, А ВЫЧИТАЮТСЯ. Подгонять их нельзя:
`g_1 - g_2 = sqrt(C_1)` тождественно, и над нашим диапазоном `C_1`
степени `sqrt(C_1)`, `1`, `1/sqrt(C_1)` почти коллинеарны — число
обусловленности выходило 6e8, а отношение «измерялось» как ровно `-1.00`
в обоих семействах, что есть подпись вырождения, а не замер.

Поэтому: плоское плечо проходит ТОТ ЖЕ отбор, обе выборки режутся на
одни и те же ячейки по `(v², C_1)`, и из каждой ячейки кривого
вычитается та же ячейка плоского. Что бы ни было источником помехи —
неточность `c_2` и `c_3`, отбор по вложению в ящик, конечность `C_1` —
оно одинаково в обоих плечах и уходит нацело. План остаётся из двух
столбцов.

ПРОВЕРЯЕТСЯ:
  1. КОНТРОЛЬ КОНВЕЙЕРА: половина плоского против другой половины.
     Кривизны там нет по построению, и подгонка обязана дать нули. Это
     проверка всей цепочки — ячеек, вычитания, весов, складного ножа.
  2. ОТНОШЕНИЕ `R_xx/R_etaeta`: ровно `+1/3` у линейного и ровно `-1/3`
     у квадратичного. Знаки РАЗНЫЕ, поэтому различать надо знак.
     Калибровочная постоянная `v_2` в отношение не входит вовсе.
  3. СКАЛЯР `R = (-R_etaeta + 3 R_xx)/a²`: у линейного обязан выйти
     НУЛЁМ — точным сокращением двух ненулевых измеренных чисел
     (`R_etaeta = 3h²`, `R_xx = h²`), — а у квадратичного `6c`.
  4. обусловленность двухстолбцового плана печатается.

  python3 pfit.py
"""
import math
import pickle

import numpy as np

from pencil import SCRATCH, VB, KFAC, V2CAL, V2ERR, V3CAL, V3ERR, truth
from pencil import IC, IX2, IX3, IV

NCB = 4          # полос по C_1


def cells(A, qc):
    """Номер ячейки (полоса по v², полоса по C_1) для каждого интервала."""
    bv = np.digitize(A[:, IV], VB) - 1
    bc = np.clip(np.digitize(A[:, IC], qc) - 1, 0, NCB - 1)
    return bv * NCB + bc


def summarize(A, cid, col, ncell):
    """Среднее, дисперсия среднего и план в каждой ячейке."""
    s = np.sqrt(A[:, IC])
    g = 1.0 / (1.0 - A[:, IV])
    m = np.full(ncell, np.nan)
    var = np.full(ncell, np.nan)
    G = np.zeros((ncell, 2))
    n = np.zeros(ncell)
    for c in range(ncell):
        k = cid == c
        nk = int(k.sum())
        if nk < 12:
            continue
        y = A[k, col]
        m[c] = y.mean()
        var[c] = y.var(ddof=1) / nk
        G[c] = [(s[k] * g[k]).mean(), (s[k] * A[k, IV] * g[k]).mean()]
        n[c] = nk
    return m, var, G, n


def fitpair(Ac, sdc, Af, sdf, col, qc, ncell, drop=None):
    """Подгонка alpha, beta по разностям ячеек. `drop` — выкинутый засев."""
    mc = np.ones(len(Ac), bool)
    mf = np.ones(len(Af), bool)
    if drop is not None:
        mc = sdc != drop
        mf = sdf != drop
    a, va, G, na = summarize(Ac[mc], cells(Ac[mc], qc), col, ncell)
    b, vb, _, nb = summarize(Af[mf], cells(Af[mf], qc), col, ncell)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None, None
    d = a[ok] - b[ok]
    w = 1.0 / (va[ok] + vb[ok])
    D = G[ok]
    Dw = D * w[:, None]
    M = D.T @ Dw
    return np.linalg.solve(M, Dw.T @ d), M


def run(res, ck, fk, tag, Ree_t, Rxx_t, R_t):
    Ac, sdc = res[ck]
    Af, sdf = res[fk]
    qc = np.quantile(np.concatenate([Ac[:, IC], Af[:, IC]]),
                     np.linspace(0, 1, NCB + 1))[1:-1]
    ncell = (len(VB) - 1) * NCB
    print(f'--- {tag} ---')
    for col, nm, vc, ve in ((IX2, 'X_2', V2CAL, V2ERR),
                            (IX3, 'X_3', V3CAL, V3ERR)):
        full, M = fitpair(Ac, sdc, Af, sdf, col, qc, ncell)
        if full is None:
            print(f'  {nm}: ячеек не хватило')
            continue
        us = np.unique(sdc)
        jk = np.array([fitpair(Ac, sdc, Af, sdf, col, qc, ncell, u)[0]
                       for u in us])
        k = len(us)

        def se(x, x0):
            return math.sqrt((k - 1) / k * ((x - x.mean()) ** 2).sum())

        al, be = full
        eal = se(jk[:, 0], al)
        ebe = se(jk[:, 1], be)
        rat = jk[:, 1] / jk[:, 0]
        r0, er = be / al, se(rat, be / al)
        # абсолютные значения: делим на калиброванное v_2 * k
        f = vc * KFAC
        ree, rxx = al / f, be / f
        sc = (-jk[:, 0] + 3 * jk[:, 1]) / f
        s0, es = (-al + 3 * be) / f, se(sc, (-al + 3 * be) / f)
        print(f'  {nm} (обусловленность плана {np.linalg.cond(M):.1f}):')
        print(f'    alpha = {al:+.5f} +- {eal:.5f},  '
              f'beta = {be:+.5f} +- {ebe:.5f}')
        rt = Rxx_t / Ree_t if Ree_t else float('nan')
        print(f'    R_xx/R_etaeta = {r0:+.3f} +- {er:.3f}   '
              f'(истина {rt:+.3f})   {abs(r0-rt)/max(er,1e-12):.1f} сигм')
        print(f'    R_etaeta l²   = {ree:+.5f} +- {eal/abs(f):.5f}   '
              f'(истина {Ree_t:+.5f})')
        print(f'    R_xx l²       = {rxx:+.5f} +- {ebe/abs(f):.5f}   '
              f'(истина {Rxx_t:+.5f})')
        print(f'    R l² (свёртка)= {s0:+.5f} +- {es:.5f}   '
              f'(истина {R_t:+.5f})   {abs(s0-R_t)/max(es,1e-12):.1f} сигм')
    print()


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    fk = [k for k in res if k[0] == 'flat'][0]
    print('набор:', {f'{k[0]} {k[1]}': len(res[k][0]) for k in res})
    print()
    print('=' * 74)
    print('1. КОНТРОЛЬ КОНВЕЙЕРА: половина плоского против другой половины.')
    print('   Кривизны нет по построению — обязаны выйти нули.')
    Af, sdf = res[fk]
    us = np.unique(sdf)
    h1 = np.isin(sdf, us[::2])
    half = {('flatA', 0.0): (Af[h1], sdf[h1]),
            ('flatB', 0.0): (Af[~h1], sdf[~h1])}
    run(half, ('flatA', 0.0), ('flatB', 0.0),
        'ПОЛОВИНА ПЛОСКОГО ПРОТИВ ПОЛОВИНЫ', 1.0, 0.0, 0.0)

    print('2. СЕМЕЙСТВА')
    for k in res:
        if k[0] == 'flat':
            continue
        Ree, Rxx, R = truth(*k)
        run(res, k, fk, f'{k[0]} {k[1]}', Ree, Rxx, R)
    print('ЧТЕНИЕ. Знак отношения обязан смениться между семействами, а')
    print('скаляр R — выйти нулём у линейного и 6c у квадратичного. Нуль')
    print('у линейного получается СОКРАЩЕНИЕМ двух ненулевых измеренных')
    print('чисел, поэтому он проверяет не слепоту замера, а его верность.')


if __name__ == '__main__':
    main()
