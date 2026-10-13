#!/usr/bin/env python3
"""РЕШАЮЩИЙ ВНУТРЕННИЙ ТЕСТ: ориентация при той же точке.

ЗАЧЕМ. Общая подгонка сказала, что счёт цепей отзывается на `R_uu` и
слеп к `R`. Но она сравнивает РАЗНЫЕ семейства, и несогласие между ними
могло бы подделать такой вывод. Нужен тест ВНУТРИ одного семейства, где
подделать нечему.

Он есть, и это ровно его мысль про карандаш цепей. В точке `eta` скаляр
`R` задан самой точкой и от направления НЕ ЗАВИСИТ. А `R_uu` зависит:
для квадратичного семейства в центре

    R_uu(v) = c (v² − 3) / (1 − v²)

то есть от `−3c` при покое до `−6.6c` при `v = 0.8`. Значит интервалы,
взятые в ОДНОМ месте, но под разным наклоном, видят ОДИН И ТОТ ЖЕ `R` и
РАЗНЫЕ `R_uu`.

ТОГДА ВЫБОР ЖЁСТКИЙ:
  если `X_2` следует за `R_uu` поперёк наклонов — отзыв идёт от `R_uu`;
  если `X_2` постоянен поперёк наклонов — отзыв шёл бы от `R`.
Третьего нет, и никакая разница между семействами сюда не вмешивается.

ЕЩЁ ДВЕ ПРОВЕРКИ ЗДЕСЬ:
  * подгонка ВНУТРИ квадратичного семейства отдельно — её `u` и `v`
    обязаны совпасть с общими;
  * то же в ОКНЕ ПРИГОДНОСТИ `|T²R| <= 0.3` и `|T²R_00| <= 0.3`, где
    первый порядок точно в силе: если промах сильного семейства в 31 %
    шёл от высших порядков, в окне он обязан уйти.

  python3 orient.py
"""
import math
import pickle

import numpy as np

from invert import SCRATCH, C1, X2, X3, AR, AU, fit, pool


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/curve.pkl', 'rb') as fh:
        res = pickle.load(fh)
    flat = [k for k in res if k[0] == 'flat']
    quad = [k for k in res if k[0] == 'quad']
    lin = [k for k in res if k[0] == 'lin']

    print('=' * 74)
    print('1. ОРИЕНТАЦИЯ ВНУТРИ КВАДРАТИЧНОГО СЕМЕЙСТВА')
    print('   Полосы по наклону. R меняется мало (он от точки), R_00 —')
    print('   сильно (он от направления). За чем следует X_2?')
    A, sd = pool(res, quad)
    # наклон виден в отношении R_00/R: оно есть (v²−3)/(6(1−v²))
    rat = A[:, AU] / A[:, AR]
    qs = np.quantile(rat, [0, 0.25, 0.5, 0.75, 1.0])
    print()
    print(f"{'R_00/R':>10}{'шт':>7}{'T²R':>9}{'T²R_00':>10}"
          f"{'X_2':>10}{'ст.ош':>9}{'X_3':>10}{'ст.ош':>9}")
    rows = []
    for i in range(4):
        m = (rat >= qs[i]) & (rat <= qs[i + 1])
        B, s = A[m], sd[m]
        n = len(B)
        # ошибка складным ножом по засевам, не по интервалам
        us = np.unique(s)
        j2 = np.array([B[s != u_][:, X2].mean() for u_ in us])
        j3 = np.array([B[s != u_][:, X3].mean() for u_ in us])
        k = len(us)
        e2 = math.sqrt((k - 1) / k * ((j2 - j2.mean()) ** 2).sum())
        e3 = math.sqrt((k - 1) / k * ((j3 - j3.mean()) ** 2).sum())
        rows.append((B[:, AR].mean(), B[:, AU].mean(),
                     B[:, X2].mean(), e2, B[:, X3].mean(), e3))
        print(f'{rat[m].mean():>10.2f}{n:>7}{B[:,AR].mean():>9.3f}'
              f'{B[:,AU].mean():>10.3f}{B[:,X2].mean():>+10.4f}{e2:>9.4f}'
              f'{B[:,X3].mean():>+10.4f}{e3:>9.4f}')
    Rr = np.array([r[0] for r in rows])
    Uu = np.array([r[1] for r in rows])
    print()
    print(f'  поперёк полос: T²R меняется на '
          f'{(Rr.max()/Rr.min()-1)*100:.0f} %, '
          f'T²R_00 — на {(abs(Uu).max()/abs(Uu).min()-1)*100:.0f} %')
    y2 = np.array([r[2] for r in rows])
    s2 = np.array([r[3] for r in rows])
    # следует ли X_2 за R_00 при почти постоянном R
    k2 = np.polyfit(Uu, y2, 1)[0]
    print(f'  наклон X_2 по T²R_00 поперёк полос: {k2:+.4f}')
    print(f'  (общая подгонка давала v = -0.0204)')
    print()

    print('1b. ТО ЖЕ В ЛИНЕЙНОМ СЕМЕЙСТВЕ, ГДЕ R ТОЖДЕСТВЕННО НУЛЬ.')
    print('    Это чистая версия теста: спутать не с чем вовсе. Наклон')
    print('    меняет R_uu = h²(3+v²)/(a⁴(1−v²)) в 3.4 раза при R = 0.')
    print('    Полосы по СЫРОМУ R_00 (он от места и наклона, не от размера).')
    AL, sL = pool(res, lin)
    raw = AL[:, 4]
    qs = np.quantile(raw, [0, 0.25, 0.5, 0.75, 1.0])
    print()
    print(f"{'R_00 l²':>11}{'шт':>7}{'C_1':>8}{'T²R_00':>10}"
          f"{'X_2':>10}{'ст.ош':>9}{'X_3':>10}{'ст.ош':>9}")
    pts = []
    for i in range(4):
        m = (raw >= qs[i]) & (raw <= qs[i + 1])
        B, s = AL[m], sL[m]
        us = np.unique(s)
        k = len(us)
        j2 = np.array([B[s != u_][:, X2].mean() for u_ in us])
        j3 = np.array([B[s != u_][:, X3].mean() for u_ in us])
        e2 = math.sqrt((k - 1) / k * ((j2 - j2.mean()) ** 2).sum())
        e3 = math.sqrt((k - 1) / k * ((j3 - j3.mean()) ** 2).sum())
        pts.append((B[:, AU].mean(), B[:, X2].mean(), e2,
                    B[:, X3].mean(), e3))
        print(f'{raw[m].mean():>11.5f}{m.sum():>7}{B[:,C1].mean():>8.0f}'
              f'{B[:,AU].mean():>10.3f}{B[:,X2].mean():>+10.4f}{e2:>9.4f}'
              f'{B[:,X3].mean():>+10.4f}{e3:>9.4f}')
    P = np.array(pts)
    d2 = P[-1, 1] - P[0, 1]
    s2d = math.hypot(P[-1, 2], P[0, 2])
    d3 = P[-1, 3] - P[0, 3]
    s3d = math.hypot(P[-1, 4], P[0, 4])
    du = P[-1, 0] - P[0, 0]
    print()
    print(f'  крайние полосы: T²R_00 меняется на {du:+.3f}')
    print(f'    X_2 меняется на {d2:+.4f} +- {s2d:.4f} '
          f'({abs(d2)/s2d:.1f} сигм)  =>  v = {d2/du:+.4f}')
    print(f'    X_3 меняется на {d3:+.4f} +- {s3d:.4f} '
          f'({abs(d3)/s3d:.1f} сигм)  =>  v = {d3/du:+.4f}')
    print('    общая подгонка давала v = -0.0204 и -0.0468')
    print()

    print('2. ПОДГОНКА ВНУТРИ ОТДЕЛЬНЫХ СЕМЕЙСТВ — обязаны совпасть')
    for nm, keys, pure in (('только квадратичные', quad, False),
                           ('только линейные (R = 0, столбца R нет)',
                            lin, True)):
        Ax, sx = pool(res, keys + flat)
        print(f'  {nm}:')
        for col, cn in ((X2, 'X_2'), (X3, 'X_3')):
            if pure:
                D = np.column_stack([Ax[:, AU], np.ones(len(Ax)),
                                     1.0 / np.sqrt(Ax[:, C1])])
                w = Ax[:, C1]
                us = np.unique(sx)
                def sol(mm):
                    Dw = D[mm] * w[mm][:, None]
                    return np.linalg.solve(D[mm].T @ Dw, Dw.T @ Ax[mm, col])
                allm = np.ones(len(Ax), bool)
                c = sol(allm)
                jk = np.array([sol(sx != u_) for u_ in us])
                k = len(us)
                e = math.sqrt((k - 1) / k
                              * ((jk[:, 0] - jk[:, 0].mean()) ** 2).sum())
                print(f'    {cn}: v = {c[0]:+.4f} +- {e:.4f} '
                      f'({abs(c[0])/e:.1f} сигм)')
            else:
                c, cv = fit(Ax, sx, col)
                e = np.sqrt(np.diag(cv))
                print(f'    {cn}: u = {c[0]:+.4f} +- {e[0]:.4f},'
                      f'   v = {c[1]:+.4f} +- {e[1]:.4f}')
    print()

    print('3. ОКНО ПРИГОДНОСТИ |T²R| <= 0.3 и |T²R_00| <= 0.3')
    Aa, sa = pool(res, quad + lin + flat)
    m = (np.abs(Aa[:, AR]) <= 0.3) & (np.abs(Aa[:, AU]) <= 0.3)
    print(f'  в окне {m.sum()} интервалов из {len(Aa)}')
    for col, nm in ((X2, 'X_2'), (X3, 'X_3')):
        c, cv = fit(Aa[m], sa[m], col)
        e = np.sqrt(np.diag(cv))
        print(f'  {nm}: u = {c[0]:+.4f} +- {e[0]:.4f} '
              f'({abs(c[0])/e[0]:.1f} сигм),   '
              f'v = {c[1]:+.4f} +- {e[1]:.4f} ({abs(c[1])/e[1]:.1f} сигм)')
    print()
    print('4. ЧИСТО ПО R_00: подгонка с ПРИНУДИТЕЛЬНЫМ u = 0')
    print('   Если так согласие не портится, скаляр R действительно лишний.')
    for col, nm in ((X2, 'X_2'), (X3, 'X_3')):
        D = np.column_stack([Aa[:, AU], np.ones(len(Aa)),
                             1.0 / np.sqrt(Aa[:, C1])])
        w = Aa[:, C1]
        Dw = D * w[:, None]
        cc = np.linalg.solve(D.T @ Dw, Dw.T @ Aa[:, col])
        r1 = Aa[:, col] - D @ cc
        chi1 = float((w * r1 ** 2).sum())
        D2 = np.column_stack([Aa[:, AR], D])
        Dw2 = D2 * w[:, None]
        cc2 = np.linalg.solve(D2.T @ Dw2, Dw2.T @ Aa[:, col])
        r2 = Aa[:, col] - D2 @ cc2
        chi2 = float((w * r2 ** 2).sum())
        print(f'  {nm}: v = {cc[0]:+.4f} при u = 0;  '
              f'добавление R снижает взвешенный остаток на '
              f'{(1-chi2/chi1)*100:.4f} %')


if __name__ == '__main__':
    main()
