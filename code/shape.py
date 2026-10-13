#!/usr/bin/env python3
"""ВЫБОР МЕЖДУ ДВУМЯ ГИПОТЕЗАМИ, а не подгонка свободных чисел.

ЗАЧЕМ ИМЕННО ТАК. Доставать `R_etaeta` и `R_xx` по отдельности дорого:
вклад `R_xx` составляет не более пятой части сигнала, и чтобы взять его
с точностью 30 %, нужен весь сигнал с точностью 6 %. А сам сигнал
УПЁРТ В ПОТОЛОК: первый порядок держится лишь до `T²R_uu ~ 0.8`,
значит `|X_2| <= |v_2| * 0.8 = 0.016`, сколько бы ни увеличивать
кривизну или размер интервала. Дальше только статистика, а она идёт как
корень.

Но вопрос, ради которого всё затевалось, вовсе не требует точных
компонент. Он такой: ЧЕМУ отзывается счёт цепей — полному `R_uu`
вдоль направления интервала или скаляру `R`? Это выбор из двух, и
данные его решают.

    ГИПОТЕЗА A (отклик на R_uu). Форма отклика по наклону задана жёстко:

        lin:   X_2 ∝ sqrt(C_1) (3 + v²)/(1 - v²)      [R = 0 тождественно]
        quad:  X_2 ∝ sqrt(C_1) (v² - 3)/(1 - v²)

      по ОДНОМУ свободному числу на семейство — общей амплитуде.

    ГИПОТЕЗА B (отклик на скаляр R). Тогда `R` от направления не
    зависит, и

        lin:   X_2 = 0 ВЕЗДЕ, свободных чисел НЕТ вовсе (R = 0)
        quad:  X_2 ∝ sqrt(C_1), без всякого хода по наклону

Линейное семейство разводит гипотезы начисто: при B оно обязано
молчать, при A — иметь сильный ход по наклону. Квадратичное добавляет
проверку на форму хода.

Помеха вычитается парно: плоское плечо проходит тот же отбор и режется
на те же ячейки по `(v², C_1)`.

КОНТРОЛЬ: половина плоского против другой половины обязана
предпочесть... ничего, то есть дать хи² около числа ячеек при НУЛЕВОЙ
амплитуде, и улучшение от гипотезы A обязано быть незначимым.

  python3 shape.py
"""
import math
import pickle

import numpy as np

from pencil import SCRATCH, VB, IC, IX2, IX3, IV

NCB = 4


def cellstats(A, qc, col):
    """Средние по ячейкам (v², C_1): значение, дисперсия среднего, план."""
    bv = np.digitize(A[:, IV], VB) - 1
    bc = np.clip(np.digitize(A[:, IC], qc) - 1, 0, NCB - 1)
    cid = bv * NCB + bc
    nc = (len(VB) - 1) * NCB
    m = np.full(nc, np.nan)
    var = np.full(nc, np.nan)
    sh = np.zeros((nc, 2))      # sqrt(C_1)/(1-v²) и sqrt(C_1) v²/(1-v²)
    for c in range(nc):
        k = cid == c
        n = int(k.sum())
        if n < 25:
            continue
        y = A[k, col]
        m[c] = y.mean()
        var[c] = y.var(ddof=1) / n
        s = np.sqrt(A[k, IC])
        g = 1.0 / (1.0 - A[k, IV])
        sh[c] = [(s * g).mean(), (s * A[k, IV] * g).mean()]
    return m, var, sh


def compare(Ac, Af, col, shapeA, shapeB, tag):
    """Сравнивает гипотезы A и B на разностях ячеек."""
    qc = np.quantile(np.concatenate([Ac[:, IC], Af[:, IC]]),
                     np.linspace(0, 1, NCB + 1))[1:-1]
    mc, vc, sh = cellstats(Ac, qc, col)
    mf, vf, _ = cellstats(Af, qc, col)
    ok = np.isfinite(mc) & np.isfinite(mf)
    d = mc[ok] - mf[ok]
    s = np.sqrt(vc[ok] + vf[ok])
    nd = int(ok.sum())
    # план каждой гипотезы: один столбец (амплитуда) или ни одного
    out = {}
    for nm, vecs in (('A (отклик на R_uu)', shapeA),
                     ('B (отклик на скаляр R)', shapeB)):
        if vecs is None:
            chi = float(((d / s) ** 2).sum())
            out[nm] = (chi, nd, None, None)
            continue
        g = (sh[ok] @ vecs)
        w = 1.0 / s ** 2
        lam = float((w * g * d).sum() / (w * g * g).sum())
        el = float(1.0 / math.sqrt((w * g * g).sum()))
        chi = float((w * (d - lam * g) ** 2).sum())
        out[nm] = (chi, nd - 1, lam, el)
    print(f'  {tag}: ячеек {nd}')
    for nm, (chi, dof, lam, el) in out.items():
        pstr = (f', амплитуда {lam:+.5f} +- {el:.5f} '
                f'({abs(lam)/max(el,1e-12):.1f} сигм)') if lam is not None \
            else ', свободных чисел нет'
        print(f'    {nm}: хи² = {chi:.1f} на {dof} степ.{pstr}')
    a = out['A (отклик на R_uu)']
    b = out['B (отклик на скаляр R)']
    dchi = b[0] - a[0]
    extra = (a[1] is not None) and (b[1] is not None)
    dof = (b[1] - a[1]) if b[1] is not None else 1
    print(f'    => A лучше B на дельта-хи² = {dchi:.1f} при {dof} лишн.'
          f' степ.  ({math.sqrt(max(dchi,0)):.1f} сигм)')
    return out


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/pencil.pkl', 'rb') as fh:
        res = pickle.load(fh)
    fk = [k for k in res if k[0] == 'flat'][0]
    Af, sdf = res[fk]
    us = np.unique(sdf)
    h1 = np.isin(sdf, us[::2])

    for col, cname in ((IX2, 'X_2'), (IX3, 'X_3')):
        print('=' * 72)
        print(cname)
        print('  КОНТРОЛЬ: половина плоского против другой половины')
        compare(Af[h1], Af[~h1], col, np.array([3.0, 1.0]), None,
                'плоское/плоское')
        for k in res:
            if k[0] == 'flat':
                continue
            if k[0] == 'lin':
                # A: R_uu ∝ (3 + v²)/(1-v²).
                # B: R = 0 тождественно, значит отклик НУЛЕВОЙ и
                #    свободных чисел нет вовсе — гипотеза без подгонки.
                sa, sb = np.array([3.0, 1.0]), None
            else:
                # A: R_uu ∝ (v² - 3)/(1-v²).
                # B: R от наклона не зависит, значит отклик ∝ sqrt(C_1),
                #    а это в наших столбцах вектор [1, -1], ибо
                #    g_1 - g_2 = sqrt(C_1) тождественно.
                sa, sb = np.array([-3.0, 1.0]), np.array([1.0, -1.0])
            compare(res[k][0], Af, col, sa, sb, f'{k[0]} {k[1]}')
        print()


if __name__ == '__main__':
    main()
