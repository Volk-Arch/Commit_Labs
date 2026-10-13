#!/usr/bin/env python3
"""ИЗБЫТОЧЕН ЛИ C_4: проверка догадки про спектр обилий.

ДОГАДКА. Генерирующий объект — спектр обилий цепей интервала
`{C_1, C_2, C_3, ...}`, а вся геометрия есть его отклонение от плоских
значений `c_k`. Уравнение поля — обращение в нуль ПЕРВОГО порядка этого
отклонения; Риччи есть ровно то, что первый порядок видит, а Вейль
появляется во втором.

ЧТО ЭТО ПРЕДСКАЗЫВАЕТ, И ЧЕМ ПРОВЕРЯЕТСЯ. В конформно-плоском
пространстве тензор Вейля равен нулю ПО ПОСТРОЕНИЮ (конформные
преобразования сохраняют световые конусы). Значит там вся кривизна —
Риччи, то есть в наших изотропных семействах ДВА числа. Тогда обилия
`C_2`, `C_3`, `C_4` обязаны быть ИЗБЫТОЧНЫ: все три должны дать одно и
то же `R_uu`, без остатка.

    если дают одно и то же  -> первый порядок исчерпывается Риччи,
                               половина догадки подтверждена;
    если C_4 несёт лишнее   -> догадка НЕВЕРНА, и это видно сразу.

C_4 СЧИТАЕТСЯ ДАРОМ. Четырёхцепочка `x1<x2<x3<x4` разбирается по
СРЕДНЕМУ ЗВЕНУ `(x2,x3)`: тогда `x1` в прошлом `x2`, `x4` в будущем
`x3`, и

    C_4 = sum_{y<z} |прошлое(y)| * |будущее(z)| = p^T M f

то есть та же `O(C_1²)`, что уже платится за `C_2` и `C_3`. Перебора
четвёрок нет.

КОНТРОЛЬ ПЛОСКОГО обязателен и здесь: `c_4` берётся из той же формулы,
что `c_2 = 1/20` и `c_3 = 1/2100`, а она проверена замером только для
`k = 2, 3`. Поэтому плоское плечо сначала подтверждает `c_4`, и лишь
потом смотрим на отклик.

  python3 c4.py [--seeds N] [--per M]
"""
import argparse
import math
import pickle
import time

import numpy as np

from chains import c_k, fits
from pencil import sprinkle, SCRATCH, KFAC, VB

C2C, C3C, C4C = c_k(2), c_k(3), c_k(4)


def counts4(I):
    """C_1, C_2, C_3, C_4 из порядка на интервале."""
    k = len(I)
    if k < 4:
        return k, 0, 0, 0
    dt = I[:, 0][None, :] - I[:, 0][:, None]
    dx = I[:, 1:][None, :, :] - I[:, 1:][:, None, :]
    M = ((dt > 0) & (dt ** 2 > (dx * dx).sum(2))).astype(np.int64)
    past = M.sum(0)
    fut = M.sum(1)
    c2 = int(M.sum())
    c3 = int((past * fut).sum())
    c4 = int(past @ M @ fut)       # p^T M f
    return k, c2, c3, c4


def harvest(fam, par, eta0, cfg, rs, want):
    W, L, lo_c, hi_c, band = cfg
    X = sprinkle(fam, par, L, 2 * eta0, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo_c / math.pi) ** 0.25
    thi = (24.0 * hi_c / math.pi) ** 0.25
    nb = len(VB) - 1
    quota = want // nb + 1
    got = [0] * nb
    out = []
    guard = 0
    while len(out) < want and guard < want * 30:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - band - thi * 1.4),
                             np.searchsorted(t, eta0 + band)))
        p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 2.2)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0:
            continue
        vv = dx2[ok] / (de[ok] ** 2)
        bi = np.digitize(vv, VB) - 1
        order = np.argsort([got[b] if 0 <= b < nb else 10 ** 9 for b in bi])
        for q in order[:8]:
            if len(out) >= want:
                break
            b = bi[q]
            if not (0 <= b < nb) or got[b] >= quota:
                continue
            jj = int(ok[q])
            f0, fs = t[j0 + jj], S[j0 + jj]
            mid = 0.5 * (p0 + f0)
            if abs(mid - eta0) > band:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, 2 * eta0):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo_c:
                continue
            Sl, tl = S[k0:k1], t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo_c <= a <= hi_c):
                continue
            _, c2, c3, c4 = counts4(np.column_stack([tl[m], Sl[m]]))
            out.append((a,
                        c2 / (C2C * a * (a - 1.0)) - 1.0,
                        c3 / (C3C * a * (a - 1.0) * (a - 2.0)) - 1.0,
                        c4 / (C4C * a * (a - 1.0) * (a - 2.0)
                              * (a - 3.0)) - 1.0,
                        float(vv[q]),
                        c2 / (a * (a - 1.0)),
                        c3 / (a * (a - 1.0) * (a - 2.0)),
                        c4 / (a * (a - 1.0) * (a - 2.0) * (a - 3.0))))
            got[b] += 1
    del X, t, S
    return out


IC, IX2, IX3, IX4, IV, IH2, IH3, IH4 = range(8)


def jk(vals):
    n = len(vals)
    m = vals.mean()
    return m, math.sqrt((n - 1) / n * ((vals - m) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--per', type=int, default=600)
    a_ = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print(f'формула даёт c_2 = {C2C:.8f} (1/{1/C2C:.1f}), '
          f'c_3 = {C3C:.3e} (1/{1/C3C:.0f}),')
    print(f'              c_4 = {C4C:.4e} (1/{1/C4C:.0f})')
    print()
    cfg = (0.0, 26.0, 200, 650, 4.0)
    arms = (('flat', 0.0, 10.0, 'ПЛОСКОЕ — контроль c_4'),
            ('lin', 0.040, 10.0, 'линейное h = 0.040, R = 0, Вейль = 0'))
    res = {}
    for fam, par, e0, tag in arms:
        rows, sdi = [], []
        for s in range(a_.seeds):
            rs = np.random.default_rng(880000 + 6151 * s + int(1e6 * par))
            r = harvest(fam, par, e0, cfg, rs, a_.per)
            rows += r
            sdi += [s] * len(r)
        A = np.array(rows)
        res[(fam, par)] = (A, np.array(sdi))
        print(f'--- {tag}: {len(A)} интервалов ---')
        if fam == 'flat':
            print('  ПРОВЕРКА ПОСТОЯННЫХ (отношение замера к формуле):')
            for i, (ih, cc, nm) in enumerate(((IH2, C2C, 'c_2'),
                                              (IH3, C3C, 'c_3'),
                                              (IH4, C4C, 'c_4'))):
                us = np.unique(sdi)
                v = np.array([A[np.array(sdi) != u][:, ih].mean()
                              for u in us])
                m, e = A[:, ih].mean(), jk(v)[1]
                print(f'    {nm}: {m:.6e} / {cc:.6e} = {m/cc:.4f}'
                      f' +- {e/cc:.4f}   ({abs(m/cc-1)/max(e/cc,1e-12):.1f} '
                      f'сигм от единицы)')
        print(f"{'v²':>8}{'шт':>7}{'C_1':>7}{'X_2':>10}{'ст.ош':>9}"
              f"{'X_3':>10}{'ст.ош':>9}{'X_4':>10}{'ст.ош':>9}")
        for i in range(len(VB) - 1):
            m = (A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
            if m.sum() < 20:
                continue
            B, s_ = A[m], np.array(sdi)[m]
            us = np.unique(s_)
            row = [B[:, IV].mean(), m.sum(), B[:, IC].mean()]
            for ix in (IX2, IX3, IX4):
                v = np.array([B[s_ != u][:, ix].mean() for u in us])
                row += [B[:, ix].mean(), jk(v)[1]]
            print(f'{row[0]:>8.3f}{row[1]:>7}{row[2]:>7.0f}'
                  f'{row[3]:>+10.4f}{row[4]:>9.4f}'
                  f'{row[5]:>+10.4f}{row[6]:>9.4f}'
                  f'{row[7]:>+10.4f}{row[8]:>9.4f}')
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
        print()
    with open(f'{SCRATCH}/c4.pkl', 'wb') as fh:
        pickle.dump(res, fh)

    # ИЗБЫТОЧНОСТЬ: остаётся ли у обилия структура по наклону?
    print('=' * 74)
    print('ИЗБЫТОЧНОСТЬ, И ТЕСТ НЕ ТРЕБУЕТ ТЕОРИИ. Для каждого обилия')
    print('отдельно подгоняется ОДНА амплитуда к форме (3+v²)/(1-v²), с')
    print('парно вычтенным плоским плечом. Если обилие видит только Риччи,')
    print('остатка по наклону быть не должно: хи² на число полос около')
    print('единицы. Если C_4 видит что-то СВЕРХ Риччи, его зависимость от')
    print('наклона будет другой, и хи² вырастет — догадка тогда неверна.')
    Ac, sc = res[('lin', 0.040)]
    Af, sf = res[('flat', 0.0)]
    print()
    print(f"{'обилие':>9}{'амплитуда':>13}{'ст.ош':>11}{'сигм':>7}"
          f"{'хи²':>8}{'полос':>7}{'хи²/полоса':>12}")
    amp = {}
    for ix, nm in ((IX2, 'C_2'), (IX3, 'C_3'), (IX4, 'C_4')):
        cells = []
        for i in range(len(VB) - 1):
            ma = (Ac[:, IV] >= VB[i]) & (Ac[:, IV] < VB[i + 1])
            mb = (Af[:, IV] >= VB[i]) & (Af[:, IV] < VB[i + 1])
            if ma.sum() < 30 or mb.sum() < 30:
                continue
            g = float(np.mean(np.sqrt(Ac[ma][:, IC])
                              * (3.0 + Ac[ma][:, IV])
                              / (1.0 - Ac[ma][:, IV])))
            us = np.unique(sc)
            jv = np.array([Ac[ma & (sc != u)][:, ix].mean()
                           - Af[mb & (sf != u)][:, ix].mean() for u in us])
            d = Ac[ma][:, ix].mean() - Af[mb][:, ix].mean()
            n = len(us)
            e = math.sqrt((n - 1) / n * ((jv - jv.mean()) ** 2).sum())
            cells.append((g, d, max(e, 1e-12)))
        C = np.array(cells)
        w = 1.0 / C[:, 2] ** 2
        lam = float((w * C[:, 0] * C[:, 1]).sum() / (w * C[:, 0] ** 2).sum())
        el = float(1.0 / math.sqrt((w * C[:, 0] ** 2).sum()))
        chi = float((w * (C[:, 1] - lam * C[:, 0]) ** 2).sum())
        dof = len(C) - 1
        amp[nm] = (lam, el)
        print(f'{nm:>9}{lam:>+13.3e}{el:>11.3e}'
              f'{abs(lam)/max(el,1e-30):>7.1f}{chi:>8.2f}{len(C):>7}'
              f'{chi/max(dof,1):>12.2f}')
    print()
    print('ИЗМЕРЕННЫЕ ОТНОШЕНИЯ ОТКЛИКОВ (универсальные числа, калибровки')
    print('не требуют; сверять с коэффициентами Роя-Синхи-Сурьи):')
    b2 = amp['C_2']
    for nm in ('C_3', 'C_4'):
        v, e = amp[nm]
        r = v / b2[0]
        er = abs(r) * math.sqrt((e / v) ** 2 + (b2[1] / b2[0]) ** 2)
        print(f'    {nm}/C_2 = {r:+.3f} +- {er:.3f}')
    print()
    print('ЧТЕНИЕ. Малое хи²/полоса у ВСЕХ трёх обилий означает, что в')
    print('конформно-плоском пространстве первый порядок исчерпывается')
    print('двумя числами Риччи, а обилия суть разные проекции тех же двух.')
    print('Это половина догадки. Вторая половина — что Вейль появляется во')
    print('втором порядке — здесь не проверяется: Вейль тут нуль по')
    print('построению, и для неё нужна НЕ конформно-плоская метрика.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
