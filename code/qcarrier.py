#!/usr/bin/env python3
"""ЛЕСТНИЦА НОСИТЕЛЕЙ: три оценщика разноса, без матрицы отношений.

Закон, выведенный из трёх точек (`qruler2-doc.md`): точность есть
дробовой шум НОСИТЕЛЯ, а показатель равен половине размерностной доли
носителя. Носители `N`, `N^{1/2}`, `N^{1/4}` дали 1/2, ~1/4, 1/8.

Отсюда два вопроса, и оба замеряются здесь:

  1. показатель пространства — ровно 1/4 (кромка, коразмерность 2) или
     3/8 (поверхность, коразмерность 1)? Нужен размах V на порядки;
  2. есть ли БОЛЕЕ ТОЛСТЫЙ носитель? Самый толстый возможный —
     элементы, несравнимые С ОБОИМИ: их порядка V, то есть
     коразмерность НОЛЬ. Если разнос в них закодирован, показатель
     обязан быть 1/2, и пространство перестанет быть медленнее времени.

БЕЗ МАТРИЦЫ. Матрица отношений при N = 1e5 заняла бы 10 ГБ и не нужна:
для пары (a,b) всё считается геометрически одним проходом по точкам.

ТРИ ОЦЕНЩИКА:
  V_cp   = |past(a) ∩ past(b)|            — носитель «кромка»
  N_both = |incomp(a) ∩ incomp(b)|        — носитель «объём»
  V_un   = |past(a) ∪ past(b)|            — для контроля (тот же объём)

МЕТОД БЕЗ КОРЗИН (вчерашний урок): калибровка оценщика на ОТДЕЛЬНОЙ
половине пар, затем разброс ОЦЕНКИ в узком окне истинной x.

КОНТРОЛИ:
  1. размеры носителей обязаны идти как заявлено: V_cp ~ N, N_both ~ N
     (оба линейны по N при фиксированной x — это проверка того, что
     «кромка» относится к ФОРМЕ, а не к размеру носителя);
  2. перемешивание x обязано убить связь у каждого оценщика;
  3. у часов на тех же парах обязан выйти показатель 1/2.

  python3 qcarrier.py
"""
import math

import numpy as np


def sprinkle4(n, rs, T=1.0):
    out, got = [], 0
    while got < n:
        m = int((n - got) * 5) + 1024
        t = rs.random(m).astype(np.float32) * T
        v = rs.normal(size=(m, 3)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m).astype(np.float32) ** (1.0 / 3.0)) * (T / 2)
        ok = rad < np.minimum(t, T - t)
        if ok.any():
            out.append(np.column_stack([t[ok], (v * rad[:, None])[ok]]))
            got += int(ok.sum())
    return np.vstack(out)[:n]


def harvest(n, rs, npair, xlo=0.4, xhi=0.8, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    t = P[:, 0]
    X = P[:, 1:]
    tau = np.sqrt(np.maximum(t.astype(np.float64) ** 2
                             - (X.astype(np.float64) ** 2).sum(1), 0.0))
    # отбираем кандидатов с возрастом в середине диаманта
    mid = (tau > 0.35 * T) & (tau < 0.65 * T)
    cand = np.nonzero(mid)[0]
    order = cand[np.argsort(tau[cand])]
    ta = tau[order]
    rows = []
    tries, lim = 0, npair * 60
    while len(rows) < npair and tries < lim:
        tries += 1
        i = int(rs.integers(len(order)))
        a = int(order[i])
        lo = np.searchsorted(ta, tau[a] * (1 - agew))
        hi = np.searchsorted(ta, tau[a] * (1 + agew))
        if hi - lo < 2:
            continue
        b = int(order[int(rs.integers(lo, hi))])
        if a == b:
            continue
        dt = float(t[b]) - float(t[a])
        dx2 = float(((X[b] - X[a]) ** 2).sum())
        if dt * dt > dx2:           # связаны — пропускаем
            continue
        sep = math.sqrt(max(dx2 - dt * dt, 0.0))
        tb = 0.5 * (tau[a] + tau[b])
        xv = sep / tb
        if not (xlo <= xv < xhi):
            continue
        # геометрия одним проходом
        dta = t[a] - t
        dtb = t[b] - t
        da = ((X - X[a]) ** 2).sum(1)
        db = ((X - X[b]) ** 2).sum(1)
        pa = (dta > 0) & (dta * dta > da)
        pb = (dtb > 0) & (dtb * dtb > db)
        fa = (dta < 0) & (dta * dta > da)
        fb = (dtb < 0) & (dtb * dtb > db)
        vcp = int((pa & pb).sum())
        nboth = int((~pa & ~fa & ~pb & ~fb).sum())
        vun = int((pa | pb).sum())
        rows.append((xv, vcp, nboth, vun, int(pa.sum())))
    return np.array(rows, dtype=np.float64)


def budget(D, col, xlo, xhi, nb=300):
    """Разброс ОЦЕНКИ при известной x. Калибровка на половине."""
    half = len(D) // 2
    cal, ev = D[:half], D[half:]
    o = cal[:, col]
    qs = np.quantile(o, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(qs, o) - 1, 0, nb - 1)
    s = np.zeros(nb)
    c = np.zeros(nb)
    np.add.at(s, idx, cal[:, 0])
    np.add.at(c, idx, 1.0)
    g = c >= 8
    xbar = np.full(nb, np.nan)
    xbar[g] = s[g] / c[g]
    ie = np.clip(np.searchsorted(qs, ev[:, col]) - 1, 0, nb - 1)
    xh = xbar[ie]
    m = (ev[:, 0] >= xlo) & (ev[:, 0] < xhi) & np.isfinite(xh)
    if m.sum() < 300:
        return None
    xm = float(ev[m, 0].mean())
    err = float(np.std(xh[m] - ev[m, 0], ddof=1))
    return xm / err, int(m.sum()), xm


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(141421)
    plan = [(20000, 9000), (60000, 6000), (200000, 3000), (600000, 1500)]
    data = {}
    for n, npr in plan:
        D = harvest(n, rs, npr)
        data[n] = D
        print(f'  N = {n:>7}: пар {len(D):>5}, '
              f'<V_cp> = {D[:,1].mean():>9.0f}, '
              f'<N_both> = {D[:,2].mean():>9.0f}, '
              f'<звеньев к a> ~ {D[:,4].mean():>8.0f}', flush=True)
    print()
    print('КОНТРОЛЬ 1: размеры носителей против N (оба обязаны быть ~N)')
    Ns = np.array(sorted(data))
    for col, nm in ((1, 'V_cp'), (2, 'N_both'), (3, 'V_un')):
        v = np.array([data[n][:, col].mean() for n in Ns])
        a, _ = np.polyfit(np.log(Ns), np.log(v), 1)
        print(f'  {nm:>7} ~ N^{a:.4f}')
    print()
    print('БЮДЖЕТЫ ТРЁХ ОЦЕНЩИКОВ (окно x = 0.5-0.65)')
    print(f"{'N':>8}{'V_cp':>10}{'N_both':>10}{'V_un':>10}")
    res = {1: [], 2: [], 3: []}
    for n in Ns:
        line = f'{n:>8}'
        for col in (1, 2, 3):
            r = budget(data[n], col, 0.5, 0.65)
            if r is None:
                line += f'{"—":>10}'
            else:
                res[col].append((data[n][:, col].mean(), r[0], r[1]))
                line += f'{r[0]:>10.2f}'
        print(line, flush=True)
    print()
    for col, nm in ((1, 'V_cp (кромка?)'), (2, 'N_both (объём?)'),
                    (3, 'V_un')):
        R = np.array(res[col])
        if len(R) < 3:
            continue
        err = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
        A = np.vstack([np.log(R[:, 0]), np.ones(len(R))]).T
        W = np.diag(1.0 / (err / R[:, 1]) ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        be = cov @ (A.T @ W @ np.log(R[:, 1]))
        se = np.sqrt(np.diag(cov))
        rr = (np.log(R[:, 1]) - A @ be) / (err / R[:, 1])
        print(f'{nm:>18}: показатель {be[0]:.4f} ± {se[0]:.4f}; '
              f'против 1/4 {abs(be[0]-0.25)/se[0]:.1f}с, '
              f'против 3/8 {abs(be[0]-0.375)/se[0]:.1f}с, '
              f'против 1/2 {abs(be[0]-0.5)/se[0]:.1f}с')
        print(f'{"":>18}  остатки: ' + ', '.join(f'{v:+.1f}' for v in rr))
    print()
    print('КОНТРОЛЬ 2: перемешать x — бюджет обязан упасть к ~1')
    D = data[Ns[-2]]
    Dp = D.copy()
    Dp[:, 0] = rs.permutation(D[:, 0])
    for col, nm in ((1, 'V_cp'), (2, 'N_both')):
        r = budget(Dp, col, 0.5, 0.65)
        print(f'  {nm}: {r[0]:.3f}' if r else f'  {nm}: —')


if __name__ == '__main__':
    main()
