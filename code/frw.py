#!/usr/bin/env python3
"""РАННЯЯ ВСЕЛЕННАЯ: плотность энергии излучения, добытая чистым счётом.

ПОЧЕМУ ИМЕННО ОНА. Посильного ВЕЩЕСТВА для нашего замера нет: нужное
число событий идёт как `3.21 rho_планк/rho`, и даже нейтронная звезда
требует 10^79. Но посильная ЭПОХА есть. У излучения во Вселенной
Фридмана `R_uu = 3/(4t²)` в планковских единицах, и при `t` от двадцати
до сотни планковских времён нужное `C_1` попадает ровно в наш диапазон.

Это ровно тот режим, о котором теория причинных множеств и говорит.

УСТРОЙСТВО, И ОНО УДАЧНО. В конформном времени излучение есть
`a(eta) = k*eta`, то есть РОВНО наше конформно-плоское семейство.
Проверено символьно (общие формулы, 0 расхождений):

    R_etaeta = 3(a'²/a² − a''/a),   R_xx = a''/a + a'²/a²,   R = 6a''/a³

при `a = k*eta` выходит `a'' = 0`, значит

    R_etaeta = 3/eta²,   R_xx = 1/eta²,   R = 0 ТОЖДЕСТВЕННО

`R = 0` здесь не устройство, а ФИЗИКА: у излучения след тензора энергии
нуль. Значит отклик обязан идти целиком от `R_uu`, как в линейном
семействе, но теперь по физической причине.

И дальше, с подстановкой `t = k eta²/2`:

    R_uu(сопутствующий) = 3/(k² eta⁴) = 3/(4t²)

— то же число, что даёт Эйнштейн из `8 pi G rho` при
`rho = 3/(32 pi G t²)`. Два независимых пути, сверено символьно.

ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ, и всё это ПРЕДСКАЗАНИЯ, а не подгонки:

  1. КОНТРОЛЬ ПЛОСКОГО с тем же отбором — нули.
  2. ХОД ПО НАКЛОНУ: `R_uu(v) = (3+v²)/(eta⁴(1-v²))`, форма задана
     жёстко, свободных чисел нет.
  3. АБСОЛЮТНАЯ ВЕЛИЧИНА: коэффициенты `v_2 = -0.0201`, `v_3 = -0.0452`
     взяты из ДРУГОГО прогона (`invert.py`, другие семейства, другой
     ящик), поэтому восстановленное `R_uu l²` сравнивается с истиной
     БЕЗ единой подгонки.
  4. ЗАКОН 1/t²: две эпохи, отношение откликов обязано дать
     `(t_2/t_1)²`. Это отношение КАЛИБРОВКИ НЕ ТРЕБУЕТ вовсе.
  5. ИТОГ: плотность энергии Вселенной `rho = R_uu/(8 pi)` из счёта.

  python3 frw.py [--seeds N] [--per M]
"""
import argparse
import math
import pickle
import time

import numpy as np

from chains import c_k, counts, fits

C2C, C3C = c_k(2), c_k(3)
SCRATCH = ('/tmp/claude-0/-home-claude/'
           '8f20417e-eecf-5054-8071-09c440210416/scratchpad')
KFAC = math.sqrt(24.0 / math.pi)
V2CAL, V3CAL = -0.0201, -0.0452          # из `invert.py`, другой прогон
VB = (0.0, 0.12, 0.26, 0.42, 0.58, 0.70)


def a_of(eta, fam, e0):
    e = np.asarray(eta, dtype=float)
    return np.ones_like(e) if fam == 'flat' else e


def R_true(eta, v2, fam):
    """Истина, ТОЛЬКО для сравнения. R_uu = (3+v²)/(eta⁴(1-v²))."""
    if fam == 'flat':
        return 0.0
    return (3.0 + v2) / (eta ** 4 * (1.0 - v2))


def sprinkle(fam, lo, hi, L, rs):
    """Постоянная ФИЗИЧЕСКАЯ плотность 1: по eta как a⁴, по x ровно."""
    e = np.linspace(lo, hi, 40000)
    w = a_of(e, fam, 0) ** 4
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(e))])
    n = rs.poisson(L ** 3 * cdf[-1])
    X = np.empty((n, 4))
    X[:, 0] = np.interp(rs.uniform(0, cdf[-1], n), cdf, e)
    X[:, 1:] = rs.uniform(0, L, (n, 3))
    return X[np.argsort(X[:, 0])]


def fits_box(p, f, lo, hi, L):
    """Ромб I(p,f) целиком внутри ящика [lo,hi] x [0,L]^3.

    Ромб лежит в шаре радиуса de/2 вокруг пространственной середины —
    строгое вложение (вывод в `chains.py`). По времени достаточно концов.
    """
    de = f[0] - p[0]
    mid = 0.5 * (p[1:] + f[1:])
    return bool(p[0] >= lo and f[0] <= hi
                and (mid - de / 2 >= 0).all() and (mid + de / 2 <= L).all())


def harvest(fam, eta0, cfg, rs, want):
    """Интервалы около eta0. Возвращает (C_1, X_2, X_3, v², mid).

    МАСШТАБ ЯЩИКА ДЕЛИТСЯ НА a(eta0). Интервал заданного ФИЗИЧЕСКОГО
    размера занимает в координатах тем меньше места, чем больше `a`,
    поэтому ящик, годный для излучения при eta0 = 5, в пять раз мал для
    плоского контроля. Первый прогон дал на плоском плече РОВНО НОЛЬ
    интервалов именно по этой причине.
    """
    Wp, Lp, lo_c, hi_c, bandp = cfg
    asc = 1.0 if fam == 'flat' else eta0
    W, L, band = Wp / asc, Lp / asc, bandp / asc
    lo, hi = eta0 - W, eta0 + W
    X = sprinkle(fam, lo, hi, L, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    # собственное время интервала: tau_физ ~ a(eta0)*tau_конф
    tlo = (24.0 * lo_c / math.pi) ** 0.25 / asc
    thi = (24.0 * hi_c / math.pi) ** 0.25 / asc
    nb = len(VB) - 1
    quota = want // nb + 1
    got = [0] * nb
    out = []
    guard = 0
    while len(out) < want and guard < want * 30:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, max(lo, eta0 - band - thi * 1.4)),
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
            if not fits_box(np.concatenate([[p0], ps]),
                            np.concatenate([[f0], fs]), lo, hi, L):
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
            _, bb, cc = counts(np.column_stack([tl[m], Sl[m]]))
            out.append((a,
                        bb / (C2C * a * (a - 1.0)) - 1.0,
                        cc / (C3C * a * (a - 1.0) * (a - 2.0)) - 1.0,
                        float(vv[q]), mid))
            got[b] += 1
    del X, t, S
    return out, got


IC, IX2, IX3, IV, IM = 0, 1, 2, 3, 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=22)
    ap.add_argument('--per', type=int, default=560)
    args = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    # W, L, C_1 от, C_1 до, полоса по mid
    # физические масштабы (в ell): полувысота, ширина, C_1 от/до, полоса
    cfg = (13.0, 26.0, 200, 650, 4.0)
    arms = (('flat', 1.0, 'ПЛОСКОЕ — контроль'),
            ('frw', 5.0, 'излучение, eta0 = 5.0'),
            ('frw', 6.5, 'излучение, eta0 = 6.5'))
    print('=' * 78)
    for fam, e0, tag in arms:
        tt = e0 ** 2 / 2.0
        extra = (f', космическое время t = {tt:.1f} планк., '
                 f'истина R_uu l² = {3/e0**4:.3e}') if fam != 'flat' else ''
        print(f'{tag}{extra}')
    print(f'засевов {args.seeds} по {args.per}, полосы v²: {VB}')
    print()
    res = {}
    for fam, e0, tag in arms:
        rows, sdi, tot = [], [], None
        for s in range(args.seeds):
            rs = np.random.default_rng(770000 + 4409 * s + int(100 * e0))
            r, got = harvest(fam, e0, cfg, rs, args.per)
            rows += r
            sdi += [s] * len(r)
            tot = got if tot is None else [x + y for x, y in zip(tot, got)]
        A = np.array(rows)
        if len(A) == 0:
            print(f'--- {tag}: интервалов НЕ НАБРАЛОСЬ — ящик не годится ---')
            print(flush=True)
            continue
        res[(fam, e0)] = (A, np.array(sdi))
        print(f'--- {tag}: {len(A)} интервалов, по полосам v² {tot} ---')
        print(f"{'v²':>8}{'шт':>7}{'C_1':>7}{'X_2':>10}{'ст.ош':>9}"
              f"{'X_3':>10}{'ст.ош':>9}")
        for i in range(len(VB) - 1):
            m = (A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
            if m.sum() < 20:
                continue
            B, s_ = A[m], np.array(sdi)[m]
            us = np.unique(s_)
            k = len(us)
            j2 = np.array([B[s_ != u][:, IX2].mean() for u in us])
            j3 = np.array([B[s_ != u][:, IX3].mean() for u in us])
            e2 = math.sqrt((k - 1) / k * ((j2 - j2.mean()) ** 2).sum())
            e3 = math.sqrt((k - 1) / k * ((j3 - j3.mean()) ** 2).sum())
            print(f'{B[:,IV].mean():>8.3f}{m.sum():>7}{B[:,IC].mean():>7.0f}'
                  f'{B[:,IX2].mean():>+10.4f}{e2:>9.4f}'
                  f'{B[:,IX3].mean():>+10.4f}{e3:>9.4f}')
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
        print()
    with open(f'{SCRATCH}/frw.pkl', 'wb') as fh:
        pickle.dump((res, cfg, VB), fh)
    print(f'  набор сохранён, время {time.time()-t0:.0f} с')
    print('  разбор: python3 frwfit.py')


if __name__ == '__main__':
    main()
