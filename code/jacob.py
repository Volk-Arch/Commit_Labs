#!/usr/bin/env python3
"""B2: НЕДОСТАЧА ПЛОЩАДИ ПРИ ФИКСИРОВАННОМ ОБЪЁМЕ — счётом.

ЗАМЫСЕЛ. Якобсон и Виссер (SciPost Phys. 7, 079 (2019)): уравнение
Эйнштейна выполнено тогда и только тогда, когда обобщённая энтропия
стационарна при фиксированном объёме в малых ромбах. Простая форма:
недостача площади границы при фиксированном объёме внутри равна энергии
внутри — круг на апельсине короче круга той же площади на бумаге.

У интервала есть ДВА независимых порядковых счёта:

    объём   C_1
    площадь c_0 — амплитуда при phi_0(k) = Г(k+1/2)/Г(k+1) в профиле
            F_k (выведено и проверено в `joint.py`: показатель 0.5156,
            отношение к площади талии постоянно до 4.4 %)

ПРЕДСКАЗАНИЕ, посчитанное самостоятельно в `predict.py` (не выужено из
статьи): для излучения `dA/A = -(1/60) tau² R_uu` с нулевым остатком по
двенадцати конфигурациям. Коэффициент есть ровно `B/2`, где `B = 1/30`
есть коэффициент Гиббонса–Солодухина: площадь талии при `a = k eta`
равна `pi tau²` ТОЧНО, поэтому вся недостача идёт от объёма, а корень
даёт половину. Свободы в коэффициенте нет.

В счёте, с `tau² = sqrt(24 C_1/pi)`:

    dc_0/c_0 = -0.0460 * sqrt(C_1) * R_uu l²

ПОЧЕМУ КАЛИБРОВКА НЕ НУЖНА. Величина ОТНОСИТЕЛЬНАЯ и берётся парным
вычитанием: плоское плечо проходит тот же отбор, те же полосы по
`(C_1, v²)`. Поэтому постоянная 1.35 в отождествлении «c_0 = площадь» и
поправка на конечный размер `O(C_1^(-1/4))` сокращаются нацело.

ДВЕ ПРОВЕРКИ, и вторая самодостаточна:
  1. против ИСТИНЫ `R_uu = 3/eta_w⁴` — предсказание без свободных чисел;
  2. против `R_uu`, добытого НЕЗАВИСИМО из обилий `C_2`, `C_3` на тех же
     самых интервалах. Это и есть уравнение поля как условие
     согласования двух счётов одного интервала.

ОГРАНИЧЕНИЕ ОТБОРА. Предсказание выведено для НЕПОДВИЖНОЙ пары: при
бусте талия перестаёт быть сферой постоянного `eta`, и площадь уже не
`4 pi r² a²`. Поэтому берутся только слабо наклонённые интервалы,
`v² < 0.25`, и это сказано заранее, а не выбрано по исходу.

КОНТРОЛЬ: половина плоского против другой половины обязана дать нуль.

  python3 jacob.py [--seeds N] [--per M]
"""
import argparse
import math
import pickle
import time

import numpy as np

from chains import c_k, fits
from joint import degrees, phi, KMAX

C2C, C3C = c_k(2), c_k(3)
SCRATCH = ('/tmp/claude-0/-home-claude/'
           '8f20417e-eecf-5054-8071-09c440210416/scratchpad')
V2CAL = -0.0201
KFAC = math.sqrt(24.0 / math.pi)
PRED = -0.0460          # из `predict.py`: -(1/60)*sqrt(24/pi)
VMAX = 0.25             # только слабо наклонённые, объявлено заранее


def a_of(e, fam):
    e = np.asarray(e, dtype=float)
    return np.ones_like(e) if fam == 'flat' else e


def sprinkle(fam, lo, hi, L, rs):
    e = np.linspace(lo, hi, 40000)
    w = a_of(e, fam) ** 4
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(e))])
    n = rs.poisson(L ** 3 * cdf[-1])
    X = np.empty((n, 4))
    X[:, 0] = np.interp(rs.uniform(0, cdf[-1], n), cdf, e)
    X[:, 1:] = rs.uniform(0, L, (n, 3))
    return X[np.argsort(X[:, 0])]


def fits_box(p, f, lo, hi, L):
    de = f[0] - p[0]
    mid = 0.5 * (p[1:] + f[1:])
    return bool(p[0] >= lo and f[0] <= hi
                and (mid - de / 2 >= 0).all() and (mid + de / 2 <= L).all())


def harvest(fam, eta0, cfg, rs, want):
    Wp, Lp, lo_c, hi_c, bandp = cfg
    asc = 1.0 if fam == 'flat' else eta0
    W, L, band = Wp / asc, Lp / asc, bandp / asc
    lo, hi = eta0 - W, eta0 + W
    X = sprinkle(fam, lo, hi, L, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo_c / math.pi) ** 0.25 / asc
    thi = (24.0 * hi_c / math.pi) ** 0.25 / asc
    out = []
    guard = 0
    while len(out) < want and guard < want * 30:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, max(lo, eta0 - band - thi)),
                             np.searchsorted(t, eta0 + band)))
        p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 1.3)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        vv = dx2 / np.maximum(de * de, 1e-30)
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi)
                            & (vv < VMAX))
        if len(ok) == 0:
            continue
        for q in rs.permutation(ok)[:6]:
            if len(out) >= want:
                break
            jj = int(q)
            f0, fs = t[j0 + jj], S[j0 + jj]
            if abs(0.5 * (p0 + f0) - eta0) > band:
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
            pas, fut = degrees(tl[m], Sl[m])
            c2 = int((fut > 0).sum() * 0)      # заполним ниже из fut
            # C_2 и C_3 из степеней: C_2 = sum fut, C_3 = sum past*fut
            c2 = int(fut.sum())
            c3 = int((pas * fut).sum())
            hf = np.bincount(fut, minlength=KMAX + 1)[:KMAX + 1]
            hp = np.bincount(pas, minlength=KMAX + 1)[:KMAX + 1]
            out.append([a, c2, c3, float(vv[jj]),
                        0.5 * (p0 + f0)] + list(hf) + list(hp))
    del X, t, S
    return out


IC, I2, I3, IV, IM = 0, 1, 2, 3, 4
HF = 5


def c0_of(rows, ks, B):
    """Амплитуда при phi_0 по профилю F_k и P_k, взвешенно по Пуассону."""
    y = np.concatenate([rows[:, HF:HF + KMAX + 1].mean(0),
                        rows[:, HF + KMAX + 1:].mean(0)])
    D = np.vstack([B, B])
    w = 1.0 / np.maximum(y, 1.0)
    Dw = D * w[:, None]
    return float(np.linalg.solve(D.T @ Dw, Dw.T @ y)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--per', type=int, default=1000)
    ap.add_argument('--eta0', type=float, default=4.5)
    a_ = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    cfg = (11.0, 24.0, 250, 700, 3.0)
    e0 = a_.eta0
    print(f'эпоха eta0 = {e0}, t = {e0**2/2:.1f} планк. времён, '
          f'истина R_uu l² = {3/e0**4:.4e}')
    print(f'предсказание: dc_0/c_0 = {PRED:+.4f} * sqrt(C_1) * R_uu l²')
    print(f'засевов {a_.seeds} по {a_.per}, только v² < {VMAX}')
    print()
    res = {}
    for fam in ('flat', 'frw'):
        rows, sdi = [], []
        for s in range(a_.seeds):
            rs = np.random.default_rng(555000 + 3313 * s
                                       + (0 if fam == 'flat' else 101))
            r = harvest(fam, e0, cfg, rs, a_.per)
            rows += r
            sdi += [s] * len(r)
        res[fam] = (np.array(rows, dtype=float), np.array(sdi))
        print(f'  {fam}: {len(rows)} интервалов   ({time.time()-t0:.0f} с)',
              flush=True)
    with open(f'{SCRATCH}/jacob.pkl', 'wb') as fh:
        pickle.dump((res, cfg, e0), fh)
    print()
    ks = np.arange(KMAX + 1)
    B = np.column_stack([phi(j, ks) for j in range(4)])
    Af, sf = res['flat']
    Ac, sc = res['frw']
    q = np.quantile(np.concatenate([Af[:, IC], Ac[:, IC]]),
                    np.linspace(0, 1, 4))

    def pair(i, keepf, keepc):
        mf = keepf & (Af[:, IC] >= q[i]) & (Af[:, IC] <= q[i + 1])
        mc = keepc & (Ac[:, IC] >= q[i]) & (Ac[:, IC] <= q[i + 1])
        if mf.sum() < 100 or mc.sum() < 100:
            return None
        return (c0_of(Ac[mc], ks, B) / c0_of(Af[mf], ks, B) - 1.0,
                Ac[mc][:, IC].mean(),
                float(np.mean(Ac[mc][:, I2]
                              / (C2C * Ac[mc][:, IC] * (Ac[mc][:, IC] - 1))
                              - 1.0)),
                float(np.mean(3.0 / Ac[mc][:, IM] ** 4)))

    print('КОНТРОЛЬ: половина плоского против другой половины')
    us = np.unique(sf)
    h1 = np.isin(sf, us[::2])
    for i in range(3):
        m1 = h1 & (Af[:, IC] >= q[i]) & (Af[:, IC] <= q[i + 1])
        m2 = (~h1) & (Af[:, IC] >= q[i]) & (Af[:, IC] <= q[i + 1])
        if m1.sum() < 100 or m2.sum() < 100:
            continue
        d = c0_of(Af[m1], ks, B) / c0_of(Af[m2], ks, B) - 1.0
        print(f'  полоса {i}: dc_0/c_0 = {d:+.5f}')
    print()
    print('ЗАМЕР: излучение против плоского, при том же C_1')
    print(f"{'C_1':>7}{'dc_0/c_0':>11}{'ст.ош':>10}"
          f"{'предск. по истине':>19}{'сигм':>7}"
          f"{'R_uu из C_2':>13}{'предск. по C_2':>16}{'сигм':>7}")
    for i in range(3):
        r = pair(i, np.ones(len(Af), bool), np.ones(len(Ac), bool))
        if r is None:
            continue
        d, c1, x2, rtrue = r
        jk = []
        for u in us:
            rr = pair(i, sf != u, sc != u)
            if rr is not None:
                jk.append(rr[0])
        jk = np.array(jk)
        k = len(jk)
        e = math.sqrt((k - 1) / k * ((jk - jk.mean()) ** 2).sum())
        p_true = PRED * math.sqrt(c1) * rtrue
        # R_uu из обилий: X_2 = v_2 * KFAC * sqrt(C_1) * R_uu l²
        r_from = x2 / (V2CAL * KFAC * math.sqrt(c1))
        p_from = PRED * math.sqrt(c1) * r_from
        print(f'{c1:>7.0f}{d:>+11.5f}{e:>10.5f}{p_true:>+19.5f}'
              f'{abs(d-p_true)/max(e,1e-12):>7.1f}'
              f'{r_from:>13.4e}{p_from:>+16.5f}'
              f'{abs(d-p_from)/max(e,1e-12):>7.1f}')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
