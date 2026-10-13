#!/usr/bin/env python3
"""МЯГКАЯ ДЫРКА: поле это или просто запрет пройти.

УЯЗВИМОЕ МЕСТО bow.py. Дырка даёт правильные знаки обеих половин
тяготения (часы медленнее 4.5 сигмы, цепи выгибаются наружу 5.0 сигмы),
но она МЕХАНИЧЕСКИ не пускает цепь внутрь, а контроль пускает. Выгиб
может быть не полем, а запретом.

РАЗЛИЧИТЕЛЬ. Разрядить трубку вместо того, чтобы её вычищать: оставить
долю f событий. При f > 0 цепь ПРОЙТИ МОЖЕТ, просто ей там хуже.

    поле   =>  и задержка, и выгиб растут ПЛАВНО при f от 1 к 0
    запрет =>  появляются только при f = 0

Контроль у каждого f — удаление того же числа событий случайно по ромбу,
так что «меньше событий вообще» вычтено.

  python3 soft.py
  python3 soft.py --fracs 1.0 0.7 0.4 0.15 0.0 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel
from delay import heights
from detour import longest_between
from bow import bow_of, pairs_same_radius


def thin(X, rad, f, rs):
    """Оставить долю f событий внутри радиуса rad. Вернуть маску."""
    r = np.linalg.norm(X[:, 1:], axis=1)
    inside = r < rad
    keep = np.ones(len(X), bool)
    ii = np.flatnonzero(inside)
    if len(ii):
        drop = rs.random(len(ii)) > f
        keep[ii[drop]] = False
    return keep


def clock_resid(X, r_lo, r_hi):
    """Средний остаток log h по log V в кольце радиусов."""
    R = cone_rel(X)
    h = heights(R).astype(np.float64)
    V = R.sum(1).astype(np.float64)
    ok = (V >= 40) & (V <= 600) & (h > 0)
    if ok.sum() < 60:
        return np.nan
    lv, lh = np.log(V[ok]), np.log(h[ok])
    res = lh - np.polyval(np.polyfit(lv, lh, 1), lv)
    r = np.linalg.norm(X[ok][:, 1:], axis=1)
    m = (r >= r_lo) & (r < r_hi)
    return float(res[m].mean()) if m.sum() > 5 else np.nan


def main():
    ap = argparse.ArgumentParser(description='Мягкая дырка')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.7, 0.4, 0.15, 0.0])
    ap.add_argument('--d', type=float, default=0.34)
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}; внутри радиуса {a.rad} остаётся доля f событий')
    print(f'часы — в кольце {a.rad:.2f}..0.40; выгиб — пары на радиусе {a.d}')
    print('контроль при каждом f: столько же событий удалено случайно')
    print('ПЛАВНЫЙ рост обеих величин при f от 1 к 0 => ПОЛЕ')
    print('появление только при f = 0                => запрет пройти')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'f':>6}{'удалено':>9}{'задержка':>11}{'сигм':>7}"
          f"{'выгиб наружу':>14}{'сигм':>7}{'цепь об':>9}{'цепь кон':>10}")

    for f in a.fracs:
        CH, CC, BH, BC, LH, LC = [], [], [], [], [], []
        nrem = []
        for t in range(a.seeds):
            rs = np.random.default_rng(2601000 + 977 * t)
            X0 = diamond(a.t, rs)
            keep = thin(X0, a.rad, f, rs)
            k = int((~keep).sum())
            nrem.append(k)
            for mk in (True, False):
                if mk:
                    X = X0[keep]
                elif k > 0:
                    ii = rs.choice(len(X0), len(X0) - k, replace=False)
                    X = X0[np.sort(ii)]
                else:
                    X = X0
                (CH if mk else CC).append(clock_resid(X, a.rad, 0.40))
                R = cone_rel(X)
                bs, ls = [], []
                for p, q in pairs_same_radius(X, a.t, a.d, 0.05, rs):
                    v = bow_of(R, X, p, q)
                    if v is not None:
                        bs.append(v - a.d)
                        ls.append(longest_between(R, p, q))
                if bs:
                    (BH if mk else BC).append(float(np.mean(bs)))
                    (LH if mk else LC).append(float(np.mean(ls)))
        def dif(h, c):
            h, c = np.array(h, float), np.array(c, float)
            h, c = h[np.isfinite(h)], c[np.isfinite(c)]
            if len(h) < 2 or len(c) < 2:
                return np.nan, np.nan
            d = h.mean() - c.mean()
            er = math.hypot(h.std() / math.sqrt(len(h)),
                            c.std() / math.sqrt(len(c)))
            return d, abs(d) / max(er, 1e-12)
        dc, zc = dif(CH, CC)
        db, zb = dif(BH, BC)
        print(f'{f:>6.2f}{np.mean(nrem):>9.0f}{dc:>+11.5f}{zc:>7.1f}'
              f'{db:>+14.5f}{zb:>7.1f}'
              f'{np.mean(LH):>9.2f}{np.mean(LC):>10.2f}', flush=True)

    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
