#!/usr/bin/env python3
"""ДОЛЯ НЕСОКРАЩЁННОГО: безразмерная величина, которую не надо угадывать.

ОТКУДА ОНА ВЗЯЛАСЬ. Весь день рушился на одном: я делил холономию на
что-то ДРУГОГО РОДА и каждый раз угадывал показатель.

  `holosource.py`  делил на масштаб^1 -> отклик оказался тавтологией
                   (`scratchpad/tautology.py`: +0.2573 в целом против
                   −0.3132 внутри корзин, парадокс Симпсона);
  `exponent.py`    мерил показатель по циклам внутри мира -> 0.0534
                   против двухточечных 0.37, отклик 2.8σ при
                   перемешанном контроле 2.4σ. Внутри мира «масштаб» есть
                   положение в ромбе, а не масштаб;
  `expladder.py`   взял показатель лестницей по N, как все стойкие
                   показатели проекта. Закон нашёлся, R^2 = 0.9999 на
                   восьмикратном N:
                       |холономия| ~ N^1.0734,  общее прошлое ~ N^1.0302
                   значит холономия/масштаб ~ N^0.04, то есть ПЛОСКАЯ;
  `matchscale.py`  сравнил при совпадающем масштабе без нормировки ->
                   перемешанный контроль −0.5489 (23.5σ) против
                   калибровочной −0.0613 (4.3σ). Контроль больше теста.

ВЫВОД ИЗ ЛЕСТНИЦЫ. Безразмерная холономия на клетку при измельчении НЕ
убывает. У связности с непрерывным пределом она обязана убывать как
кривизна на площадь клетки. Плоская означает беспорядочную фазу —
непрерывного предела нет, и откликаться нечем.

ЧТО МЕРИМ ЗДЕСЬ. У холономии есть делитель ТОГО ЖЕ РОДА — сумма модулей
её собственных слагаемых:

    kappa(цикл) = |сумма связности по обходу| / сумма |связности|

Безразмерна по построению: одни единицы сверху и снизу, одна зависимость
от плотности, никакого показателя угадывать не нужно.

    kappa = 0        полное сокращение (телескопирующаяся фаза)
    kappa ~ 1/sqrt(n) случайные слагаемые, беспорядочная фаза
    kappa -> 0 при N  непрерывный предел ЕСТЬ
    kappa плоско по N беспорядочная фаза, предела нет

ЭТАП 1: kappa по лестнице N. Эталоны рядом: перемешанная (случайные
слагаемые) и телескопирующаяся (обязана дать ровно 0).
ЭТАП 2: отвечает ли kappa на разрежение. Парно, с корзинами по масштабу
и без них, и с теми же двумя контролями.

  python3 kappa.py
  python3 kappa.py --ns 500 1000 2000 4000 --stage2 --seeds 20
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, antisym_causal
from response import warped_diamond


def kap(cyc, A):
    """Доля несокращённого: |сумма| / сумма модулей. Безразмерна."""
    v = np.array([A[cyc[t], cyc[(t + 1) % len(cyc)]]
                  for t in range(len(cyc))], float)
    s = np.abs(v).sum()
    return (abs(v.sum()) / s) if s > 1e-12 else float('nan')


def table(X, f_lo, f_hi, samp, ncyc, rs, absolute=False):
    """На каждый цикл: масштаб, kappa калибр / перемеш / телеск."""
    R = cone_rel(X)
    L = links_of(R)
    U = L | L.T
    A = R.astype(np.float32)
    CP = A @ A.T
    G = antisym_causal((CP - (A.T @ A)).astype(np.float64), R)
    sz = A.sum(1)
    TE = sz[:, None] - sz[None, :]
    idx = rs.permutation(len(X))
    SH = G[np.ix_(idx, idx)]
    SH = 0.5 * (SH - SH.T)
    rr = np.linalg.norm(X[:, 1:], axis=1)
    rmax = 1.0 if absolute else rr.max()
    pool = np.flatnonzero((rr >= f_lo * rmax) & (rr < f_hi * rmax) &
                          (U.sum(1) >= 3))
    if len(pool) < 25:
        del R, L, U, A, CP, G, SH, TE
        return None
    pool = rs.choice(pool, min(samp, len(pool)), replace=False)
    rows = []
    for v in pool:
        v = int(v)
        for c in cycles_through(U, v, ncyc, rs):
            s = float(np.mean([CP[c[k], c[(k + 1) % len(c)]]
                               for k in range(len(c))]))
            rows.append((s, kap(c, G), kap(c, SH), kap(c, TE)))
    del R, L, U, A, CP, G, SH, TE
    rows = [r for r in rows if np.isfinite(r[1]) and np.isfinite(r[2])]
    return np.array(rows) if len(rows) >= 60 else None


def within(D, C, col, nb, mincell):
    """Разница внутри корзин по масштабу; kappa уже безразмерна."""
    ed = np.unique(np.quantile(np.concatenate([D[:, 0], C[:, 0]]),
                               np.linspace(0, 1, nb + 1)))
    if len(ed) < 3:
        return float('nan')
    ed[0] -= 1e-9
    ed[-1] += 1e-9
    bd = np.digitize(D[:, 0], ed) - 1
    bc = np.digitize(C[:, 0], ed) - 1
    num, den = 0.0, 0.0
    for b in range(len(ed) - 1):
        x, y = D[bd == b, col], C[bc == b, col]
        if len(x) < mincell or len(y) < mincell:
            continue
        w = min(len(x), len(y))
        num += (x.mean() - y.mean()) * w
        den += w
    return num / den if den > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser(description='Доля несокращённого')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[500, 1000, 2000, 4000])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--shell', type=float, nargs=2, default=[0.45, 0.75])
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--samp', type=int, default=300)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--bins', type=int, default=6)
    ap.add_argument('--mincell', type=int, default=30)
    ap.add_argument('--seeds', type=int, default=8)
    ap.add_argument('--stage2', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.shell
    print('kappa = |сумма связности по обходу| / сумма модулей')
    print('безразмерна по построению: нормировку угадывать не нужно')
    print(f'циклы четвёрные, случайные слагаемые дали бы ~{1/math.sqrt(4):.3f}')
    print()
    print(f'ЭТАП 1: лестница N = {a.ns}, оболочка r/rmax {lo}..{hi}, '
          f'засевов {a.seeds}')
    print(f"{'N':>7}{'kappa калибр':>14}{'± с.о.':>9}{'перемеш':>10}"
          f"{'телеск':>9}{'масштаб':>9}{'циклов':>8}")

    ks = []
    for N in a.ns:
        kg, kshuf, kt, sc, nc = [], [], [], [], 0
        for t in range(a.seeds):
            rs = np.random.default_rng(7401000 + 977 * t)
            X = warped_diamond(N, a.t, 0.0, 0.45, rs)
            M = table(X, lo, hi, a.samp, a.ncyc,
                      np.random.default_rng(59 + t))
            del X
            if M is None:
                continue
            kg.append(M[:, 1].mean()); kshuf.append(M[:, 2].mean())
            kt.append(np.nanmax(M[:, 3])); sc.append(M[:, 0].mean())
            nc += len(M)
        if not kg:
            print(f'{N:>7}  мало данных')
            ks.append(None)
            continue
        ks.append(float(np.mean(kg)))
        se = np.std(kg, ddof=1) / math.sqrt(len(kg)) if len(kg) > 1 else 0.0
        print(f'{N:>7}{np.mean(kg):>14.4f}{se:>9.4f}{np.mean(kshuf):>10.4f}'
              f'{np.nanmax(kt):>9.1e}{np.mean(sc):>9.1f}{nc:>8}', flush=True)

    ok = [i for i in range(len(ks)) if ks[i]]
    if len(ok) >= 3:
        x = np.log([a.ns[i] for i in ok])
        y = np.log([ks[i] for i in ok])
        b, c = np.polyfit(x, y, 1)
        yp = b * x + c
        r2 = 1 - ((y - yp) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-18)
        print()
        print(f'  kappa ~ N^{b:+.4f}   R^2 = {r2:.4f}')
        print('  показатель < 0 заметно => холономия сокращается всё лучше,')
        print('    непрерывный предел ЕСТЬ, связность слабая')
        print('  показатель ~ 0 => беспорядочная фаза, предела нет')

    if not a.stage2:
        print(f'\n  время {time.time()-t0:.0f} с   (ЭТАП 2: --stage2)')
        return

    sl, sh_ = a.rad, a.rad + 0.16
    print()
    print(f'ЭТАП 2: отвечает ли kappa на разрежение. N={a.n}, '
          f'дырка {a.rad}, слой {sl:.2f}..{sh_:.2f}')
    print(f"{'f':>6}{'удал':>6}{'kappa дырка':>13}{'контроль':>10}"
          f"{'разница':>10}{'сигм':>7}{'знак':>7}{'в корзинах':>12}"
          f"{'сигм':>7}{'перемеш разн':>14}{'сигм':>7}")

    for f in a.fracs:
        D_, C_, W_, S_ = [], [], [], []
        k = 0
        for t in range(a.seeds):
            rs = np.random.default_rng(7501000 + 977 * t)
            X0 = warped_diamond(a.n, a.t, 0.0, 0.45, rs)
            rr = np.linalg.norm(X0[:, 1:], axis=1)
            ins = np.flatnonzero(rr < a.rad)
            drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
            k = len(drop)
            keep = np.ones(len(X0), bool)
            keep[drop] = False
            D = table(X0[keep], sl, sh_, a.samp, a.ncyc,
                      np.random.default_rng(71 + t), absolute=True)
            if k > 0:
                ii = np.sort(rs.choice(len(X0), len(X0) - k, replace=False))
                C = table(X0[ii], sl, sh_, a.samp, a.ncyc,
                          np.random.default_rng(71 + t), absolute=True)
            else:
                C = D
            del X0
            if D is None or C is None:
                continue
            D_.append(D[:, 1].mean()); C_.append(C[:, 1].mean())
            W_.append(within(D, C, 1, a.bins, a.mincell))
            S_.append(D[:, 2].mean() - C[:, 2].mean())
        if len(D_) < 4:
            print(f'{f:>6.2f}  мало данных')
            continue

        def rep(v):
            v = np.array([x for x in v if np.isfinite(x)])
            if len(v) < 2:
                return float('nan'), float('nan'), 0, 0
            se = v.std(ddof=1) / math.sqrt(len(v))
            sg = abs(v.mean()) / se if se > 1e-12 else float('nan')
            sm = int((v > 0).sum() if v.mean() > 0 else (v < 0).sum())
            return v.mean(), sg, sm, len(v)

        dd = np.array(D_) - np.array(C_)
        m, sg, sm, nn = rep(dd)
        mw, sw, _, _ = rep(W_)
        msh, ssh, _, _ = rep(S_)
        print(f'{f:>6.2f}{k:>6}{np.mean(D_):>13.4f}{np.mean(C_):>10.4f}'
              f'{m:>+10.4f}{sg:>7.1f}{sm}/{nn:<5}{mw:>+12.4f}{sw:>7.1f}'
              f'{msh:>+14.4f}{ssh:>7.1f}', flush=True)

    print('\n  разница держится в корзинах, перемешанная молчит => ОТКЛИК')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
