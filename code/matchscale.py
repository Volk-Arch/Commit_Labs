#!/usr/bin/env python3
"""Холономия против вещества ПРИ СОВПАДАЮЩЕМ МАСШТАБЕ — без нормировки.

ПУТЬ К ЭТОМУ ЗАМЕРУ (три ошибки, каждая исправлена предыдущим шагом).

1. `holosource.py` делил холономию на масштаб и получил отклик на
   разрежение. `scratchpad/tautology.py` показал, что это тавтология:
   при одинаковом знаменателе знак переворачивается (+0.2573 в целом
   против −0.3132 внутри корзин, парадокс Симпсона).

2. Его мысль: ошибка показала, где искать, потому что ПОКАЗАТЕЛЬ
   нормировки не требует. `exponent.py` мерил показатель по циклам
   внутри мира — вышло 0.0534 при двухточечной оценке 0.37, отклик 2.8σ
   при перемешанном контроле 2.4σ. Негодно: внутри одного мира общее
   прошлое цикла есть почти целиком ПОЛОЖЕНИЕ в ромбе.

3. `expladder.py` взял показатель лестницей по N, как взяты все стойкие
   показатели проекта. Закон нашёлся, R^2 = 0.9999 на восьмикратном N:

       |холономия| ~ N^1.0734,   общее прошлое ~ N^1.0302

   то есть при измельчении холономия/масштаб инвариантна (показатель
   1.0419). Деление на масштаб^1 было ВЕРНЫМ — но для измельчения.
   Внутри мира правит другой показатель, 0.37. Два разных показателя, и
   я их смешал.

ОТСЮДА ЗАМЕР. Нормировка не нужна вовсе, если сравнивать при
СОВПАДАЮЩЕМ масштабе — так же, как `response.py` сравнивает при
совпадающей высоте. Берём сырую |холономию|, делим циклы на корзины по
общему прошлому (границы ОБЩИЕ для обоих миров, по объединённым
квантилям), внутри корзины сравниваем дырку с контролем, и усредняем
разницы по корзинам.

Внутрикорзинная разница делится на СРЕДНЕЕ ПО КОРЗИНЕ ОБОИХ МИРОВ —
это не нормировка проверяемой величины, а приведение корзин к общему
знаменателю: делитель один и тот же для дырки и для контроля.

КОНТРОЛИ:
  f = 1.0        — дырки нет, разница обязана быть нулём;
  перемешанная   — обязана молчать при любом f;
  телескопир.    — холономия ровно 0;
  знак по засевам — отдельной колонкой, чтобы видеть, не держится ли
                   среднее на одном мире.

  python3 matchscale.py
  python3 matchscale.py --fracs 1.0 0.5 0.0 --seeds 24 --bins 6
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through, holonomy, antisym_causal
from response import warped_diamond


def cycle_table(X, r_lo, r_hi, samp, ncyc, rs):
    """На каждый цикл: масштаб и |холономия| трёх фаз."""
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
    pool = np.flatnonzero((rr >= r_lo) & (rr < r_hi) & (U.sum(1) >= 3))
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
            rows.append((s, abs(holonomy(c, G)), abs(holonomy(c, SH)),
                         abs(holonomy(c, TE))))
    del R, L, U, A, CP, G, SH, TE
    return np.array(rows) if len(rows) >= 60 else None


def within(D, C, col, nb, mincell):
    """Взвешенная разница внутри корзин по масштабу; делитель общий."""
    ed = np.unique(np.quantile(np.concatenate([D[:, 0], C[:, 0]]),
                               np.linspace(0, 1, nb + 1)))
    if len(ed) < 3:
        return float('nan'), []
    ed[0] -= 1e-9
    ed[-1] += 1e-9
    bd = np.digitize(D[:, 0], ed) - 1
    bc = np.digitize(C[:, 0], ed) - 1
    num, den, per = 0.0, 0.0, []
    for b in range(len(ed) - 1):
        x, y = D[bd == b, col], C[bc == b, col]
        if len(x) < mincell or len(y) < mincell:
            continue
        pooled = np.concatenate([x, y]).mean()
        if pooled <= 1e-12:
            continue
        w = min(len(x), len(y))
        rel = (x.mean() - y.mean()) / pooled
        num += rel * w
        den += w
        per.append((0.5 * (ed[b] + ed[b + 1]), len(x), len(y), rel, w))
    return (num / den if den > 0 else float('nan')), per


def main():
    ap = argparse.ArgumentParser(description='Отклик при совпадающем масштабе')
    ap.add_argument('--n', type=int, default=2600)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--fracs', type=float, nargs='+',
                    default=[1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--samp', type=int, default=320)
    ap.add_argument('--ncyc', type=int, default=6)
    ap.add_argument('--bins', type=int, default=6)
    ap.add_argument('--mincell', type=int, default=30)
    ap.add_argument('--seeds', type=int, default=20)
    a = ap.parse_args()

    t0 = time.time()
    lo, hi = a.rad, a.rad + 0.16
    print(f'N={a.n}, дырка радиуса {a.rad}, замер в слое {lo:.2f}..{hi:.2f}')
    print('СЫРАЯ |холономия|, корзины по общему прошлому, границы общие')
    print(f'корзин {a.bins}, минимум {a.mincell} циклов на ячейку, '
          f'засевов {a.seeds}')
    print('нормировки проверяемой величины НЕТ')
    print()
    print(f"{'f':>6}{'удал':>6}{'разн. калибр':>14}{'сигм':>7}"
          f"{'знак один':>11}{'перемеш':>10}{'сигм':>7}{'телеск':>9}"
          f"{'корзин':>8}")

    detail = {}
    for f in a.fracs:
        dg, ds, tmax, nb = [], [], 0.0, []
        for t in range(a.seeds):
            rs = np.random.default_rng(7301000 + 977 * t)
            X0 = warped_diamond(a.n, a.t, 0.0, 0.45, rs)
            rr = np.linalg.norm(X0[:, 1:], axis=1)
            ins = np.flatnonzero(rr < a.rad)
            drop = ins[rs.random(len(ins)) > f] if len(ins) else ins
            k = len(drop)
            keep = np.ones(len(X0), bool)
            keep[drop] = False
            D = cycle_table(X0[keep], lo, hi, a.samp, a.ncyc,
                            np.random.default_rng(41 + t))
            if k > 0:
                ii = np.sort(rs.choice(len(X0), len(X0) - k, replace=False))
                C = cycle_table(X0[ii], lo, hi, a.samp, a.ncyc,
                                np.random.default_rng(41 + t))
            else:
                C = D
            del X0
            if D is None or C is None:
                continue
            g, per = within(D, C, 1, a.bins, a.mincell)
            s, _ = within(D, C, 2, a.bins, a.mincell)
            tmax = max(tmax, D[:, 3].max(), C[:, 3].max())
            if np.isfinite(g):
                dg.append(g); nb.append(len(per))
                if t == 0:
                    detail[f] = per
            if np.isfinite(s):
                ds.append(s)
        if len(dg) < 4:
            print(f'{f:>6.2f}  мало данных')
            continue
        d = np.array(dg)
        se = d.std(ddof=1) / math.sqrt(len(d))
        sg = abs(d.mean()) / se if se > 1e-12 else float('nan')
        same = int((d > 0).sum() if d.mean() > 0 else (d < 0).sum())
        s = np.array(ds) if ds else np.array([0.0, 0.0])
        ses = s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else float('nan')
        print(f'{f:>6.2f}{k:>6}{d.mean():>+14.4f}{sg:>7.1f}'
              f'{same}/{len(d):<9}{s.mean():>+10.4f}'
              f'{(abs(s.mean())/ses if ses and ses > 1e-12 else float("nan")):>7.1f}'
              f'{tmax:>9.1e}{np.mean(nb):>8.1f}', flush=True)

    for f, per in detail.items():
        if f != min(detail):
            continue
        print(f'\n  по корзинам при f={f:.2f} (засев 0):')
        print(f"{'масштаб':>10}{'дырка n':>9}{'контр n':>9}"
              f"{'отн. разница':>14}")
        for c, nx, ny, rel, _ in per:
            print(f'{c:>10.1f}{nx:>9}{ny:>9}{rel:>+14.4f}')

    print('\n  монотонно по f, при f=1 нуль, перемешанная молчит => ОТКЛИК')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
