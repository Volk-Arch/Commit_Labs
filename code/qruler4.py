#!/usr/bin/env python3
"""ДВА ПРОГОНА: показатель при ФИКСИРОВАННОЙ x, и вторая линейка.

ОШИБКА `qruler3.py`, пойманная остатками (+1.7, −0.1, −7.1, +6.7, +4.1):
корзины по V_общ связаны с x — чем больше общее прошлое, тем ближе пара
(⟨x⟩ падала 0.761 -> 0.228). То есть «показатель по V» смешивал
зависимость от плотности с зависимостью от расстояния. Здесь x
фиксируется узким окном, а меняется ТОЛЬКО плотность засева.

ВТОРАЯ ЛИНЕЙКА. Протяжённый объект есть цепь. Для пробного события на
расстоянии r от неё число событий цепи, НЕСРАВНИМЫХ с ним, равно
W = 2 lambda r (событие цепи в момент t_c несравнимо с пробным, если
|t_c - t_x| < r). Это точная геометрия, не подгонка.

W есть счёт, значит при пуассоновской цепи шум sqrt(W), и

    бюджет = r/delta_r = sqrt(W)

ЧЕСТНОЕ СРАВНЕНИЕ. sqrt(W) безгранично по W, но сама W растёт лишь как
длина цепи, а длиннейшая цепь в засеве идёт как N^{1/4} при d = 4.
Значит в пересчёте на ПОЛНЫЙ ресурс бюджет второй линейки ~ N^{1/8}.
Это замеряется здесь же.

КОНТРОЛИ второй линейки:
  1. <W> обязано равняться 2 lambda r без подгонки;
  2. дисперсия W обязана быть пуассоновской (Var = <W>);
  3. перемешивание r обязано убить связь;
  4. длиннейшая цепь в засеве обязана идти как N^{1/4}.

  python3 qruler4.py
"""
import math

import numpy as np

from qruler3 import sprinkle4, rel_blocks


def harvest_tag(n, rs, npair, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    Rc = rel_blocks(P)
    Vp = Rc.sum(axis=0)
    tau = np.sqrt(np.maximum(P[:, 0].astype(np.float64) ** 2
                             - (P[:, 1:].astype(np.float64) ** 2).sum(1),
                             0.0))
    rows = []
    tries, lim = 0, npair * 300
    while len(rows) < npair and tries < lim:
        tries += 1
        a, b = int(rs.integers(n)), int(rs.integers(n))
        if a == b or Rc[a, b] or Rc[b, a] or Vp[a] < 40 or Vp[b] < 40:
            continue
        if abs(tau[a] / max(tau[b], 1e-9) - 1.0) > agew:
            continue
        dt = float(P[b, 0]) - float(P[a, 0])
        dx2 = float(((P[b, 1:] - P[a, 1:]) ** 2).sum())
        sep = math.sqrt(max(dx2 - dt * dt, 0.0))
        tb = 0.5 * (tau[a] + tau[b])
        vcp = int((Rc[:, a] & Rc[:, b]).sum())
        rows.append((sep / tb, vcp / (0.5 * (Vp[a] + Vp[b])), vcp, n))
    return np.array(rows)


def cond_budget(xs, us, nq=9, nmin=15):
    qs = np.quantile(us, np.linspace(0, 1, nq))
    res = []
    for j in range(len(qs) - 1):
        m = (us >= qs[j]) & (us <= qs[j + 1])
        if m.sum() < nmin:
            continue
        res.append(xs[m] - np.median(xs[m]))
    if not res:
        return None
    rr = np.concatenate(res)
    return float(xs.mean()) / float(np.std(rr, ddof=1)), len(rr)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(5150)
    print('=== ПРОГОН 1: бюджет при ФИКСИРОВАННОЙ x, меняется плотность ===')
    parts = []
    for n, k in ((1000, 5), (2000, 4), (4000, 3), (8000, 2)):
        for _ in range(k):
            parts.append(harvest_tag(n, rs, 40000))
        print(f'  N = {n}: пар {sum(len(p) for p in parts[-k:])}',
              flush=True)
    D = np.vstack(parts)
    print()
    for xlo, xhi in ((0.35, 0.5), (0.5, 0.7)):
        print(f'  окно x = {xlo}-{xhi}:')
        print(f"      {'N':>7}{'пар':>8}{'<V_общ>':>10}{'бюджет':>9}")
        rows = []
        for n in (1000, 2000, 4000, 8000):
            s = ((D[:, 3] == n) & (D[:, 0] >= xlo) & (D[:, 0] < xhi))
            if s.sum() < 300:
                continue
            r = cond_budget(D[s, 0], D[s, 1])
            if r is None:
                continue
            rows.append((n, float(D[s, 2].mean()), r[0], r[1]))
            print(f'      {n:>7}{int(s.sum()):>8}{D[s,2].mean():>10.0f}'
                  f'{r[0]:>9.2f}', flush=True)
        if len(rows) >= 3:
            R = np.array([(x[1], x[2], x[3]) for x in rows])
            err = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
            A = np.vstack([np.log(R[:, 0]), np.ones(len(R))]).T
            W = np.diag(1.0 / (err / R[:, 1]) ** 2)
            cov = np.linalg.inv(A.T @ W @ A)
            be = cov @ (A.T @ W @ np.log(R[:, 1]))
            se = np.sqrt(np.diag(cov))
            resid = (np.log(R[:, 1]) - A @ be) / (err / R[:, 1])
            print(f'      -> бюджет ~ V^{be[0]:.4f} ± {se[0]:.4f}, '
                  f'остатки: ' + ', '.join(f'{v:+.1f}' for v in resid))
    print()
    print('=== ПРОГОН 2: вторая линейка, W = 2 lambda r ===')
    T, lam = 1.0, 4000.0
    print('КОНТРОЛЬ 1-2: <W> против 2 lambda r, и пуассоновость')
    print(f"{'r':>9}{'<W> замер':>12}{'2 lambda r':>12}"
          f"{'Var(W)/<W>':>13}{'проб':>7}")
    rows = []
    for r in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        ws, rs_true = [], []
        for _ in range(4000):
            nch = rs.poisson(lam * T)
            tc = rs.random(nch) * T
            tx = 0.3 + 0.4 * rs.random()
            w = int(np.sum(np.abs(tc - tx) < r))
            ws.append(w)
            rs_true.append(r)
        w = np.array(ws, dtype=float)
        print(f'{r:>9.3f}{w.mean():>12.2f}{2*lam*r:>12.2f}'
              f'{w.var(ddof=1)/max(w.mean(),1e-9):>13.3f}{len(w):>7}')
        rows.append((r, w.mean()))
    print()
    print('БЮДЖЕТ ВТОРОЙ ЛИНЕЙКИ: условный разброс r при заданном W')
    print(f"{'<W>':>10}{'проб':>8}{'усл. разброс r/r':>18}"
          f"{'бюджет':>9}{'прогноз sqrt(W)':>17}")
    allw, allr = [], []
    for _ in range(60000):
        rr = 10 ** (rs.random() * 2.0 - 3.0)      # r от 0.001 до 0.1
        nch = rs.poisson(lam * T)
        tc = rs.random(nch) * T
        tx = 0.3 + 0.4 * rs.random()
        allw.append(int(np.sum(np.abs(tc - tx) < rr)))
        allr.append(rr)
    Wv = np.array(allw, dtype=float)
    Rv = np.array(allr)
    out = []
    for lo, hi in ((4, 10), (10, 25), (25, 60), (60, 150), (150, 400),
                   (400, 900)):
        s = (Wv >= lo) & (Wv < hi)
        if s.sum() < 200:
            continue
        lr = np.log(Rv[s])
        sp = float(np.std(lr - np.median(lr), ddof=1))
        wm = float(Wv[s].mean())
        out.append((wm, 1.0 / sp, int(s.sum())))
        print(f'{wm:>10.1f}{int(s.sum()):>8}{sp:>18.5f}'
              f'{1/sp:>9.2f}{math.sqrt(wm):>17.2f}')
    print()
    print('КОНТРОЛЬ 3: перемешать r — бюджет обязан упасть до ~1')
    perm = rs.permutation(len(Rv))
    s = (Wv >= 25) & (Wv < 60)
    lr = np.log(Rv[perm][s])
    print(f'  перемешанный: {1.0/float(np.std(lr - np.median(lr), ddof=1)):.3f}')
    O = np.array(out)
    A = np.vstack([np.log(O[:, 0]), np.ones(len(O))]).T
    be = np.linalg.lstsq(A, np.log(O[:, 1]), rcond=None)[0]
    print()
    print(f'ЗАКОН ВТОРОЙ ЛИНЕЙКИ: бюджет ~ W^{be[0]:.4f} '
          f'(прогноз 0.5), коэффициент {math.exp(be[1]):.3f}')
    print()
    print('КОНТРОЛЬ 4: длиннейшая цепь в засеве против N^{1/4}')
    for n in (500, 1000, 2000, 4000):
        P = sprinkle4(n, rs, 1.0)
        Rc = rel_blocks(P)
        h = np.zeros(n, dtype=int)
        order = np.argsort(P[:, 0])
        for j in order:
            pre = np.nonzero(Rc[:, j])[0]
            h[j] = (h[pre].max() + 1) if len(pre) else 0
        print(f'  N = {n:>5}: цепь {h.max()+1:>4}, '
              f'/N^0.25 = {(h.max()+1)/n**0.25:.3f}')


if __name__ == '__main__':
    main()
