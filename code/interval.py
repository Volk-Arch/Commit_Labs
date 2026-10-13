#!/usr/bin/env python3
"""ЛОКАЛЬНЫЕ петли: обходы внутри ИНТЕРВАЛА, а не по связям.

ЧТО ЭТО ИСПРАВЛЯЕТ. `locality.py` намерил структурный факт, рушащий всю
холономическую программу в прежнем виде:

    собственное время связи ~ N^-0.2095   при ell ~ N^-0.2500
    |Δx| связи              ~ N^-0.0160   (0.445 -> 0.432)
    поперечник четв. цикла  ~ N^-0.0204   (0.644 -> 0.619)

Связи светоподобны, а не коротки: собственное время сжимается как велит
дискретность, а пространственная протяжённость стоит на месте. Четвёрный
цикл имеет поперечник 0.62 в ромбе радиуса 1.0 — обход охватывает две
трети мира при ЛЮБОМ измельчении. Холономия по нему — глобальная
величина, потому перемешанный контроль весь день и выходил ей равен
(`matchscale.py` 23.5σ против 4.3σ, `kappa.py` 4.9σ против 5.6σ).

Калибровочная теория на решётке держится на кривизна * площадь -> 0.
Нужны петли, которые СЖИМАЮТСЯ.

ОТКУДА ИХ ВЗЯТЬ. В причинных множествах локальность добывают интервалом
Александрова I(a,b) = {z : a < z < b}. Если держать его ЧИСЛО СОБЫТИЙ
постоянным, его собственное время идёт как n ~ rho * tau^4, то есть
tau ~ (n/rho)^(1/4) ~ N^(-1/4), и пространственная протяжённость тоже,
потому что интервал — настоящая малая область, в отличие от связи. Так же
устроено действие Бенинказы–Доукера: выглядит нелокальным, а локальность
берёт из счёта малых интервалов n0..n3.

ЧТО МЕРИМ. Лестница N. Пары a < b с числом событий интервала в
фиксированной полосе. Внутри интервала — свой граф связей и свои
четвёрные обходы. Поперечник и площадь — показателями по N.

ПРЕДСКАЗАНИЯ, расходящиеся по знаку:
    работает  поперечник ~ N^-0.25 (как ell), площадь ~ N^-0.50;
    не работает — опять около нуля, и тогда локальности в порядке нет
                  вовсе, а не только в графе связей.

Рядом, для прямого сравнения, те же величины по НЕОГРАНИЧЕННЫМ циклам.

  python3 interval.py
  python3 interval.py --ns 400 800 1600 3200 --card 10 24 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from topo import cone_rel
from holo2 import links_of, cycles_through
from response import warped_diamond


def sizes(X, idx):
    """Поперечник и площадь (в плоскости 1-2) набора событий."""
    P = X[list(idx)]
    d = 0.0
    for p in range(len(idx)):
        for q in range(p + 1, len(idx)):
            d = max(d, float(np.linalg.norm(P[p, 1:] - P[q, 1:])))
    ar = abs(sum(P[t, 1] * P[(t + 1) % len(idx), 2] -
                 P[t, 2] * P[(t + 1) % len(idx), 1]
                 for t in range(len(idx)))) / 2
    return d, float(ar)


def one(N, T, clo, chi, npair, ncyc, seed):
    """Размеры обходов: внутри интервалов и без ограничения."""
    rs = np.random.default_rng(7801000 + 977 * seed)
    X = warped_diamond(N, T, 0.0, 0.45, rs)
    R = cone_rel(X)          # R[i,j] = «j раньше i»
    A = R.astype(np.float32)
    CP = (A @ A.T)           # CP[i,j] = |общее прошлое|
    # число событий интервала между j (раньше) и i (позже)
    INT = (A.T @ A.T).T * 0  # заготовка нужной формы
    INT = (R.astype(np.float32) @ R.astype(np.float32))
    # INT[i,j] = #{z : R[i,z] и R[z,j]} = #{z : z<i и j<z} = |I(j,i)|
    rel = np.argwhere(R)
    if len(rel) < 50:
        return None
    card = INT[rel[:, 0], rel[:, 1]]
    ok = np.flatnonzero((card >= clo) & (card <= chi))
    if len(ok) < 8:
        return None
    ok = rs.choice(ok, min(npair, len(ok)), replace=False)
    dia_i, ar_i, nci = [], [], 0
    for p in ok:
        i, j = int(rel[p, 0]), int(rel[p, 1])
        mem = np.flatnonzero(R[:, j] & R[i, :])
        sub = np.concatenate([[j], mem, [i]])
        if len(sub) < 5:
            continue
        Rs = R[np.ix_(sub, sub)]
        Ls = links_of(Rs)
        Us = Ls | Ls.T
        deg = Us.sum(1)
        vs = np.flatnonzero(deg >= 3)
        if len(vs) == 0:
            continue
        for v in vs[:4]:
            for c in cycles_through(Us, int(v), ncyc,
                                    np.random.default_rng(17 + seed)):
                d, ar = sizes(X, sub[list(c)])
                dia_i.append(d); ar_i.append(ar); nci += 1
    if nci < 30:
        return None
    L = links_of(R)
    U = L | L.T
    pool = np.flatnonzero(U.sum(1) >= 3)
    pool = rs.choice(pool, min(120, len(pool)), replace=False)
    dia_f, ar_f = [], []
    for v in pool:
        for c in cycles_through(U, int(v), ncyc,
                                np.random.default_rng(19 + seed)):
            d, ar = sizes(X, list(c))
            dia_f.append(d); ar_f.append(ar)
    del R, A, CP, INT, L, U, X
    vol = math.pi / 24.0 * T ** 4
    return (float(np.mean(dia_i)), float(np.mean(ar_i)),
            float(np.mean(dia_f)) if dia_f else float('nan'),
            float(np.mean(ar_f)) if ar_f else float('nan'),
            float((vol / N) ** 0.25), nci)


def fit(ns, vals):
    ok = [k for k in range(len(vals))
          if vals[k] is not None and np.isfinite(vals[k]) and vals[k] > 1e-14]
    if len(ok) < 3:
        return float('nan'), float('nan')
    x = np.log([ns[k] for k in ok])
    y = np.log([vals[k] for k in ok])
    b, c = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    return float(b), float(1 - ((y - b * x - c) ** 2).sum() / ss) \
        if ss > 1e-18 else (float(b), 1.0)


def main():
    ap = argparse.ArgumentParser(description='Петли внутри интервала')
    ap.add_argument('--ns', type=int, nargs='+',
                    default=[400, 800, 1600, 3200])
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--card', type=int, nargs=2, default=[10, 24])
    ap.add_argument('--npair', type=int, default=90)
    ap.add_argument('--ncyc', type=int, default=4)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    print(f'лестница N = {a.ns}, ромб T={a.t}')
    print(f'интервалы с числом событий {a.card[0]}..{a.card[1]} — '
          f'постоянным при всех N')
    print(f'засевов {a.seeds}')
    print('ПРЕДСКАЗАНИЕ, если локальность берётся интервалом:')
    print('  поперечник ~ N^-0.25 (как ell), площадь ~ N^-0.50')
    print()
    cols = ['попереч. в инт.', 'площадь в инт.', 'попереч. свободн.',
            'площадь свободн.', 'ell']
    print(f"{'N':>7}" + ''.join(f'{c:>19}' for c in cols) + f"{'обходов':>9}")

    acc = [[] for _ in cols]
    for N in a.ns:
        rows = []
        for t in range(a.seeds):
            r = one(N, a.t, a.card[0], a.card[1], a.npair, a.ncyc, t)
            if r:
                rows.append(r)
        if not rows:
            print(f'{N:>7}  мало данных')
            for k in range(len(cols)):
                acc[k].append(None)
            continue
        M = np.array(rows)
        line = f'{N:>7}'
        for k in range(len(cols)):
            v = float(np.nanmean(M[:, k]))
            acc[k].append(v)
            line += f'{v:>19.5f}'
        print(line + f'{int(M[:, 5].sum()):>9}', flush=True)

    print()
    be, _ = fit(a.ns, acc[4])
    for k, c in enumerate(cols):
        b, r2 = fit(a.ns, acc[k])
        print(f'  {c:<19} ~ N^{b:+.4f}   R^2 = {r2:.4f}')
    bdi, _ = fit(a.ns, acc[0])
    bai, _ = fit(a.ns, acc[1])
    bdf, _ = fit(a.ns, acc[2])
    print()
    print(f'  эталон локальности: ell ~ N^{be:+.4f}, площадь ~ '
          f'N^{2*be:+.4f}')
    print(f'  в интервале:  поперечник {bdi:+.4f}, площадь {bai:+.4f}')
    print(f'  без ограничения: поперечник {bdf:+.4f}')
    print()
    if abs(bdi - be) < 0.09 and abs(bdf) < 0.09:
        print('  ПЕТЛИ В ИНТЕРВАЛЕ СЖИМАЮТСЯ, свободные нет => локальность')
        print('  в порядке ЕСТЬ, но живёт в интервалах, а не в связях')
    elif abs(bdi) < 0.09:
        print('  и в интервале не сжимаются => локальности в порядке нет')
    else:
        print('  промежуточный случай — смотреть на числа')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
