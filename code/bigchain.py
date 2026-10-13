#!/usr/bin/env python3
"""Задержка от массы в большом мире: без матрицы N x N.

Плотная матрица отношений упирается в память уже на десятках тысяч
событий. Для этого измерения она не нужна:

  интервал   фильтр по координатам, O(N*k), матрица не строится
  цепь       матрица только для событий ВНУТРИ интервала

Так достигается настоящее слабое поле: масса M в сотнях событий против
интервала в тысячах, то есть M/V_bg ~ 0.05, тогда как прежний предел
был 0.23.

Пары выбираются так, чтобы тело входило в интервал целиком: a раньше
всех точек шара, b позже всех. Тогда n_add = M точно, и сравнение с
контролем идёт при одинаковой массе.

  python3 bigchain.py --k 4
  python3 bigchain.py --k 6 --n 60000 --pairs 60
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of
from chaingrav import chain_len

CENTER = 0.5


def build(N, M, k, r, seed):
    """Фон плюс два способа добавить M событий."""
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, k)).astype(np.float32)
    v = rs.normal(size=(M, k))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = r * rs.random(M) ** (1.0 / k)
    Xbody = (CENTER + v * rad[:, None]).astype(np.float32)
    Xunif = rs.random((M, k)).astype(np.float32)
    return Xb, Xbody, Xunif


def pick_pairs(Xb, k, r, n_pairs, rs):
    """Пары, гарантированно накрывающие шар: a < center-r, b > center+r."""
    lo, hi = CENTER - r, CENTER + r
    ia = np.flatnonzero((Xb < lo).all(1))
    ib = np.flatnonzero((Xb > hi).all(1))
    if len(ia) == 0 or len(ib) == 0:
        return []
    na = min(n_pairs, len(ia), len(ib))
    return list(zip(rs.choice(ia, na, replace=False),
                    rs.choice(ib, na, replace=False)))


def one_pair(Xb, Xadd, N, a, b, vol_cap):
    """tau, объём фона, число и высота добавленных внутри интервала."""
    xa, xb = Xb[a], Xb[b]
    mb = (Xb > xa).all(1) & (Xb < xb).all(1)
    mb[a] = mb[b] = False
    ib = np.flatnonzero(mb)
    if len(ib) > vol_cap:                 # слишком большой интервал — пропуск
        return None
    if Xadd is None:
        ia = np.empty(0, dtype=np.int64)
        Xsub = np.vstack([xa[None], Xb[ib], xb[None]])
    else:
        ma = (Xadd > xa).all(1) & (Xadd < xb).all(1)
        ia = np.flatnonzero(ma)
        Xsub = np.vstack([xa[None], Xb[ib], Xadd[ia], xb[None]])
    R = rel_of(Xsub)
    tau = chain_len(R) - 1
    if len(ia):
        off = 1 + len(ib)
        sl = slice(off, off + len(ia))
        h_add = chain_len(R[sl, sl])
    else:
        h_add = 0
    return tau, len(ib), len(ia), h_add


def run(N, M, k, r, n_pairs, seed, vol_cap):
    Xb, Xbody, Xunif = build(N, M, k, r, seed)
    rs = np.random.default_rng(seed + 1)
    pairs = pick_pairs(Xb, k, r, n_pairs, rs)
    rows = []
    for a, b in pairs:
        base = one_pair(Xb, None, N, a, b, vol_cap)
        if base is None:
            continue
        bod = one_pair(Xb, Xbody, N, a, b, vol_cap)
        uni = one_pair(Xb, Xunif, N, a, b, vol_cap)
        if bod is None or uni is None:
            continue
        rows.append((base[0], base[1],
                     bod[0], bod[2], bod[3],
                     uni[0], uni[2], uni[3]))
    return np.array(rows, dtype=np.int64)


def report(A, M, k, label):
    if len(A) == 0:
        print(f'  {label}: пар не набралось')
        return None
    tau0, vbg = A[:, 0].astype(float), A[:, 1].astype(float)
    dt_b = (A[:, 2] - A[:, 0]).astype(float)
    dt_u = (A[:, 5] - A[:, 0]).astype(float)
    hb, hu = A[:, 4].astype(float), A[:, 7].astype(float)
    print(f'  {label}: пар {len(A)}, V_bg {vbg.mean():.0f}, '
          f'M/V_bg {M/vbg.mean():.3f}, tau фона {tau0.mean():.1f}')
    print(f"{'':>6}{'прирост':>12}{'высота массы':>15}"
          f"{'прирост/высота':>17}")
    out = {}
    for nm, dt, h in (('тело', dt_b, hb), ('контроль', dt_u, hu)):
        ok = h > 0
        ratio = dt[ok] / h[ok]
        m = ratio.mean() if ok.sum() else float('nan')
        se = ratio.std() / np.sqrt(max(ok.sum(), 1)) if ok.sum() else float('nan')
        print(f'{nm:>6}{dt.mean():>12.3f}{h.mean():>15.2f}'
              f'{f"{m:.4f} ± {se:.4f}":>17}')
        out[nm] = (m, se, int(ok.sum()))
    d = out['тело'][0] - out['контроль'][0]
    s = np.hypot(out['тело'][1], out['контроль'][1])
    print(f'{"":>6}разность коэффициентов: {d:.4f} ± {s:.4f}  '
          f'({abs(d)/s if s > 0 else float("nan"):.1f}σ)')
    return out


def main():
    ap = argparse.ArgumentParser(description='Задержка от массы в большом мире')
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--m', type=int, default=200)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--r', type=float, default=0.05)
    ap.add_argument('--pairs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=4000)
    a = ap.parse_args()

    t0 = time.time()
    A = run(a.n, a.m, a.k, a.r, a.pairs, a.seed, a.volcap)
    print(f'N={a.n}, M={a.m}, k={a.k}, радиус {a.r}, '
          f'потолок объёма {a.volcap}')
    report(A, a.m, a.k, f'k={a.k}')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
