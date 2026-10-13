#!/usr/bin/env python3
"""Веса БД выведены для ПУАССОНА. Решётка — не Пуассон (размеры интервалов —
гребёнка), поэтому оператор с весами (1,−2,1) на ней смещён (растёт ~N).
Вопрос по-честному: существуют ли веса w_m (по размеру интервала m) и c,
при которых на золотой решётке оператор даёт □f точно для всех квадратичных f
— и тогда как ведёт себя разброс по N?

Подгонка на одной решётке (N≈5000), проверка без переподгонки на N до 1.6e5.
Контроль: то же на засеве.
"""
import sys, math, time, numpy as np
sys.path.insert(0, '/home/claude')
from fastwave import tx
from goldlat import gold_points

MMAX = 12
def sums_at(U, V, idx, order_u, pos, fvals):
    p = pos[idx]; cand_i = order_u[:p]
    cand = cand_i[V[cand_i] < V[idx]][::-1]
    S = np.zeros((MMAX + 1, len(fvals)))
    if len(cand) == 0: return S
    b = V[idx] - V[cand]; a = U[idx] - U[cand]
    alive = np.ones(len(cand), bool); lay = []
    for _ in range(MMAX + 1):
        bb = np.where(alive, b, np.inf)
        rmin = np.minimum.accumulate(np.concatenate([[np.inf], bb[:-1]]))
        L = alive & (bb < rmin)
        if not L.any(): break
        lay.append(np.nonzero(L)[0]); alive &= ~L
    ci = np.concatenate(lay)
    cnt = np.array([np.count_nonzero((a < a[j]) & (b < b[j])) for j in ci])
    keep = cnt <= MMAX; ci, cnt = ci[keep], cnt[keep]
    ys = cand[ci]
    for k, f in enumerate(fvals):
        np.add.at(S[:, k], cnt, f[ys])
    return S

def collect(P, npts, rs):
    U, V = P[:, 0], P[:, 1]; n = len(P); t, x = tx(P)
    order_u = np.argsort(U); pos = np.empty(n, int); pos[order_u] = np.arange(n)
    ok = np.nonzero((t > 0.3) & (t < 0.7) & (np.abs(x) < 0.15))[0]
    pick = ok if len(ok) <= npts else rs.choice(ok, npts, replace=False)
    # функции и их □ (□ = 4 ∂u∂v):  1,u,v -> 0;  u², v² -> 0;  uv -> 4
    F = [np.ones(n), U, V, U * U, V * V, U * V]
    box = np.array([0, 0, 0, 0, 0, 4.0])
    rows = []
    for idx in pick:
        S = sums_at(U, V, idx, order_u, pos, F)
        fx = np.array([f[idx] for f in F])
        rows.append((fx, S))
    return 2.0 * n, rows, box

def design(rho, rows, box, ms):
    A = []; y = []
    for fx, S in rows:
        for k in range(len(fx)):
            A.append(rho * np.concatenate([[fx[k]], S[ms, k]])); y.append(box[k])
    return np.array(A), np.array(y)

def apply(rho, rows, box, ms, w):
    # значение оператора на uv (□=4) и на 1 (□=0) в каждой точке
    out = {k: [] for k in ('1', 'u²', 'uv')}
    for fx, S in rows:
        for lab, k in (('1', 0), ('u²', 3), ('uv', 5)):
            out[lab].append(rho * (w[0] * fx[k] + S[ms, k] @ w[1:]))
    return {k: np.array(v) for k, v in out.items()}

rs = np.random.default_rng(11)
def lat(side):
    u, v = gold_points(float(side)); return np.stack([u / side, v / side], 1)

for lab, gen in (('ЗОЛОТАЯ', lat), ('ЗАСЕВ', lambda s: rs.random((int(round(len(gold_points(float(s))[0]))), 2)))):
    rho, rows, box = collect(gen(70), 600, rs)
    S0 = np.array([S[:, 0] for _, S in rows]).mean(0)
    ms = [m for m in range(MMAX + 1) if S0[m] > 0.01]
    A, y = design(rho, rows, box, ms)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f'{lab}: размеры с ненулевым числом {ms}')
    print('   веса: c=%+.4f  ' % w[0] + '  '.join(f'w{m}={x:+.4f}' for m, x in zip(ms, w[1:])))
    print(f"   {'N':>8} | {'B·1: среднее':>13} {'σ':>8} | {'B·u²: ср.':>11} {'σ':>8} | {'B·uv (цель 4)':>14} {'σ':>8}")
    for side in (40, 70, 125, 220, 400):
        P = gen(side); rho2, rows2, _ = collect(P, 400, rs)
        r = apply(rho2, rows2, box, ms, w)
        print(f'   {len(P):>8} | {r["1"].mean():>13.3f} {r["1"].std():>8.3f} | {r["u²"].mean():>11.3f} {r["u²"].std():>8.3f} | {r["uv"].mean():>14.3f} {r["uv"].std():>8.3f}')
        sys.stdout.flush()
