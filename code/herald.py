#!/usr/bin/env python3
"""СВЕТОВОДЫ ВНУТРИ ЗАМКНУТОЙ СИСТЕМЫ, ИСТОЧНИК С ОПОВЕЩЕНИЕМ.

1) Пара рождается в неизвестном месте полосы источника и летит по замкнутой системе «мир + тень»
   (closed2.py: все звенья, скрытые ноги, резервуар, «поворот = i») — динамика унитарна точно.
2) В событиях A₁, A₂ (Алиса) и B₁, B₂ (Боб) начинаются световоды. Захват: световод забирает симметричную
   моду входящих ног события, c = ⟨s|v_in⟩, s = (1, …, 1)/√D (собственная мода матрицы Гровера);
   остальное идёт дальше свободно. Захват — один из выходов унитарного перехода, норма не теряется.
3) Оповещение: «по частице в световодах Алисы и Боба». Это случается ДО любых настроек (пластинки — внутри
   световодов, позже), поэтому отбор законен (как оповещённый источник в опытах без лазейки детектирования).
4) Дальше — только световоды с двумя крестами и двумя пластинками (railwalk2.py): переход точно унитарен,
   детекторы на концах ловят всё — после оповещения отбора нет.
Меряем: запутанность оповещённой пары, S идеальный и S на приборах, вероятность оповещения η,
разнесённость установок Алисы и Боба."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closed2 import Closed, CH
import railwalk2 as RW
from beamsplit import smax
sys.argv = _a

S = Closed(turn=True); X = S.X; n = S.n; w = S.w
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)

def run_record(inj, record):
    ncol = len(next(iter(inj.values())))
    amp = np.zeros((S.E, ncol), complex); rec = {}
    for k in S.order:
        li, lo, Um = S.legs[k]; D = len(Um)
        vin = np.zeros((D, ncol), complex); vin[:len(li)] = amp[li]
        if k in inj: vin[len(li) if len(li) < D else 0] += inj[k]
        if k in record: rec[k] = vin.sum(0) / math.sqrt(D)          # ⟨s|v_in⟩
        vout = Um @ vin
        for idx, e in enumerate(lo): amp[e] = vout[idx]
    return rec

src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]; N = len(src)
cands = [k for k in range(n) if 1.8 <= Xa[k, 0] <= 3.4 and 0.3 <= Xa[k, 1] <= 2.0]
quads = []
for A1 in cands:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
    if min(B1, B2) < 0: continue
    quads.append((A1, A2, B1, B2))
rec = run_record({s: np.eye(N)[i].astype(complex) for i, s in enumerate(src)}, {k for q in quads for k in q})
print(f'замкнутая система: событий {n}; источников {N}; четвёрок A₁A₂B₁B₂ (точные зеркала) {len(quads)}')
rows = []
for (A1, A2, B1, B2) in quads:
    cA = np.stack([rec[A1], rec[A2]]); cB = np.stack([rec[B1], rec[B2]])
    M = cA @ cB.T                                  # M_ij = Σ_s c_Ai(s) c_Bj(s)
    eta = 2 * (np.abs(M) ** 2).sum() / N
    if eta == 0: continue
    Mn = M / np.linalg.norm(M); C = 2 * abs(np.linalg.det(Mn)); Sid = smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj()))
    rows.append((Sid, C, eta, A1, A2, B1, B2, Mn))
rows.sort(key=lambda r: -r[0])
print(f'оповещённая пара: C макс {max(r[1] for r in rows):.3f}, S ид. макс {rows[0][0]:.3f}; S ид. > 2 у {sum(r[0] > 2.0005 for r in rows)} из {len(rows)}')
for Sid, C, eta, A1, A2, B1, B2, Mn in rows[:6]:
    r1 = RW.rail_of(A1, 4.5); cells, _, _ = RW.crosses(r1)
    r1b = [CH.img(w, k, CH.P, CH.P) for k in r1]
    rails_A = r1 + [CH.img(w, k, H, Hs) for k in r1]; rails_B = r1b + [CH.img(w, k, CH.P, CH.P) for k in [CH.img(w, k, H, Hs) for k in r1]]
    caus = sum(S.R[a, b] or S.R[b, a] for a in rails_A for b in rails_B if a >= 0 and b >= 0)
    S_dev = RW.two_cell_S(Mn, restarts=30) if len(cells) >= 2 else float('nan')
    print(f'  A₁ ({Xa[A1, 0]:.2f}, {Xa[A1, 1]:+.2f}): C {C:.3f}, S ид. {Sid:.3f}, на световодах S = {S_dev:.3f}; '
          f'вероятность оповещения η = {eta:.2e}; крестов {len(cells)}; причинных связей световодов Алиса–Боб {caus}')
    sys.stdout.flush()
