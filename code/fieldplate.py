#!/usr/bin/env python3
"""ПЛАСТИНКА ИЗ ПОЛЯ. Внешняя пластинка e^{iα} на втором пути — единственное «комплексное» место в опыте.
Её можно сделать полем: фаза e^{iα} на каждом звене, пересекающем короткий пространственноподобный
отрезок Σ над A₂ (в системе Алисы), лежащий в будущем A₂, но вне будущего A₁. Каждый путь из A₂ к портам
пересекает Σ ровно один раз, путь из A₁ — ни разу. Это калибровочный потенциал вдоль отрезка; его поле F
сосредоточено в двух концах отрезка (два точечных потока ±α) — как у эффекта Ааронова–Бома.
Проверяем: T_поле = T·diag(1, e^{iα}) точно? Делитель остаётся без потерь? Ломается ли вещественность?"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a
K, LABK, rho = 2, 1, 4.0
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
w = CH.sym_world(rho, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), LABK); X = w.X; n = w.n
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
I, J = LK['I'], LK['J']
worst = 0; rows = 0; mixed = 0; ucheck = 0
for A1 in [k for k in range(n) if 2.0 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.0]:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    tA, xA1, xA2 = Xa[A2, 0], Xa[A1, 1], Xa[A2, 1]
    tX = tA + abs(xA1 - xA2) / 2; tS = tA + 0.5 * (tX - tA); half = tS - tA
    # звенья, пересекающие Σ: T̃ = tS, x̃ ∈ [xA2 − half, xA2 + half]
    t0, t1 = Xa[I, 0], Xa[J, 0]
    cr = (t0 < tS) & (t1 >= tS)
    xc = Xa[I, 1] + (Xa[J, 1] - Xa[I, 1]) * (tS - t0) / np.where(t1 - t0 == 0, 1, t1 - t0)
    cut = cr & (xc >= xA2 - half) & (xc <= xA2 + half)
    for m0 in (0, 1):
        for alpha in (math.pi / 2, 1.0):
            fld = np.where(cut, np.exp(1j * alpha), 1.0)
            Z1, Z2 = CH.chiral(w, LK, R, A1, m0), CH.chiral(w, LK, R, A2, 1 - m0)
            F1, F2 = CH.chiral(w, LK, R, A1, m0, field=fld), CH.chiral(w, LK, R, A2, 1 - m0, field=fld)
            for m1 in (0, 1):
                for p in range(n):
                    if not (4.0 <= Xa[p, 0] <= 6.5 and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
                    q = CH.img(w, p, H, Hs)
                    if q < 0: continue
                    T0 = np.array([[Z1[p, m1], Z2[p, m1]], [Z1[q, 1 - m1], Z2[q, 1 - m1]]])
                    TF = np.array([[F1[p, m1], F2[p, m1]], [F1[q, 1 - m1], F2[q, 1 - m1]]])
                    if np.abs(T0).max() == 0: continue
                    worst = max(worst, np.abs(TF - T0 @ np.diag([1, np.exp(1j * alpha)])).max() / np.abs(T0).max())
                    u = 2 * abs(np.linalg.det(TF)) / (np.abs(TF) ** 2).sum(); ucheck = max(ucheck, 1 - u); rows += 1
            # вещественность после поля: доля событий, где у ψ «смешанный» класс
            a = np.abs(F2).max(); mixed = max(mixed, (np.minimum(np.abs(F2.real), np.abs(F2.imag)) > 1e-9 * a).mean())
    print(f'A1 ({Xa[A1,0]:.2f},{Xa[A1,1]:+.2f}): звеньев под полем {cut.sum()}; пока макс |T_поле − T·diag(1,e^iα)| = {worst:.1e}'); sys.stdout.flush()
print(f'== делителей {rows}: T_поле = T·diag(1, e^iα) с точностью {worst:.1e}; 1 − качество макс {ucheck:.1e}; '
      f'событий со «смешанным» (комплексным) классом амплитуды после поля: до {mixed:.0%}')
