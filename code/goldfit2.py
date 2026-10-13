"""Подгонка с нормальной обусловленностью: неизвестные W = ρ·w, уравнения
делятся на масштаб функции. Смотрим, достижима ли точность хотя бы на
ТОЙ ЖЕ решётке, на которой подгоняли (невязка на обучении)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from goldfit import collect, MMAX, lat
rs = np.random.default_rng(3)
for side in (40, 70, 125):
    P = lat(side); rho, rows, box = collect(P, 800, rs)
    S0 = np.array([S[:, 0] for _, S in rows]).mean(0)
    ms = [m for m in range(MMAX + 1) if S0[m] > 0.01]
    A, y = [], []
    for fx, S in rows:
        for k in range(6):
            sc = 1.0
            A.append(rho * np.concatenate([[fx[k]], S[ms, k]]) / sc); y.append(box[k] / sc)
    A, y = np.array(A), np.array(y)
    col = np.abs(A).max(0); col[col == 0] = 1
    Wn, *_ = np.linalg.lstsq(A / col, y, rcond=None)
    pred = (A / col) @ Wn
    res = (pred - y).reshape(-1, 6)
    print(f'N={len(P):6d}  невязка на обучении (σ по точкам):  B·1 {res[:,0].std():8.3f}   B·u {res[:,1].std():8.3f}   '
          f'B·u² {res[:,3].std():8.3f}   B·uv {res[:,5].std():8.3f} (цель 4, ср. {pred.reshape(-1,6)[:,5].mean():.3f})')
