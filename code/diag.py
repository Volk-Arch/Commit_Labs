"""Где шум в цене изгиба и подтверждение артефакта окна."""
import numpy as np
from weightmass import box_sprinkle, tube, bend, per_ev, n_for

EPS, M, L, N = 0.2, 400, 1.0, n_for(1.0)
print(f'N={N}, M={M}, eps={EPS}')
print('dS_bg = сумма по ФОНОВЫМ событиям, изогнутый минус прямой')
print('(при delta=0 обязан быть ровно нуль)')
print()
print(f"{'delta':>7}{'окно 0.5':>22}{'весь фон':>22}")
for dl in (0.0, 0.1, 0.2, 0.3):
    w, f = [], []
    for t in range(30):
        rs = np.random.default_rng(300 + 17*t)
        bg = box_sprinkle(N, L, rs); ob = tube(M, 0.05, rs)
        win = np.linalg.norm(bg[:, 1:], axis=1) < 0.5
        s0 = per_ev(np.vstack([ob, bg]), EPS)[M:]
        s1 = per_ev(np.vstack([bend(ob, dl), bg]), EPS)[M:]
        w.append(float(s1[win].sum() - s0[win].sum()))
        f.append(float(s1.sum() - s0.sum()))
    g = lambda v: f'{np.mean(v):>13.3f} ±{np.std(v)/np.sqrt(len(v)):<7.3f}'
    print(f'{dl:>7.1f}{g(w):>22}{g(f):>22}')
