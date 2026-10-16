"""То же, сглаженно: источник «вправо» — гауссово пятно на одном слое; сравниваем отклик «влево» блуждания и −i·m·Kᵀf, усреднённые
по полоскам x на поздних слоях (точечные амплитуды на зерне шумят)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, fermions as Fe
for L, m in ((16, 1.0), (16, 0.5), (20, 1.0)):
    w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Fe.Walk(w); K = Pr.johnston11(w.X, 4.0, m)
    lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
    f = np.where(src, np.exp(-x * x / 4.5), 0); aR, aL, _ = W.run(m, f, 0 * f)
    kf = -1j * m * (K.T @ f) * (1.0 / 1.0)
    bins = np.arange(-10, 10.01, 1.0); rows = []
    for tl in lv[(lv > ts + 3) & (lv < L - 4)][::3]:
        Z = np.abs(t - tl) < 0.6
        for a, b in zip(bins[:-1], bins[1:]):
            S = Z & (x >= a) & (x < b)
            if S.sum(): rows.append((aL[S].mean(), kf[S].mean(), (a + b) / 2, tl))
    A = np.array(rows); aw, kk = A[:, 0], A[:, 1]
    r = np.vdot(kk, aw) / np.vdot(kk, kk); corr = abs(np.vdot(kk, aw)) / (np.linalg.norm(kk) * np.linalg.norm(aw))
    # фазовая скорость: где максимум |·| по x на позднем слое
    print(f'L = {L}, m = {m}: сглаженная корреляция {corr:.3f}, множитель |r| = {abs(r):.3f}, фаза r {np.degrees(np.angle(r)):+.0f}°; c нитей {W.c:.3f}', flush=True)
