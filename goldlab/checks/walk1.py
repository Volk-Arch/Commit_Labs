"""ДИРАК НА КРИСТАЛЛЕ, шаг 1: «световые нити». Каждое событие должно лежать ровно на одной нити, бегущей вправо, и одной — влево:
у события ровно один шаг-вправо в будущее и ровно один предшественник справа (то же влево). Это задача о паросочетании:
событие i → связь в будущее с Δx > 0, каждая цель — не более одного раза, сумма Δt — наименьшая (связи — из порядка).
Тогда квантовое блуждание «монета 2×2 в каждом событии» унитарно: что входит по двум нитям, то и выходит по двум."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.optimize import linear_sum_assignment
from goldlab import worlds as Wd, causal as Cs
def threads(X, side, big=1e6, cap=1.6):
    n = len(X); Cr = Cs.causal(X); la, lb = np.nonzero(Cr); dx = X[lb, 1] - X[la, 1]; dt = X[lb, 0] - X[la, 0]   # шаг — любое «раньше», не только связь
    ok = (dx * side > 1e-9) & (dt < cap); C = np.full((n, n), big); C[la[ok], lb[ok]] = dt[ok]
    r, c = linear_sum_assignment(C); nxt = -np.ones(n, int); good = C[r, c] < big; nxt[r[good]] = c[good]
    return nxt
if __name__ == '__main__':
    for L, cap in ((12, 1.3), (12, 1.6), (12, 2.0), (16, 1.6)):
        w = Wd.gold11(4.0, L); X = w.X; n = w.n; R = threads(X, +1, cap=cap); Lt = threads(X, -1, cap=cap)
        inner = np.abs(X[:, 0]) + np.abs(X[:, 1]) < L - 2
        prevR = np.zeros(n, int); prevR[R[R >= 0]] += 1; prevL = np.zeros(n, int); prevL[Lt[Lt >= 0]] += 1
        sr = X[R[R >= 0]] - X[R >= 0]; sl = X[Lt[Lt >= 0]] - X[Lt >= 0]
        full = inner & (R >= 0) & (Lt >= 0) & (prevR == 1) & (prevL == 1)
        print(f'L = {L}, предел шага {cap}: событий {n}; внутри с полным набором (вход/выход вправо и влево): {full.sum() / inner.sum():.1%}; '
              f'шаг вправо: Δt {np.median(sr[:, 0]):.2f}, скорость {np.median(sr[:, 1] / sr[:, 0]):.2f} (макс {np.max(sr[:, 0]):.2f}); '
              f'влево: Δt {np.median(sl[:, 0]):.2f}, скорость {np.median(sl[:, 1] / sl[:, 0]):.2f}', flush=True)
