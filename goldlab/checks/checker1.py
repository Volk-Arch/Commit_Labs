"""ШАХМАТКА ФЕЙНМАНА НА КРИСТАЛЛЕ, шаг 1: граф «самых световых» связей. У каждого события — связь вправо R (наименьший Δv) и
влево Λ (наименьший Δu) среди его связей в будущее. Вопрос: плоский ли этот граф (рёбра пересекаются только в событиях)?
Если да — сумма по НЕВСТРЕЧАЮЩИМСЯ путям = определитель (ЛГВ) точно, и «запрет встречи» = Паули без оговорок."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
def light_graph(X):
    la, lb = Cs.staircase11(X); U = X[:, 0] + X[:, 1]; V = X[:, 0] - X[:, 1]; n = len(X); R = -np.ones(n, int); Lm = -np.ones(n, int)
    for i in range(n):
        js = lb[la == i]
        if len(js) == 0: continue
        R[i] = js[np.argmin(V[js] - V[i])]; Lm[i] = js[np.argmin(U[js] - U[i])]
    return R, Lm
def crossings(X, E):
    P, Q = X[E[:, 0]], X[E[:, 1]]; cnt = 0
    def orient(a, b, c): return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    for k in range(len(E)):
        a, b = P[k], Q[k]; c, d = P[k + 1:], Q[k + 1:]
        share = (E[k + 1:, 0] == E[k, 0]) | (E[k + 1:, 0] == E[k, 1]) | (E[k + 1:, 1] == E[k, 0]) | (E[k + 1:, 1] == E[k, 1])
        x = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a[None]) * orient(c, d, b[None]) < 0) & ~share
        cnt += int(x.sum())
    return cnt
for L in (8, 12):
    w = Wd.gold11(4.0, L); X = w.X; R, Lm = light_graph(X); ok = R >= 0
    E = np.unique(np.r_[np.c_[np.nonzero(ok)[0], R[ok]], np.c_[np.nonzero(ok)[0], Lm[ok]]], axis=0)
    la, lb = Cs.staircase11(X); Eall = np.c_[la, lb]
    same = (R[ok] == Lm[ok]).mean()
    print(f'L = {L}: событий {w.n}; всех связей {len(Eall)}, пересечений между ними {crossings(X, Eall)}; световых рёбер {len(E)}, пересечений {crossings(X, E)}; R = Λ у {same:.1%} событий', flush=True)

print('— короткие шаги: вправо/влево — связь с наименьшим Δt по каждую сторону —')
for L in (8, 12):
    w = Wd.gold11(4.0, L); X = w.X; la, lb = Cs.staircase11(X); n = w.n; Rs = -np.ones(n, int); Ls = -np.ones(n, int)
    for i in range(n):
        js = lb[la == i]; dx = X[js, 1] - X[i, 1]; dt = X[js, 0] - X[i, 0]
        r = js[dx > 1e-9]; l = js[dx < -1e-9]
        if len(r): Rs[i] = r[np.argmin(dt[dx > 1e-9])]
        if len(l): Ls[i] = l[np.argmin(dt[dx < -1e-9])]
    E = np.unique(np.r_[np.c_[np.nonzero(Rs >= 0)[0], Rs[Rs >= 0]], np.c_[np.nonzero(Ls >= 0)[0], Ls[Ls >= 0]]], axis=0)
    st = X[E[:, 1]] - X[E[:, 0]]
    print(f'L = {L}: рёбер {len(E)}, пересечений {crossings(X, E)}; |скорость| шага: медиана {np.median(np.abs(st[:, 1] / st[:, 0])):.2f}, Δt медиана {np.median(st[:, 0]):.2f}', flush=True)
