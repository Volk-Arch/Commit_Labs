"""ШАХМАТКА ФЕЙНМАНА НА КРИСТАЛЛЕ, шаг 2: плоская «световая сеть» и зигзаг-частица.
Сеть: кандидаты — связи в будущее (по порядку), сначала короткие; связь добавляется, если не пересекает уже взятые
(в 1+1 «пересекаются» определяется порядком: двумерный порядок = два световых упорядочения). У каждого события — шаг вправо
и шаг влево. Частица идёт по сети; поворот (смена направления) — множитель −i·m·Δt, прямо — 1 (Фейнман: шахматка = Дирак 1+1).
Проверка частицы: часы (частота покоя ≈ m?)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
def orient(a, b, c): return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
def planar_net(X):
    la, lb = Cs.staircase11(X); n = len(X); dt = X[lb, 0] - X[la, 0]; dx = X[lb, 1] - X[la, 1]; o = np.argsort(dt, kind='stable')
    R = -np.ones(n, int); Lf = -np.ones(n, int); P = np.zeros((0, 2)); Q = np.zeros((0, 2)); EA = np.zeros(0, int); EB = np.zeros(0, int)
    for e in o:
        if abs(dx[e]) < 1e-9: continue                                           # вертикальные связи — не шаги «влево/вправо»
        i, j = la[e], lb[e]; side = R if dx[e] > 0 else Lf
        if side[i] >= 0: continue
        a, b = X[i], X[j]
        if len(EA):
            share = (EA == i) | (EA == j) | (EB == i) | (EB == j)
            cross = (orient(a, b, P) * orient(a, b, Q) < 0) & (orient(P, Q, a[None]) * orient(P, Q, b[None]) < 0) & ~share
            if cross.any(): continue
        side[i] = j; P = np.vstack([P, a]); Q = np.vstack([Q, b]); EA = np.r_[EA, i]; EB = np.r_[EB, j]
    return R, Lf
if __name__ == '__main__':
    for L in (8, 12):
        w = Wd.gold11(4.0, L); X = w.X; R, Lf = planar_net(X); inner = np.abs(X[:, 0]) + np.abs(X[:, 1]) < L - 2
        sr = X[R[R >= 0]] - X[R >= 0]; sl = X[Lf[Lf >= 0]] - X[Lf >= 0]
        print(f'L = {L}: внутри есть шаг вправо у {(R[inner] >= 0).mean():.1%}, влево у {(Lf[inner] >= 0).mean():.1%}; '
              f'скорость вправо {np.median(sr[:, 1] / sr[:, 0]):.2f}, влево {np.median(sl[:, 1] / sl[:, 0]):.2f}; Δt {np.median(np.r_[sr[:, 0], sl[:, 0]]):.2f}', flush=True)
