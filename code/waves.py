"""Волны и масса на кристалле: пропагатор Джонстона K_m = aL(I − abL)⁻¹ (сумма по путям из
связей), a = √ρ/(2π√6), b = −m²/ρ. Внутри светового конуса непрерывная теория даёт
ρ·⟨K⟩(τ) → G(τ) = −m·J₁(mτ)/(4πτ): волна «звенит» с частотой m — это и есть масса частицы.
Сравниваем золотой кристалл (общая система) и посыпку: среднее и разброс по парам."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from scipy.special import j1
from scipy.linalg import expm, solve_triangular
from gold4 import points_in_box, r_for_density
t0 = time.time(); rng = np.random.default_rng(21)
NEXP = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
H = (24 * NEXP / math.pi) ** 0.25                     # высота ромба
def lorentz():
    G = np.zeros((4, 4)); G[0, 1:] = G[1:, 0] = rng.normal(0, 0.4, 3)
    R = np.zeros((4, 4)); w = rng.normal(0, 1, 3)
    R[1, 2], R[2, 1], R[1, 3], R[3, 1], R[2, 3], R[3, 2] = w[0], -w[0], w[1], -w[1], w[2], -w[2]
    return expm(G) @ expm(R)
def in_diamond(P): return np.abs(P[:, 0] - H / 2) + np.linalg.norm(P[:, 1:], axis=1) <= H / 2
def crystal():
    L = lorentz(); Li = np.linalg.inv(L)
    corners = np.array([[t, x, y, z] for t in (0, H) for x in (-H/2, H/2) for y in (-H/2, H/2) for z in (-H/2, H/2)]) @ Li.T
    lo, hi = corners.min(0) - 0.5, corners.max(0) + 0.5; s = rng.uniform(0, 300, 4)
    P = points_in_box([(lo[k] + s[k], hi[k] + s[k]) for k in range(4)], r_for_density(1.0))
    P = (P - s) @ L.T; return P[in_diamond(P)]
def poisson():
    n = rng.poisson(NEXP * 16 / (2 * math.pi / 3) * (2 * math.pi / 3) / 16 * 16 / 1)  # не используем
    box = np.column_stack([rng.uniform(0, H, 40000), rng.uniform(-H/2, H/2, (40000, 3))])
    # плотность 1: в кубе объёма H·H³ ожидаем H⁴ точек
    m = rng.poisson(H ** 4); box = np.column_stack([rng.uniform(0, H, m), rng.uniform(-H/2, H/2, (m, 3))])
    return box[in_diamond(box)]
def propagators(P, masses):
    P = P[np.argsort(P[:, 0])]; n = len(P)
    d = P[None, :, :] - P[:, None, :]
    C = (d[..., 0] > 0) & (d[..., 0] ** 2 > (d[..., 1:] ** 2).sum(-1))
    tau = np.sqrt(np.maximum(d[..., 0] ** 2 - (d[..., 1:] ** 2).sum(-1), 0))
    Cf = C.astype(np.float32); L = C & ~((Cf @ Cf) > 0.5)
    a = 1 / (2 * math.pi * math.sqrt(6)); out = {}
    for m in masses:
        b = -m ** 2
        A = np.eye(n) - a * b * L                       # верхнетреугольная (порядок по времени)
        K = a * L.astype(float) @ solve_triangular(A, np.eye(n), lower=False)
        out[m] = K
    return P, C, tau, L, out
bins = np.arange(2, H, 1.0)
res = {}
for name, gen in (('кристалл', crystal), ('посыпка', poisson)):
    acc = {m: [[] for _ in bins[:-1]] for m in (0.3, 0.6)}
    for rep in range(3):
        P = gen(); P, C, tau, L, Ks = propagators(P, (0.3, 0.6))
        # пары: нижний край (первые 15 % по времени) → всё в будущем
        src = np.arange(int(0.15 * len(P)))
        for m, K in Ks.items():
            T = tau[src]; V = K[src]; Cm = C[src]
            for i in range(len(bins) - 1):
                sel = Cm & (T >= bins[i]) & (T < bins[i + 1])
                acc[m][i].extend(V[sel].tolist())
        print(f'  {name} повтор {rep}: N = {len(P)}, связей на событие {L.sum()/len(P):.1f}   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
    res[name] = acc
print(f'\nρ·⟨K_m⟩ по τ против непрерывной −mJ₁(mτ)/(4πτ); в скобках — разброс по парам (σ)')
for m in (0.3, 0.6):
    print(f'm = {m}:')
    print(f'{"τ":>5} | {"теория":>9} | {"кристалл":>20} | {"посыпка":>20}')
    for i in range(len(bins) - 1):
        tc = 0.5 * (bins[i] + bins[i + 1]); th = -m * j1(m * tc) / (4 * math.pi * tc)
        row = f'{tc:5.1f} | {th:+9.5f}'
        for name in ('кристалл', 'посыпка'):
            v = np.array(res[name][m][i]); row += f' | {v.mean():+9.5f} ({v.std():.4f})' if len(v) > 20 else ' |       —'
        print(row)
np.save('/home/claude/data/waves_res.npy', np.array([res], dtype=object), allow_pickle=True)
