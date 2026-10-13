"""Множитель в ΔV·ΔΛ из кристалла: через энтропию горизонта.
Энтропия горизонта = A/(4ℓ_p²). В причинном множестве её считают «молекулами
горизонта» (Доу–Соркин): пара x ≺ y, x снаружи горизонта, y внутри, обе в прошлом
среза Σ (T < 0), y — максимален (ничего после него до Σ), а у x в будущем до Σ
только y. Число молекул = c·A/ℓ²  ⇒  ℓ²/ℓ_p² = 4c  ⇒  α = ½(ℓ_p/ℓ)²/√ν = 1/(8c√ν).
Горизонт — нулевая плоскость T = X (горизонт ускоренного наблюдателя).
Миры: посыпка Пуассона, золотой кристалл (в общей системе), кристалл с размытым краем."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
from gold4 import points_in_box, r_for_density
from scipy.linalg import expm
t0 = time.time(); rng = np.random.default_rng(11)
TD, XW, YW, M = 3.0, 3.0, 10.0, 3.0         # глубина, полуширина по X, сторона площадки, запас
def region(P):
    T, X, Y, Z = P.T
    return P[(T < 0) & (T > -TD) & (np.abs(X) < XW) & (Y > -M) & (Y < YW + M) & (Z > -M) & (Z < YW + M)]

def lorentz():
    G = np.zeros((4, 4)); v = rng.normal(0, 0.35, 3); G[0, 1:] = G[1:, 0] = v
    R = np.zeros((4, 4)); w = rng.normal(0, 1.0, 3)
    R[1, 2], R[2, 1], R[1, 3], R[3, 1], R[2, 3], R[3, 2] = w[0], -w[0], w[1], -w[1], w[2], -w[2]
    return expm(G) @ expm(R)

def crystal(fuzzy_w=0.0):
    L = lorentz(); Li = np.linalg.inv(L)
    # куб в системе кристалла, покрывающий область после преобразования
    corners = np.array([[t, x, y, z] for t in (-TD, 0) for x in (-XW, XW) for y in (-M, YW + M) for z in (-M, YW + M)])
    C = corners @ Li.T; lo, hi = C.min(0) - 0.5, C.max(0) + 0.5
    shift = rng.uniform(0, 300, 4)
    if fuzzy_w == 0:
        P = points_in_box([(lo[k] + shift[k], hi[k] + shift[k]) for k in range(4)], r_for_density(1.0))
    else:
        from hidden import points_lab, prob, r_for
        r = r_for(fuzzy_w); P, s, _ = points_lab([(lo[k] + shift[k], hi[k] + shift[k]) for k in range(4)], r * (1 + fuzzy_w))
        P = P[rng.random(len(P)) < prob(s, r, fuzzy_w)]
    return region((P - shift) @ L.T)

def poisson():
    vol = TD * 2 * XW * (YW + 2 * M) ** 2; n = rng.poisson(vol)
    P = np.column_stack([rng.uniform(-TD, 0, n), rng.uniform(-XW, XW, n), rng.uniform(-M, YW + M, n), rng.uniform(-M, YW + M, n)])
    return P

def molecules(P):
    n = len(P); fut_cnt = np.zeros(n, int); only = np.full(n, -1)
    for i in range(0, n, 500):
        d = P[None, :, :] - P[i:i + 500, None, :]            # d[a, b] = P_b − P_a
        later = (d[..., 0] > 0) & (d[..., 0] ** 2 > (d[..., 1:] ** 2).sum(-1))
        fut_cnt[i:i + 500] = later.sum(1)
        one = later.sum(1) == 1
        only[i:i + 500][one] = later[one].argmax(1)
    T, X = P[:, 0], P[:, 1]
    inside = T > X
    mol = 0
    for xi in np.where((fut_cnt == 1) & ~inside)[0]:
        y = only[xi]
        if inside[y] and fut_cnt[y] == 0 and 0 <= P[y, 2] < YW and 0 <= P[y, 3] < YW: mol += 1
    return mol / YW ** 2

res = {}
for name, gen, reps in (('Пуассон', poisson, 12), ('золотой кристалл', crystal, 12), ('кристалл, край w=0.3', lambda: crystal(0.3), 12)):
    cs = []
    for k in range(reps):
        P = gen(); cs.append(molecules(P))
    cs = np.array(cs); res[name] = cs
    print(f'{name:22s}: молекул на ℓ² площади c = {cs.mean():.3f} ± {cs.std(ddof=1)/math.sqrt(reps):.3f}  (точек ~{len(P)})   [{time.time()-t0:.0f} с]')
    sys.stdout.flush()
print()
for name, cs in res.items():
    c = cs.mean(); l = math.sqrt(4 * c)
    for nu in (0.33, 1.0):
        print(f'{name:22s}, ν={nu}: ℓ/ℓ_p = {l:.2f},  α = 1/(8c√ν) = {1/(8*c*math.sqrt(nu)):.3f}   (данные: 0.02–0.04)')
