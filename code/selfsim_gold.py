"""Самоподобие золотого квазикристалла: события с тенью ≤ 0.618·r — это весь
мир, увеличенный в τ раз? Проверка на точках и на порядке, три уровня."""
import sys, math, numpy as np
sys.path.insert(0, '.')
from gold4 import points_in_box, r_for_density, TAU
from scipy.spatial import cKDTree
def world(r, scale):
    P = points_in_box([(-1.02 * scale, 1.02 * scale)] * 4, r)
    return P[np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= scale * (1 + 1e-12)]
def mink(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    return dt > dx
r = r_for_density(1500 / (2 * math.pi / 3))
W0 = world(r, 1.0)
for k in (1, 2, 3):
    s = TAU ** k
    Wk = world(r * TAU ** -k, s)              # мир с тенью в τᵏ раз меньше, в ромбе в τᵏ раз больше
    A = W0 * s                                 # исходный мир, увеличенный в τᵏ
    d, mp = cKDTree(Wk).query(A)
    same_pts = len(Wk) == len(A) and d.max() < 1e-9 and len(set(mp)) == len(mp)
    same_order = same_pts and (mink(A) == mink(Wk)[np.ix_(mp, mp)]).all()
    print(f'τ^{k}: событий {len(A)} и {len(Wk)}; точки совпадают: {same_pts} (макс. расхождение {d.max():.1e}); порядок совпадает: {same_order}')
# доля «грубых» событий в том же ромбе
Wc = W0[[True] * 0 or slice(None)]
P = points_in_box([(-1.02, 1.02)] * 4, r * 0.618033988749895)
P = P[np.abs(P[:, 0]) + np.linalg.norm(P[:, 1:], axis=1) <= 1]
print(f'в том же ромбе событий с тенью ≤ 0.618r: {len(P)} из {len(W0)} = {len(P)/len(W0):.3f} (ожидание τ⁻⁴ = {TAU**-4:.3f})')
