#!/usr/bin/env python3
"""«Иррациональная» решётка 1+1: точки (u, v) = (a + bτ, a + bτ'), a,b ∈ Z,
τ = золотое сечение, τ' = −1/τ. Это обычная решётка (кристалл), но
умножение на τ² переводит её в себя: (u,v) -> (τ²u, v/τ²) — это БУСТ
(быстрота ln τ² = 0.962). Решётка с точной дискретной симметрией Лоренца.
Нормировано на плотность 1. Сравнение с Пуассоном и квадратной решёткой.
"""
import sys, numpy as np
sys.path.insert(0, '/home/claude')
from core import links_of
tau = (1 + 5 ** .5) / 2; tp = -1 / tau
s = 5 ** -0.25                                   # covolume √5 -> плотность 1

def gold_points(U):
    """все точки решётки в квадрате [0,U]^2 (световые координаты)"""
    B = int(U / s * 1.5) + 5
    a, b = np.meshgrid(np.arange(-3 * B, 3 * B), np.arange(-2 * B, 2 * B), indexing='ij')
    u = s * (a + b * tau); v = s * (a + b * tp)
    m = (u >= 0) & (u <= U) & (v >= 0) & (v <= U)
    return u[m], v[m]

rs = np.random.default_rng(2); V = 40.0; M = 3000
print('1. ЧИСЛО ТОЧЕК В РОМБЕ ОБЪЁМА 40 ПРИ БЫСТРОТЕ η (Пуассон: 40 ± 6.3 всегда)')
GU, GV = gold_points(400.0)
order = np.argsort(GU); GU, GV = GU[order], GV[order]
for eta in (0, 0.5, 1, 2, 3, 4):
    du, dv = np.sqrt(V) * np.exp(eta), np.sqrt(V) * np.exp(-eta)
    c = np.empty(M)
    for m in range(M):
        u0 = rs.uniform(0, 400 - du); v0 = rs.uniform(0, 400 - dv)
        lo, hi = np.searchsorted(GU, [u0, u0 + du])
        vv = GV[lo:hi]; c[m] = np.count_nonzero((vv >= v0) & (vv <= v0 + dv))
    sq = (np.floor(rs.uniform(0,1,M) + du) ) * (np.floor(rs.uniform(0,1,M) + dv))
    print(f'   η={eta:3.1f}  золотая: {c.mean():5.1f} ± {c.std():4.1f}  пустых {np.mean(c==0):.2f}   |  квадратная: {sq.mean():5.1f} ± {sq.std():5.1f}  пустых {np.mean(sq==0):.2f}')

print('\n2. СВЯЗЕЙ НА СОБЫТИЕ (среднее по внутренним точкам) против размера мира N')
for U in (12, 20, 32, 50, 80):
    u, v = gold_points(float(U)); N = len(u)
    R = (u[None, :] < u[:, None]) & (v[None, :] < v[:, None])
    Lg = links_of(R).sum(1)
    up, vp = rs.uniform(0, U, N), rs.uniform(0, U, N)
    Rp = (up[None, :] < up[:, None]) & (vp[None, :] < vp[:, None])
    Lp = links_of(Rp).sum(1)
    k = int(np.sqrt(N)); ui = np.round(rs.uniform(0, 1, 1))  # квадратная решётка: ровно 2 (кроме края)
    print(f'   N={N:5d}  ln N={np.log(N):4.2f}   золотая {Lg.mean():5.2f} (макс {Lg.max()})   Пуассон {Lp.mean():5.2f}   квадратная (u,v): 2')
