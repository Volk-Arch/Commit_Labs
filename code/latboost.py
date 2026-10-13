#!/usr/bin/env python3
"""Почему не решётка: число точек в причинном ромбе фиксированного объёма
при разных скоростях наблюдателя (быстрота η). 1+1, плотность 1.

Ромб в световых координатах — прямоугольник Δu × Δv с ΔuΔv = V
(Δu = √V e^η, Δv = √V e^−η): тот же объём, другой наблюдатель.
У честной дискретности число точек должно зависеть только от V.
"""
import numpy as np
rs = np.random.default_rng(1)
V, M = 40.0, 4000                       # объём ромба, число случайных ромбов
tau = (1 + 5 ** .5) / 2

def counts_lattice(pts_fn, eta):
    du, dv = np.sqrt(V) * np.exp(eta), np.sqrt(V) * np.exp(-eta)
    c = np.empty(M)
    for m in range(M):
        u0, v0 = rs.uniform(0, 1000, 2)
        c[m] = pts_fn(u0, v0, du, dv)
    return c

def square_null(u0, v0, du, dv):        # решётка Z² в световых координатах
    return (np.floor(u0 + du) - np.floor(u0)) * (np.floor(v0 + dv) - np.floor(v0))

S2 = np.sqrt(2)
def square_rest(u0, v0, du, dv):        # квадратная решётка в (t, x), шаг 1
    # точки t,x ∈ Z  ->  u = (t+x)/√2, v = (t−x)/√2, плотность 1
    T = np.arange(np.floor((u0 + v0) / S2) - 2, np.ceil((u0 + du + v0 + dv) / S2) + 3)
    n = 0
    for t in T:
        # x такие, что u ∈ [u0,u0+du], v ∈ [v0,v0+dv]
        lo = max(S2 * u0 - t, t - S2 * (v0 + dv)); hi = min(S2 * (u0 + du) - t, t - S2 * v0)
        if hi >= lo: n += np.floor(hi) - np.ceil(lo) + 1
    return n

# Фибоначчи: квазирешётка на прямой (длины 1 и 1/τ), средний шаг 1
def fib_points(L):
    w = []; x = 0.0
    for k in range(1, int(L * 1.2) + 10):
        x += 1.0 if np.floor((k + 1) / tau) - np.floor(k / tau) == 1 else 1 / tau
        w.append(x)
    w = np.array(w); return w / (w[-1] / len(w))
FIB = fib_points(3000)
def fib_null(u0, v0, du, dv):           # квазикристалл-произведение в световых координатах
    a = np.searchsorted(FIB, u0 + du) - np.searchsorted(FIB, u0)
    b = np.searchsorted(FIB, v0 + dv) - np.searchsorted(FIB, v0)
    return a * b

print(f'Объём ромба V={V:.0f}. Пуассон: среднее {V:.0f}, дисперсия {V:.0f} при ЛЮБОЙ быстроте.')
print(f"{'быстрота η':>11}{'Δu/Δv':>9} |{'Пуассон':>16} |{'решётка в (u,v)':>18} |{'решётка в (t,x)':>18} |{'Фибоначчи (u,v)':>18}")
for eta in (0, 1, 2, 3, 4):
    out = []
    pois = rs.poisson(V, M); out.append(pois)
    for f in (square_null, square_rest, fib_null):
        out.append(counts_lattice(f, eta))
    s = ''.join(f' |{c.mean():7.1f} ±{c.var()**.5:6.1f}  ' for c in out)
    frac0 = [np.mean(c == 0) for c in out]
    print(f'{eta:>11}{np.exp(2*eta):>9.0f}{s}  пустых: ' + ' '.join(f'{z:.2f}' for z in frac0))
