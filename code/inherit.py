"""Самоподобный шум «по родословной»: в мире де Ситтера каждый шаг τ рождает слой
событий в τ⁴ раз больше прежнего. Монетка нового события с вероятностью q
копирует монетку его предка (x/τ), иначе бросается заново. Отклонение слоя:
d_{k+1} = q·τ⁴·d_k + √((1−q)·n_{k+1})·ξ. Λ_k = (Σ d)/V_k.
Слежение за H² (|Λ|·t² постоянно) и смены знака на шаг τ."""
import numpy as np, math
TAU = (1 + 5 ** .5) / 2; K = 60; R4 = TAU ** 4
n = R4 ** np.arange(K); V = np.cumsum(n); t = V ** 0.25
rs = np.random.default_rng(4)
print(f'порог 1/τ² = {1/TAU**2:.3f}')
for q in (0.0, 0.2, 0.3, 0.35, 0.37, 0.4, 0.5, 1.0):
    reps = 2000; d = np.zeros((reps, K)); d[:, 0] = rs.normal(0, 1, reps)
    for k in range(1, K):
        d[:, k] = q * R4 * d[:, k - 1] + math.sqrt((1 - q) * n[k]) * rs.normal(size=reps)
    D = np.cumsum(d, axis=1); L = D / V
    a = np.median(np.abs(L) * t ** 2, axis=0)
    trend = np.polyfit(np.log(t[20:]), np.log(a[20:]), 1)[0]
    flips = np.mean(np.sum(np.diff(np.sign(L[:, 20:]), axis=1) != 0, axis=1)) / (K - 21)
    print(f'q = {q:.2f}:  |Λ|·t² ~ t^{trend:+.2f} (слежение: 0)   смен знака на шаг τ: {flips:.2f}')
