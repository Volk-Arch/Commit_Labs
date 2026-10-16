import numpy as np, sys
from twist import make, geodesic, field
kap = 0.05; S, tang, frame = make(kap, True); F = field(frame, 0, 0); L = 6.0
def geo0(rho, Om, th0, n):
    def frame2(T, x):
        N1, N2 = frame(T, x); c, s = np.cos(th0), np.sin(th0); return [c * N1 + s * N2, -s * N1 + c * N2]
    return geodesic(S, frame2, rho, Om, L, n)
for rho, Om in ((1.0, 0.5), (0.5, 1.0), (0.25, 2.0), (0.125, 4.0)):
    n = int(600 * max(1, Om / 0.5))
    d = []
    for sgn in (+1, -1):
        d.append(np.mean([geo0(rho, sgn * Om, th0, n)[-1, 1] for th0 in np.arange(4) * np.pi / 2]))
    g = 1 / np.sqrt(1 - rho * rho * Om * Om); q = rho * rho * Om * g
    pred = q * F * L * L * g
    print(f'ρ={rho} Ω={Om} (q = ρ²θ̇ = {q:.4f}): разность x(+)−x(−) = {d[0]-d[1]:+.5f}, Лоренц {pred:+.5f}, отношение {(d[0]-d[1])/pred:.3f}'); sys.stdout.flush()
