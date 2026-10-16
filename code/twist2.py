import numpy as np, sys
from twist import make, geodesic, connection, field
kap = 0.05; S, tang, frame = make(kap, True); F = field(frame, 0, 0)
print('A в начале координат', connection(frame, 0, 0), 'F', F)
def geo0(rho, Om, L, th0, n=1500):
    # старт с угла th0: сдвигаем угол в параметризации
    import twist
    def S2(T, x): return S(T, x)
    def frame2(T, x):
        N1, N2 = frame(T, x); c, s = np.cos(th0), np.sin(th0); return [c * N1 + s * N2, -s * N1 + c * N2]
    return geodesic(S2, frame2, rho, Om, L, n)
for rho, Om in ((1.0, 0.5), (1.0, 0.25), (0.5, 0.5), (1.0, 0.0)):
    d = []
    for sgn in (+1, -1):
        xs = [geo0(rho, sgn * Om, 6.0, th0)[-1, 1] for th0 in np.arange(8) * np.pi / 4]
        d.append(np.mean(xs))
    g = 1 / np.sqrt(1 - rho * rho * Om * Om); q = rho * rho * Om * g
    print(f'ρ={rho} Ω={Om}: среднее по 8 стартовым углам x(+) = {d[0]:+.5f}, x(−) = {d[1]:+.5f}, разность {d[0]-d[1]:+.5f};  Лоренц q·F·L²·γ = {q * F * 36 * g:+.5f}')
