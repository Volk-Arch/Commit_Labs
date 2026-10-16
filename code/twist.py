#!/usr/bin/env python3
"""КАЛИБРОВОЧНОЕ ПОЛЕ ИЗ ЗАКРУТКИ ОКНА (1+1, объемлющее R^{1,3}, тень пространственная).

Срез Σ: (T, x) -> (T, x, h1, h2),  h1 = κ(T² + x²)/2,  h2 = κ T x.
Формы изгиба K1 = κ·1, K2 = κ·σx. Гаусс: вклады взаимно гасятся => Σ плоский (до κ⁴).
Риччи: [K1, K2] ≠ 0 => нормали закручиваются, связность нормального расслоения A_μ
и её кривизна F = ∂_T A_x − ∂_x A_T ≈ ±2κ² (однородное «поле»).

1) считаем A, F и внутреннюю кривизну численно;
2) «заряд»: частица, крутящаяся в окне (окружность радиуса ρ в нормальной плоскости),
   — геодезическая на 3-мерной поверхности Σ × S¹ в плоском объемлющем пространстве.
   Сравниваем сдвиг по x для вращения +Ω и −Ω: сила Лоренца даёт противоположные знаки.
   Контроль: h2 = 0 (изгиб без закрутки) — разность обязана быть нулём по симметрии θ → −θ.
"""
import numpy as np, sys
ETA = np.diag([1.0, -1, -1, -1])
def ip(a, b): return np.einsum('...i,ij,...j->...', a, ETA, b)

def make(kappa, twist=True):
    def S(T, x):
        return np.array([T, x, kappa * (T * T + x * x) / 2, kappa * T * x if twist else 0.0])
    def tang(T, x, e=1e-5):
        return np.array([(S(T + e, x) - S(T - e, x)) / (2 * e), (S(T, x + e) - S(T, x - e)) / (2 * e)])
    def frame(T, x):
        Dt = tang(T, x); g = np.array([[ip(a, b) for b in Dt] for a in Dt]); gi = np.linalg.inv(g)
        N = []
        for v in (np.array([0, 0, 1.0, 0]), np.array([0, 0, 0, 1.0])):
            c = gi @ np.array([ip(a, v) for a in Dt]); v = v - c @ Dt
            for u in N: v = v - ip(v, u) / ip(u, u) * u
            N.append(v / np.sqrt(-ip(v, v)))
        return N
    return S, tang, frame

def connection(frame, T, x, e=1e-4):
    """A_μ = ⟨∂_μ N1, N2⟩ (знак: нормали пространственные, ⟨N,N⟩ = −1)"""
    N1p, _ = frame(T + e, x); N1m, _ = frame(T - e, x); _, N2 = frame(T, x)
    AT = ip((N1p - N1m) / (2 * e), N2)
    N1p, _ = frame(T, x + e); N1m, _ = frame(T, x - e)
    Ax = ip((N1p - N1m) / (2 * e), N2)
    return np.array([AT, Ax])

def field(frame, T, x, e=1e-3):
    return (connection(frame, T + e, x)[1] - connection(frame, T - e, x)[1]) / (2 * e) - (connection(frame, T, x + e)[0] - connection(frame, T, x - e)[0]) / (2 * e)

def metric2(tang, T, x):
    Dt = tang(T, x); return np.array([[ip(a, b) for b in Dt] for a in Dt])

def gauss(tang, T, x, e=1e-3):
    """скалярная кривизна 2-метрики численно (формула Бриоски через g и производные)"""
    def g(T, x): return metric2(tang, T, x)
    # Christoffel по конечным разностям, затем R
    def chris(T, x):
        dg = np.array([(g(T + e, x) - g(T - e, x)) / (2 * e), (g(T, x + e) - g(T, x - e)) / (2 * e)])  # dg[c,a,b] = ∂_c g_ab
        gi = np.linalg.inv(g(T, x))
        return 0.5 * np.einsum('ad,bdc->abc', gi, dg.transpose(1, 0, 2) + dg.transpose(1, 2, 0) - dg.transpose(0, 1, 2).transpose(1, 2, 0) * 0 - np.einsum('cab->abc', dg) * 0) if False else _chris(g, T, x, e)
    return _ricci_scalar(g, T, x, e)

def _chris(g, T, x, e):
    G = g(T, x); gi = np.linalg.inv(G)
    d = [(g(T + e, x) - g(T - e, x)) / (2 * e), (g(T, x + e) - g(T, x - e)) / (2 * e)]
    Gam = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                Gam[a, b, c] = 0.5 * sum(gi[a, d_] * (d[b][d_, c] + d[c][d_, b] - d[d_][b, c]) for d_ in range(2))
    return Gam

def _ricci_scalar(g, T, x, e):
    Gam = _chris(g, T, x, e)
    dG = [(_chris(g, T + e, x, e) - _chris(g, T - e, x, e)) / (2 * e), (_chris(g, T, x + e, e) - _chris(g, T, x - e, e)) / (2 * e)]
    Ric = np.zeros((2, 2))
    for b in range(2):
        for c in range(2):
            Ric[b, c] = sum(dG[a][a, b, c] - dG[c][a, b, a] + sum(Gam[a, a, d] * Gam[d, b, c] - Gam[a, c, d] * Gam[d, b, a] for d in range(2)) for a in range(2))
    return np.einsum('ab,ab->', np.linalg.inv(g(T, x)), Ric)

def geodesic(S, frame, rho, Om, L, n=6000):
    """геодезическая на Σ × S¹ (точка на окружности радиуса rho в нормальной плоскости)."""
    def X(y): T, x, th = y; N1, N2 = frame(T, x); return S(T, x) + rho * (np.cos(th) * N1 + np.sin(th) * N2)
    def G(y, e=1e-5):
        J = np.array([(X(y + e * np.eye(3)[k]) - X(y - e * np.eye(3)[k])) / (2 * e) for k in range(3)])
        return J @ ETA @ J.T
    def acc(y, v, e=1e-4):
        g = G(y); gi = np.linalg.inv(g)
        dg = np.array([(G(y + e * np.eye(3)[k]) - G(y - e * np.eye(3)[k])) / (2 * e) for k in range(3)])  # dg[k,a,b]
        Gam = 0.5 * np.einsum('ad,kdb->akb', gi, dg) * 0
        Gam = np.zeros((3, 3, 3))
        for a in range(3):
            Gam[a] = 0.5 * np.einsum('d,dbc->bc', gi[a], dg.transpose(1, 0, 2) + dg.transpose(1, 2, 0) - dg)
        return -np.einsum('abc,b,c->a', Gam, v, v)
    y = np.array([0.0, 0.0, 0.0]); v = np.array([1.0, 0.0, Om]); v /= np.sqrt(v @ G(y) @ v)
    h = L / n; traj = [y.copy()]
    for _ in range(n):
        k1y, k1v = v, acc(y, v)
        k2y, k2v = v + h / 2 * k1v, acc(y + h / 2 * k1y, v + h / 2 * k1v)
        k3y, k3v = v + h / 2 * k2v, acc(y + h / 2 * k2y, v + h / 2 * k2v)
        k4y, k4v = v + h * k3v, acc(y + h * k3y, v + h * k3v)
        y = y + h / 6 * (k1y + 2 * k2y + 2 * k3y + k4y); v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v); traj.append(y.copy())
    return np.array(traj)

if __name__ == '__main__':
    kap = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    for twist in (True, False):
        S, tang, frame = make(kap, twist)
        print(f'\n{"закрутка (h2 = κTx)" if twist else "контроль (h2 = 0)"}, κ = {kap}')
        for (T, x) in ((0.0, 0.0), (2.0, 1.0), (-1.5, 3.0)):
            print(f'  в ({T:+.1f},{x:+.1f}): F = {field(frame, T, x):+.6f}  (2κ² = {2 * kap * kap:.6f}),  скалярная кривизна R = {_ricci_scalar(lambda a, b: metric2(tang, a, b), T, x, 1e-3):+.2e}')
    print('\nчастица на окружности радиуса ρ в окне, вращение ±Ω; сдвиг по x при T = L')
    for twist in (True, False):
        S, tang, frame = make(kap, twist)
        for rho, Om, L in ((1.0, 0.5, 6.0), (1.0, 0.25, 6.0), (0.5, 1.0, 6.0)):
            tp = geodesic(S, frame, rho, Om, L, 1500); tm = geodesic(S, frame, rho, -Om, L, 1500)
            print(f'  {"закрутка" if twist else "контроль"} ρ={rho} Ω={Om}: x(+Ω) = {tp[-1,1]:+.5f}, x(−Ω) = {tm[-1,1]:+.5f}, разность {tp[-1,1] - tm[-1,1]:+.5f}   (T в конце {tp[-1,0]:.2f})')
