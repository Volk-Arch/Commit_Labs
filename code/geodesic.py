"""Точная геодезическая Шварцшильда (экватор) из (0, R, φ=0) в (T, R, φ=π/2): стрельба."""
import numpy as np, math
from scipy.integrate import solve_ivp
def shoot(R, w, L, M=1.0, phi_end=math.pi / 2):
    A0 = 1 - 2 * M / R; E = math.sqrt(w * w + A0 * (1 + L * L / R ** 2))
    def rhs(s, y):
        t, r, ph, v = y; A = 1 - 2 * M / r
        return [E / A, v, L / r ** 2, -M / r ** 2 + L ** 2 / r ** 3 - 3 * M * L ** 2 / r ** 4]
    ev = lambda s, y: y[2] - phi_end; ev.terminal = True; ev.direction = 1
    sol = solve_ivp(rhs, (0, 1e5), [0, R, 0, w], events=ev, rtol=1e-10, atol=1e-10, dense_output=True, max_step=0.5)
    return sol
def bvp(R, T, M=1.0):
    Lc = math.sqrt(M * R) / math.sqrt(1 - 3 * M / R); x = np.array([0.0, Lc])
    def F(x):
        sol = shoot(R, x[0], x[1])
        if not sol.t_events[0].size: return np.array([1e3, 1e3])
        y = sol.y_events[0][0]; return np.array([y[1] - R, y[0] - T])
    for _ in range(40):
        f = F(x); J = np.zeros((2, 2)); h = 1e-6
        for k in range(2):
            dx = np.zeros(2); dx[k] = h; J[:, k] = (F(x + dx) - f) / h
        step = np.linalg.solve(J, -f); x = x + np.clip(step, -0.2, 0.2)
        if np.abs(f).max() < 1e-8: break
    sol = shoot(R, x[0], x[1]); s_end = sol.t_events[0][0]
    ss = np.linspace(0, s_end, 2000); Y = sol.sol(ss)
    t, r = Y[0], Y[1]
    rmean_t = np.trapezoid(r, t) / t[-1]        # среднее по координатному времени
    return dict(w=x[0], L=x[1], rmean=float(rmean_t), xy=np.column_stack([r * np.cos(Y[2]), r * np.sin(Y[2])]), tau=float(s_end))
if __name__ == '__main__':
    for R in (8, 12, 16):
        TK = (math.pi / 2) * math.sqrt(R ** 3)
        print(R, [round(bvp(R, f * TK)['rmean'], 2) for f in (0.8, 0.9, 1.0, 1.1, 1.2)])
