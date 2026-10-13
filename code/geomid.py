"""Точная геодезическая для дуги 135°: радиус в середине дуги при T = f·T_K."""
import math, numpy as np, geodesic as g, json, sys
def mid(R, f, dphi=0.75 * math.pi):
    TK = dphi * math.sqrt(R ** 3); x = np.array([0.0, math.sqrt(R) / math.sqrt(1 - 3 / R)])
    def F(x):
        sol = g.shoot(R, x[0], x[1], phi_end=dphi)
        if not sol.t_events[0].size: return np.array([1e3, 1e3])
        y = sol.y_events[0][0]; return np.array([y[1] - R, y[0] - f * TK])
    for _ in range(60):
        fv = F(x); J = np.zeros((2, 2))
        for k in range(2):
            dx = np.zeros(2); dx[k] = 1e-6; J[:, k] = (F(x + dx) - fv) / 1e-6
        x = x + np.clip(np.linalg.solve(J, -fv), -0.2, 0.2)
        if abs(fv).max() < 1e-8: break
    sol = g.shoot(R, x[0], x[1], phi_end=dphi); s = sol.t_events[0][0]; ss = np.linspace(0, s, 4000); Y = sol.sol(ss)
    w = np.abs(Y[2] - dphi / 2) < 0.15
    return float(Y[1][w].mean()), np.column_stack([Y[1] * np.cos(Y[2]), Y[1] * np.sin(Y[2])])
if __name__ == '__main__':
    out = {}
    for R in (8.0, 12.0):
        row = []
        for f in (0.7, 0.85, 1.0, 1.15, 1.3):
            m, xy = mid(R, f); row.append(round(m, 3)); out[f'{R}_{f}'] = xy[::40].round(3).tolist()
        out[str(R)] = row; print(R, row)
    json.dump(out, open('/home/claude/data/geomid.json', 'w'))
