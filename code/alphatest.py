"""Одна Λ на конус при α из молекул горизонта (0.7–1.4): какая доля миров проходит SN и BAO?"""
import sys, math, numpy as np
sys.argv = sys.argv[:1]
import everpresent as ep
from frozen import solve, past_cone, OMN, OMR
from frozenbao import chi_bao, chi_sn, rows
np.seterr(all='ignore')
it = (rows[:, 1] + rows[:, 2]).argmin(); ctot0 = rows[it, 1] + rows[it, 2]
csn0 = min(ep.lcdm_chi2(o) for o in np.linspace(0.2, 0.45, 51))
for alpha in (0.03, 0.1, 0.3, 0.7, 1.0, 1.4):
    xs = np.linspace(1e-4, 0.3 / alpha * 1.6, 1500) if alpha > 0.05 else np.linspace(0.005, 3, 600)
    dx = xs[1] - xs[0]; g_sn = g_all = 0.0; win = []
    for xi in xs:
        L = solve(alpha, xi); V, Hn = past_cone(L)
        h0 = 67.7 * Hn; Om = OMN / Hn ** 2
        if not (60 < h0 < 80 and 0.2 < Om < 0.45): continue
        w = math.exp(-xi ** 2 / 2) / math.sqrt(2 * math.pi) * dx
        if chi_sn(L, Hn) - csn0 < 4: g_sn += w
        if chi_bao(L, Hn) + chi_sn(L, Hn) - ctot0 < 4: g_all += w; win.append(xi)
    print(f'α = {alpha:4.2f}: SN Δχ²<4 {g_sn*100:5.2f} %; SN+BAO Δχ²<4 {g_all*100:5.2f} %' + (f'  (ξ ≈ {np.mean(win):.3f})' if win else ''))
    sys.stdout.flush()
