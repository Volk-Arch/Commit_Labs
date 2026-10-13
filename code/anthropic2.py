"""Типичность нашей Λ среди наблюдателей: какая доля наблюдателей (с Λ > 0) видит Λ меньше нашей."""
import sys, math, numpy as np
sys.argv_orig = list(sys.argv); sys.argv = sys.argv[:1]
from anthropic import growth, DC
from frozen import solve, past_cone, OMN
np.seterr(all='ignore')
L_ours = 3 * 0.69 * 1.0      # Λ/3 = Ω_Λ H0², в ед. H_ном (≈ наша)
for alpha in [float(a) for a in (sys.argv_orig[1:] or ['0.03', '1.0'])]:
    xs = np.linspace(1e-5, min(4, 0.6 / alpha), 1200); dx = xs[1] - xs[0]
    Ls = []; ws = {S: [] for S in (1.5, 2.5, 4.0)}
    for xi in xs:
        try:
            L = solve(alpha, xi); D1 = growth(L)
        except Exception:
            L, D1 = np.inf, 0.0            # Λ так велика, что галактик нет
        Ls.append(L)
        for S in ws: ws[S].append(math.exp(-xi ** 2 / 2) * (math.erfc(DC / (math.sqrt(2) * S * D1)) if D1 > 0 else 0.0))
    Ls = np.array(Ls)
    line = f'α = {alpha}: '
    for S, w in ws.items():
        w = np.array(w); line += f' S={S}: доля наблюдателей с Λ < нашей = {(w * (Ls < L_ours)).sum() / w.sum() * 100:.0f} %;'
        med = Ls[np.searchsorted(np.cumsum(w) / w.sum(), 0.5)]
        line += f' медиана Λ/Λ_наша = {med / L_ours:.1f}, вес на краю {w[-1]/w.max():.1e} |'
    print(line); sys.stdout.flush()
