"""Данные для «галереи вселенных» на странице: для α = 1 (вывод) и 0.03 (подгонка) — по сетке
зерна ξ: Λ/Λ_наша, Ω_Λ, Ω_m, H0, доля вещества в галактиках (S = 2.5), согласие с SN+BAO."""
import sys, math, json, numpy as np
sys.argv = sys.argv[:1]
from frozen import solve, past_cone, OMN, OMR
from frozenbao import chi_bao, chi_sn, rows
from anthropic import growth, DC
np.seterr(all='ignore')
it = (rows[:, 1] + rows[:, 2]).argmin(); ctot0 = rows[it, 1] + rows[it, 2]
L_ours = 3 * 0.69
out = {}
for alpha in (1.0, 0.03):
    xs = np.sinh(np.linspace(-np.arcsinh(3 / 0.02), np.arcsinh(3 / 0.02), 900)) * 0.02
    rowsout = []
    for xi in xs:
        if abs(xi) < 1e-9: continue
        rec = dict(xi=round(float(xi), 6))
        if xi * alpha > 0.9:                       # Λ огромна: галактик нет, считать дорого
            rec.update(kind='empty'); rowsout.append(rec); continue
        try:
            L = solve(alpha, xi)
        except Exception:
            L = np.nan
        if not np.isfinite(L):
            rec.update(kind='crunch'); rowsout.append(rec); continue
        V, Hn = past_cone(L); D1 = growth(L); f = math.erfc(DC / (math.sqrt(2) * 2.5 * D1))
        h0 = 67.7 * Hn; Om = OMN / Hn ** 2; OL = L / 3 / Hn ** 2
        ok = bool((60 < h0 < 80 and 0.2 < Om < 0.45) and (chi_bao(L, Hn) + chi_sn(L, Hn) - ctot0 < 4))
        rec.update(kind='ok', lam=round(L / L_ours, 5), OL=round(OL, 4), Om=round(Om, 4), H0=round(h0, 2), gal=round(f, 5), match=ok)
        rowsout.append(rec)
    out[str(alpha)] = rowsout
    print(alpha, len(rowsout), sum(r.get('match', False) for r in rowsout)); sys.stdout.flush()
json.dump(out, open('/home/claude/data/gallery.json', 'w'), ensure_ascii=False, separators=(',', ':'))
