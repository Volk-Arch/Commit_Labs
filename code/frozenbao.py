"""Одна Λ на прошлый конус (frozen-doc) против BAO DESI DR2 (+ сверхновые Pantheon+).
Ранняя вселенная стандартная (Λ при рекомбинации ~1e−9), поэтому r_d стандартный:
ω_m = 0.142, ω_b = 0.0222 ⇒ r_d ≈ 147.4 Мпк (аппроксимация Aubourg 2015)."""
import sys; sys.argv = sys.argv[:1]
import numpy as np, math
import everpresent as ep
from frozen import solve, past_cone, OMN, OMR
np.seterr(all='ignore')
C = 299792.458; HN = 67.7; DHN = C / HN                       # Мпк
RD = 147.05 * (0.0222 / 0.02236) ** -0.13 * (0.142 / 0.1432) ** -0.23
B = np.loadtxt('/home/claude/data/bao/desi_gaussian_bao_ALL_GCcomb_mean.txt', dtype=str)
BZ = B[:, 0].astype(float); BV = B[:, 1].astype(float); BQ = B[:, 2]
BC = np.linalg.inv(np.loadtxt('/home/claude/data/bao/desi_gaussian_bao_ALL_GCcomb_cov.txt'))

def bao_model(L, Hn):
    zz = np.linspace(0, 2.5, 5001)
    E = np.sqrt(OMN * (1 + zz) ** 3 + OMR * (1 + zz) ** 4 + L / 3)   # H/H_ном
    DM = DHN * np.concatenate([[0], np.cumsum(0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zz))])
    out = []
    for z, q in zip(BZ, BQ):
        dm = np.interp(z, zz, DM); dh = DHN / np.interp(z, zz, E)
        out.append({'DM_over_rs': dm, 'DH_over_rs': dh, 'DV_over_rs': (z * dm ** 2 * dh) ** (1 / 3)}[q] / RD)
    return np.array(out)
def chi_bao(L, Hn):
    d = BV - bao_model(L, Hn); return d @ BC @ d
def chi_sn(L, Hn):
    a = np.exp(ep.LNA_LATE); E = np.sqrt(OMN * a ** -3 + OMR * a ** -4 + L / 3)[None] / Hn
    return ep.chi2(ep.mu_from_E(ep.LNA_LATE, E))[0]

# эталон: плоская ΛCDM с тем же ω_m, H0 свободен (то есть Λ свободна) — лучший возможный
Ls = np.linspace(1.2, 3.0, 181); rows = []
for L in Ls:
    Hn = math.sqrt(OMN + OMR + L / 3); rows.append((L, chi_bao(L, Hn), chi_sn(L, Hn), Hn))
rows = np.array(rows)
ib, isn, it = rows[:, 1].argmin(), rows[:, 2].argmin(), (rows[:, 1] + rows[:, 2]).argmin()
f = lambda r: f'H0 = {HN*r[3]:.1f}, Ω_m = {OMN/r[3]**2:.3f}, Ω_Λ = {r[0]/3/r[3]**2:.3f}'
print(f'r_d = {RD:.1f} Мпк; 13 точек BAO')
print(f'ΛCDM (ω_m=0.142): лучшая по BAO   χ²_BAO = {rows[ib,1]:.1f}  при {f(rows[ib])}')
print(f'                  лучшая по SN    χ²_SN = {rows[isn,2]:.1f}  при {f(rows[isn])}  (её χ²_BAO = {rows[isn,1]:.1f})')
print(f'                  совместно       χ² = {rows[it,1]:.1f} + {rows[it,2]:.1f}  при {f(rows[it])}')
ctot0 = rows[it, 1] + rows[it, 2]; cb0 = rows[ib, 1]

if __name__ == '__main__':
    xs = np.linspace(0.005, 3, 600); wg = np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi) * (xs[1] - xs[0])
    for alpha in (0.02, 0.025, 0.03, 0.035, 0.04):
        gb = gt = 0.0; xin = []
        for xi, w in zip(xs, wg):
            L = solve(alpha, xi); V, Hn = past_cone(L)
            cb = chi_bao(L, Hn); ct = cb + chi_sn(L, Hn)
            if cb - cb0 < 4: gb += w
            if ct - ctot0 < 4: gt += w; xin.append(xi)
        print(f'α = {alpha:.3f}: миров с Δχ²_BAO<4 {gb*100:.1f} %; с Δχ²(BAO+SN)<4 {gt*100:.1f} %'
              + (f'  (ξ ∈ [{min(xin):.2f}, {max(xin):.2f}])' if xin else ''))
        sys.stdout.flush()
