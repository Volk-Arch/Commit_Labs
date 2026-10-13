"""Λ из правила подстановки вместо монетки: ΛV = 8πα·D(V), D(V) = √V·g(ln V + φ),
g — профиль отклонения счёта у подстановки с |λ2| = √λ1 (самоподобен: g(x+ln λ1) = sign(λ2)·g(x)).
Сравнение со сверхновыми, те же условия, что в cosmo-doc: H0 = 60…80, 0.2 < Ω_m < 0.45."""
import sys; sys.argv = sys.argv[:1]
import numpy as np, math
import everpresent as ep
from substit import build
np.seterr(all='ignore')
TAU = ep.TAU
RULES = {'золотая a→acd b→cd c→abc d→ab (τ², −τ)': ([0, 2, 3], [2, 3], [0, 1, 2], [0, 1]),
         'a→aaab b→abbb (4, +2)': ([0, 0, 0, 1], [0, 1, 1, 1])}

def profile(rule, N=6_000_000):
    s = build(rule, N); k = len(rule)
    M = np.array([[rule[j].count(i) for j in range(k)] for i in range(k)])
    w, V = np.linalg.eig(M); o = np.argsort(-abs(w)); l1, l2 = w[o[0]].real, w[o[1]].real
    f = np.mean(s == 0); D = np.cumsum(s == 0) - f * np.arange(1, N + 1)
    P = math.log(abs(l1)); x = np.linspace(math.log(N) - 2 * P - 0.05, math.log(N) - 0.05, 2000)
    L = np.exp(x).astype(int); g = D[L - 1] / np.sqrt(L)
    sg = np.sign(l2)
    gper = 0.5 * (g[:1000] + sg * g[1000:])          # один период, усреднённый по двум
    chk = np.corrcoef(g[:1000], sg * g[1000:])[0, 1]
    return x[:1000] - x[0], gper / np.std(gper), P, sg, chk

def run(xg, gg, P, sg, alpha, om, phases, nstep=4000, a0=1e-5):
    K = len(phases); lna = np.linspace(math.log(a0), 0, nstep); dl = lna[1] - lna[0]
    a = math.exp(lna[0]); H = math.sqrt(ep.OM_R) / a ** 2; t = 1 / (2 * H)
    A = a / math.sqrt(t)
    S = [np.full(K, A ** (3 - m) * 2 ** m * t ** ((5 + m) / 2) / ((5 + m) / 2)) for m in range(4)]
    eta = np.full(K, 2 * t / a); V = np.full(K, (32 * math.pi / 3) * t ** 4 / 140)
    def g(lv):
        y = lv + phases; k = np.floor(y / P); r = y - k * P
        return np.interp(r, xg, gg) * sg ** (k % 2)
    Lam = 8 * math.pi * alpha * g(np.log(V)) / np.sqrt(V)
    Hs = np.zeros((K, nstep)); alive = np.ones(K, bool); irec = int(np.argmin(abs(lna - math.log(1 / 1101)))); OLrec = 0
    for i, x in enumerate(lna):
        a = math.exp(x)
        H2 = om * a ** -3 + ep.OM_R * a ** -4 + Lam / 3
        alive &= H2 > 0; Hc = np.sqrt(np.maximum(H2, 1e-300)); Hs[:, i] = Hc
        if i == irec: OLrec = Lam / 3 / np.maximum(H2, 1e-300)
        dt = dl / Hc; eta = eta + dt / a; w = a ** 3 * dt
        S[0] += w; S[1] += w * eta; S[2] += w * eta ** 2; S[3] += w * eta ** 3
        V = np.maximum((4 * math.pi / 3) * (eta ** 3 * S[0] - 3 * eta ** 2 * S[1] + 3 * eta * S[2] - S[3]), V)
        Lam = 8 * math.pi * alpha * g(np.log(V)) / np.sqrt(V)
    E = Hs / Hs[:, -1:]; keep = lna >= ep.LNA_LATE[0] - 0.01
    return lna[keep], E[:, keep], alive, Hs[:, -1], OLrec

c0 = min(ep.lcdm_chi2(o) for o in np.linspace(0.2, 0.45, 26))
for name, rule in RULES.items():
    xg, gg, P, sg, chk = profile(rule)
    phases = np.linspace(0, 2 * P, 200, endpoint=False)
    best = np.full(len(phases), np.inf); rec = np.zeros(len(phases))
    for alpha in np.arange(0.002, 0.0241, 0.002):
        for om in (0.24, 0.28, 0.32, 0.36, 0.40):
            lna, E, alive, h0r, OL = run(xg, gg, P, sg, alpha, om, phases)
            c = np.full(len(phases), np.inf)
            if alive.any(): c[alive] = ep.chi2(ep.mu_from_E(lna, E[alive]))
            h0 = 67.7 * h0r; ome = om / h0r ** 2
            c = np.where((h0 > 60) & (h0 < 80) & (ome > 0.2) & (ome < 0.45), c, np.inf)
            u = c < best; best[u] = c[u]; rec[u] = OL[u]
    D = best - c0; ok = np.isfinite(D)
    print(f'{name}: период по ln V {P:.2f}, самоподобие профиля r = {chk:.3f}; фаз {len(phases)}, допустимых {ok.sum()}; '
          f'лучше ΛCDM {np.sum(D < 0)}; Δχ²<4 {np.sum(D < 4)}; лучший Δχ² {D.min():+.1f}'
          + (f'; Ω_Λ на рекомбинации у лучшего {rec[D.argmin()]:+.2f}' if ok.any() else ''))
    sys.stdout.flush()
