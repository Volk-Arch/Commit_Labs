"""Петля «конус + тень», чинящая ошибки, на кристалле с размытым краем тени.
Мир — одна реализация (монетки брошены один раз). Часы — линии из точек
кристалла (без прореживания). Вопрос: чинит ли петля сдвиги так же, как при
резком крае (fixgold-doc: 0.1ℓ → 2 %, 93 % дома)?"""
import sys, math, time, numpy as np
sys.argv = [sys.argv[0], '1200']
sys.path.insert(0, '.')
import fixgold as F
from fuzzy import points_shadow, prob, r_for
from scipy.spatial import cKDTree

def run(sigma, w, rs, world_seed, steps=5):
    lines = F.make_lines(12)
    Rw = F.R * r_for(w) / r_for(0.0)
    cand, s = points_shadow([(-1.05, 1.05)] * 4, Rw * (1 + w))
    wr = np.random.default_rng(world_seed)
    Lam = cand[wr.random(len(cand)) < prob(s, Rw, w)]
    Lam = Lam[np.abs(Lam[:, 0]) + np.linalg.norm(Lam[:, 1:], axis=1) <= 1]
    dd, _ = cKDTree(np.vstack(lines)).query(Lam); Lam = Lam[dd > 1e-9]
    N = len(Lam)
    lab = np.concatenate([np.full(N, -1)] + [np.full(len(L), i) for i, L in enumerate(lines)])
    tick = np.concatenate([np.full(N, -1)] + [np.arange(len(L)) for L in lines])
    R_true = F.mink(np.vstack([Lam] + lines))
    P = np.vstack([Lam] + lines).copy(); P[:N] += rs.normal(0, sigma * F.ELL, (N, 4)); Rm = F.mink(P)
    hist = [(Rm ^ R_true)[:N, :N].sum() / R_true[:N, :N].sum()]
    tree = cKDTree(Lam)
    for st in range(steps):
        Q, ok = F.reconstruct(Rm, lab, tick, lines)
        E = Q[:N].copy(); E[~ok[:N]] = P[:N][~ok[:N]]
        dist, idx = tree.query(E, k=8); taken = np.zeros(N, bool); newE = E.copy()
        for e in np.argsort(dist[:, 0]):
            for c in range(8):
                j = idx[e, c]
                if not taken[j]: taken[j] = True; newE[e] = Lam[j]; break
        E = newE; P = np.vstack([E] + lines); Rm = F.mink(P)
        hist.append((Rm ^ R_true)[:N, :N].sum() / R_true[:N, :N].sum())
    home = (np.linalg.norm(E - Lam, axis=1) < 1e-9).mean()
    return N, hist, home

t0 = time.time()
for w in (0.0, 0.1, 0.3):
    for sigma in (0.0, 0.1, 0.2):
        outs = [run(sigma, w, np.random.default_rng(100 + s), 500 + s) for s in range(2)]
        H = np.mean([o[1] for o in outs], 0); home = np.mean([o[2] for o in outs])
        print(f'w={w}  σ={sigma}ℓ  N={outs[0][0]}  ошибка: ' + ' → '.join(f'{h:.3f}' for h in H) + f'   дома: {home:.3f}   [{time.time()-t0:.0f} с]')
        sys.stdout.flush()
