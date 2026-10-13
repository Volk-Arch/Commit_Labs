#!/usr/bin/env python3
"""ОБЕ СТРОКИ В ОДНОЙ ПЕТЛЕ: конус (радар) + «кто существует» (тень в шаре).

Мир — золотой квазикристалл Λ в ромбе. Часы — НЕ внешние: 12 мировых линий
из точек самого Λ (неподвижное место x₀ ∈ Z[τ]³ с малой тенью, тики — все
t, разрешённые тенью; шаг ≈ 0.75ℓ, неравномерный, фибоначчиев).

Шаг петли:
  1. радар по ПОРЯДКУ до линий -> время и радиусы -> MDS, синхронизация, сферы;
  2. перевод в рамку Λ по самим линиям (Прокруст: линии знают свои метки);
  3. ВТОРАЯ СТРОКА: каждое событие садится на ближайшую свободную точку Λ;
  4. новый порядок — конус на занятых точках.
Старт: Λ, у которого события сдвинуты шумом σ·ℓ (порядок испорчен).
Вопрос: возвращается ли петля ТОЧНО к порядку Λ (каждое событие на своё
место)? Контроль: та же петля без шага 3 (только конус).
"""
import sys, math, time
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/claude'); sys.path.insert(0, '.')
from gold4 import coord_values, r_for_density, ST, SI, TAU, TP, points_in_box

V4 = 2 * math.pi / 3
NTARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
RHO = NTARGET / V4; R = r_for_density(RHO); ELL = RHO ** -0.25

def golden_near(target, maxb=40):
    """число a+bτ возле target с наименьшей тенью |a+bτ'|"""
    best = None
    for b in range(-maxb, maxb + 1):
        a = round(target - b * TAU)
        for aa in (a - 1, a, a + 1):
            c, cp = aa + b * TAU, aa + b * TP
            if abs(c - target) < 0.08:
                if best is None or abs(cp) < abs(best[1]): best = (c, cp)
    return best

def make_lines(nl):
    """линии в точках с координатами из {0, ±0.618} = {0, ±(−1+τ)}: тень мала,
    значит на линии много разрешённых тиков"""
    g = TAU - 1; gp = TP - 1                      # 0.618 и его тень −1.618
    pos = [(0, 0, 0), (g, 0, 0), (-g, 0, 0), (0, g, 0), (0, -g, 0), (0, 0, g), (0, 0, -g),
           (g, g, 0), (-g, 0, g), (0, -g, -g), (g, 0, -g), (-g, g, 0)][:nl]
    lines = []
    for x0 in pos:
        x0 = np.array(x0, float); sh2 = sum((gp if abs(v) > 0 else 0.0) ** 2 for v in x0)
        rt = math.sqrt(max(R * R - sh2, 0)) / SI
        t, tp = coord_values(-3.0 / ST, 3.0 / ST, rt)
        T = np.sort(t * ST)
        lines.append(np.column_stack([T, np.repeat(x0[None], len(T), 0)]))
    return lines

def mink(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    Rm = dt > dx; np.fill_diagonal(Rm, False); return Rm

def radar(Rm, lab, tick, lines):
    """радар в показаниях часов линий (линия знает свои тики): середины
    промежутков, где лежат испускание и приём — несмещённо"""
    nl = len(lines); n = len(Rm); T = np.full((n, nl), np.nan); Rr = np.full((n, nl), np.nan)
    for i in range(nl):
        idx = np.flatnonzero(lab == i); idx = idx[np.argsort(tick[idx])]; tt = lines[i][:, 0]
        before = Rm[:, idx]; after = Rm[idx, :].T
        hp, hf = before.any(1), after.any(1)
        p = np.where(hp, len(idx) - 1 - np.argmax(before[:, ::-1], 1), -1); f = np.where(hf, np.argmax(after, 1), -1)
        ok = hp & hf & (p + 1 < len(tt)) & (f >= 1)
        te = (tt[p[ok]] + tt[np.minimum(p[ok] + 1, len(tt) - 1)]) / 2
        tr = (tt[np.maximum(f[ok] - 1, 0)] + tt[f[ok]]) / 2
        T[ok, i] = (te + tr) / 2; Rr[ok, i] = np.maximum((tr - te) / 2, 0)
        own = lab == i; T[own, i] = tt[tick[own]]; Rr[own, i] = 0
    return T, Rr

def reconstruct(Rm, lab, tick, lines):
    nl = len(lines)
    T, Rr = radar(Rm, lab, tick, lines)
    D = np.zeros((nl, nl)); off = np.zeros(nl)
    for j in range(nl):
        own = lab == j
        for i in range(nl):
            if i != j: D[i, j] = np.nanmedian(Rr[own, i])
        if j: off[j] = np.nanmedian(T[own, 0] - T[own, j])
    D = (D + D.T) / 2
    J = np.eye(nl) - 1 / nl; B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); A = V[:, ::-1][:, :3] * np.sqrt(np.maximum(w[::-1][:3], 1e-12))
    Ts = T + off[None]; ok = np.isfinite(Rr).sum(1) >= 6
    Tm = np.nanmean(Ts, 1); X = np.zeros((len(Rm), 3))
    for it in range(15):
        d = X[:, None, :] - A[None]; dist = np.linalg.norm(d, axis=-1) + 1e-9
        res = np.where(np.isfinite(Rr), dist - Rr, 0.0); Jm = np.where(np.isfinite(Rr)[..., None], d / dist[..., None], 0.0)
        H = np.einsum('nij,nik->njk', Jm, Jm) + 1e-6 * np.eye(3); g = np.einsum('nij,ni->nj', Jm, res)
        X -= np.linalg.solve(H, g[..., None])[..., 0]
    # перевод в рамку Λ: Прокруст по линиям (масштаб, поворот, сдвиг), время — по часам линии 0
    Ltrue = np.array([L[0, 1:] for L in lines])
    mu_a, mu_l = A.mean(0), Ltrue.mean(0)
    U, S, Vt = np.linalg.svd((A - mu_a).T @ (Ltrue - mu_l))
    Rot = U @ Vt; s = S.sum() / ((A - mu_a) ** 2).sum()
    Xp = s * (X - mu_a) @ Rot + mu_l
    Tp = Tm
    return np.column_stack([Tp, Xp]), ok

def run(sigma, snap, rs, steps=6):
    lines = make_lines(12)
    Lam = points_in_box([(-1.05, 1.05)] * 4, R)
    Lam = Lam[np.abs(Lam[:, 0]) + np.linalg.norm(Lam[:, 1:], axis=1) <= 1]
    # события = Λ в ромбе без точек линий
    linepts = np.vstack(lines)
    tree_l = cKDTree(linepts); dd, _ = tree_l.query(Lam); Lam = Lam[dd > 1e-9]
    N = len(Lam)
    P_true = np.vstack([Lam] + lines)
    lab = np.concatenate([np.full(N, -1)] + [np.full(len(L), i) for i, L in enumerate(lines)])
    tick = np.concatenate([np.full(N, -1)] + [np.arange(len(L)) for L in lines])
    R_true = mink(P_true)
    P = P_true.copy(); P[:N] += rs.normal(0, sigma * ELL, (N, 4))
    Rm = mink(P)
    hist = [(Rm ^ R_true)[:N, :N].sum() / R_true[:N, :N].sum()]
    tree = cKDTree(Lam)
    home = np.zeros(N)
    for st in range(steps):
        Q, ok = reconstruct(Rm, lab, tick, lines)
        E = Q[:N].copy(); E[~ok[:N]] = P[:N][~ok[:N]]
        if snap:
            k = min(8, N); dist, idx = tree.query(E, k=k)
            taken = np.zeros(N, bool); newE = np.empty_like(E); order = np.argsort(dist[:, 0])
            for e in order:
                for c in range(k):
                    j = idx[e, c]
                    if not taken[j]: taken[j] = True; newE[e] = Lam[j]; break
                else:
                    newE[e] = E[e]
            E = newE
        P = np.vstack([E] + lines); Rm = mink(P)
        hist.append((Rm ^ R_true)[:N, :N].sum() / R_true[:N, :N].sum())
        home = np.linalg.norm(E - Lam, axis=1) < 1e-9
    return N, hist, home.mean()

if __name__ == '__main__':
    print(f'Λ: ρ={RHO:.0f}, r={R:.2f}, ℓ={ELL:.3f};  12 линий из точек самого Λ')
    t0 = time.time()
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.5):
        for snap in (False, True):
            outs = [run(sigma, snap, np.random.default_rng(100 + s)) for s in range(2)]
            N = outs[0][0]; H = np.mean([o[1] for o in outs], 0); home = np.mean([o[2] for o in outs])
            lab = 'конус + тень' if snap else 'только конус'
            print(f'σ={sigma:.1f}ℓ  {lab:13s} N={N}  ошибка порядка по шагам: ' + ' → '.join(f'{h:.3f}' for h in H)
                  + (f'   на своём месте: {home:.3f}' if snap else '') + f'   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
