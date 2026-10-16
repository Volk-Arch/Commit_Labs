"""ЭТАПЫ ВСЕЛЕННОЙ ИЗ ПОРЯДКА: уравнение состояния w по одному отношению «раньше» и счёту (`weos-doc`, `exact-doc`).

Мир — посыпка в конформно-плоской вселенной ds² = a(η)²(dη² − dx²), 3+1. При постоянном w масштабный фактор задан одним
числом: a = (1 + hΔη/p)^p, p = 2/(1 + 3w) (h = a′/a в середине; при w = −1/3 — предел e^{hΔη}). Этапы:
  w = +1/3 — излучение (a ∝ η);  w = +1 — жёсткая материя (a ∝ √η);  w = −1/3 — «кривизна» (a = e^{hη});
  w = −1 — тёмная энергия, де Ситтер (a ∝ 1/(η∞ − η)).
Из порядка мерятся только счёты в причинном ромбе: C₁ (событий), X₂ = C₂/(c₂C₁²) − 1, X₃ = C₃/(c₃C₁³) − 1 (связанные пары и
тройки) — и сравниваются с ТОЧНЫМ откликом (квадратура по ромбу с весом a⁴, без разложения). Подгоняется одно число w.
Данные: наборы 5–9 окт. (`data/eos_arms.npz`: излучение 364 мира, жёсткое 60, экспонента 420, плоское 104) и новый
набор де Ситтера (`harvest`, 16 окт.)."""
import math
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), 'data')
KFAC = math.sqrt(24.0 / math.pi)
CFG = (26.0, 20.0, 10.0, 1.6, 200, 900)          # ящик L, T; η₀; полоса |mid − η₀|; C₁ от, до
VB = (0.0, 0.15, 0.30, 0.45, 0.60, 0.72, 0.85)   # полосы набора по v²
VBB = (0.0, 0.45, 0.65, 0.85001); CBB = (200.0, 400.0, 650.0, 900.0001)
IC, IX2, IX3, IV, IM = 0, 1, 2, 3, 4
TRUE_W = {'lin': 1 / 3, 'root': 1.0, 'exp': -1 / 3, 'ds': -1.0, 'flat': None}
NAMES = {'lin': 'излучение (w = 1/3)', 'root': 'жёсткая материя (w = 1)', 'exp': 'экспонента (w = −1/3)', 'ds': 'де Ситтер (w = −1)', 'flat': 'плоское (контроль)'}

def c_k(k, n=4):
    """доля k-цепей в ромбе плоского n-мерия: C_k = c_k·C₁^k."""
    g = math.gamma
    return g(n + 1) ** (k - 1) * g(n / 2) * g(n) / (2 ** (k - 1) * k * g(k * n / 2) * g((k + 1) * n / 2))

def a_ps(p, s_, eta0=10.0):
    """степенной закон с известным началом отсчёта: a = (1 + Δη/s)^p, s = η₀ − η_нач (s < 0 — сингулярность в будущем,
    как у де Ситтера); темп h = p/s. Так был измерен w в weos-doc (известно «время с начала»)."""
    return lambda e: np.maximum(1.0 + (np.asarray(e, float) - eta0) / s_, 1e-12) ** p

def w_of_p(p): return (2.0 / p - 1.0) / 3.0

def a_w(w, h, eta0=10.0):
    """масштабный фактор при постоянном w с темпом h в η₀ (a(η₀) = 1)."""
    if h == 0: return lambda e: np.ones_like(np.asarray(e, float))
    if abs(w + 1 / 3) < 1e-9: return lambda e: np.exp(h * (np.asarray(e, float) - eta0))
    p = 2.0 / (1.0 + 3.0 * w)
    return lambda e: np.maximum(1.0 + h * (np.asarray(e, float) - eta0) / p, 1e-12) ** p

# ---------- набор (посыпка и счёты в ромбах) ----------
def sprinkle(afun, L, T, rs):
    e = np.linspace(0.0, T, 40000); wgt = afun(e) ** 4
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (wgt[1:] + wgt[:-1]) * np.diff(e))]); n = rs.poisson(L ** 3 * cdf[-1])
    X = np.empty((n, 4)); X[:, 0] = np.interp(rs.uniform(0, cdf[-1], n), cdf, e); X[:, 1:] = rs.uniform(0, L, (n, 3))
    return X[np.argsort(X[:, 0])]

def fits(p, f, L, T):
    de = f[0] - p[0]; mid = 0.5 * (p[1:] + f[1:])
    return bool(p[0] >= 0 and f[0] <= T and (mid - de / 2 >= 0).all() and (mid + de / 2 <= L).all())

def counts(I):
    k = len(I)
    if k < 3: return k, 0, 0
    dt = I[:, 0][None, :] - I[:, 0][:, None]; dx = I[:, 1:][None, :, :] - I[:, 1:][:, None, :]
    M = (dt > 0) & (dt ** 2 > (dx * dx).sum(2)); past = M.sum(0); fut = M.sum(1)
    return k, int(M.sum()), int((past * fut).sum())

def harvest(afun, rs, want, cfg=CFG):
    """интервалы одного мира с равномерным набором по наклону v²: строки (C₁, X₂, X₃, v², середина)."""
    L, T, eta0, half, lo, hi = cfg; X = sprinkle(afun, L, T, rs); t = np.ascontiguousarray(X[:, 0]); S = np.ascontiguousarray(X[:, 1:])
    tlo, thi = (24.0 * lo / math.pi) ** 0.25, (24.0 * hi / math.pi) ** 0.25; nb = len(VB) - 1; quota = want // nb + 1; got = [0] * nb; out = []; guard = 0
    C2C, C3C = c_k(2), c_k(3)
    while len(out) < want and guard < want * 25:
        guard += 1; ip = int(rs.integers(np.searchsorted(t, eta0 - half - thi), np.searchsorted(t, eta0 + half))); p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo); j1 = np.searchsorted(t, p0 + thi * 2.2)
        if j1 - j0 < 40: continue
        de = t[j0:j1] - p0; d = S[j0:j1] - ps; dx2 = np.einsum('ij,ij->i', d, d); tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0: continue
        v2 = dx2[ok] / (de[ok] ** 2); bi = np.digitize(v2, VB) - 1
        for q in np.argsort([got[b] if 0 <= b < nb else 10 ** 9 for b in bi])[:8]:
            if len(out) >= want: break
            b = bi[q]
            if not (0 <= b < nb) or got[b] >= quota: continue
            jj = int(ok[q]); f0, fs = t[j0 + jj], S[j0 + jj]; mid = 0.5 * (p0 + f0)
            if abs(mid - eta0) > half or not fits(np.r_[p0, ps], np.r_[f0, fs], L, T): continue
            k0 = np.searchsorted(t, p0, 'right'); k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo: continue
            Sl, tl = S[k0:k1], t[k0:k1]; u = Sl - ps; wv = fs - Sl
            m = ((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u)) & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', wv, wv)); a = int(m.sum())
            if not (lo <= a <= hi): continue
            _, bb, cc = counts(np.column_stack([tl[m], Sl[m]]))
            out.append((a, bb / (C2C * a * (a - 1.0)) - 1.0, cc / (C3C * a * (a - 1.0) * (a - 2.0)) - 1.0, float(v2[q]), mid)); got[b] += 1
    return out

# ---------- точный отклик (квадратура по ромбу, без разложения) ----------
def _mk(a, b): return a[:, 0] * b[:, 0] - np.einsum('ij,ij->i', a[:, 1:], b[:, 1:])

def _ball(rs, n, dim, R):
    g = rs.normal(size=(n, dim)); g /= np.linalg.norm(g, axis=1)[:, None]; return g * (R * rs.random(n) ** (1.0 / dim))[:, None]

def _rest_diamond(rs, n, T, d=4):
    z = rs.random(n); half = z < 0.5; y = np.where(half, 2 * z, 2 * (1 - z)) ** (1.0 / d); s = np.where(half, 0.5 * y, 1.0 - 0.5 * y) * T
    return s, _ball(rs, n, d - 1, np.minimum(s, T - s))

def _sub_sample(a, q, rs, d=4):
    dd = q - a; t2 = _mk(dd, dd); ok = t2 > 1e-12; a, dd, t2 = a[ok], dd[ok], t2[ok]; tau = np.sqrt(t2); m = len(a); e0 = dd / tau[:, None]
    B = np.zeros((m, d - 1, d)); seed = np.zeros((m, d - 1, d))
    for i in range(d - 1): seed[:, i, i + 1] = 1.0
    for i in range(d - 1):
        v = seed[:, i, :].copy(); v -= _mk(v, e0)[:, None] * e0
        v[~np.isfinite(v)] = 0
        for j in range(i): v -= (-_mk(v, B[:, j, :]))[:, None] * B[:, j, :]
        v /= np.sqrt(np.maximum(-_mk(v, v), 1e-300))[:, None]; B[:, i, :] = v
    s, wv = _rest_diamond(rs, m, 1.0, d); p = a + (s * tau)[:, None] * e0
    for i in range(d - 1): p = p + (wv[:, i] * tau)[:, None] * B[:, i, :]
    good = np.isfinite(p).all(1)                     # почти световые направления: базис вырождается — такие точки отбрасываются
    okf = ok.copy(); okf[np.nonzero(ok)[0][~good]] = False
    return p[good], tau[good], okf

def exact_x(afun, v, tau, mid, n, rs, d=4):
    """точные ⟨X₂⟩, ⟨X₃⟩ для ромба собственного времени τ, наклона v, середины mid при масштабном факторе afun."""
    gam = 1.0 / math.sqrt(1.0 - v * v); amid = float(afun(np.array([mid]))[0])
    wgt = lambda pt: (afun(mid + ((pt[:, 0] - 0.5 * tau) + v * pt[:, 1]) * gam) / amid) ** 4
    s0, w0 = _rest_diamond(rs, n, tau, d); x = np.column_stack([s0, w0]); q = np.zeros((n, d)); q[:, 0] = tau; wx = wgt(x)
    y, txq, ok1 = _sub_sample(x, q, rs, d); wy = wgt(y)
    x2 = (wx[ok1] * txq ** d * wy).mean() / (c_k(2) * tau ** d * wx.mean() ** 2) - 1.0
    z, tyq, ok2 = _sub_sample(y, q[ok1], rs, d); wz = wgt(z)
    x3 = (wx[ok1][ok2] * txq[ok2] ** d * wy[ok2] * tyq ** d * wz).mean() / (c_k(3) * tau ** (2 * d) * wx.mean() ** 3) - 1.0
    return x2, x3

# ---------- подгонка w ----------
def load_arms():
    z = np.load(os.path.join(DATA, 'eos_arms.npz')); arms = {}
    for fam in ('lin', 'root', 'exp', 'flat', 'ds'):
        if f'{fam}_A' in z: arms[fam] = (z[f'{fam}_A'].astype(float), z[f'{fam}_sd'], float(z[f'{fam}_h']))
    return arms

def cells(A, sd, minrows=150):
    out = []
    for i in range(len(VBB) - 1):
        for j in range(len(CBB) - 1):
            m = (A[:, IV] >= VBB[i]) & (A[:, IV] < VBB[i + 1]) & (A[:, IC] >= CBB[j]) & (A[:, IC] < CBB[j + 1])
            if m.sum() < minrows: continue
            us = np.unique(sd[m]); vals, errs = [], []
            for col in (IX2, IX3):
                jk = np.array([A[m & (sd != u), col].mean() for u in us]); errs.append(math.sqrt((len(us) - 1) / len(us) * ((jk - jk.mean()) ** 2).sum())); vals.append(A[m, col].mean())
            out.append({'v2': A[m, IV].mean(), 'c1': A[m, IC].mean(), 'mid': A[m, IM].mean(), 'obs': np.array(vals), 'err': np.array(errs), 'msk': m})
    return out

def fit_w(A, sd, family, wgrid, n=300000, blocks=2, seed=314159, log=None):
    """χ² по сетке параметра (ошибка опоры включена), параболическое уточнение, ошибка — складной нож по мирам.
    family(параметр) → a(η)."""
    wgrid = np.asarray(wgrid, float); rs = np.random.default_rng(seed); cl = cells(A, sd); obs = np.array([c['obs'] for c in cl]); err = np.array([c['err'] for c in cl])
    R = np.zeros((len(wgrid), len(cl), 2)); E = np.zeros_like(R)
    for a_, w in enumerate(wgrid):
        af = family(w)
        for b_, c in enumerate(cl):
            tau = math.sqrt(KFAC * math.sqrt(c['c1'])); rr = np.array([exact_x(af, math.sqrt(c['v2']), tau, c['mid'], n, np.random.default_rng(seed + 1000 * b_ + k_)) for k_ in range(blocks)])   # общие случайные числа по сетке — гладкая кривая χ²
            R[a_, b_] = rr.mean(0); E[a_, b_] = rr.std(0, ddof=1) / math.sqrt(blocks)
        if log: log(f'  опора w = {w:+.3f} готова')
    def best(o):
        v = np.array([(((R[a_] - o) ** 2) / (err ** 2 + E[a_] ** 2)).sum() for a_ in range(len(wgrid))]); v = np.where(np.isfinite(v), v, np.inf); k = int(np.argmin(v))
        if 0 < k < len(wgrid) - 1:
            c2, c1, _ = np.polyfit(wgrid[k - 1:k + 2], v[k - 1:k + 2], 2)
            ww = float(np.clip(-c1 / (2 * c2), wgrid[k - 1], wgrid[k + 1])) if c2 > 0 else wgrid[k]
        else: ww = wgrid[k]
        return ww, v[k], v
    w0, chi, curve = best(obs); us = np.unique(sd); jk = []
    for u in us:
        jk.append(best(np.array([[A[c['msk'] & (sd != u), IX2].mean(), A[c['msk'] & (sd != u), IX3].mean()] for c in cl]))[0])
    jk = np.array(jk); se = math.sqrt((len(us) - 1) / len(us) * ((jk - jk.mean()) ** 2).sum())
    return {'par': float(w0), 'w': float(w0), 'err': float(se), 'chi2': float(chi), 'dof': int(obs.size - 1), 'curve': curve.tolist(), 'grid': list(map(float, wgrid)), 'cells': len(cl), 'worlds': len(us)}

S_ORIGIN = 1.0 / 0.035                              # конформное время от начала расширения (наборы 5–9 окт.)
PLAN = {   # этап: (семейство, сетка параметра, параметр → w, истина параметра)
    'lin': (lambda p: a_ps(p, S_ORIGIN), [0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.20, 1.45], w_of_p, 1.0),
    'root': (lambda p: a_ps(p, S_ORIGIN), [0.2, 0.27, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00], w_of_p, 0.5),
    'flat': (lambda p: a_ps(p, S_ORIGIN), [0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.20, 1.45], w_of_p, None),
    'ds': (lambda p: a_ps(p, -S_ORIGIN), [-1.6, -1.4, -1.2, -1.0, -0.85, -0.7, -0.55, -0.4], w_of_p, -1.0),
    'exp': (lambda w: a_w(w, 0.03), [-0.9, -0.7, -0.55, -0.45, -1 / 3, -0.2, -0.1, 0.0, 0.2], lambda w: w, -1 / 3),
}

def measure(fam, arms, n=300000, log=None):
    """w этапа из порядка: подгонка параметра своего семейства, перевод в w (ошибка — переносом)."""
    A, sd, _ = arms[fam]; family, grid, conv, ptrue = PLAN[fam]; r = fit_w(A, sd, family, grid, n=n, log=log)
    p0, e = r['par'], r['err']; r['w'] = float(conv(p0)); r['werr'] = float(abs(conv(p0 + e) - conv(p0 - e)) / 2) if e > 0 else 0.0
    r['wtrue'] = None if ptrue is None else float(conv(ptrue)); return r
