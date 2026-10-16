"""НАСКОЛЬКО ТОЧНЫМ ДОЛЖНО БЫТЬ ЗОЛОТОЕ СЕЧЕНИЕ (вопрос пользователя). То же правило «событие есть ⟺ тень в шаре», но вместо τ —
(а) дроби Фибоначчи 3/2, 5/3, 8/5, 13/8 (кристалл становится периодическим), (б) другие «единицы» θ с сопряжённым −1/θ:
серебряное 1 + √2, бронзовое (3 + √13)/2. Плотность везде 4 (подбирается по счёту). Сравниваем: размерность, буст (переходит ли
мир сам в себя при η = arcch θ), частоту покоя m = 1 (масса кристалла), скорость нитей, синглет Белла, укрупнение в θ."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from goldlab import worlds as Wd, experiments as E, fermions as Fe, causal as Cs
def numbers(lo, hi, rlo, rhi, th, thc):
    D = th - thc; C, Cp = [], []
    for b in range(math.floor((lo - rhi) / D) - 2, math.ceil((hi - rlo) / D) + 3):
        alo = max(math.ceil(lo - b * th), math.ceil(rlo - b * thc)); ahi = min(math.floor(hi - b * th), math.floor(rhi - b * thc))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1, dtype=float); C.append(a + b * th); Cp.append(a + b * thc)
    return (np.concatenate(C), np.concatenate(Cp)) if C else (np.empty(0), np.empty(0))
def world(th, thc, rho, L, r=None):
    ST, SI = math.sqrt(abs(th)), math.sqrt(abs(thc)); D = (th - thc) ** 2; r = r or math.sqrt(D * rho / math.pi)
    for _ in range(3):
        c, cp = numbers(-L / ST, L / ST, -r / SI, r / SI, th, thc); T, Tq = c * ST, cp * SI; d, dp = numbers(-L, L, -r, r, th, thc)
        i, j = np.nonzero(Tq[:, None] ** 2 + dp[None, :] ** 2 <= r * r); X = np.stack([T[i], d[j]], 1); sh = Tq[i] ** 2 + dp[j] ** 2
        key = np.round(X, 7); _, u = np.unique(key, axis=0, return_index=True); X, sh = X[u], sh[u]          # дроби: одинаковые точки — одно событие
        keep = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L; X, sh = X[keep], sh[keep]
        dens = len(X) / (2 * L * L); r *= math.sqrt(rho / dens)
    o = np.argsort(X[:, 0], kind='stable'); return Wd.World(X[o], 'custom', rho, {'L': L}), sh[o], r
def boost_frac(w, th):
    if th < 1.0001: return float('nan')
    eta = math.acosh(th); ch, sh_ = math.cosh(eta), math.sinh(eta); X = w.X
    inner = np.abs(X[:, 0]) + np.abs(X[:, 1]) < 0.35 * w.meta['L']; P = X[inner]
    Q = np.c_[P[:, 0] * ch - P[:, 1] * sh_, -P[:, 0] * sh_ + P[:, 1] * ch]; S = {tuple(p) for p in np.round(X, 6)}
    return float(np.mean([tuple(q) in S for q in np.round(Q, 6)]))
cases = [('τ (золото)', (1 + 5 ** .5) / 2, -2 / (1 + 5 ** .5))] + [(f'{p}/{q}', p / q, -q / p) for p, q in ((3, 2), (5, 3), (8, 5), (13, 8))] + \
        [('1 + √2 (серебро)', 1 + 2 ** .5, 1 - 2 ** .5), ('(3 + √13)/2 (бронза)', (3 + 13 ** .5) / 2, (3 - 13 ** .5) / 2)]
orig_g, orig_m = Wd.gold11, Wd.make
print('число | событий (L=12) | размерность | буст η = arcch θ | частота покоя m=1 (L=24) | скорость нитей | синглет | укрупнение в θ: событий мелк/крупн (θ²)')
for name, th, thc in cases:
    w12, sh12, r12 = world(th, thc, 4.0, 12.0)
    Wd.gold11 = lambda rho=4.0, L=12.0, th=th, thc=thc: world(th, thc, rho, L)[0]; Wd.make = lambda kind='gold11', **kw: Wd.gold11(kw.get('rho', 4.0), kw.get('L', 12.0))
    try:
        dim = E.run('world', {'L': 12})['numbers'][3]['value']
        disp = E.run('dispersion', {'L': 24})['numbers'][0]['value']
        bell = E.run('bell', {'L': 10})['numbers'][0]['value']
    finally:
        Wd.gold11, Wd.make = orig_g, orig_m
    c = Fe.Walk(world(th, thc, 4.0, 16.0)[0]).c
    coarse = sh12 <= (r12 / abs(th)) ** 2 if th > 1.0001 else np.zeros(len(sh12), bool); ratio = len(sh12) / max(coarse.sum(), 1)
    print(f'{name:22s} | {w12.n:5d} | {dim:.3f} | {boost_frac(w12, th):.2f} | {disp:.3f} | {c:.3f} | {bell:.4f} | {ratio:.3f} ({th * th:.3f})', flush=True)
