"""3+1: «нет почти светоподобных пар». Теорема для мира-тени: если окно тени — шар
формы −Q'(тени) ≤ r² (а это ровно наше окно: 0.618t'² + |x'|² = −(τ't'² − |x'|²)),
то интервал Q пары — целое алгебраическое число с |Q·Q'| ≥ 1, значит
|ΔT² − Δx²| ≥ 1/(4r²) для ЛЮБОЙ пары. Нормировка min|Q|·√ρ не зависит от масштаба.
Гарантия ∝ 1/(D·√|N(θ)|): D — дискриминант поля, θ — коэффициент при t².
Сравниваем поля √5 (наше), √2, √3, √13 и контроли: посыпка Пуассона, Z⁴, «иррациональная» решётка."""
import numpy as np, math, sys
from scipy.spatial import cKDTree
FIELDS = {   # имя: (ω, ω', D, θ = p + q ω)
    'золото √5, θ=τ':      ((1 + 5 ** .5) / 2, (1 - 5 ** .5) / 2, 5, (0, 1)),
    'серебро √2, θ=1+√2':  (2 ** .5, -2 ** .5, 8, (1, 1)),
    '√3, θ=1+√3':          (3 ** .5, -3 ** .5, 12, (1, 1)),
    '√13, θ=(1+√13)/2':    ((1 + 13 ** .5) / 2, (1 - 13 ** .5) / 2, 13, (0, 1)),
}
def coord_values(lo, hi, rint, w, wp):
    D = w - wp
    C, Cp = [], []
    for b in range(math.floor((lo - rint) / D) - 2, math.ceil((hi + rint) / D) + 3):
        alo = max(math.ceil(lo - b * w), math.ceil(-rint - b * wp)); ahi = min(math.floor(hi - b * w), math.floor(rint - b * wp))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1); C.append(a + b * w); Cp.append(a + b * wp)
    return (np.concatenate(C), np.concatenate(Cp)) if C else (np.empty(0), np.empty(0))
def model_set(name, L, rho):
    w, wp, D, (p, q) = FIELDS[name]; th, thp = p + q * w, p + q * wp
    assert th > 0 and thp < 0
    r = (rho * D ** 2 * math.sqrt(th * -thp) / (math.pi ** 2 / 2)) ** 0.25
    st, si = math.sqrt(th), math.sqrt(-thp)
    t, tp = coord_values(-L / st, L / st, r / si, w, wp)
    vals = [(t * st, (tp * si) ** 2)] + [(lambda c: (c[0], c[1] ** 2))(coord_values(-L, L, r, w, wp)) for _ in range(3)]
    P = np.zeros((1, 4)); Wt = np.zeros(1)
    for k, (c, w2) in enumerate(vals):
        P = np.repeat(P, len(c), 0); Wt = np.repeat(Wt, len(c))
        P[:, k] = np.tile(c, len(P) // len(c)); Wt += np.tile(w2, len(Wt) // len(c))
        keep = Wt <= r * r; P, Wt = P[keep], Wt[keep]
    return P, r, D * math.sqrt(th * -thp)
def min_interval(P):
    """минимум |ΔT² − |Δx|²| по всем парам (кусками)."""
    best = np.inf
    for i in range(0, len(P), 400):
        d = P[i:i + 400, None, :] - P[None, :, :]
        q = np.abs(d[..., 0] ** 2 - (d[..., 1:] ** 2).sum(-1))
        q[np.all(d == 0, axis=-1)] = np.inf
        best = min(best, q.min())
    return best
rng = np.random.default_rng(0)
rho = 1.0
print('min|интервал|·√ρ по всем парам в ящике [−L, L]⁴, ρ = 1  (больше — надёжнее порядок)')
print(f'{"мир":26s} ' + '  '.join(f'L={L:<5}' for L in (2.5, 3.2, 4.0)) + '   гарантия теоремы')
for name in FIELDS:
    row = []
    for L in (2.5, 3.2, 4.0):
        P, r, k = model_set(name, L, rho); row.append((min_interval(P) * math.sqrt(rho), len(P)))
    g = math.sqrt(math.pi ** 2 / 2) / (4 * k)
    print(f'{name:26s} ' + '  '.join(f'{m:.4f}({n})' for m, n in row) + f'   ≥ {g:.4f}')
    sys.stdout.flush()
# контроли
row = []
for L in (2.5, 3.2, 4.0):
    n = rng.poisson(rho * (2 * L) ** 4); P = rng.uniform(-L, L, (n, 4)); row.append((min_interval(P), n))
print(f'{"посыпка Пуассона":26s} ' + '  '.join(f'{m:.1e}({n})' for m, n in row) + '   → 0')
row = []
for L in (2.5, 3.2, 4.0):
    g = np.arange(-math.floor(L), math.floor(L) + 1); P = np.array(np.meshgrid(g, g, g, g)).reshape(4, -1).T.astype(float)
    row.append((min_interval(P), len(P)))
print(f'{"Z⁴":26s} ' + '  '.join(f'{m:.1e}({n})' for m, n in row) + '   = 0 (есть светоподобные пары)')
# «иррациональная» решётка: Z⁴, повёрнутая случайным лоренцевым бустом и вращением
from scipy.linalg import expm
G = np.zeros((4, 4)); G[0, 1:] = G[1:, 0] = rng.normal(0, 0.4, 3); R = np.zeros((4, 4)); R[1, 2], R[2, 1] = 0.7, -0.7; R[1, 3], R[3, 1] = 0.3, -0.3
Lm = expm(G) @ expm(R)
row = []
for L in (2.5, 3.2, 4.0):
    g = np.arange(-math.ceil(2 * L), math.ceil(2 * L) + 1); Z = np.array(np.meshgrid(g, g, g, g)).reshape(4, -1).T.astype(float)
    P = Z @ Lm.T; P = P[np.all(np.abs(P) <= L, axis=1)]; row.append((min_interval(P), len(P)))
print(f'{"Z⁴ под иррац. бустом":26s} ' + '  '.join(f'{m:.1e}({n})' for m, n in row) + '   → 0 (Маргулис)')
