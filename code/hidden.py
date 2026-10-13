"""Шум от наблюдателя = его незнание скрытых фаз.
Кристалл с размытым краем тени (w), но монетки НЕ случайны: у события скрытая фаза
θ = F(зерно, метки события) — нелинейная золотая «кошка», прогнанная по 8 целым
меткам события (a_i, b_i). Событие есть ⟺ θ₁ < P(s). Зерно — единственная скрытая
переменная мира.
Наблюдатель фиксирован (его ромб и всё видимое строение кристалла фиксированы),
меняем только зерно → дисперсия числа событий в его прошлом (ромбе).
Сравниваем: (а) с Σ P(1−P) — независимые монетки; (б) с разбросом по разным
наблюдателям при одном зерне (то, что видно «снаружи»)."""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from gold4 import ST, SI, r_for_density, TAU, TP
t0 = time.time()
OM = np.array([1 / TAU, 1 / TAU ** 2]); EPS = 0.6

def coord_values_lab(lo, hi, rint):
    C, Cp, L = [], [], []
    for b in range(math.floor((lo - rint) / 5 ** .5) - 1, math.ceil((hi + rint) / 5 ** .5) + 2):
        alo = max(math.ceil(lo - b * TAU), math.ceil(-rint - b * TP)); ahi = min(math.floor(hi - b * TAU), math.floor(rint - b * TP))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1); C.append(a + b * TAU); Cp.append(a + b * TP); L.append(np.stack([a, np.full_like(a, b)], 1))
    return np.concatenate(C), np.concatenate(Cp), np.concatenate(L)

def points_lab(box, rmax):
    vals = []
    for k, (lo, hi) in enumerate(box):
        if k == 0: c, cp, l = coord_values_lab(lo / ST, hi / ST, rmax / SI); vals.append((c * ST, (cp * SI) ** 2, l))
        else: c, cp, l = coord_values_lab(lo, hi, rmax); vals.append((c, cp ** 2, l))
    P = np.zeros((1, 4)); W = np.zeros(1); Lb = np.zeros((1, 0), int)
    for k, (c, w2, l) in enumerate(vals):
        m = len(P); P = np.repeat(P, len(c), 0); W = np.repeat(W, len(c)); Lb = np.repeat(Lb, len(c), 0)
        P[:, k] = np.tile(c, m); W += np.tile(w2, m); Lb = np.hstack([Lb, np.tile(l, (m, 1))])
        keep = W <= rmax ** 2; P, W, Lb = P[keep], W[keep], Lb[keep]
    return P, np.sqrt(W), Lb

def prob(s, r, w): return np.clip((r * (1 + w) - s) / (2 * w * r), 0, 1)
def r_for(w):
    r0 = r_for_density(1.0)
    def dens(r):
        s = np.linspace(0, r * (1 + w), 4001); return np.trapezoid(prob(s, r, w) * 4 * s ** 3, s) / r0 ** 4
    lo, hi = 0.5 * r0, 1.5 * r0
    for _ in range(60):
        mid = 0.5 * (lo + hi); (lo, hi) = (mid, hi) if dens(mid) < 1 else (lo, mid)
    return mid

def phase(labels, seeds):
    """θ для всех событий × всех зёрен: (S, n, 2). Два прохода золотой кошки по 8 меткам."""
    S = len(seeds); n = len(labels)
    th = np.broadcast_to(seeds[:, None, :], (S, n, 2)).copy()
    for rep in range(2):
        for k in range(labels.shape[1]):
            th = (th + labels[None, :, k, None] * OM) % 1.0
            p = EPS / (2 * math.pi) * np.sin(2 * math.pi * th[..., 0])
            th = np.stack([2 * th[..., 0] + th[..., 1] + p, th[..., 0] + th[..., 1] + p], -1) % 1.0
    return th

if __name__ == '__main__':
    w = 0.3; r = r_for(w); rmax = r * (1 + w)
    rng = np.random.default_rng(5); rnd = np.array([.48, -.71, .52]); rnd /= np.linalg.norm(rnd)
    S = 120; seeds = rng.random((S, 2))
    print(f'w = {w}, r = {r:.3f}; скрытых зёрен {S}')
    for name, eta, nv in (('покой', 0.0, np.array([1., 0, 0])), ('буст 1.5', 1.5, rnd)):
        for nexp in (3000, 20000):
            tau0 = (24 * nexp / math.pi) ** 0.25; G.tau0 = tau0
            # (а) один наблюдатель, меняем только скрытое зерно
            a = rng.uniform(0, 500, 4); b = a + tau0 * G.boost_vec(eta, nv)
            Pp, s, Lb = points_lab(G.interval_box(a, b, eta, nv), rmax)
            ins = G.inside(Pp, a, b); Pp, s, Lb = Pp[ins], s[ins], Lb[ins]
            pr = prob(s, r, w)
            cnt = np.zeros(S)
            for c in range(0, S, 20):
                th = phase(Lb, seeds[c:c + 20]); cnt[c:c + 20] = (th[..., 0] < pr[None]).sum(1)
            var_h = cnt.var(ddof=1); bern = (pr * (1 - pr)).sum(); mean = cnt.mean()
            kurt = ((cnt - mean) ** 4).mean() / cnt.var() ** 2
            # (б) одно зерно, разные наблюдатели
            c2 = []
            for m in range(40):
                a = rng.uniform(0, 500, 4); b = a + tau0 * G.boost_vec(eta, nv)
                Pp, s, Lb = points_lab(G.interval_box(a, b, eta, nv), rmax)
                ins = G.inside(Pp, a, b); Pp, s, Lb = Pp[ins], s[ins], Lb[ins]
                th = phase(Lb, seeds[:1])[0]; c2.append((th[:, 0] < prob(s, r, w)).sum())
            c2 = np.array(c2)
            print(f'{name:9s} n≈{nexp}: (а) скрытое: среднее {mean:.0f}, дисперсия/среднее {var_h/mean:.3f}, '
                  f'у незав. монеток {bern/mean:.3f}, эксцесс {kurt:.2f} | (б) по наблюдателям: дисперсия/среднее {c2.var(ddof=1)/c2.mean():.3f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
