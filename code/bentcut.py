#!/usr/bin/env python3
"""ИЗОГНУТЫЙ СРЕЗ. Золотой мир 1+1 без лоскутов: гнём не кристалл, а сам срез.

Точка — пара чисел t, c из Z[τ] (как в gold4, только 1+1). У каждого есть тень.
Естественная форма — след:  Tr(τt² − c²) = T² − x² − T'² − x'²,
где T = √τ·t, x = c (наш мир) и T' = t'/√τ, x' = c' (тень). Это плоское
пространство R^{1,3} с ОДНИМ временем: все теневые направления пространственные.

Плоский мир: срез {T' = x' = 0}, событие есть ⟺ тень (T', x') в круге радиуса r.
Изогнутый мир: срез Σ — гипербола де Ситтера в (T, x, x'):
    −T² + x² + (x' + ℓ)² = ℓ².
Событие есть ⟺ его смещение от Σ по нормали (n, T') лежит в том же круге.
Положение события — его проекция на Σ. Порядок — хорда в R^{1,3}, на гиперболе
он совпадает с причинностью де Ситтера точно. Швов и склейки нет.

Проверка: число событий в причинном ромбе высоты τ против точной площади
ромба в dS2 (кривизна R = 2/ℓ²). Контроли: плоский срез и засев Пуассона на Σ.
"""
import sys, math, time, numpy as np
TAU = (1 + 5 ** .5) / 2; TP = -1 / TAU
ST, SI = math.sqrt(TAU), math.sqrt(-TP)
S5 = 5 ** .5

def zt_range(lo, hi, plo, phi):
    """все c = a + bτ с lo ≤ c ≤ hi и plo ≤ c' ≤ phi; возвращает (c, c')."""
    C, Cp = [], []
    for b in range(math.floor((lo - phi) / S5) - 1, math.ceil((hi - plo) / S5) + 2):
        alo = max(math.ceil(lo - b * TAU), math.ceil(plo - b * TP))
        ahi = min(math.floor(hi - b * TAU), math.floor(phi - b * TP))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1, dtype=float); C.append(a + b * TAU); Cp.append(a + b * TP)
    if not C: return np.empty(0), np.empty(0)
    return np.concatenate(C), np.concatenate(Cp)

def gd(s): return 2 * np.arctan(np.tanh(s / 2))          # конформное время dS2

def world(ell, r, sm, chm, off=(0.0, 0.0)):
    """события на изогнутом срезе: |σ| ≤ sm, |χ| ≤ chm. Возвращает (σ, χ, X) с X=(T,x,Z) на Σ."""
    pad = 1 + r / ell
    Tm = ell * math.sinh(sm) * pad + r
    Rm = ell * math.cosh(sm) * pad + r                   # радиус в плоскости (x, Z)
    t, tp = zt_range(-Tm / ST, Tm / ST, (off[0] - r) / SI, (off[0] + r) / SI); T, Tq = t * ST, tp * SI - off[0]
    xlo, xhi = -Rm * math.sin(chm) - r, Rm * math.sin(chm) + r
    Zlo, Zhi = (ell - r) * math.cos(chm) - r, Rm
    c, cp = zt_range(xlo, xhi, Zlo - ell + off[1], Zhi - ell + off[1]); Z = cp - off[1] + ell
    out = []
    for k in range(0, len(T), 200):                      # пары (время × место), кусками
        Tk, Tqk = T[k:k + 200, None], Tq[k:k + 200, None]
        q = -Tk ** 2 + c[None, :] ** 2 + Z[None, :] ** 2
        ok = q > 0
        n = np.where(ok, np.sqrt(np.where(ok, q, 1)), 0) - ell
        acc = ok & (Tqk ** 2 + n ** 2 <= r * r)
        i, j = np.nonzero(acc)
        s = ell / (ell + n[i, j])
        out.append(np.stack([Tk[i, 0] * s, c[j] * s, Z[j] * s], 1))
    X = np.concatenate(out)
    sig = np.arcsinh(X[:, 0] / ell); chi = np.arctan2(X[:, 1], X[:, 2])
    m = (np.abs(sig) <= sm) & (np.abs(chi) <= chm)
    return sig[m], chi[m], X[m]

def flat_world(r, L):
    """плоский золотой мир 1+1 (контроль): (T, x) в квадрате [−L, L]²."""
    t, tp = zt_range(-L / ST, L / ST, -r / SI, r / SI); T, Tq = t * ST, tp * SI
    c, cp = zt_range(-L, L, -r, r)
    acc = Tq[:, None] ** 2 + cp[None, :] ** 2 <= r * r
    i, j = np.nonzero(acc)
    return np.stack([T[i], c[j]], 1)

def exact_area(ell, tau):
    """точная площадь причинного ромба высоты τ в dS2 радиуса ℓ."""
    s0 = tau / (2 * ell)
    s = np.linspace(-s0, s0, 200001)
    arg = (1 + np.abs(np.sinh(s)) * math.sinh(s0)) / (np.cosh(s) * math.cosh(s0))
    f = np.cosh(s) * 2 * np.arccos(np.clip(arg, -1, 1))
    return ell ** 2 * np.trapezoid(f, s)

def iso(th, bs, bx):
    """изометрия dS2 = преобразование Лоренца в (T, x, Z): поворот x–Z, буст T–Z, буст T–x."""
    R = np.array([[1, 0, 0], [0, math.cos(th), math.sin(th)], [0, -math.sin(th), math.cos(th)]])
    Bs = np.array([[math.cosh(bs), 0, math.sinh(bs)], [0, 1, 0], [math.sinh(bs), 0, math.cosh(bs)]])
    Bx = np.array([[math.cosh(bx), math.sinh(bx), 0], [math.sinh(bx), math.cosh(bx), 0], [0, 0, 1]])
    return R @ Bs @ Bx

def mdot(a, b): return -a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]

def sc(X, ell):  # (σ, χ) точки на Σ
    return np.arcsinh(X[..., 0] / ell), np.arctan2(X[..., 1], X[..., 2])

def count_diamonds(X, ell, tau, M, rs, sm, chm, lim=(0.45, 0.25, 0.45)):
    s0 = tau / (2 * ell)
    p0 = np.array([ell * math.sinh(-s0), 0, ell * math.cosh(s0)]); q0 = np.array([ell * math.sinh(s0), 0, ell * math.cosh(s0)])
    eg, em = gd(sm), chm
    res, where = [], []
    tries = 0
    while len(res) < M and tries < 50 * M:
        tries += 1
        L = iso(rs.uniform(-lim[0], lim[0]), rs.uniform(-lim[1], lim[1]), rs.uniform(-lim[2], lim[2]))
        p, q = L @ p0, L @ q0
        (sp, cpp), (sq, cq) = sc(p, ell), sc(q, ell)
        ep, eq = gd(sp), gd(sq)                          # вершины в конформных координатах
        side = [((ep + eq) / 2 + (cq - cpp) / 2 * s, ) for s in (1, -1)]
        e1 = (ep + eq) / 2 + (cq - cpp) / 2; c1 = (cpp + cq) / 2 + (eq - ep) / 2
        e2 = (ep + eq) / 2 - (cq - cpp) / 2; c2 = (cpp + cq) / 2 - (eq - ep) / 2
        if max(abs(ep), abs(eq), abs(e1), abs(e2)) > eg or max(abs(cpp), abs(cq), abs(c1), abs(c2)) > em: continue
        inn = (mdot(X, p) >= ell ** 2) & (X[:, 0] > p[0]) & (mdot(X, q) >= ell ** 2) & (X[:, 0] < q[0])
        res.append(inn.sum()); where.append((cpp + cq) / 2)
    return np.array(res, float), np.array(where)

def poisson_on_sigma(ell, rho, sm, chm, rs):
    A = ell ** 2 * 2 * math.sinh(sm) * 2 * chm
    n = rs.poisson(rho * A)
    sig = np.arcsinh(rs.uniform(-math.sinh(sm), math.sinh(sm), n)); chi = rs.uniform(-chm, chm, n)
    return np.stack([ell * np.sinh(sig), ell * np.cosh(sig) * np.sin(chi), ell * np.cosh(sig) * np.cos(chi)], 1)

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    ELLS = [float(e) for e in (sys.argv[2] if len(sys.argv) > 2 else '10000,40,20,12').split(',')]
    TAUS = [float(e) for e in (sys.argv[3] if len(sys.argv) > 3 else '4,8,12').split(',')]
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 30; M = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    r = math.sqrt(5 * rho / math.pi); rs = np.random.default_rng(11); t0 = time.time()
    print(f'плотность {rho}, окно r = {r:.3f}; {K} сдвигов фазона × {M} ромбов на каждый')
    for ell in ELLS:
        sm = min(1.0, 2.2 * max(TAUS) / ell); chm = sm
        lim = (min(.45, .5 * sm), min(.25, .3 * sm), .45)
        rho_eff = rho * (1 + r * r / (4 * ell * ell))
        C = {t: [] for t in TAUS}; Pn = {t: [] for t in TAUS}; dens = []
        A = ell ** 2 * 2 * math.sinh(sm) * 2 * chm
        for k in range(K):
            sig, chi, X = world(ell, r, sm, chm, tuple(rs.uniform(-5, 5, 2))); dens.append(len(X) / A)
            Xp = poisson_on_sigma(ell, rho, sm, chm, rs)
            for tau in TAUS:
                C[tau].append(count_diamonds(X, ell, tau, M, rs, sm, chm, lim)[0])
                Pn[tau].append(count_diamonds(Xp, ell, tau, M, rs, sm, chm, lim)[0])
        dens = np.array(dens)
        print(f'\nℓ = {ell:g}: плотность на Σ {dens.mean():.4f} ± {dens.std() / math.sqrt(K):.4f} (ожидание {rho_eff:.4f})  [{time.time() - t0:.0f} с]')
        for tau in TAUS:
            V0 = tau ** 2 / 2; th = exact_area(ell, tau) / V0 - 1
            c = np.array(C[tau]); p = np.array(Pn[tau])
            dc = c.mean(1) / (rho_eff * V0) - 1; dp = p.mean(1) / (rho * V0) - 1
            print(f'  τ={tau:4.1f} (τ/ℓ={tau / ell:.2f}): точно {th:+.5f} | кристалл {dc.mean():+.5f} ± {dc.std() / math.sqrt(K):.5f} '
                  f'(на ромб ±{c.std(1).mean():.2f}) | Пуассон {dp.mean():+.5f} ± {dp.std() / math.sqrt(K):.5f} (на ромб ±{p.std(1).mean():.2f})')
            sys.stdout.flush()
