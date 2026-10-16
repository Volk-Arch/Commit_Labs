"""ПОЛЯ И СРЕЗ.
* Потенциалы 1+1 для «уголка»: V-образное поле заряда, мягкий Кулон (s-волна водорода), однородное.
* Изогнутый срез в R^{1,1} ⊕ R^k: нормальная рамка, связность A_μ = ⟨N_a, ∂_μN_b⟩, кривизна F = dA + [A, A],
  уравнение Риччи (F = [K, K]), голономия петель. k = 2 — электромагнетизм (заряд = число оборотов), k = 4 — неабелево.
* Уравнения среза Редже–Тейтельбойма для статического сферического винтового среза (slice-doc §5–6)."""
import math
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

# ---------- потенциалы (U — потенциальная энергия частицы; P′ = U; dU = U′) ----------
def vshape(F, Xc=0.0):
    """U = F|x − Xc| (F > 0 — притяжение к Xc, запирание)."""
    return (lambda x: F * (x - Xc) * np.abs(x - Xc) / 2, lambda x: F * np.sign(x - Xc), lambda x: F * np.abs(x - Xc))

def coulomb_soft(alpha, a):
    """U = −α/√(x² + a²) (нечётный сектор = s-волна водорода)."""
    return (lambda x: -alpha * np.arcsinh(x / a), lambda x: alpha * x / (x * x + a * a) ** 1.5, lambda x: -alpha / np.sqrt(x * x + a * a))

def uniform(E):
    return (lambda x: -E * x * x / 2, lambda x: -E * np.ones_like(x), lambda x: -E * x)

# ---------- изогнутый срез и связность нормалей ----------
def slice_twist(k=0.055):
    """абелева закрутка (gauge-doc): h = κ((T² + x²)/2, Tx) в R², кривизна −2κ²."""
    return lambda T, x: np.array([k * (T * T + x * x) / 2, k * T * x])

def slice_rotating(k=0.15, om=1.0):
    """неабелев срез в R⁴: плоскости изгиба поворачиваются от места к месту."""
    return lambda T, x: k * ((T * T + x * x) / 2 * np.array([np.cos(om * x), 0, np.sin(om * x), 0]) + T * x * np.array([0, np.cos(om * T), 0, np.sin(om * T)]))

def slice_fixed4(k=0.15):
    return lambda T, x: k * ((T * T + x * x) / 2 * np.array([1.0, 0, 0, 0]) + T * x * np.array([0, 1.0, 0, 0]))

def _eta(kdim): return np.diag([1.0] + [-1.0] * (1 + kdim))

def frame(h, T, x, d=1e-5):
    hv = h(T, x); kd = len(hv); eta = _eta(kd); emb = lambda T_, x_: np.r_[T_, x_, h(T_, x_)]
    eT = (emb(T + d, x) - emb(T - d, x)) / (2 * d); ex = (emb(T, x + d) - emb(T, x - d)) / (2 * d)
    ip = lambda u, v: u @ eta @ v; tang = [eT, ex]; G = np.array([[ip(u, v) for v in tang] for u in tang]); Gi = np.linalg.inv(G)
    proj = lambda v: v - sum(c * e for c, e in zip(Gi @ np.array([ip(u, v) for u in tang]), tang))
    N = []
    for a in range(kd):
        v = proj(np.eye(2 + kd)[2 + a])
        for w in N: v = v - (ip(v, w) / ip(w, w)) * w
        nn = -ip(v, v)
        if nn <= 0: raise ValueError('срез не лоренцев в этой точке')
        N.append(v / math.sqrt(nn))
    return np.array(N), G, eta

def connection(h, T, x, d=1e-4):
    N, _, eta = frame(h, T, x); ip = lambda u, v: -(u @ eta @ v); kd = len(N)
    dT = (frame(h, T + d, x)[0] - frame(h, T - d, x)[0]) / (2 * d); dx = (frame(h, T, x + d)[0] - frame(h, T, x - d)[0]) / (2 * d)
    return (np.array([[ip(N[a], dT[b]) for b in range(kd)] for a in range(kd)]), np.array([[ip(N[a], dx[b]) for b in range(kd)] for a in range(kd)]))

def curvature(h, T, x, d=1e-3):
    AT, Ax = connection(h, T, x)
    dAx = (connection(h, T + d, x)[1] - connection(h, T - d, x)[1]) / (2 * d); dAT = (connection(h, T, x + d)[0] - connection(h, T, x - d)[0]) / (2 * d)
    return dAx - dAT + (AT @ Ax - Ax @ AT)

def ricci(h, T, x, d=1e-4):
    N, G, eta = frame(h, T, x); Gi = np.linalg.inv(G); ip = lambda u, v: u @ eta @ v; kd = len(N)
    y = lambda T_, x_: np.r_[T_, x_, h(T_, x_)]
    dTT = (y(T + d, x) - 2 * y(T, x) + y(T - d, x)) / d ** 2; dxx = (y(T, x + d) - 2 * y(T, x) + y(T, x - d)) / d ** 2
    dTx = (y(T + d, x + d) - y(T + d, x - d) - y(T - d, x + d) + y(T - d, x - d)) / (4 * d * d)
    K = np.array([[[-ip(D, N[a]) for D in row] for row in ((dTT, dTx), (dTx, dxx))] for a in range(kd)])
    return np.array([[np.einsum('ls,l,s->', Gi, K[a, 0], K[b, 1]) - np.einsum('ls,l,s->', Gi, K[a, 1], K[b, 0]) for b in range(kd)] for a in range(kd)])

def holonomy(h, path, nsteps=40):
    kd = len(h(0.0, 0.0)); U = np.eye(kd)
    for (T0, x0), (T1, x1) in zip(path[:-1], path[1:]):
        for s in range(nsteps):
            Tm = T0 + (s + 0.5) / nsteps * (T1 - T0); xm = x0 + (s + 0.5) / nsteps * (x1 - x0)
            AT, Ax = connection(h, Tm, xm); U = expm(-(AT * (T1 - T0) + Ax * (x1 - x0)) / nsteps) @ U
    return U

def lasso(c, r=0.6, base=(0.0, 0.0), n=24):
    ring = [(c[0] + r * math.cos(2 * math.pi * k / n), c[1] + r * math.sin(2 * math.pi * k / n)) for k in range(n + 1)]
    return [base, ring[0]] + ring[1:] + [base]

# ---------- уравнения среза Редже–Тейтельбойма (статический сферический винтовой срез) ----------
_E7 = np.diag([1, -1, -1, -1, -1, -1, -1.0])
def rt_geom(r, A, Ap, Bp, App, Bpp, w):
    f = 1 - w * w * A * A; h = 1 + Ap * Ap + Bp * Bp
    fp = -2 * w * w * A * Ap; fpp = -2 * w * w * (Ap * Ap + A * App); hp = 2 * (Ap * App + Bp * Bpp)
    Gtt = (1 - 1 / h) / r ** 2 + hp / (r * h * h); Grr = (1 - 1 / h) / r ** 2 - fp / (r * f * h)
    Gthth = -(fpp / (2 * f * h) - fp * fp / (4 * f * f * h) - fp * hp / (4 * f * h * h) + (fp / f - hp / h) / (2 * r * h))
    eps, pr, pt = Gtt, -Grr, -Gthth
    T = np.array([[1, 0, 0, 0, 0, A * w, 0], [0, 1, 0, 0, Ap, 0, Bp], [0, 0, 0, -r, 0, 0, 0], [0, 0, r, 0, 0, 0, 0.]])
    N = np.linalg.svd(T @ _E7)[2][4:]
    d2 = {'tt': np.array([0, 0, 0, 0, -A * w * w, 0, 0]), 'rr': np.array([0, 0, 0, 0, App, 0, Bpp]), 'th': np.array([0, -r, 0, 0, 0, 0, 0.])}
    res = [(eps / f) * (d2['tt'] @ _E7 @ n) + (pr / h) * (d2['rr'] @ _E7 @ n) + (pt / r ** 2) * 2 * (d2['th'] @ _E7 @ n) for n in N]
    return np.array(res), (eps, pr, pt, f, h, fp)

def rt_schw(r, M, w):
    A = math.sqrt(2 * M / r) / w; Ap = -0.5 * A / r; Bp = math.sqrt(2 * M / (r - 2 * M) - Ap * Ap); App = 0.75 * A / r ** 2
    return A, Ap, Bp, App, (-2 * M / (r - 2 * M) ** 2 - 2 * Ap * App) / (2 * Bp)

def rt_integrate(M=1.0, w=1.0, r0=10.0, r1=200.0, dA=0.0, dAp=0.0, dB=0.0, n=1200):
    A, Ap, Bp, App, Bpp = rt_schw(r0, M, w); s = np.array([A * (1 + dA), Ap * (1 + dAp), Bp * (1 + dB)]); g = np.array([App, Bpp])
    rs = np.geomspace(r0, r1, n); out = []
    def acc(r, ss, g0):
        sol = least_squares(lambda z: rt_geom(r, ss[0], ss[1], ss[2], z[0], z[1], w)[0] * 1e3, g0, xtol=1e-14, ftol=1e-14, gtol=1e-14); return sol.x
    for i in range(n - 1):
        r, hh = rs[i], rs[i + 1] - rs[i]
        def D(rr, ss):
            nonlocal g
            g = acc(rr, ss, g); return np.array([ss[1], g[0], g[1]])
        try:
            k1 = D(r, s); k2 = D(r + hh / 2, s + hh / 2 * k1); k3 = D(r + hh / 2, s + hh / 2 * k2); k4 = D(r + hh, s + hh * k3)
        except Exception: break
        s = s + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        _, (eps, pr, pt, f, h, fp) = rt_geom(rs[i + 1], *s, *g, w)
        if f <= 0 or not np.all(np.isfinite(s)): break
        out.append((rs[i + 1], f, h, rs[i + 1] * fp / (2 * f), pt / pr if pr else np.nan, s[2], eps, pr))
    return np.array(out)

# ---------- поле ядра из порядка (1+1): ½ за каждое событие ядра-трубки в прошлом конусе ----------
def order_field11(w, a=1.0, g=0.8, fit_t=(None, None), fit_x=10.0, step=0.05):
    """Потенциальная энергия частицы U(t, x) = −g·½·N(t, x)/(λ·2a), N — число событий трубки |x| < a в прошлом конусе точки
    (запаздывающий Лоренц: A₀ = G_ret·j, G_ret = ½). Однородная по x часть α·t + β вычитается — калибровка (функция одного t),
    точка отсчёта энергии. Возвращает (U(t, x) на сетке, U′ в событиях, U в событиях, (α, β, F — наклон по |x|))."""
    from .causal import causal
    from .propagate import GridField
    t, x, X = w.t, w.x, w.X; L = max(np.abs(t).max(), np.abs(x).max()); tube = np.nonzero(np.abs(x) < a)[0]; te, xe = t[tube], x[tube]
    lam = len(tube) / (2 * L - 2 * a) / (2 * a); c = -g * 0.5 / (lam * 2 * a)
    tg = np.arange(-L - 0.5, L + 0.51, step); xg = np.arange(-L - 0.5, L + 0.51, step); T, Xg = np.meshgrid(tg, xg, indexing='ij'); V = np.zeros(T.shape)
    for r in range(0, len(tg), 40):
        tt, xx = T[r:r + 40], Xg[r:r + 40]; acc = np.zeros(tt.shape)
        for s in range(0, len(te), 64): acc += ((tt[..., None] - te[s:s + 64]) > np.abs(xx[..., None] - xe[s:s + 64])).sum(-1)
        V[r:r + 40] = c * acc
    Uev = c * causal(X)[tube].sum(0); t0 = -np.inf if fit_t[0] is None else fit_t[0]; t1 = np.inf if fit_t[1] is None else fit_t[1]
    S = (t > t0) & (t < t1) & (np.abs(x) > a + 0.3) & (np.abs(x) < fit_x)
    (al, be, F), *_ = np.linalg.lstsq(np.c_[t[S], np.ones(S.sum()), np.abs(x[S])], Uev[S], rcond=None)
    V = V - al * T - be; Uf = GridField(tg, xg, V); dUev = GridField(tg, xg, np.gradient(V, xg, axis=1))(t, x)
    return Uf, dUev, Uev - al * t - be, (float(al), float(be), float(F))

def order_cloud11(w, rho_fun, g, tmin, step=0.1, xmax=10.0, fit_x=10.0):
    """Поле размазанного заряда (облако электрона с плотностью ρ(x), ∫ρ = 1) из порядка: U = g·½·Σ_{z ≺ точка} ρ(x_z)/ρ_соб
    (ρ_соб — плотность событий). Ожидание: однородная часть α·t + β плюс c·M(x), M(x) = −∫ρ(x′)|x − x′|dx′, c = g/2.
    Вычитается только α·t + β (калибровка). Возвращает (U на сетке, U′ в событиях, c, α)."""
    from .propagate import GridField
    t, x = w.t, w.x; L = max(np.abs(t).max(), np.abs(x).max()); dens = w.n / (2 * L * L)
    sel = np.abs(x) < xmax; tz, xz = t[sel], x[sel]; wz = rho_fun(xz) / dens
    tg = np.arange(tmin - 1.0, L + 0.51, step); xg = np.arange(-L - 0.5, L + 0.51, step); T, Xg = np.meshgrid(tg, xg, indexing='ij')
    tf, xf = T.ravel(), Xg.ravel(); V = np.empty(tf.shape)
    for s in range(0, len(tf), 4000):
        V[s:s + 4000] = ((tf[s:s + 4000, None] - tz[None]) > np.abs(xf[s:s + 4000, None] - xz[None])) @ wz
    V = 0.5 * g * V.reshape(T.shape)
    xs = np.linspace(-xmax, xmax, 801); rs = rho_fun(xs); rs = rs / np.trapezoid(rs, xs); M = lambda xx: -np.trapezoid(rs[None] * np.abs(np.asarray(xx)[:, None] - xs[None]), xs, axis=1)
    Uf0 = GridField(tg, xg, V); S = (t > tmin + 1) & (t < L - 4) & (np.abs(x) < fit_x); Uev = Uf0(t[S], x[S])
    (al, be, c), *_ = np.linalg.lstsq(np.c_[t[S], np.ones(S.sum()), M(x[S])], Uev, rcond=None)
    V = V - al * T - be; Uf = GridField(tg, xg, V); dUev = GridField(tg, xg, np.gradient(V, xg, axis=1))(t, x)
    return Uf, dUev, float(c), float(al)
