"""Атом 3+1 в режиме без гребёнки (m = 1/τ² относительно зерна, dirac-doc §27). Свет, массивная частица и поле облака —
одним правилом: симметрично размытый световой конус (fuzzy.py, β = 0.115, голая масса из ядра). Эталоны — непрерывная теория
ТЕМ ЖЕ прибором (окно H, шар R, вес формы 1s). Все пробы масштабированы на размер атома a₀ = 1/(αM)."""
import math, numpy as np
import scipy.linalg as sl, scipy.optimize as so
from . import worlds as Wd, fuzzy as Fz
from .core import r_for_density

TAU = (1 + 5 ** 0.5) / 2
AL, A = 0.45, 0.5
SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]]); SZ = np.array([[1, 0], [0, -1]], complex)
BETA = 0.115
SHAPE = 1                     # 1 — гауссиан; 3 — (3/2 − x)e^{−x}, без δ″ (β = 0.083)

# ---------------- непрерывная теория тем же прибором ----------------
def _radial(M, R, H, h=0.04, Rbox=None, dt=0.01, Uextra=None, m2=None, cz=0.0, Lz=2.0, pauli=0.0, trunc=False):
    """trunc=True — как в дискретном мире: вне шара R нет событий, значит нет ни вставок массы, ни потенциала (свободный безмассовый
    свет снаружи остаётся — он идёт одним шагом ядра)."""
    Rbox = Rbox or max(4 * R, 40.0); r = h * np.arange(1, int(Rbox / h) + 1); N = len(r); U = -AL / np.sqrt(r * r + A * A)
    if Uextra is not None: U = U + Uextra(r)
    prof = np.exp(-AL * M * r); ball = r < R; m2 = M * M if m2 is None else m2
    def lap(u, l):
        d = np.empty_like(u); d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2; d[0] = (u[1] - 2 * u[0]) / h ** 2; d[-1] = (-2 * u[-1] + u[-2]) / h ** 2
        return d - l * (l + 1) / r ** 2 * u
    fz = cz * r * np.exp(-r / Lz) / np.sqrt(3); dU = AL * r / (r * r + A * A) ** 1.5; fp = pauli * 1j * dU
    th = (r < R).astype(float) if trunc else np.ones(N); U = U * th; fz = fz * th; fp = fp * th; m2v = m2 * th
    def F(y):
        us, up, vs, vp = y
        a_s = 2j * U * vs + U * U * us + lap(us, 0) - m2v * us + (fz + fp) * up
        a_p = 2j * U * vp + U * U * up + lap(up, 1) - m2v * up + (fz + fp) * us
        return np.array([vs, vp, a_s, a_p])
    y = np.array([np.zeros(N, complex), np.zeros(N, complex), (r * prof).astype(complex), np.zeros(N, complex)]); ts = []; sig = []; t = 0.0
    while t < H - 0.2:
        k1 = F(y); k2 = F(y + dt / 2 * k1); k3 = F(y + dt / 2 * k2); k4 = F(y + dt * k3); y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        ts.append(t); sig.append(np.sum((r * prof * y[0])[ball]) * h)
    ts = np.array(ts); sig = np.array(sig); s = (ts > 1.5) & (ts < H - 0.3); tt = ts[s]; w = np.sin(np.pi * (tt - tt.min()) / np.ptp(tt)) ** 2
    om = np.linspace(0.15, 1.3, 2301); S = np.abs(np.exp(-1j * np.outer(om, tt)) @ (w * sig[s])) ** 2; i = int(np.argmax(S)); a, b, c = S[i - 1], S[i], S[i + 1]
    return om[i] + 0.5 * (a - c) / (a - 2 * b + c) * (om[1] - om[0])

def cloud(M, e2=AL / 2, N=2000, Rb=80.0):
    """КГ 1s массы M в поле ядра; заряд облака (E − U)|φ|², его потенциал e²·[Q(r)/r + ∫_r σ/r′]; J — точный сдвиг уровня."""
    h = Rb / N; r = h * np.arange(1, N + 1); U = -AL / np.sqrt(r * r + A * A)
    D = (np.diag(np.full(N, 2.0)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / h ** 2
    lam = lambda om, Ut: sl.eigh(D + np.diag(M * M - (om - Ut) ** 2), eigvals_only=True, subset_by_index=[0, 0])[0]
    level = lambda Ut: so.brentq(lambda o: lam(o, Ut), 0.3 * M, 0.99999 * M)
    E = level(U); _, v = sl.eigh(D + np.diag(M * M - (E - U) ** 2), subset_by_index=[0, 0]); u = v[:, 0]
    sig = (E - U) * u * u; sig /= np.sum(sig) * h; Q = np.cumsum(sig) * h; out = np.cumsum((sig / r)[::-1])[::-1] * h
    Uc = e2 * (Q / r + out); return r, sig / (4 * np.pi * r * r), Uc, E, level(U + Uc) - E

def cont_refs(M, R, H, trunc=False):
    cz = 0.12 * M * M / ((2 / M) / math.e); Lz = 2 / M; kw = dict(trunc=trunc)
    E1 = _radial(M, R, H, **kw); E2 = _radial(M, R, H, m2=0.81 * M * M, **kw); Es = _radial(M, R, H, cz=cz, Lz=Lz, **kw); Ed = _radial(M, R, H, pauli=1.0, **kw)
    rr, sg, Uc, Ec, Jc = cloud(M); Ej = _radial(M, R, H, Uextra=lambda q: np.interp(q, rr, Uc), **kw)
    return dict(E=E1, slope=(E1 - E2) / (0.19 * M * M), stark=Es - E1, spin=Ed - E1, J=Ej - E1, J_exact=Jc)

# ---------------- кристалл / посыпка ----------------
def make_world(rho, H, R, shift=None, poisson_seed=None):
    if poisson_seed is not None:
        rng = np.random.default_rng(poisson_seed); n = rng.poisson(rho * (H + 0.7) * (2 * R) ** 3); P = np.c_[rng.uniform(-0.6, H + 0.1, n), rng.uniform(-R, R, (n, 3))]
        P = P[np.linalg.norm(P[:, 1:], axis=1) <= R]; return np.ascontiguousarray(P[np.argsort(P[:, 0])])
    return np.ascontiguousarray(Wd.gold31(rho, 'cylinder', H=H, Rc=R, Rs=50.0, shift=shift).X)

class Atom:
    def __init__(self, X, rho, M, H, R):
        self.X, self.rho, self.M, self.H, self.R = X, rho, M, H, R; n = self.n = len(X)
        self.r = np.linalg.norm(X[:, 1:], axis=1); self.t = X[:, 0]; self.prof = np.exp(-AL * M * self.r); self.sel = (self.t > 1.5) & (self.t < H - 0.3)
        self.alpha = rho * math.pi / 24; self.coef = math.sqrt(rho) / (2 * math.pi * math.sqrt(6)); self.tc2 = (2.8 * rho ** -0.25) ** 2
        self.rg = np.linspace(0, 2 * R + 1, 4001); self.Unuc = -AL / np.sqrt(self.rg ** 2 + A * A)
        t_, d_ = Fz.mean_kernel_delta(rho, BETA, shape=SHAPE); G = lambda w: np.trapezoid(np.exp(1j * w * t_) * d_, t_)
        self.mb2 = lambda m: m * m / (1 - m * m * G(m).real)
        rh = np.where(self.r[:, None] > 1e-9, X[:, 1:] / np.maximum(self.r, 1e-9)[:, None], 0)
        self.sr = rh[:, 0, None, None] * SX + rh[:, 1, None, None] * SY + rh[:, 2, None, None] * SZ
        self.dU = AL * self.r / (self.r ** 2 + A * A) ** 1.5; self.eye = np.broadcast_to(np.eye(2), (n, 2, 2))
        self.om = np.linspace(0.15, 1.3, 2301)
    def src(self, x0=0.0, spin=0):
        rr = np.linalg.norm(self.X[:, 1:] - np.array([x0, 0, 0]), axis=1)
        return np.ascontiguousarray(np.where((self.t < 0.45)[:, None], np.exp(-AL * self.M * rr)[:, None] * np.eye(2)[spin][None, :], 0).astype(np.complex128))
    def respond(self, B, f=None, Ug=None):
        """быстрый счёт (индекс клеток + слои времени); фаза ядра — точный логарифм, иначе — таблица U(r)."""
        if not hasattr(self, 'idx'): self.idx = Fz.build_index(self.X, 2.0)
        f = self.src() if f is None else f; B = np.ascontiguousarray(B, dtype=np.complex128)
        if Ug is None: return Fz.respond_fast(self.X, self.idx, self.tc2, self.alpha, self.coef, np.array([AL, A]), self.Unuc, f, B, BETA, SHAPE, use_phase=2)
        return Fz.respond_fast(self.X, self.idx, self.tc2, self.alpha, self.coef, self.rg, Ug, f, B, BETA, SHAPE, use_phase=1)
    def peak(self, sig):
        S = Fz._spec(self.t[self.sel], self.prof[self.sel] * sig[self.sel], self.om); i = int(np.argmax(S)); a, b, c = S[max(i - 1, 0)], S[i], S[min(i + 1, len(S) - 1)]
        den = a - 2 * b + c; return self.om[i] + (0.5 * (a - c) / den * (self.om[1] - self.om[0]) if den != 0 else 0.0)
    def growth(self, sig):
        ed = np.arange(0, self.H + 1e-9, 1.0); k = np.digitize(self.t, ed) - 1; ok = (k >= 0) & (k < len(ed) - 1); cnt = np.bincount(k[ok], minlength=len(ed) - 1)
        amp = np.sqrt(np.bincount(k[ok], np.abs(self.prof * sig)[ok] ** 2, minlength=len(ed) - 1) / np.maximum(cnt, 1)); tc = 0.5 * (ed[1:] + ed[:-1]); s = (tc > 2) & (tc < self.H - 1) & (cnt > 0)
        return float(np.polyfit(tc[s], np.log(amp[s] + 1e-300), 1)[0])
    def B(self, m, extra=None, pauli=0.0):
        b = -self.mb2(m) * self.eye if extra is None else (-self.mb2(m) + extra)[:, None, None] * self.eye
        return (b + pauli * 1j * self.dU[:, None, None] * self.sr) / self.rho
    def measure(self):
        M = self.M; e = np.array([1, 0]); s1 = self.respond(self.B(M)) @ e; E1 = self.peak(s1); E2 = self.peak(self.respond(self.B(0.9 * M)) @ e)
        Lz = 2 / M; g = self.X[:, 3] * np.exp(-self.r / Lz); c = 0.12 * M * M / np.abs(g).max(); Es = self.peak(self.respond(self.B(M, extra=c * g)) @ e)
        Ed = self.peak(self.respond(self.B(M, pauli=1.0)) @ e)
        return dict(E=E1, growth=self.growth(s1), slope=(E1 - E2) / (0.19 * M * M), stark=Es - E1, spin=Ed - E1)
    def pauli(self, d=None):
        """два электрона без взаимодействия: Ψ = φ_A⊗φ_B − φ_B⊗φ_A, проекция на 1s⊗1s по слоям времени."""
        M = self.M; d = d or 0.7 / (AL * M); Bx = self.B(M, pauli=1.0)
        A_ = self.respond(Bx, self.src(+d, 0)); Bu = self.respond(Bx, self.src(-d, 0)); Bd = self.respond(Bx, self.src(-d, 1))
        ed = np.arange(1.5, self.H - 0.3 + 1e-9, 0.25); k = np.digitize(self.t, ed) - 1; ok = (k >= 0) & (k < len(ed) - 1); tc = 0.5 * (ed[1:] + ed[:-1])
        sl_ = lambda ps, cc: np.bincount(k[ok], (self.prof * ps[:, cc])[ok].real, minlength=len(tc)) + 1j * np.bincount(k[ok], (self.prof * ps[:, cc])[ok].imag, minlength=len(tc))
        def pk(sig, om):
            w_ = np.sin(np.pi * (tc - tc.min()) / np.ptp(tc)) ** 2; S = np.abs(np.exp(-1j * np.outer(om, tc)) @ (w_ * sig)) ** 2; return om[int(np.argmax(S))]
        a0, a1, bu0, bd0, bd1 = sl_(A_, 0), sl_(A_, 1), sl_(Bu, 0), sl_(Bd, 0), sl_(Bd, 1)
        Fuu = a0 * bu0 - bu0 * a0; Buu = a0 * bu0 + bu0 * a0; Fud = a0 * bd1 - bd0 * a1; om2 = np.linspace(0.3, 2.6, 2301)
        return dict(uu=float(np.abs(Fuu).max() / np.abs(Buu).max()), ud=pk(Fud, om2), bb=pk(Buu, om2), E1=pk(a0, self.om))

def cloud_field_from_order(M, rho=1.0, S=None, per=10, seed=3, radii=None):
    """потенциал облака ИЗ ПОРЯДКА тем же размытым конусом (фотон = тот же конус): U(x) = 4π·e²/ρ · Σ_y K(x, y)·σ(y)."""
    rr, sg, Uc, _, _ = cloud(M); a0 = 1 / (AL * M); S = S or min(4.0 * a0, 16.0); T = S + 4.0
    G = np.ascontiguousarray(Wd.points_in_box31([(-T, 0.6), (-S, S), (-S, S), (-S, S)], Wd.r_for_density31(rho)))
    rad = np.linalg.norm(G[:, 1:], axis=1); sig = np.interp(rad, rr, sg, right=0.0)
    radii = radii or [x * a0 for x in (0.15, 0.4, 0.7, 1.0, 1.5, 2.0, 2.6)]; rng = np.random.default_rng(seed); rows = []
    for R_ in radii:
        idx = np.nonzero((np.abs(G[:, 0]) < 0.3) & (np.abs(rad - R_) < 0.25))[0]; tg = rng.choice(idx, min(per, len(idx)), replace=False).astype(np.int64)
        v = Fz.field_static(G, (2.8 * rho ** -0.25) ** 2, rho * math.pi / 24, math.sqrt(rho) / (2 * math.pi * math.sqrt(6)), BETA, sig, tg, SHAPE) * 4 * math.pi * (AL / 2) / rho
        rows.append((R_, float(v.mean()), float(v.std() / math.sqrt(len(v))), float(np.interp(R_, rr, Uc))))
    return np.array(rows), rr, Uc


# ---------------- потенциал как вставка при энергии E (dirac-doc §28) ----------------
# Фаза Вильсона на длинных почти световых шагах даёт только «прямолинейный пролёт»: силу (изгиб путей) размытый конус с редкими
# остановками не воспроизводит. Для стационарного состояния с энергией E потенциал точно равен вставке массы:
# (E − U)² − k² − m² = 0  ⇔  E² − k² − (m² + 2EU − U²) = 0. Уровень — неподвижная точка ω(E) = E. Это та же «энергия в полюсе»,
# что у векторной вершины пары (§17): заряд чувствует поле через (E − U).
def _fixed_point(omega, E0, tol=2e-4, it=6):
    w0 = omega(E0); E1 = w0; w1 = omega(E1)
    for _ in range(it):
        if abs(w1 - E1) < tol: break
        E2 = E1 - (w1 - E1) * (E1 - E0) / ((w1 - E1) - (w0 - E0)); E0, w0, E1 = E1, w1, E2; w1 = omega(E1)
    return E1

def energy_level(at, pauli=0.0, E0=None):
    """уровень 1s на мире at (Atom) методом вставки при энергии E; поле — только внутри мира (как есть)."""
    U = -AL / np.sqrt(at.r ** 2 + A * A); f = at.src(); e = np.array([1, 0])
    if not hasattr(at, 'idx'): at.idx = Fz.build_index(at.X, 2.0)
    def omega(E):
        b = -(at.mb2(at.M) + 2 * E * U - U * U); B = (b[:, None, None] * at.eye + pauli * 1j * at.dU[:, None, None] * at.sr) / at.rho
        phi = Fz.respond_fast(at.X, at.idx, at.tc2, at.alpha, at.coef, at.rg, at.Unuc, f, np.ascontiguousarray(B, dtype=np.complex128), BETA, SHAPE, use_phase=0) @ e
        return at.peak(phi)
    return _fixed_point(omega, E0 or 0.8 * at.M)

def energy_level_cont(M, R, H, pauli=0.0, E0=None, h=0.04):
    """то же в непрерывной теории тем же прибором; источник и вставки — только внутри шара R (как в дискретном мире)."""
    r = h * np.arange(1, int(max(4 * R, 40) / h) + 1); N = len(r); th = (r < R).astype(float); U = -AL / np.sqrt(r * r + A * A) * th
    prof = np.exp(-AL * M * r); ball = r < R; fp = pauli * 1j * AL * r / (r * r + A * A) ** 1.5 * th
    def lap(u, l):
        d = np.empty_like(u); d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2; d[0] = (u[1] - 2 * u[0]) / h ** 2; d[-1] = (-2 * u[-1] + u[-2]) / h ** 2
        return d - l * (l + 1) / r ** 2 * u
    def omega(E):
        m2 = (M * M + 2 * E * U - U * U) * th
        def F(y):
            us, up, vs, vp = y; return np.array([vs, vp, lap(us, 0) - m2 * us + fp * up, lap(up, 1) - m2 * up + fp * us])
        y = np.array([np.zeros(N, complex), np.zeros(N, complex), (r * prof * th).astype(complex), np.zeros(N, complex)]); ts = []; sig = []; t = 0.0; dt = 0.01
        while t < H - 0.2:
            k1 = F(y); k2 = F(y + dt / 2 * k1); k3 = F(y + dt / 2 * k2); k4 = F(y + dt * k3); y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
            ts.append(t); sig.append(np.sum((r * prof * y[0])[ball]) * h)
        ts = np.array(ts); sig = np.array(sig); s = (ts > 1.5) & (ts < H - 0.3); tt = ts[s]; w = np.sin(np.pi * (tt - tt.min()) / np.ptp(tt)) ** 2
        om = np.linspace(0.15, 1.3, 2301); S = np.abs(np.exp(-1j * np.outer(om, tt)) @ (w * sig[s])) ** 2; i = int(np.argmax(S)); a, b, c = S[i - 1], S[i], S[i + 1]
        return om[i] + 0.5 * (a - c) / (a - 2 * b + c) * (om[1] - om[0])
    return _fixed_point(omega, E0 or 0.8 * M)


# ---------------- один проход + гармоническая инверсия (dirac-doc §29) ----------------
def shell_channels(at, phi, nsh=6, dtb=0.5):
    """каналы — шаровые слои по r; среднее φ в слое и шаге времени × объём слоя (на кристалле — по числу событий: гребёнка)."""
    from . import hinv as Hv
    shells = np.linspace(0, at.R, nsh + 1); sh = np.digitize(at.r, shells) - 1; ed = np.arange(1.5, at.H - 0.3 + 1e-9, dtb); kb = np.digitize(at.t, ed) - 1; Y = []
    for c in range(nsh):
        ok = (sh == c) & (kb >= 0) & (kb < len(ed) - 1); cnt = np.bincount(kb[ok], minlength=len(ed) - 1)
        v = (np.bincount(kb[ok], phi[ok].real, minlength=len(ed) - 1) + 1j * np.bincount(kb[ok], phi[ok].imag, minlength=len(ed) - 1)) / np.maximum(cnt, 1)
        Y.append(v * (4 / 3 * np.pi * (shells[c + 1] ** 3 - shells[c] ** 3)))
    return np.array(Y), dtb

def level_hinv(at, phi):
    from . import hinv as Hv
    Y, dtb = shell_channels(at, phi); return Hv.level(Y, dtb, 0.05, at.M)[0]

def energy_level_hinv(at, pauli=0.0, E0=None):
    """уровень вставкой при энергии E; частота — гармонической инверсией (а не пиком спектра)."""
    U = -AL / np.sqrt(at.r ** 2 + A * A); f = at.src(); e = np.array([1, 0])
    if not hasattr(at, 'idx'): at.idx = Fz.build_index(at.X, 2.0)
    def omega(E):
        b = -(at.mb2(at.M) + 2 * E * U - U * U); B = (b[:, None, None] * at.eye + pauli * 1j * at.dU[:, None, None] * at.sr) / at.rho
        return level_hinv(at, Fz.respond_fast(at.X, at.idx, at.tc2, at.alpha, at.coef, at.rg, at.Unuc, f, np.ascontiguousarray(B, dtype=np.complex128), BETA, SHAPE, use_phase=0) @ e)
    return _fixed_point(omega, E0 or 0.8 * at.M, tol=3e-4)

def cont_level_hinv(M, R, H, pauli=0.0, trunc=True, h=0.04):
    """непрерывная теория (фаза — как положено: (i∂t − U)²), тот же прибор: слои, инверсия. trunc — источник и поле только в шаре R."""
    from . import hinv as Hv
    r = h * np.arange(1, int(max(4 * R, 40) / h) + 1); N = len(r); th = (r < R).astype(float) if trunc else np.ones(N)
    U = -AL / np.sqrt(r * r + A * A) * th; prof = np.exp(-AL * M * r); m2 = M * M * th; fp = pauli * 1j * AL * r / (r * r + A * A) ** 1.5 * th
    def lap(u, l):
        d = np.empty_like(u); d[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / h ** 2; d[0] = (u[1] - 2 * u[0]) / h ** 2; d[-1] = (-2 * u[-1] + u[-2]) / h ** 2
        return d - l * (l + 1) / r ** 2 * u
    def F(y):
        us, up, vs, vp = y
        return np.array([vs, vp, 2j * U * vs + U * U * us + lap(us, 0) - m2 * us + fp * up, 2j * U * vp + U * U * up + lap(up, 1) - m2 * up + fp * us])
    y = np.array([np.zeros(N, complex), np.zeros(N, complex), (r * prof * th).astype(complex), np.zeros(N, complex)]); t = 0.0; dt = 0.01
    shells = np.linspace(0, R, 7); out = []; ts = []
    while t < H - 0.2:
        k1 = F(y); k2 = F(y + dt / 2 * k1); k3 = F(y + dt / 2 * k2); k4 = F(y + dt * k3); y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        if abs(t / 0.5 - round(t / 0.5)) < 1e-6: ts.append(t); out.append([np.sum((r * y[0])[(r >= a) & (r < b)]) * h for a, b in zip(shells[:-1], shells[1:])])
    ts = np.array(ts); Y = np.array(out).T; s = ts > 1.5; return Hv.level(Y[:, s], 0.5, 0.05, M)[0]
