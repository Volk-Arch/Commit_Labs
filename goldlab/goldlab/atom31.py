"""Атом 3+1 на золотом кристалле: один электрон со спином (спинор Вейля = КГ + член Паули), ансамбль сдвигов фазона,
два электрона (Паули, «гелий» в первом порядке). Опыты: spin31, ens31, pauli31 (dirac-doc §20–§22)."""
import math, numpy as np
import scipy.linalg as sl, scipy.optimize as so
from . import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob
from .core import r_for_density

AL, A, M = 0.45, 0.5, 1.0                         # ядро −α/√(r² + a²), масса
SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]]); SZ = np.array([[1, 0], [0, -1]], complex)
U_NUC = lambda q: -AL / np.sqrt(q * q + A * A)

class Atom:
    """мир-цилиндр 3+1 с ядром в начале координат; связи по экономному правилу; фаза — прямая нить Вильсона."""
    def __init__(self, rho, H=12.0, Rc=5.0, shift=None, rho_eff=None):
        self.rho, self.H = rho, H; self.w = Wd.gold31(rho, 'cylinder', H=H, Rc=Rc, Rs=3.0, shift=shift); X = self.X = self.w.X; self.n = self.w.n
        self.I, self.J = Cs.links_sparse(X, rho, 1.6); self.re = rho if rho_eff is None else rho_eff
        self.r = np.linalg.norm(X[:, 1:], axis=1); self.t = X[:, 0]; self.prof = np.exp(-AL * self.r); self.src0 = self.t < 0.45
        self.sel = (self.t > 1.5) & (self.t < H - 0.3); self.ph = Pr.wilson31(X, self.I, self.J, U_NUC)
        rh = np.where(self.r[:, None] > 1e-9, X[:, 1:] / np.maximum(self.r, 1e-9)[:, None], 0)
        self.sr = rh[:, 0, None, None] * SX + rh[:, 1, None, None] * SY + rh[:, 2, None, None] * SZ
        self.dU = AL * self.r / (self.r ** 2 + A * A) ** 1.5; self.eye = np.broadcast_to(np.eye(2), (self.n, 2, 2))
    def B(self, m2=M * M, extra=None, pauli=0.0):
        """вставка на событиях: −(m² − extra)/ρ + pauli·i(σ·∇U)/ρ (pauli = ±1 — две киральности, 0 — КГ)."""
        b = -m2 * self.eye if extra is None else (-m2 + extra)[:, None, None] * self.eye
        return (b + pauli * 1j * self.dU[:, None, None] * self.sr) / self.re
    def respond(self, Bx, f=None, spin=0, phase=None):
        if f is None: f = np.where(self.src0[:, None], self.prof[:, None] * np.eye(2)[spin][None, :], 0)
        return Pr.johnston31_respond_spinor(self.n, self.I, self.J, self.re, M, f, self.ph if phase is None else phase, Bx)
    def peak(self, sig, om=None):
        om = np.linspace(0.5, 1.3, 1601) if om is None else om
        S = Ob.spectrum(self.t[self.sel], self.prof[self.sel] * sig[self.sel], om, 'hann'); pk = Ob.peaks(om, S, 0.0); return pk[int(np.argmax([h for _, h in pk]))][0]
    def growth(self, sig):
        ed = np.arange(0, self.H + 1e-9, 1.0); k = np.digitize(self.t, ed) - 1; ok = (k >= 0) & (k < len(ed) - 1)
        amp = np.sqrt(np.bincount(k[ok], np.abs(self.prof * sig)[ok] ** 2, minlength=len(ed) - 1) / np.maximum(np.bincount(k[ok], minlength=len(ed) - 1), 1))
        tc = 0.5 * (ed[1:] + ed[:-1]); s = (tc > 2) & (tc < self.H - 1); return float(np.polyfit(tc[s], np.log(amp[s] + 1e-300), 1)[0])
    def measure(self):
        """уровень 1s (КГ), темп роста, наклон dE/dm², второй порядок (вставка z·e^{−r/2}), спиновый сдвиг Дирак − КГ."""
        e = np.array([1, 0]); s1 = self.respond(self.B()) @ e; E1 = self.peak(s1); E2 = self.peak(self.respond(self.B(0.81)) @ e)
        g = self.X[:, 3] * np.exp(-self.r / 2.0); c = 0.12 / np.abs(g).max(); Es = self.peak(self.respond(self.B(extra=c * g)) @ e)
        Ed = self.peak(self.respond(self.B(pauli=1.0)) @ e)
        return dict(E=E1, growth=self.growth(s1), slope=(E1 - E2) / 0.19, stark=Es - E1, spin=Ed - E1)

def random_shift(rho, rng):
    v = rng.normal(size=4); return tuple(v * r_for_density(rho, 4) * rng.random() ** 0.25 / np.linalg.norm(v))

# ---- непрерывные эталоны ----
CONT_SAME_INSTRUMENT = {'E': 0.89899, 'slope': 0.511, 'stark': -0.00259, 'spin': +0.00682, 'J': 0.06901}   # окно 12, шар 5 (instr_cont.py)
def cloud(e2=0.225, N=1500, R=50.0):
    """КГ 1s в поле ядра; заряд облака (E − U)|φ|²; его потенциал e²·[Q(r)/r + ∫_r σ dr′/r′]; J — сдвиг уровня от облака."""
    h = R / N; r = h * np.arange(1, N + 1); U = U_NUC(r)
    D = (np.diag(np.full(N, 2.0)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / h ** 2
    lam = lambda om, Ut: sl.eigh(D + np.diag(M * M - (om - Ut) ** 2), eigvals_only=True, subset_by_index=[0, 0])[0]
    level = lambda Ut: so.brentq(lambda o: lam(o, Ut), 0.5, 0.99999)
    E = level(U); _, v = sl.eigh(D + np.diag(M * M - (E - U) ** 2), subset_by_index=[0, 0]); u = v[:, 0]
    sig = (E - U) * u * u; sig /= np.sum(sig) * h; Q = np.cumsum(sig) * h; out = np.cumsum((sig / r)[::-1])[::-1] * h
    Uc = e2 * (Q / r + out); return r, sig / (4 * np.pi * r * r), Uc, E, level(U + Uc) - E

def field_from_order(rr, sig_r, rho=2.0, e2=0.225, S=10.0, radii=(0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0), per=12, seed=3):
    """потенциал статического облака ИЗ ПОРЯДКА: U(x) = 4π·e²·(√ρ/(2π√6))/ρ · Σ_{связи i → x} σ(r_i) (то же правило, что Кулон из связей)."""
    G = Wd.points_in_box31([(-(S + 4.0), 0.6), (-S, S), (-S, S), (-S, S)], Wd.r_for_density31(rho)); rad = np.linalg.norm(G[:, 1:], axis=1)
    sg = np.interp(rad, rr, sig_r, right=0.0); tc2 = (1.6 * rho ** -0.25) ** 2; coef = 4 * math.pi * e2 * math.sqrt(rho) / (2 * math.pi * math.sqrt(6)) / rho
    rng = np.random.default_rng(seed); rows = []
    for R in radii:
        idx = np.nonzero((np.abs(G[:, 0]) < 0.3) & (np.abs(rad - R) < 0.25))[0]; vals = []
        for j in rng.choice(idx, min(per, len(idx)), replace=False):
            d = G[j] - G; tau2 = d[:, 0] ** 2 - (d[:, 1:] ** 2).sum(1); cnd = np.nonzero((d[:, 0] > 0) & (tau2 > 1e-12) & (tau2 < tc2))[0]
            Q = G[cnd]; e = Q[None, :, :] - Q[:, None, :]; prec = (e[..., 0] > 0) & (e[..., 0] ** 2 - (e[..., 1:] ** 2).sum(-1) > 1e-12)
            vals.append(coef * sg[cnd[~prec.any(1)]].sum())
        rows.append((R, float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))))
    return np.array(rows)
