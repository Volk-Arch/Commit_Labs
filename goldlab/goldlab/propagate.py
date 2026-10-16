"""ПРОПАГАТОРЫ И ВАКУУМ.
* Джонстон 1+1: K = Φ(I − BΦ)⁻¹, Φ = ½·C (∘ фазы поля), B — вставки в узлах (−m²/ρ и сила поля).
* Световой «уголок» (1+1, точная связь поля при любой силе): фаза ∫U dt по пути y → (u_x, v_y) → x, вставка −iU′
  (абелево) или −(m² − F_Tx)/ρ (матрица, неабелево). Выведено в maxwell-doc §12, gauge-doc.
* Джонстон 3+1 по связям: Φ = (√ρ/(2π√6))·L ∘ U, прямая нить Вильсона на связи (ромб связи тонок).
* Вакуум SJ: W± — положительная/отрицательная части iΔ = i(K − K†). Частица — ветвь W₊ (вращается как e^{+iEt}).
* Фейнман: G_F = c·[W₊θ(t_j − t_i) + W₋θ(t_i − t_j)], c из K ≈ c(W₊ − W₋) на «вперёд».
Соглашение о матрицах: K[i, j] — амплитуда из i в j; отклик на источник f: φ = Kᵀ f."""
import math
import numpy as np
from scipy.linalg import eigh
from .causal import causal

def johnston11(X, rho, m, Phi=None, B=None):
    if Phi is None: Phi = 0.5 * causal(X)
    n = len(X); Bv = np.full(n, -m * m / rho) if B is None else B
    return np.linalg.solve((np.eye(n) - Phi * Bv[None, :]).T, Phi.T).T            # K = (I − ΦB)⁻¹Φ

def corner_total(P, ty, xy, tx, xx, side=1):
    """∫U dt по световому уголку для статического U(x), P′ = U. side = 1: y → (u_x, v_y) → x (сначала влево, потом вправо):
    P(x_y) + P(x_x) − 2P(x_c); side = −1 — зеркальный уголок через (u_y, v_x): 2P(x_c′) − P(x_y) − P(x_x)."""
    if side == 1:
        xc = 0.5 * ((ty + xy) - (tx - xx)); return P(xy) + P(xx) - 2 * P(xc)
    xc = 0.5 * ((tx + xx) - (ty - xy)); return 2 * P(xc) - P(xy) - P(xx)

def corner_dense(X, P, dU, m, rho):
    """Φ (фазы уголка) и вставки B для статического потенциала U (P — первообразная, dU — производная)."""
    t, x = X[:, 0], X[:, 1]; C = causal(X)
    tot = corner_total(P, t[:, None], x[:, None], t[None, :], x[None, :])
    return 0.5 * C * np.exp(1j * tot), -(m * m - 1j * dU(x)) / rho

def corner_respond(X, P, dU, m, rho, f, side=1, ins=1.0):
    """отклик φ = Kᵀ f прямой подстановкой по времени (без больших матриц; X отсортирован по t).
    Скаляр КГ: side = 1, ins = 1 (вставка −iU′). Компоненты Дирака (dirac-doc §12): ins = 0, side = ±1 — только масса."""
    t, x = X[:, 0], X[:, 1]; n = len(X); phi = np.zeros(n, complex); src = np.asarray(f, complex).copy()
    B = -(m * m - 1j * ins * dU(x)) / rho
    for j in range(n):
        idx = np.nonzero(t[j] - t[:j] > np.abs(x[j] - x[:j]) + 1e-12)[0]
        if len(idx): phi[j] = 0.5 * (np.exp(1j * corner_total(P, t[idx], x[idx], t[j], x[j], side)) * src[idx]).sum()
        src[j] = src[j] + B[j] * phi[j]
    return phi

def johnston31_respond(n, I, J, rho, m, f, phase=None):
    """3+1: отклик по связям (I, J отсортированы по J), φ_j = Σ_{i→j} coef·U·(f_i + bφ_i)."""
    order = np.argsort(J, kind='stable'); I, J = I[order], J[order]; ph = np.ones(len(I)) if phase is None else phase[order]
    starts = np.searchsorted(J, np.arange(n + 1)); coef = math.sqrt(rho) / (2 * math.pi * math.sqrt(6)); b = -m * m / rho
    phi = np.zeros(n, complex); src = np.asarray(f, complex).copy(); w = coef * ph
    for j in range(n):
        s0, s1 = starts[j], starts[j + 1]
        if s1 > s0: phi[j] = (w[s0:s1] * src[I[s0:s1]]).sum()
        src[j] = src[j] + b * phi[j]
    return phi

def wilson31(X, I, J, Ufun, npts=4):
    """фаза ∫U dt по прямой связи (Гаусс–Лежандр), U(r) — сферический потенциал."""
    gx, gw = np.polynomial.legendre.leggauss(npts); A, B = X[I], X[J]; tot = np.zeros(len(I))
    for xg, wg in zip(gx, gw):
        Q = A + 0.5 * (xg + 1) * (B - A); tot += 0.5 * wg * Ufun(np.linalg.norm(Q[:, 1:], axis=1))
    return np.exp(1j * tot * (B[:, 0] - A[:, 0]))

def sj(K):
    """W₊, W₋ (обе ≥ 0) из iΔ = i(K − K†)."""
    lam, V = eigh(1j * (K - K.conj().T))
    Wp = (V[:, lam > 0] * lam[lam > 0]) @ V[:, lam > 0].conj().T
    Wm = -(V[:, lam < 0] * lam[lam < 0]) @ V[:, lam < 0].conj().T
    return Wp, Wm, lam

def feynman(K, t, ts_src=None, f=None):
    """G_F = c·[W₊θ(t_j − t_i) + W₋θ(t_i − t_j)]; c — из K ≈ c(W₊ − W₋) на «вперёд» (≈ −i)."""
    Wp, Wm, _ = sj(K); fwd = t[None, :] > t[:, None]; bwd = t[None, :] < t[:, None]
    a = (K * fwd).ravel(); b = ((Wp - Wm) * fwd).ravel(); c = (b.conj() @ a) / (b.conj() @ b)
    resid = np.linalg.norm(a - c * b) / np.linalg.norm(a)
    return c * (Wp * fwd + Wm * bwd), c, resid

# ---------- поле общего вида U(t, x) (например, из порядка): фаза — квадратура вдоль уголка ----------
def corner_phase_general(Ufun, ty, xy, tx, xx, side=1, k=6):
    """∫U dt вдоль светового уголка y → c → x для U(t, x) общего вида (Гаусс–Лежандр по k точек на каждом луче).
    side = 1: c = (u_x, v_y) — сначала влево; side = −1: c = (u_y, v_x)."""
    if side == 1: tc = 0.5 * ((ty + xy) + (tx - xx)); xc = 0.5 * ((ty + xy) - (tx - xx))
    else: tc = 0.5 * ((tx + xx) + (ty - xy)); xc = 0.5 * ((tx + xx) - (ty - xy))
    gx, gw = np.polynomial.legendre.leggauss(k); tot = 0.0
    for (a0, b0, a1, b1) in ((ty, xy, tc, xc), (tc, xc, tx, xx)):
        for g, wg in zip(gx, gw):
            q = 0.5 * (g + 1); tot = tot + 0.5 * wg * Ufun(a0 + q * (a1 - a0), b0 + q * (b1 - b0)) * (a1 - a0)
    return tot

def corner_respond_general(X, Ufun, dUev, m, rho, f, side=1, ins=1.0, k=6):
    """как corner_respond, но поле U(t, x) любое (вектор-функция), вставка −i·ins·U′ по значениям dUev в событиях."""
    t, x = X[:, 0], X[:, 1]; n = len(X); phi = np.zeros(n, complex); src = np.asarray(f, complex).copy()
    B = -(m * m - 1j * ins * np.asarray(dUev)) / rho
    for j in range(n):
        idx = np.nonzero(t[j] - t[:j] > np.abs(x[j] - x[:j]) + 1e-12)[0]
        if len(idx): phi[j] = 0.5 * (np.exp(1j * corner_phase_general(Ufun, t[idx], x[idx], t[j], x[j], side, k)) * src[idx]).sum()
        src[j] = src[j] + B[j] * phi[j]
    return phi

class GridField:
    """поле на равномерной сетке (t, x) с билинейной интерполяцией (быстро, векторно)."""
    def __init__(self, tg, xg, V):
        self.t0, self.dt, self.x0, self.dx, self.V = tg[0], tg[1] - tg[0], xg[0], xg[1] - xg[0], V
    def __call__(self, t, x):
        a = np.clip((t - self.t0) / self.dt, 0, self.V.shape[0] - 1.000001); b = np.clip((x - self.x0) / self.dx, 0, self.V.shape[1] - 1.000001)
        i = a.astype(int); j = b.astype(int); fa = a - i; fb = b - j; V = self.V
        return (V[i, j] * (1 - fa) * (1 - fb) + V[i + 1, j] * fa * (1 - fb) + V[i, j + 1] * (1 - fa) * fb + V[i + 1, j + 1] * fa * fb)

def johnston31_respond_spinor(n, I, J, rho, m, f, phase=None, Bx=None):
    """3+1, двухкомпонентное поле (спинор Вейля): отклик по связям, φ_j = Σ_{i→j} coef·U·(f_i + B_i φ_i), B_i — матрица 2×2
    (масса −m²/ρ на диагонали + член Паули ±i(σ·∇U)/ρ). f — (n, 2)."""
    order = np.argsort(J, kind='stable'); I, J = I[order], J[order]; ph = np.ones(len(I)) if phase is None else phase[order]
    starts = np.searchsorted(J, np.arange(n + 1)); coef = math.sqrt(rho) / (2 * math.pi * math.sqrt(6))
    B = np.broadcast_to((-m * m / rho) * np.eye(2), (n, 2, 2)) if Bx is None else Bx
    phi = np.zeros((n, 2), complex); src = np.asarray(f, complex).copy(); w = coef * ph
    for j in range(n):
        s0, s1 = starts[j], starts[j + 1]
        if s1 > s0: phi[j] = w[s0:s1] @ src[I[s0:s1]]
        src[j] = src[j] + B[j] @ phi[j]
    return phi
