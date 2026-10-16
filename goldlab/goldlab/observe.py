"""НАБЛЮДАЕМЫЕ И ТЕОРИЯ ДЛЯ СРАВНЕНИЯ.
* spectrum / peaks — спектр сигнала по событиям с окном Тьюки (ветвь частицы e^{+iEt}: вес e^{−iωt}).
* kg_levels — уровни Клейна–Гордона u″ = (m² − (E − U)²)u (стрельба, чётные/нечётные); salpeter_levels — пара.
* polar light (полярное событие, как в песочнице) и пара из распада накачки для опыта Белла; CHSH по Хородецким."""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def tukey(tt, frac=0.15):
    u = (tt - tt.min()) / max(np.ptp(tt), 1e-12)
    return np.where(u < frac, np.sin(np.pi * u / (2 * frac)) ** 2, np.where(u > 1 - frac, np.sin(np.pi * (1 - u) / (2 * frac)) ** 2, 1.0))

def spectrum(tt, sig, om, window='tukey'):
    w = tukey(tt) if window == 'tukey' else np.sin(np.pi * (tt - tt.min()) / np.ptp(tt)) ** 2
    E = np.exp(-1j * np.outer(om, tt)); return np.abs(E @ (w * sig)) ** 2

def peaks(om, S, thr=0.05):
    out = []
    for i in range(1, len(om) - 1):
        if S[i] > S[i - 1] and S[i] > S[i + 1] and S[i] > thr * S.max():
            a, b, c = S[i - 1], S[i], S[i + 1]; out.append((float(om[i] + 0.5 * (a - c) / (a - 2 * b + c) * (om[1] - om[0])), float(S[i] / S.max())))
    return out

def kg_levels(U, m=1.0, parity='odd', Emin=0.4, Emax=0.9995, X=60.0, n=600):
    def tail(E):
        y0 = [0.0, 1.0] if parity == 'odd' else [1.0, 0.0]
        sol = solve_ivp(lambda s, y: [y[1], (m * m - (E - U(s)) ** 2) * y[0]], (0, X), y0, rtol=1e-10, atol=1e-14)
        return sol.y[0, -1] / (abs(sol.y[0]).max() + 1e-300)
    Es = np.linspace(Emin, Emax, n); v = [tail(e) for e in Es]
    return [float(brentq(tail, Es[i], Es[i + 1])) for i in range(len(Es) - 1) if np.sign(v[i]) != np.sign(v[i + 1])]

def kg_levels_linear(F, m=1.0, parity='odd', Emax=2.6, n=1500):
    """Уровни КГ для U = F|x| (spectro1d.py). Линейный векторный потенциал не запирает навсегда (парадокс Клейна):
    уровни — квазисвязанные, стрельба до середины «щели Клейна» между точками поворота (E − m)/F и (E + m)/F."""
    def tail(E):
        x1, x2 = (E - m) / F, (E + m) / F; xm = min(0.5 * (x1 + x2), x1 + 7)
        y0 = [0.0, 1.0] if parity == 'odd' else [1.0, 0.0]
        sol = solve_ivp(lambda s, y: [y[1], (m * m - (E - F * s) ** 2) * y[0]], (0, xm), y0, rtol=1e-10, atol=1e-12)
        return sol.y[0, -1]
    Es = np.linspace(m + 1e-4, Emax, n); v = [tail(e) for e in Es]
    return [float(brentq(tail, Es[i], Es[i + 1])) for i in range(len(Es) - 1) if np.sign(v[i]) != np.sign(v[i + 1])]

def salpeter_levels(m1, m2, F, parity='odd', nlev=3):
    N, L = 1024, 60.0; rg = (np.arange(N) - N // 2) * (L / N); p = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    col = np.real(np.fft.ifft(np.sqrt(p * p + m1 * m1) + np.sqrt(p * p + m2 * m2)))
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N; E, V = np.linalg.eigh(col[idx] + np.diag(F * np.abs(rg))); out = []
    for k in range(len(E)):
        v = V[:, k]; sym = np.dot(v[1:], v[1:][::-1]) / np.dot(v[1:], v[1:])
        if (sym > 0) == (parity == 'even'): out.append(float(E[k] - m1 - m2))
        if len(out) == nlev: break
    return out

def centers(t, x, psi, times, half=0.5, win=None):
    out = []
    for tt in times:
        S = np.abs(t - tt) < half
        if win is not None: S &= np.abs(x - win(tt)) < 4.5
        p = np.abs(psi[S]) ** 2; out.append(float((p * x[S]).sum() / p.sum()) if p.sum() > 0 else float('nan'))
    return np.array(out)

# ---------- полярный свет (полярное событие, песочница) ----------
class PolarLight:
    def __init__(self, X, la, lb, m0=2.0, kernel='mink'):
        self.X, self.la, self.lb = X, la, lb; n = len(X); E = len(la); self.E = E
        self.inL = [[] for _ in range(n)]; self.outL = [[] for _ in range(n)]
        for e in range(E): self.outL[la[e]].append(e); self.inL[lb[e]].append(e)
        dv = X[lb] - X[la]; self.mats = []
        for k in range(n):
            li, lo = self.inL[k], self.outL[k]
            if not li or not lo: self.mats.append(None); continue
            o, i = dv[lo], dv[li]
            if kernel == 'mink': K = np.exp(-(np.outer(o[:, 0], i[:, 0]) - np.outer(o[:, 1], i[:, 1])) / m0)
            else:
                tau = lambda a: np.sqrt(np.maximum(a[..., 0] ** 2 - a[..., 1] ** 2, 0)); s = o[:, None, :] + i[None, :, :]
                K = np.exp(-(tau(s) - tau(o)[:, None] - tau(i)[None, :]) / m0)
            lam, Q = np.linalg.eigh(K.T @ K); keep = lam > 1e-11 * lam.max()
            S = (Q[:, keep] / np.sqrt(lam[keep])) @ Q[:, keep].T; self.mats.append(K @ S)
    def run(self, src):
        amp = np.zeros(self.E); shadow = 0.0
        for k in range(len(self.X)):
            li, lo = self.inL[k], self.outL[k]
            if not lo: continue
            vin = amp[li] if li else np.zeros(0)
            vout = self.mats[k] @ vin if li else np.zeros(len(lo))
            shadow += (vin ** 2).sum() - (vout ** 2).sum()
            if k == src: vout = vout + 1 / math.sqrt(len(lo))
            amp[lo] = vout
        return amp, shadow
    def arrival(self, amp, k): return float(sum(amp[e] for e in self.inL[k]))

def chsh_max(psi):
    """наибольшее S для вещественной двухкубитной ψ = (ψ00, ψ01, ψ10, ψ11): 2√(λ₁ + λ₂), λ — собств. числа TᵀT."""
    a, b, c, d = psi
    T = np.array([[2 * (a * d + b * c), 0, a * a - b * b + c * c - d * d], [0, -2 * (a * d - b * c), 0], [2 * (a * c - b * d), 0, a * a - b * b - c * c + d * d]])
    lam = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]; return float(2 * math.sqrt(max(lam[0] + lam[1], 0)))

def correlation(psi, al, be):
    ra, rb = math.radians(al), math.radians(be)
    va = [(math.cos(ra), math.sin(ra)), (-math.sin(ra), math.cos(ra))]; vb = [(math.cos(rb), math.sin(rb)), (-math.sin(rb), math.cos(rb))]; e = 0.0
    for xx in range(2):
        for yy in range(2):
            amp = va[xx][0] * (vb[yy][0] * psi[0] + vb[yy][1] * psi[1]) + va[xx][1] * (vb[yy][0] * psi[2] + vb[yy][1] * psi[3]); e += (1 if xx == yy else -1) * amp * amp
    return e
