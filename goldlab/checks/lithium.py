"""«ЛИТИЙ» 1+1: САМОСОГЛАСОВАННОЕ ПОЛЕ НА КРИСТАЛЛЕ (метод Хартри — так начинается химия).
Ядро +3, три электрона; всё взаимодействие — линейное поле из порядка (`field11.py`): ядро V = (3e²/2)|x|, между электронами
−(e²/2)|Δx|. Каждый электрон движется в среднем поле: V_эфф = V_ядра + (N−1)/N · V_H[ρ], V_H(x) = −(e²/2)∫|x − y|ρ(y)dy.
На кристалле: уровни — спектроскопией блуждания в V_эфф, орбитали — Фурье отклика на частоте уровня (по полоскам x),
ρ = Σ|ψ_k|² по трём занятым (Паули: по одному на уровень). Итерации до согласования. Опора — тот же метод в непрерывном Дираке."""
import sys, os, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from walk4 import Walk
from goldlab import worlds as Wd, observe as Ob
L = float(os.environ.get('L', 30)); e2 = 0.1; Zn = 3.0; m = 1.0; Ne = 3; NIT = int(os.environ.get('NIT', 6))
w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Walk(w)
xb = np.linspace(-40, 40, 641); dxb = xb[1] - xb[0]                      # сетка для плотности и потенциала
def VH_of(rho): return -(e2 / 2) * (np.abs(xb[:, None] - xb[None, :]) @ rho) * dxb
def make_int(Vtab):
    gx, gw = np.polynomial.legendre.leggauss(6)
    def f(t0, x0, t1, x1):
        tot = 0
        for g, wg in zip(gx, gw):
            q = 0.5 * (g + 1); xs = x0 + q * (x1 - x0); v = np.interp(xs, xb, Vtab)
            tot = tot + 0.5 * wg * v
        return tot * (t1 - t0)
    return f
lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
sel = (t > ts + 2) & (t < L - 4) & (np.abs(x) < 12); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
win = np.sin(np.pi * (t[sel] - t[sel].min()) / np.ptp(t[sel])) ** 2
om = np.linspace(0.9, 2.2, 2601)
def crystal_levels(Vtab):
    ph = W.phases(make_int(Vtab)); (Ec, _), shapes = cont_levels(Vtab, shapes=True); Es, dens = [], []
    for k in range(3):
        g = np.interp(x, xg, shapes[k]); gs = np.where(src, g, 0); aR, aL, _ = W.run(m, gs, gs, ph); psi = (aR + aL)[sel]
        s = np.zeros(len(tu), complex); np.add.at(s, inv, (g[sel] * psi)); S = Ob.spectrum(tu, s, -om); S /= S.max()
        pk = Ob.peaks(om, S, 0.1); E = min(pk, key=lambda z: abs(z[0] - Ec[k]))[0] if pk else np.nan; Es.append(E)
        b = np.clip(np.round((x[sel] - xb[0]) / dxb).astype(int), 0, len(xb) - 1); prof = np.zeros(len(xb), complex); cnt = np.zeros(len(xb))
        np.add.at(prof, b, win * psi * np.exp(1j * E * t[sel])); np.add.at(cnt, b, win)
        d = np.abs(prof / np.maximum(cnt, 1e-9)) ** 2; d = np.convolve(d, np.ones(5) / 5, 'same'); dens.append(d / (d.sum() * dxb))
    return tuple(Es), dens
# непрерывный Дирак — тот же метод
N, Lg = 384, 48.0; xg = (np.arange(N) - N / 2) * (Lg / N); hg = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=hg)
Dg = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0))
def cont_levels(Vtab, shapes=False):
    Vg = np.interp(xg, xb, Vtab, left=np.nan, right=np.nan); Vg = np.where(np.isnan(Vg), (Zn * e2 / 2) * np.abs(xg) + (Vtab[0] - (Zn * e2 / 2) * abs(xb[0])), Vg)
    H = np.block([[-1j * W.c * Dg + np.diag(Vg), m * np.eye(N)], [m * np.eye(N), 1j * W.c * Dg + np.diag(Vg)]]); ev, vec = np.linalg.eigh(H)
    loc = lambda i: (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) @ xg ** 2
    idx = [i for i in np.argsort(ev) if ev[i] > 0 and loc(i) < 40][:3]
    dens = [np.interp(xb, xg, (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) / hg) for i in idx]
    if shapes:
        sh = []
        for i in idx:
            f = vec[:N, i] + vec[N:, i]; f = np.real(f * np.exp(-1j * np.angle(f[np.argmax(np.abs(f))]))); sh.append(f / np.abs(f).max())
        return (tuple(ev[idx]), dens), sh
    return tuple(ev[idx]), dens
Vnuc = (Zn * e2 / 2) * np.abs(xb)
for name, solver in (('кристалл', crystal_levels), ('непрерывный', cont_levels)):
    Veff = Vnuc.copy(); hist = []
    for it in range(NIT):
        (E1, E2, E3), dens = solver(Veff); rho = sum(dens); VH = VH_of(rho) * (Ne - 1) / Ne
        Etot = E1 + E2 + E3 - 0.5 * (rho @ VH) * dxb
        hist.append((E1, E2, E3, Etot)); Veff = 0.5 * Veff + 0.5 * (Vnuc + VH)          # смешивание для устойчивости
        if it == 0: E0 = E1 + E2 + E3
    print(f'{name} (c = {W.c:.3f}): без отталкивания E₁ + E₂ + E₃ = {E0:.4f}; итерации, полная энергия: ' + ' → '.join(f'{h[3]:.4f}' for h in hist) +
          f';  уровни в конце {hist[-1][0]:.4f}, {hist[-1][1]:.4f}, {hist[-1][2]:.4f}', flush=True)
