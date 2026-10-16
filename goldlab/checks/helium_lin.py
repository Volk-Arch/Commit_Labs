"""«ГЕЛИЙ» 1+1, ПОЛЕ ИЗ САМИХ ЗАРЯДОВ: ядро (заряд +Z) и электроны (−1) действуют друг на друга линейным потенциалом,
который в 1+1 даёт сам порядок (½ за отношение, `field11.py`: наклон −0.49 против −0.5): ядро V = (Ze²/2)|x|, электроны U = −(e²/2)|Δx|.
Прежнее описание: два фермиона со спином в поле ядра −α/√(x² + a²) с ВЗАИМНЫМ отталкиванием U = β/√(Δx² + b²)
(модельный выбор — мягкий Кулон, как у ядра; взято руками). Двухчастичное блуждание по нитям: амплитуда A[e₁, e₂] по парам
«рёбер» (событие + направление). Монета и фаза ядра — для каждой частицы; фаза отталкивания e^{−iU·Δt} — на каждом промежутке
между слоями времени (система кристалла — нити и так в ней). Старт антисимметричный (фермионы). Чтение — на каждом слое.
Сравнение: β = 0 (E₁ + E₂ на том же мире) и первый порядок E₁ + E₂ + J − K по непрерывным орбиталям Дирака."""
import sys, os, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from walk4 import Walk, soft_coulomb_int
from goldlab import worlds as Wd, observe as Ob
L = float(os.environ.get('L', 24)); e2 = float(os.environ.get('E2', 0.1)); Zn = 2.0; alpha = Zn * e2 / 2; beta = e2 / 2; m = 1.0; T0 = time.time()
w = Wd.gold11(4.0, L); t, x, n = w.t, w.x, w.n; W = Walk(w)
def lin_int(t0, x0, t1, x1, k=8):
    gx, gw = np.polynomial.legendre.leggauss(k); tot = 0
    for g, wg in zip(gx, gw):
        q = 0.5 * (g + 1); tot = tot + 0.5 * wg * alpha * np.abs(x0 + q * (x1 - x0))
    return tot * (t1 - t0)
phR, phL = W.phases(lin_int)
E = 2 * n; nxt = np.empty(E, int); nxt[0::2] = W.R; nxt[1::2] = W.L
pR = -np.ones(n, int); pR[W.R[W.R >= 0]] = np.nonzero(W.R >= 0)[0]; pL = -np.ones(n, int); pL[W.L[W.L >= 0]] = np.nonzero(W.L >= 0)[0]
cz, sz = np.cos(m * W.dtb), np.sin(m * W.dtb); ph = np.empty(E, complex); ph[0::2] = phR; ph[1::2] = phL
tS = np.repeat(t, 2); xS = np.repeat(x, 2); tE = np.where(nxt >= 0, t[np.maximum(nxt, 0)], np.inf); xE = np.where(nxt >= 0, x[np.maximum(nxt, 0)], 0)
lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]
# непрерывная опора (считается заранее: орбитали нужны и как фильтры чтения — частоту они не задают)
N, Lg = 512, 64.0; xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
D = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)); V = np.diag(alpha * np.abs(xg))
Hc = np.block([[-1j * W.c * D + V, m * np.eye(N)], [m * np.eye(N), 1j * W.c * D + V]]); ev, vec = np.linalg.eigh(Hc)
loc = lambda i: (np.abs(vec[:N, i]) ** 2 + np.abs(vec[N:, i]) ** 2) @ xg ** 2
b_idx = [i for i in np.argsort(ev) if ev[i] > 0 and loc(i) < 30][:2]; E1, E2 = ev[b_idx]; p1, p2 = vec[:, b_idx[0]], vec[:, b_idx[1]]
dens = lambda u, v: (np.conj(u[:N]) * v[:N] + np.conj(u[N:]) * v[N:]) / h
Ux = -beta * np.abs(xg[:, None] - xg[None, :])
J = np.real(dens(p1, p1) @ Ux @ dens(p2, p2)) * h * h; K = np.real(dens(p1, p2) @ Ux @ dens(p2, p1)) * h * h
def orbf(p):
    f = np.real((p[:N] + p[N:]) * np.exp(-1j * np.angle((p[:N] + p[N:])[np.argmax(np.abs(p[:N] + p[N:]))]))); return lambda xx: np.interp(xx, xg, f)
he, ho = orbf(p1), orbf(p2)

def run(beta, sgn=-1):
    A = np.zeros((E, E), np.complex64); act = []; sF, sEE, tt = [], [], []
    for k, tl in enumerate(lv):
        Z = np.nonzero(np.abs(t - tl) < 1e-9)[0]
        if tl > ts + 1.5 and tl < L - 4:                                   # чтение: обе частицы прибыли на этот слой
            inR = np.where(pR[Z] >= 0, 2 * pR[Z], -1); inL = np.where(pL[Z] >= 0, 2 * pL[Z] + 1, -1)
            amp = np.zeros((len(Z), len(Z)), complex)
            for i1 in (inR, inL):
                for i2 in (inR, inL):
                    ok1, ok2 = i1 >= 0, i2 >= 0; blk = np.zeros((len(Z), len(Z)), complex)
                    blk[np.ix_(ok1, ok2)] = A[np.ix_(i1[ok1], i2[ok2])]; amp += blk
            xz = x[Z]; PF = np.outer(he(xz), ho(xz)) + sgn * np.outer(ho(xz), he(xz)); PE = np.outer(he(xz), he(xz))
            sF.append((PF * amp).sum()); sEE.append((PE * amp).sum()); tt.append(tl)
        # шаг монеты: частица 1 (строки), затем 2 (столбцы)
        for z in Z:
            eR, eL = (2 * pR[z] if pR[z] >= 0 else -1), (2 * pL[z] + 1 if pL[z] >= 0 else -1)
            for axis in (0, 1):
                M = A if axis == 0 else A.T
                vR = M[eR].copy() if eR >= 0 else 0; vL = M[eL].copy() if eL >= 0 else 0
                if eR >= 0: M[eR] = 0
                if eL >= 0: M[eL] = 0
                M[2 * z] = (cz[z] * vR - 1j * sz[z] * vL) * ph[2 * z] if nxt[2 * z] >= 0 else 0
                M[2 * z + 1] = (-1j * sz[z] * vR + cz[z] * vL) * ph[2 * z + 1] if nxt[2 * z + 1] >= 0 else 0
        if abs(tl - ts) < 1e-9:                                            # старт: фермионы из x = −1.5 и +1.5 (ветвь частицы)
            f1 = np.exp(-(x[Z] + 2.0) ** 2 / 3); f2 = np.exp(-(x[Z] - 2.0) ** 2 / 3); S = np.outer(f1, f2) + sgn * np.outer(f2, f1)
            for d1 in (0, 1):
                for d2 in (0, 1):
                    e1 = 2 * Z + d1; e2 = 2 * Z + d2; A[np.ix_(e1, e2)] += (0.5 * S * np.outer(ph[e1], ph[e2])).astype(np.complex64)
        # отталкивание на промежутке до следующего слоя
        act = [e for e in act if tE[e] > tl + 1e-9] + [e for z in Z for e in (2 * z, 2 * z + 1) if nxt[e] >= 0]
        if beta and k + 1 < len(lv) and act:
            t1 = lv[k + 1]; tm = 0.5 * (tl + t1); ac = np.array(act)
            xm = xS[ac] + (xE[ac] - xS[ac]) * (tm - tS[ac]) / (tE[ac] - tS[ac])
            Uphase = np.exp(-1j * (-beta) * np.abs(xm[:, None] - xm[None, :]) * (t1 - tl)).astype(np.complex64)   # одноимённые: −(e²/2)|Δx|
            A[np.ix_(ac, ac)] *= Uphase
    return np.array(tt), np.array(sF), np.array(sEE)
om = np.linspace(-3.6, -1.8, 3601)
def pk(tt, s):
    S = Ob.spectrum(tt, s, om); S /= S.max(); return [(-e, q) for e, q in Ob.peaks(om, S, 0.1)]
print(f'мир {L}: событий {n}, рёбер {E} (массив пар {E * E * 8 / 1e9:.2f} ГБ); c = {W.c:.3f}; e² = {e2}, Z = {Zn}: ядро {alpha:.3f}|x|, электроны −{beta:.3f}|Δx|', flush=True)
for b_, sg in ((0.0, -1), (beta, -1)):
    tt, sF, sEE = run(b_, sg); print('   ' + ('ФЕРМИОНЫ' if sg < 0 else 'БОЗОНЫ'), end=' ')
    print(f'β = {b_}: фермионы (нижняя + следующая): энергии пары ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in pk(tt, sF)) +
          f';  «обе внизу»: |амплитуда| ≤ {np.abs(sEE).max():.1e} (относит. {np.abs(sEE).max() / np.abs(sF).max():.1e}); {time.time() - T0:.0f} с', flush=True)
print(f'непрерывный Дирак (c = {W.c:.3f}), V = {alpha:.3f}|x|: E₁ = {E1:.4f}, E₂ = {E2:.4f}, E₁ + E₂ = {E1 + E2:.4f}; первый порядок U = −{beta:.3f}|Δx|: J = {J:+.4f}, K = {K:+.4f} ⇒ фермионы {E1 + E2 + J - K:.4f} (сдвиг {J - K:+.4f})', flush=True)
