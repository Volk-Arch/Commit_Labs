"""ПРОВЕРКА ПАРЫ: откуда 0.422 (§21) против 0.389 (каркас). Гипотеза: в twopropF.py множитель c брался подгонкой по
спектру (|c| ≈ 1.03–1.04), а точное тождество K_вперёд = −i(W₊ − W₋) даёт |c| = 1. Варианты: C=fit|exact, MEAS=strip|level."""
import os, sys, math, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, causal as Cs
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres
L = float(os.environ.get('L', 20)); CM = os.environ.get('C', 'exact'); MEAS = os.environ.get('MEAS', 'strip'); par = os.environ.get('PAR', 'odd')
w = Wd.gold11(4.0, L); t, x, n = w.t, w.x, w.n; rho, dl, F = 4.0, float(os.environ.get('DL', 0.3)), 0.08; m1, m2 = float(os.environ.get('M1', 1.0)), float(os.environ.get('M2', 1.0)); m = m1
I1, I2 = np.nonzero((np.abs(t[:, None] - t[None, :]) < dl) & (np.abs(x[:, None] - x[None, :]) < 10.0)); k = I1 != I2; I1, I2 = I1[k], I2[k]
tb = 0.5 * (t[I1] + t[I2]); o = np.argsort(tb, kind='stable'); I1, I2, tb = I1[o], I2[o], tb[o]; M = len(I1); ts = -L + 7
fwd = t[None, :] > t[:, None]; bwd = t[None, :] < t[:, None]
def GF(m):
    K = Pr.johnston11(w.X, rho, m); Wp, Wm, _ = Pr.sj(K)
    if CM == 'fit':                                    # как в twopropF.py: c из ветви частицы по спектру отклика
        f = np.where(np.abs(t - ts) < 0.45, np.exp(-x * x / 8), 0); a = K.T @ f; b = (Wp * fwd).T @ f
        sel = (t > ts + 3) & (np.abs(x) < 5); om = np.linspace(m - 0.3, m + 0.6, 181)
        br = lambda y: np.array([(y[sel] * np.exp(-1j * q * t[sel])).sum() for q in om]); A_, B_ = br(a), br(b); c = (np.conj(B_) @ A_) / (np.conj(B_) @ B_)
    else:
        c = -1j
    print(f'm = {m}: c = {c.real:+.4f}{c.imag:+.4f}i (|c| = {abs(c):.4f})', flush=True)
    return c * (Wp * fwd + Wm * bwd)
G1 = GF(m1); G2 = G1 if m2 == m1 else GF(m2)
print(f'L = {L}, n = {n}, пар {M}; мера {MEAS}; массы {m1}, {m2}', flush=True)
def A(v):
    Zt = csr_matrix((v, (I1, I2)), shape=(n, n)).T @ G1; out = np.empty(M, complex)
    for s in range(0, M, 6000): sl = slice(s, s + 6000); out[sl] = (Zt[:, I1[sl]] * G2[:, I2[sl]]).sum(0)
    return out
r = x[I1] - x[I2]; R = 0.5 * (x[I1] + x[I2]); h = (r if par == 'odd' else 1.0) * np.exp(-r * r / 4.5)
f = np.where((np.abs(t[I1] - ts) < 0.45) & (np.abs(t[I2] - ts) < 0.45), h * np.exp(-R * R / 18), 0).astype(complex); b = A(f)
if MEAS == 'voronoi':                                 # мера из кристалла: ячейка (Δt слоя) × (Δx₁ ячейки Вороного вдоль слоя) × (Δx₂)
    from goldlab import core
    LV = list(core.layers(w.X)); tl_ = np.array([q for q, _ in LV]); gp = np.diff(tl_); half = 0.5 * (np.r_[gp[0], gp] + np.r_[gp, gp[-1]])
    dxv = np.zeros(n); hv = np.zeros(n)
    for k, (q, idx) in enumerate(LV):
        o = idx[np.argsort(x[idx])]; hv[o] = half[k]
        if len(o) == 1: dxv[o] = 1.0 / (rho * half[k]); continue
        d = np.diff(x[o]); dxv[o] = 0.5 * (np.r_[d[0], d] + np.r_[d, d[-1]])
    meas = 1.0 / (hv[I1] * dxv[I1] * dxv[I2])
elif MEAS == 'count':                                  # мера из кристалла: настоящая плотность пар (подсчёт), а не 2δρ² «как в континууме»
    box = np.nonzero((np.abs(t) < 0.3 * L) & (np.abs(x) < 0.25 * L))[0]
    q = np.mean([((np.abs(t - t[i]) < dl) & (np.abs(x - x[i]) < 10.0)).sum() - 1 for i in box]) / (rho * 2 * dl * 20.0)
    print(f'плотность пар / 2δρ² = {q:.3f}', flush=True); meas = 2 * dl * rho ** 2 * q
elif MEAS == 'level':
    meas = rho * Cs.time_levels(t, rho)[I1]
else:
    meas = 2 * dl * rho ** 2
wv = -1j * 4 * m1 * m2 * F * np.abs(r) / meas; T0 = time.time(); it = [0]
op = LinearOperator((M, M), matvec=lambda v: v - A(wv * v), dtype=complex)
CAP = int(os.environ.get('CAP', 300))                   # предел итераций: дальше честно пишем «не сошлось»
def cb(rk):
    it[0] += 1
    if it[0] % 10 == 0: print(f'   итерация {it[0]}: невязка {rk:.1e}, {time.time() - T0:.0f} с', flush=True)
psi, info = gmres(op, b, rtol=1e-4, restart=CAP, maxiter=1, callback=cb, callback_type='pr_norm')
res = np.linalg.norm(op @ psi - b) / np.linalg.norm(b)
sel = (tb > ts + 1.5) & (tb < L - 5); om = np.linspace(m1 + m2 - 0.4, m1 + m2 + 1.2, 1601)
np.savez(f'sig_L{L:g}_{MEAS}{dl:g}_{m1:g}_{m2:g}_{par}.npz', tt=tb[sel], sig=(h * np.exp(-R * R / 18))[sel] * psi[sel], res=res)
_k = np.round(tb[sel], 6); _tu, _inv = np.unique(_k, return_inverse=True); _s = np.zeros(len(_tu), complex); np.add.at(_s, _inv, (h * np.exp(-R * R / 18))[sel] * psi[sel])
S = Ob.spectrum(_tu, _s, om); S /= S.max(); pk = Ob.peaks(om - m1 - m2, S, 0.1)      # по уникальным временам (память)
print(('СОШЛОСЬ' if res < 2e-4 else 'НЕ СОШЛОСЬ (предел итераций)') + f'; GMRES {it[0]} итераций, невязка {res:.1e}, {time.time() - T0:.0f} с; пики: ' + ' '.join(f'{e:.3f}({hh:.2f})' for e, hh in pk), flush=True)
print('Солпитер:', ' '.join(f'{e:.3f}' for e in Ob.salpeter_levels(m1, m2, F, par)), flush=True)
