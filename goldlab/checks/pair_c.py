"""ПРОВЕРКА ПАРЫ: откуда 0.422 (§21) против 0.389 (каркас). Гипотеза: в twopropF.py множитель c брался подгонкой по
спектру (|c| ≈ 1.03–1.04), а точное тождество K_вперёд = −i(W₊ − W₋) даёт |c| = 1. Варианты: C=fit|exact, MEAS=strip|level."""
import os, sys, math, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, causal as Cs
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres
L = float(os.environ.get('L', 20)); CM = os.environ.get('C', 'exact'); MEAS = os.environ.get('MEAS', 'strip'); par = os.environ.get('PAR', 'odd')
w = Wd.gold11(4.0, L); t, x, n = w.t, w.x, w.n; rho, dl, F, m = 4.0, 0.3, 0.08, 1.0
I1, I2 = np.nonzero((np.abs(t[:, None] - t[None, :]) < dl) & (np.abs(x[:, None] - x[None, :]) < 10.0)); k = I1 != I2; I1, I2 = I1[k], I2[k]
tb = 0.5 * (t[I1] + t[I2]); o = np.argsort(tb, kind='stable'); I1, I2, tb = I1[o], I2[o], tb[o]; M = len(I1); ts = -L + 7
K = Pr.johnston11(w.X, rho, m); Wp, Wm, _ = Pr.sj(K); fwd = t[None, :] > t[:, None]; bwd = t[None, :] < t[:, None]
if CM == 'fit':                                    # как в twopropF.py: c из ветви частицы по спектру отклика
    f = np.where(np.abs(t - ts) < 0.45, np.exp(-x * x / 8), 0); a = K.T @ f; b = (Wp * fwd).T @ f
    sel = (t > ts + 3) & (np.abs(x) < 5); om = np.linspace(m - 0.3, m + 0.6, 181)
    br = lambda y: np.array([(y[sel] * np.exp(-1j * q * t[sel])).sum() for q in om]); A_, B_ = br(a), br(b); c = (np.conj(B_) @ A_) / (np.conj(B_) @ B_)
else:
    c = -1j
print(f'L = {L}, n = {n}, пар {M}; c = {c.real:+.4f}{c.imag:+.4f}i (|c| = {abs(c):.4f}), мера {MEAS}', flush=True)
G = c * (Wp * fwd + Wm * bwd)
def A(v):
    Zt = csr_matrix((v, (I1, I2)), shape=(n, n)).T @ G; out = np.empty(M, complex)
    for s in range(0, M, 6000): sl = slice(s, s + 6000); out[sl] = (Zt[:, I1[sl]] * G[:, I2[sl]]).sum(0)
    return out
r = x[I1] - x[I2]; R = 0.5 * (x[I1] + x[I2]); h = (r if par == 'odd' else 1.0) * np.exp(-r * r / 4.5)
f = np.where((np.abs(t[I1] - ts) < 0.45) & (np.abs(t[I2] - ts) < 0.45), h * np.exp(-R * R / 18), 0).astype(complex); b = A(f)
meas = rho * Cs.time_levels(t, rho)[I1] if MEAS == 'level' else 2 * dl * rho ** 2
wv = -1j * 4 * m * m * F * np.abs(r) / meas; T0 = time.time(); it = [0]
op = LinearOperator((M, M), matvec=lambda v: v - A(wv * v), dtype=complex)
psi, info = gmres(op, b, rtol=1e-4, restart=250 if MEAS == 'level' else 60, maxiter=4, callback=lambda rk: it.__setitem__(0, it[0] + 1), callback_type='pr_norm')
res = np.linalg.norm(op @ psi - b) / np.linalg.norm(b)
sel = (tb > ts + 1.5) & (tb < L - 5); om = np.linspace(2 * m - 0.4, 2 * m + 1.2, 1601)
S = Ob.spectrum(tb[sel], (h * np.exp(-R * R / 18))[sel] * psi[sel], om); S /= S.max(); pk = Ob.peaks(om - 2 * m, S, 0.1)
print(f'GMRES {it[0]} итераций, невязка {res:.1e}, {time.time() - T0:.0f} с; пики: ' + ' '.join(f'{e:.3f}({hh:.2f})' for e, hh in pk), flush=True)
print('Солпитер:', ' '.join(f'{e:.3f}' for e in Ob.salpeter_levels(m, m, F, par)), flush=True)
