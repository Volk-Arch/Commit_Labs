"""ПАРА БЕЗ «ОДНОГО МОМЕНТА»: ковариантная лестница на кристалле. Только порядок:
взаимодействие двух событий z₁ (частица 1) и z₂ (частица 2) = (F/2)·[z₁ и z₂ несравнимы — ни одно не раньше другого].
В непрерывном пределе ∫dt₂ (F/2)·θ(|Δx| − |Δt|) = F·|Δx| — та же линейная сила, но без выбора «сейчас» (несравнимость не
зависит от системы отсчёта). Мера — 1/ρ на событие (двумерная ячейка), никаких слоёв.
Амплитуда пары — матрица Ψ[k₁, k₂] по ВСЕМ парам событий: Ψ = G₁ᵀ (S + V∘Ψ) G₂, G — Фейнман из SJ (c = −i точно).
Решение — BiCGSTAB (мало векторов памяти). Считывание — в системе покоя пары (одинаковые слои), как раньше."""
import os, sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, core
from scipy.sparse.linalg import LinearOperator, bicgstab
L = float(os.environ.get('L', 12)); m1, m2 = float(os.environ.get('M1', 1)), float(os.environ.get('M2', 1)); F = float(os.environ.get('F', 0.08)); par = os.environ.get('PAR', 'odd')
CAP = int(os.environ.get('CAP', 200)); rho = 4.0
w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n; T0 = time.time()
fwd = t[None, :] > t[:, None]; bwd = t[None, :] < t[:, None]
def GF(m):
    Wp, Wm, _ = Pr.sj(Pr.johnston11(w.X, rho, m)); return -1j * (Wp * fwd + Wm * bwd)
G1 = GF(m1); G2 = G1 if m2 == m1 else GF(m2)
C = core.precedes(w.X); INC = ~(C | C.T); np.fill_diagonal(INC, False)                     # несравнимые пары — из порядка
RMAX = float(os.environ.get('RMAX', 10.0)); INC &= np.abs(x[:, None] - x[None, :]) < RMAX    # как в прежнем опыте: дальше F·r > 2m — рвётся струна (рождение пар), лестница этого не описывает
kap = -1j * 4 * m1 * m2 * (F / 2) / rho ** 2; V = kap * INC
print(f'L = {L}, событий {n}, несравнимых пар {INC.sum()}, F = {F}, массы {m1}, {m2}; {time.time() - T0:.0f} с', flush=True)
ts = -L + 7; Rr = x[:, None] - x[None, :]; Rc = 0.5 * (x[:, None] + x[None, :])
slab = (np.abs(t - ts) < 0.45); h = (Rr if par == 'odd' else 1.0) * np.exp(-Rr ** 2 / 4.5) * np.exp(-Rc ** 2 / 18)
S = np.where(slab[:, None] & slab[None, :], h, 0).astype(complex); np.fill_diagonal(S, 0)
A = lambda X: G1.T @ X @ G2
b = A(S).ravel(); it = [0]
def mv(v):
    return v - A(V * v.reshape(n, n)).ravel()
op = LinearOperator((n * n, n * n), matvec=mv, dtype=complex)
def cb(xk):
    it[0] += 1
    if it[0] % 5 == 0: print(f'   итерация {it[0]}: {time.time() - T0:.0f} с', flush=True)
psi, info = bicgstab(op, b, rtol=1e-4, maxiter=CAP, callback=cb)
res = np.linalg.norm(mv(psi) - b) / np.linalg.norm(b); Psi = psi.reshape(n, n)
# считывание: пары на одном слое (система покоя пары), проекция на ту же форму h, мера Δx не нужна для положения пика
same = np.abs(t[:, None] - t[None, :]) < 1e-9; np.fill_diagonal(same, False); sel = same & (t[:, None] > ts + 1.5) & (t[:, None] < L - 5)
I1, I2 = np.nonzero(sel); tt = t[I1]; sg = h[I1, I2] * Psi[I1, I2]
k = np.round(tt, 6); tu, inv = np.unique(k, return_inverse=True); s = np.zeros(len(tu), complex); np.add.at(s, inv, sg)
np.savez(f'cov_L{L:g}_F{F:g}_{m1:g}_{m2:g}_{par}.npz', tu=tu, s=s, res=res)
om = np.linspace(m1 + m2 - 0.4, m1 + m2 + 1.2, 1601); Sp = Ob.spectrum(tu, s, om); Sp /= Sp.max(); pk = Ob.peaks(om - m1 - m2, Sp, 0.1)
print(('СОШЛОСЬ' if res < 2e-4 else f'НЕ СОШЛОСЬ (info {info})') + f'; итераций {it[0]}, невязка {res:.1e}, {time.time() - T0:.0f} с; пики: ' + ' '.join(f'{e:.3f}({q:.2f})' for e, q in pk), flush=True)
print('Солпитер:', ' '.join(f'{e:.3f}' for e in Ob.salpeter_levels(m1, m2, F, par)), flush=True)
