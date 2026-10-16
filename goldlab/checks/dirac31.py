"""Атом со спином в 3+1. Спинор Дирака в киральном базисе = два спинора Вейля; каждый подчиняется уравнению Клейна–Гордона
с членом Паули: (D_t² − ∇² + m²)χ = ±i(σ·∇U)χ. На кристалле: пропагатор Джонстона по связям с нитью Вильсона (как у водорода
3+1) для двухкомпонентного поля и матричная вставка B = −(m² ∓ iσ·∇U)/ρ. Проверки: (1) уровень 1s Дирака против КГ и
непрерывного Дирака; (2) обе киральности дают одно и то же; (3) спин вверх и вниз — одно и то же (вырождение);
(4) переворот спина s-волной не читается (σ·r̂ переводит s в p)."""
import sys, os, math, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from goldlab import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob
H = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0; Rc = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0; al = float(sys.argv[3]) if len(sys.argv) > 3 else 0.45; a = 0.5; m = 1.0; T0 = time.time()
lg = lambda *s: print(f'[{time.time() - T0:5.0f} c]', *s, flush=True)
U = lambda r: -al / np.sqrt(r * r + a * a); dU = lambda r: al * r / (r * r + a * a) ** 1.5
# ---- непрерывные эталоны ----
def dirac_1s(kappa=-1):
    def tail(E):
        def rhs(r, y):
            g, f = y; V = -al / math.sqrt(r * r + a * a); return [-kappa / r * g + (E - V + m) * f, kappa / r * f - (E - V - m) * g]
        r0 = 1e-4; V0 = -al / a; y0 = [r0, -(E - V0 - m) * r0 * r0 / 3] if kappa == -1 else [r0 * r0, (E - V0 + m) * r0 / 3]
        s = solve_ivp(rhs, (r0, 40.0), y0, rtol=1e-10, atol=1e-14); return s.y[0, -1] / (np.abs(s.y[0]).max() + 1e-300)
    Es = np.linspace(0.3, 0.9995, 500); v = [tail(e) for e in Es]
    return [brentq(tail, Es[i], Es[i + 1]) for i in range(len(Es) - 1) if np.sign(v[i]) != np.sign(v[i + 1])]
Ed = dirac_1s(-1); Ep = dirac_1s(1); kg = Ob.kg_levels(lambda s: -al / math.sqrt(s * s + a * a), 1.0, "odd", 0.3, 0.9995)
lg(f'непрерывные: КГ s {np.round(kg[:2], 4)}; Дирак s½ (κ = −1) {np.round(Ed[:2], 4)}; Дирак p½ (κ = +1) {np.round(Ep[:1], 4)}')
# ---- мир ----
w = Wd.gold31(1.0, 'cylinder', H=H, Rc=Rc, Rs=3.0); X = w.X; cache = f'/home/claude/code/goldlab_checks/links31_H{H:.0f}_R{Rc:.0f}.npz'
if os.path.exists(cache): z = np.load(cache); I, J = z['I'], z['J']
else: I, J = Cs.links_sparse(X, 1.0, 1.6); np.savez(cache, I=I, J=J)
lg(f'мир: {w.n} событий, {len(I)} связей'); ph = Pr.wilson31(X, I, J, U)
r = np.linalg.norm(X[:, 1:], axis=1); a0 = 1 / al; prof = np.exp(-r / a0); src0 = X[:, 0] < 0.45
sel = (X[:, 0] > 1.5) & (X[:, 0] < H - 0.3); tt = X[sel, 0]; om = np.linspace(0.3, 1.6, 2601)
def peak(sig):
    S = Ob.spectrum(tt, sig, om, 'hann'); pk = Ob.peaks(om, S, 0.0); return pk[int(np.argmax([h for _, h in pk]))][0], S
phi = Pr.johnston31_respond(w.n, I, J, 1.0, m, np.where(src0, prof, 0), ph); Ekg, Skg = peak(prof[sel] * phi[sel]); lg(f'КГ на кристалле: {Ekg:.4f}')
rh = np.where(r[:, None] > 1e-9, X[:, 1:] / np.maximum(r, 1e-9)[:, None], 0)
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]], complex)
sr = rh[:, 0, None, None] * sx + rh[:, 1, None, None] * sy + rh[:, 2, None, None] * sz          # σ·r̂ в каждом событии
res = {}
for sgn in (+1, -1):
    Bx = (-m * m / 1.0) * np.eye(2)[None] + sgn * 1j * (dU(r)[:, None, None] * sr) / 1.0
    for spin, e in (('вверх', np.array([1, 0])), ('вниз', np.array([0, 1]))):
        if sgn == -1 and spin == 'вниз': continue
        f = np.where(src0[:, None], prof[:, None] * e[None, :], 0)
        ps = Pr.johnston31_respond_spinor(w.n, I, J, 1.0, m, f, ph, Bx)
        Ein, S1 = peak(prof[sel] * (ps[sel] @ e)); flip = np.abs(prof[sel] * (ps[sel] @ (1 - e))).sum() / np.abs(prof[sel] * (ps[sel] @ e)).sum()
        res[(sgn, spin)] = Ein; lg(f'Дирак, киральность {"+" if sgn > 0 else "−"}, спин {spin}: 1s {Ein:.4f}; переворот спина в s-чтении {flip:.2e}')
lg(f'ИТОГ H={H:.0f} R={Rc:.0f}: кристалл КГ {Ekg:.4f}, Дирак {res[(1, "вверх")]:.4f} / {res[(-1, "вверх")]:.4f} / вниз {res[(1, "вниз")]:.4f}; '
   f'непрерывные КГ {kg[0]:.4f}, Дирак {Ed[0]:.4f}; сдвиг Дирак − КГ: кристалл {res[(1, "вверх")] - Ekg:+.4f}, непрерывный {Ed[0] - kg[0]:+.4f}')
