"""ПАУЛИ В ДЕЙСТВИИ: «атом» с двумя одинаковыми частицами со спином (блуждание по нитям) в поле ядра V = −α/√(x² + a²).
1) Уровни одной частицы (спектроскопия ударом) против непрерывного Дирака с той же скоростью нитей c.
2) Две частицы без взаимного взаимодействия (первое приближение — как «независимые электроны»), стартуют из разных мест.
   Амплитуда пары: фермионы Ψ = ψ₁⊗ψ₂ − ψ₂⊗ψ₁, бозоны — с плюсом. Чтение: «обе на чётной (нижней) орбитали» и «одна на чётной,
   другая на нечётной». У фермионов первое — тождественный ноль (Паули), низшая энергия пары E₁ + E₂; у бозонов — 2E₁."""
import sys, os, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from walk4 import Walk, soft_coulomb_int
from goldlab import worlds as Wd, observe as Ob
L = float(os.environ.get('L', 30)); alpha = float(os.environ.get('AL', 0.5)); a = float(os.environ.get('A', 1.0)); m = 1.0
w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Walk(w); ph = W.phases(soft_coulomb_int(alpha, a))
lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
print(f'мир {L}, событий {w.n}; нити: полных {W.full:.1%}, c = {W.c:.3f}; ядро α = {alpha}, a = {a}; источник — слой {ts:.2f} ({src.sum()} событий)', flush=True)
# непрерывный Дирак: H = −i c σ_z ∂x + m σ_x + V (киральный базис), сетка
N, Lg = 1200, 60.0; xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N
D = (np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)) / (2 * h)
H = np.block([[-1j * W.c * D + np.diag(-alpha / np.sqrt(xg ** 2 + a * a)), m * np.eye(N)], [m * np.eye(N), 1j * W.c * D + np.diag(-alpha / np.sqrt(xg ** 2 + a * a))]])
ev = np.linalg.eigvalsh(H); bound = sorted(e for e in ev if 0 < e < m)
ev0 = np.linalg.eigvalsh(np.block([[-1j * W.c * D, m * np.eye(N)], [m * np.eye(N), 1j * W.c * D]])); gap0 = min(e for e in ev0 if e > 0)
print(f'Дирак (непрерывный, c = {W.c:.3f}): связанные уровни частицы ' + ', '.join(f'{e:.4f}' for e in bound[:4]) + f'  (порог {gap0:.4f} на сетке)', flush=True)
# одна частица: удары из двух мест
sel = (t > ts + 2) & (t < L - 4) & (np.abs(x) < 8); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
he, ho = np.exp(-x * x / 4), x * np.exp(-x * x / 4)                     # чтение: чётная и нечётная орбитали
def proj(hx, aR, aL):
    s = np.zeros(len(tu), complex); np.add.at(s, inv, (hx * (aR + aL))[sel]); return s   # (1, 1) — ветвь частицы
amps = {}
for k, x0 in ((1, -1.5), (2, +1.5)):
    g = np.where(src, np.exp(-(x - x0) ** 2 / 2), 0); aR, aL, leak = W.run(m, g, g, ph)
    amps[k] = {'e': proj(he, aR, aL), 'o': proj(ho, aR, aL)}
om = np.linspace(-2.6, -0.3, 2301)
def peaks(s, lo=-1.0, hi=-0.3):
    S = Ob.spectrum(tu, s, om); S /= S.max(); return [(-e, q) for e, q in Ob.peaks(om, S, 0.08)]
for k in (1, 2):
    print(f'частица {k}: чётное чтение — энергии ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in peaks(amps[k]['e'])) +
          f';  нечётное — ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in peaks(amps[k]['o'])), flush=True)
A = amps
F_ee = A[1]['e'] * A[2]['e'] - A[2]['e'] * A[1]['e']; B_ee = A[1]['e'] * A[2]['e'] + A[2]['e'] * A[1]['e']
F_eo = A[1]['e'] * A[2]['o'] - A[2]['e'] * A[1]['o']; B_eo = A[1]['e'] * A[2]['o'] + A[2]['e'] * A[1]['o']
print(f'ФЕРМИОНЫ, обе на чётной орбитали: |амплитуда| ≤ {np.abs(F_ee).max():.1e} (тождественный ноль — Паули)')
for name, s in (('бозоны, обе на чётной', B_ee), ('бозоны, чётная + нечётная', B_eo), ('фермионы, чётная + нечётная', F_eo)):
    print(f'{name}: энергии пары ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in peaks(s)), flush=True)
