"""ТРЕТИЙ ЭЛЕКТРОН: три одинаковые частицы со спином в поле ядра (без взаимного отталкивания — первое приближение).
Амплитуда — определитель Слэтера; чтение на трёх «орбиталях»: чётная h₀ = e^{−x²/4}, нечётная h₁ = x·e^{−x²/4}, вторая чётная
h₂ = (x² − 2)·e^{−x²/4}. Ждём низшую энергию тройки E₁ + E₂ + E₃ (у бозонов было бы 3E₁)."""
import sys, os, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from walk4 import Walk, soft_coulomb_int
from goldlab import worlds as Wd, observe as Ob
L = float(os.environ.get('L', 30)); alpha, a, m = 0.5, 1.0, 1.0
w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Walk(w); ph = W.phases(soft_coulomb_int(alpha, a))
lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
sel = (t > ts + 2) & (t < L - 4) & (np.abs(x) < 9); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
H = [np.exp(-x * x / 4), x * np.exp(-x * x / 4), (x * x - 2) * np.exp(-x * x / 4)]
A = np.zeros((3, 3, len(tu)), complex)
for p, x0 in enumerate((-2.0, 0.3, 2.0)):
    g = np.where(src, np.exp(-(x - x0) ** 2 / 2), 0); aR, aL, _ = W.run(m, g, g, ph)
    for r in range(3):
        s = np.zeros(len(tu), complex); np.add.at(s, inv, (H[r] * (aR + aL))[sel]); A[r, p] = s
om = np.linspace(-3.4, -0.3, 3101)
def pk(s):
    S = Ob.spectrum(tu, s, om); S /= S.max(); return [(-e, q) for e, q in Ob.peaks(om, S, 0.1)]
for r, nm in enumerate(('h₀', 'h₁', 'h₂')):
    print(f'одна частица, чтение {nm}: ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in pk(A[r, 1] + A[r, 0])), flush=True)
det = np.array([np.linalg.det(A[:, :, i]) for i in range(len(tu))])
print('три фермиона (определитель Слэтера): энергии тройки ' + ', '.join(f'{e:.4f}({q:.2f})' for e, q in pk(det)), flush=True)
# непрерывный Дирак
N, Lg = 512, 64.0; xg = (np.arange(N) - N / 2) * (Lg / N); h = Lg / N; kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
D = np.real_if_close(np.fft.ifft(1j * kf[:, None] * np.fft.fft(np.eye(N), axis=0), axis=0)); V = np.diag(-alpha / np.sqrt(xg ** 2 + a * a))
ev = np.linalg.eigvalsh(np.block([[-1j * W.c * D + V, m * np.eye(N)], [m * np.eye(N), 1j * W.c * D + V]])); bnd = sorted(e for e in ev if 0 < e < m)
print(f'непрерывный Дирак (c = {W.c:.3f}): уровни ' + ', '.join(f'{e:.4f}' for e in bnd[:4]) + f'; E₁ + E₂ + E₃ = {sum(bnd[:3]):.4f}', flush=True)
