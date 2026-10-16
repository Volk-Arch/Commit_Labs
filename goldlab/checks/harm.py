"""ГАРМОНИЧЕСКИЙ АНАЛИЗ сигнала пары: sig(t) ≈ Σ a_k e^{−iE_k t}. Фурье разрешает лишь ~2π/T (0.24–0.34 — сравнимо с
расстоянием между уровнями 0.425 и 0.733), поэтому пик Фурье — смесь уровней. Здесь частоты подгоняются (вариационная проекция:
при заданных E амплитуды — линейный МНК), стартуя с пиков широкого спектра. Теория (Солпитер) в подгонку не входит."""
import sys, numpy as np
from scipy.optimize import least_squares
def analyse(fn, m1=1.0, m2=1.0, K=None, band=(1.2, 4.0)):
    d = np.load(fn); tt0, sig0 = d['tt'], d['sig']
    key = np.round(tt0, 6); tt, inv = np.unique(key, return_inverse=True); sig = np.zeros(len(tt), complex); np.add.at(sig, inv, sig0)   # проекция на слой: s(t)
    tt = tt - tt.min()
    om = np.linspace(band[0], band[1], 2801); u = tt / tt.max(); win = np.sin(np.pi * u) ** 2
    S = np.abs(np.exp(-1j * np.outer(om, tt)) @ (win * sig)) ** 2; S /= S.max()
    pk = [om[i] for i in range(1, len(om) - 1) if S[i] > S[i - 1] and S[i] > S[i + 1] and S[i] > 0.02]
    def resid(E):
        B = np.exp(-1j * np.outer(tt, E)); a = np.linalg.lstsq(B, sig, rcond=None)[0]; r = sig - B @ a
        return np.r_[r.real, r.imag], a
    best = None
    for k in range(1, (K or min(len(pk), 5)) + 1):
        E0 = np.array(sorted(pk, key=lambda e: -S[np.argmin(abs(om - e))])[:k])
        sol = least_squares(lambda E: resid(E)[0], E0, x_scale=0.05)
        r, a = resid(sol.x); rel = np.linalg.norm(r) / np.linalg.norm(np.r_[sig.real, sig.imag])
        o = np.argsort(sol.x); E, a = sol.x[o], a[o]; w = np.abs(a) ** 2 / (np.abs(a) ** 2).sum()
        line = f'  K = {k}: остаток {rel:.3f}; уровни E − m₁ − m₂: ' + ', '.join(f'{e - m1 - m2:+.3f} (доля {q:.2f})' for e, q in zip(E, w))
        print(line); best = (E - m1 - m2, w)
    return pk, best
if __name__ == '__main__':
    for fn in sys.argv[1:]:
        m1, m2 = (0.6, 1.2) if '_0.6_1.2_' in fn else (1.0, 1.0)
        print(fn); pk, _ = analyse(fn, m1, m2); print('  пики Фурье (E − m₁ − m₂):', ', '.join(f'{p - m1 - m2:+.3f}' for p in pk))
