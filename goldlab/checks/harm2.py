"""Считывание уровня пары с мерой: s(t_слоя) = Σ_{пары слоя} Δx₁Δx₂·h·ψ (интеграл по r, R, а не сырая сумма — число событий на
слое скачет квазипериодически и без меры даёт ложные боковые частоты). Затем: спектр и подгонка частот (обе ветви ±)."""
import sys, re, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from scipy.optimize import least_squares
from goldlab import worlds as Wd, core
def cells(L, rho=4.0):
    w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n; dl = 0.3
    I1, I2 = np.nonzero((np.abs(t[:, None] - t[None, :]) < dl) & (np.abs(x[:, None] - x[None, :]) < 10.0)); k = I1 != I2; I1, I2 = I1[k], I2[k]
    tb = 0.5 * (t[I1] + t[I2]); o = np.argsort(tb, kind='stable'); I1, I2, tb = I1[o], I2[o], tb[o]; ts = -L + 7
    sel = (tb > ts + 1.5) & (tb < L - 5); dxv = np.zeros(n)
    for q, idx in core.layers(w.X):
        o2 = idx[np.argsort(x[idx])]
        if len(o2) == 1: dxv[o2] = np.nan; continue
        d = np.diff(x[o2]); dxv[o2] = 0.5 * (np.r_[d[0], d] + np.r_[d, d[-1]])
    return tb[sel], dxv[I1[sel]] * dxv[I2[sel]]
def fit(tt, s, m1, m2, label):
    tt = tt - tt.min(); u = tt / tt.max(); win = np.sin(np.pi * u) ** 2
    om = np.linspace(-5, 5, 2001); S = np.abs(np.exp(-1j * np.outer(om, tt)) @ (win * s)) ** 2; S /= S.max()
    pk = [om[i] for i in range(1, len(om) - 1) if S[i] > S[i - 1] and S[i] > S[i + 1] and S[i] > 0.03]
    pos = [p for p in pk if p > 0]; print(f'  {label}: пики Фурье (E − m₁ − m₂, положит.): ' + ', '.join(f'{p - m1 - m2:+.3f}({S[np.argmin(abs(om - p))]:.2f})' for p in pos) +
          ' | отрицат.: ' + ', '.join(f'{p:+.2f}' for p in pk if p < 0))
    def resid(E):
        B = np.exp(-1j * np.outer(tt, E)); a = np.linalg.lstsq(B, s, rcond=None)[0]; r = s - B @ a; return np.r_[r.real, r.imag], a
    E0 = np.array(sorted(pk, key=lambda e: -S[np.argmin(abs(om - e))])[:4]); sol = least_squares(lambda E: resid(E)[0], E0, x_scale=0.05)
    r, a = resid(sol.x); rel = np.linalg.norm(r) / np.linalg.norm(np.r_[s.real, s.imag]); o = np.argsort(sol.x); w = np.abs(a[o]) ** 2 / (np.abs(a) ** 2).sum()
    print(f'     подгонка {len(E0)} частот: остаток {rel:.3f}; ' + ', '.join((f'{e - m1 - m2:+.3f}' if e > 0 else f'[{e:+.2f}]') + f' ({q:.2f})' for e, q in zip(sol.x[o], w)))
for fn in sys.argv[1:]:
    L = float(re.search(r'sig_L([0-9.]+)_', fn).group(1)); m1, m2 = (0.6, 1.2) if '_0.6_1.2_' in fn else (1.0, 1.0)
    d = np.load(fn); tt0, sig0 = d['tt'], d['sig']; tsel, wcell = cells(L); assert len(tsel) == len(tt0) and np.allclose(tsel, tt0)
    print(fn)
    for label, wt in (('сырая сумма', np.ones(len(tt0))), ('с мерой Δx₁Δx₂', np.nan_to_num(wcell))):
        key = np.round(tt0, 6); tt, inv = np.unique(key, return_inverse=True); s = np.zeros(len(tt), complex); np.add.at(s, inv, wt * sig0)
        fit(tt, s, m1, m2, label)
