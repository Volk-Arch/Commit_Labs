"""Подгонка ρ⟨K⟩(τ) формой A·(−m J₁(mτ)/(4πτ)): какая масса «звенит» на самом деле."""
import numpy as np, math
from scipy.special import j1
from scipy.optimize import curve_fit
res = np.load('/home/claude/data/waves_res.npy', allow_pickle=True)[0]
H = (24 * 5000 / math.pi) ** 0.25; bins = np.arange(2, H, 1.0); tc = 0.5 * (bins[:-1] + bins[1:])
f = lambda t, A, m: -A * m * j1(m * t) / (4 * math.pi * t)
for m in (0.3, 0.6):
    for name in ('кристалл', 'посыпка'):
        y, e, t = [], [], []
        for i in range(len(tc)):
            v = np.array(res[name][m][i])
            if len(v) > 50 and tc[i] >= 3.5: y.append(v.mean()); e.append(v.std() / math.sqrt(len(v))); t.append(tc[i])
        p, c = curve_fit(f, np.array(t), np.array(y), p0=[1, m], sigma=np.array(e))
        print(f'm = {m}: {name:9s} масса звона m_эфф = {p[1]:.3f} ± {math.sqrt(c[1,1]):.3f} ({p[1]/m*100:.0f} % от заданной), амплитуда {p[0]:.2f}')
