import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import fields as Fl
kap = 0.055; h = Fl.slice_twist(kap); L = 16
for (T, x) in ((0, 0), (-6, 1), (-5, 0), (5, 5), (-10, 4), (12, -3), (-14, 1)):
    try: print((T, x), Fl.curvature(h, T, x) if hasattr(Fl, 'curvature') else None)
    except Exception as e: print((T, x), 'ошибка', e)
g = np.linspace(-L - 1, L + 1, 161); bad = 0
AT = np.zeros((161, 161)); Ax = np.zeros((161, 161))
for i, Tq in enumerate(g):
    for j, xq in enumerate(g):
        try: a_, b_ = Fl.connection(h, Tq, xq); AT[i, j], Ax[i, j] = a_[0, 1], b_[0, 1]
        except ValueError: bad += 1
dg = g[1] - g[0]; Fg = np.gradient(Ax, dg, axis=0) - np.gradient(AT, dg, axis=1)
ins = np.abs(g[:, None]) + np.abs(g[None, :]) <= L
print('ошибок рамки', bad, '; F на сетке внутри ромба: среднее', Fg[ins].mean(), 'мин', Fg[ins].min(), 'макс', Fg[ins].max())
for T in (-12, -6, 0, 6, 12):
    i = np.argmin(abs(g - T)); print('T', T, ' F(x) при x = −8,−4,0,4,8:', [round(Fg[i, np.argmin(abs(g - xx))], 5) for xx in (-8, -4, 0, 4, 8)])
