import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import observe as Ob
for fn in sys.argv[1:]:
    d = np.load(fn); k = np.round(d['tt'], 6); tu, inv = np.unique(k, return_inverse=True); s = np.zeros(len(tu), complex); np.add.at(s, inv, d['sig'])
    m1, m2 = (0.6, 1.2) if '_0.6_1.2_' in fn else (1.0, 1.0); om = np.linspace(m1 + m2 - 0.4, m1 + m2 + 1.2, 1601)
    S = Ob.spectrum(tu, s, om); S /= S.max(); pk = Ob.peaks(om - m1 - m2, S, 0.1)
    print(f'{fn}: невязка GMRES {float(d["res"]):.1e}; пики: ' + ' '.join(f'{e:.3f}({h:.2f})' for e, h in pk))
