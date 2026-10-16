"""ИЗМЕРИТЕЛЬ НА ИЗВЕСТНОЙ ЗАДАЧЕ: одна частица m = 1 во внешнем поле U = F|x| (точная связь — световой уголок), тот же
удар x·e^{−x²/4.5} на слое t_s = −L + 7, то же окно и тот же Фурье, что у пары. Теория — КГ (квазисвязанные, щель Клейна)."""
import os, sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, fields as Fl, observe as Ob
F = float(os.environ.get('F', 0.08)); kg = [e - 1 for e in Ob.kg_levels_linear(F, 1.0, 'odd', 2.6)[:2]]
for L in [float(a) for a in sys.argv[1:]]:
    w = Wd.gold11(4.0, L); t, x = w.t, w.x; ts = -L + 7; P, dU, U = Fl.vshape(F)
    g = x * np.exp(-x * x / 4.5); phi = Pr.corner_respond(w.X, P, dU, 1.0, 4.0, np.where(np.abs(t - ts) < 0.45, g, 0))
    sel = (t > ts + 1.5) & (t < L - 5); k = np.round(t[sel], 6); tu, inv = np.unique(k, return_inverse=True); s = np.zeros(len(tu), complex); np.add.at(s, inv, (g * phi)[sel])
    om = np.linspace(0.6, 2.2, 1601); S = Ob.spectrum(tu, s, om); S /= S.max(); pk = Ob.peaks(om - 1, S, 0.1)
    print(f'L = {L:g}: одно тело в F|x|: пики ' + ' '.join(f'{e:.3f}({q:.2f})' for e, q in pk) + f'   (КГ {kg[0]:.3f}, {kg[1]:.3f})', flush=True)
