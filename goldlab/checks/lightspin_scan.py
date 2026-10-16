import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
g = lambda s: (1 + s) * np.exp(-s * s / 2)
for m, F in ((1.0, 0.4), (1.0, 0.6), (1.0, 0.2), (0.7, 0.25)):
    kg = sorted(Ob.kg_levels_linear(F, m, 'even', m + 0.9, n=300) + Ob.kg_levels_linear(F, m, 'odd', m + 0.9, n=300))[0]
    _, ev, _, _, _ = Fe.dirac_orbitals(lambda s: F * np.abs(s), 1.0, m, nlev=2, N=512, Lg=80, locmax=60)
    print(f'm={m} F={F}: непрерывные КГ {kg:.4f} Дирак {ev[0]:.4f} сдвиг {ev[0] - kg:+.4f}', flush=True)
    for L in (26.0, 30.0, 34.0):
        w = Wd.gold11(4.0, L); t, x = w.t, w.x; Pf, dU, U = Fl.vshape(F); ts = -L + 10; om = np.linspace(m - 0.1, m + 1.2, 2601); E = []
        for side, ins in ((1, 1.0), (1, 0.0), (-1, 0.0)):
            phi = Pr.corner_respond(w.X, Pf, dU, m, 4.0, np.where(np.abs(t - ts) < 0.45, g(x), 0), side, ins)
            sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); S = Ob.spectrum(t[sel], g(x[sel]) * phi[sel], om); S /= S.max()
            E.append(max(Ob.peaks(om, S, 0.05), key=lambda z: z[1])[0])
        print(f'   L={L:.0f}: КГ {E[0]:.4f}  уголок {E[1]:.4f} ({E[1] - E[0]:+.4f})  зеркальный {E[2]:.4f} ({E[2] - E[0]:+.4f})', flush=True)
