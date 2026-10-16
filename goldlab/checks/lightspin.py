"""Дирак 1+1 без нитей: компонента = отклик Джонстона по световому уголку БЕЗ вставки −iU′ (уголок и зеркальный уголок).
Скаляр КГ = уголок + вставка. Сравнение уровней в U = F|x| с непрерывным Дираком и КГ (масса кристалла)."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe
F = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4; L = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0; m = 1.0
w = Wd.gold11(4.0, L); t, x = w.t, w.x; Pf, dU, U = Fl.vshape(F); ts = -L + 10; print('n =', w.n, flush=True)
g = lambda s: (1 + s) * np.exp(-s * s / 2); om = np.linspace(0.9, 2.6, 1701); out = {}
for nm, side, ins in (('КГ (уголок + вставка)', 1, 1.0), ('Дирак: уголок', 1, 0.0), ('Дирак: зеркальный уголок', -1, 0.0)):
    t0 = time.time(); phi = Pr.corner_respond(w.X, Pf, dU, m, 4.0, np.where(np.abs(t - ts) < 0.45, g(x), 0), side, ins)
    sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); S = Ob.spectrum(t[sel], g(x[sel]) * phi[sel], om); S /= S.max()
    pk = [(round(e, 4), round(h, 2)) for e, h in Ob.peaks(om, S) if h > 0.05]; out[nm] = (S, pk); print(nm, pk, f'{time.time() - t0:.0f} c', flush=True)
for mc in (1.0, 1.041):
    kg = sorted(Ob.kg_levels_linear(F, mc, 'even', 2.6, n=400) + Ob.kg_levels_linear(F, mc, 'odd', 2.6, n=400))
    _, ev, _, _, _ = Fe.dirac_orbitals(lambda s: F * np.abs(s), 1.0, mc, nlev=5, N=512, Lg=80, locmax=60)
    print(f'непрерывные, m = {mc}: КГ', np.round(kg[:4], 4), ' Дирак', np.round(ev[:4], 4))
np.savez(f'/home/claude/code/goldlab_checks/lightspin_F{F}_L{L:.0f}.npz', om=om, **{f's{i}': out[k][0] for i, k in enumerate(out)})
