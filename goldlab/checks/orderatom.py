"""Атом Дирака целиком из порядка: поле ядра = ½ за каждое событие трубки-ядра в прошлом конусе (Лоренц, A₀ = G_ret·j),
фаза — интеграл этого поля вдоль светового уголка; компоненты Дирака — без вставки, скаляр — со вставкой −iU′.
Калибровка: вычитается только однородная по x часть α·t + β (функция одного t — чистая калибровка, точка отсчёта энергии)."""
import sys, time, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, fields as Fl, propagate as Pr, observe as Ob, fermions as Fe, causal as Cs
mode = sys.argv[1] if len(sys.argv) > 1 else 'order'; Ft = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4; L = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
a = 1.0; m = 1.0; w = Wd.gold11(4.0, L); t, x, X = w.t, w.x, w.X; ts = -L + 10; print('n =', w.n, 'режим', mode, flush=True)
tg = np.arange(-L - 0.5, L + 0.51, 0.05); xg = np.arange(-L - 0.5, L + 0.51, 0.05)
if mode == 'order':
    tube = np.nonzero(np.abs(x) < a)[0]; te, xe = t[tube], x[tube]; lam = len(tube) / (2 * L - 2 * a) / (2 * a)
    g = 2 * Ft                                                   # величина заряда: ядро·электрон (руками), подобрана под наклон Ft
    def count(tt, xx):
        out = np.zeros(tt.shape)
        for s in range(0, len(te), 64):
            out += ((tt[..., None] - te[s:s + 64]) > np.abs(xx[..., None] - xe[s:s + 64])).sum(-1)
        return out
    T, Xg = np.meshgrid(tg, xg, indexing='ij'); V = np.empty(T.shape)
    for r in range(0, len(tg), 40): V[r:r + 40] = -g * 0.5 * count(T[r:r + 40], Xg[r:r + 40]) / (lam * 2 * a)
    Uev = -g * 0.5 * Cs.causal(X)[tube].sum(0) / (lam * 2 * a)
    S = (t > ts) & (t < L - 4) & (np.abs(x) > a + 0.3) & (np.abs(x) < 10)
    A = np.c_[t[S], np.ones(S.sum()), np.abs(x[S])]; (al, be, Fm), *_ = np.linalg.lstsq(A, Uev[S], rcond=None)
    print(f'поле из порядка: U ≈ {al:+.4f}·t {be:+.3f} + {Fm:.4f}·|x|  (калибровка: вычитаем α·t + β)', flush=True)
    V = V - al * T - be; Uev = Uev - al * t - be; Uf = Pr.GridField(tg, xg, V)
    dV = np.gradient(V, xg, axis=1); dUev = Pr.GridField(tg, xg, dV)(t, x); Fuse = Fm
else:
    Uf = lambda tt, xx: Ft * np.abs(xx); dUev = Ft * np.sign(x); Fuse = Ft
gs = lambda s: (1 + s) * np.exp(-s * s / 2); om = np.linspace(m - 0.1, m + 1.2, 2601); E = []
for nm, side, ins in (('КГ', 1, 1.0), ('уголок', 1, 0.0), ('зеркальный', -1, 0.0)):
    t0 = time.time(); phi = Pr.corner_respond_general(X, Uf, dUev, m, 4.0, np.where(np.abs(t - ts) < 0.45, gs(x), 0), side, ins)
    sel = (t > ts + 1.5) & (t < L - 8) & (np.abs(x) < 8); Sp = Ob.spectrum(t[sel], gs(x[sel]) * phi[sel], om); Sp /= Sp.max()
    pk = sorted([(round(e, 4), round(h, 2)) for e, h in Ob.peaks(om, Sp, 0.05)], key=lambda z: -z[1])[:3]; E.append(pk[0][0]); print(nm, pk, f'{time.time() - t0:.0f} c', flush=True)
kg = sorted(Ob.kg_levels_linear(Fuse, m, 'even', m + 0.9, n=300) + Ob.kg_levels_linear(Fuse, m, 'odd', m + 0.9, n=300))[0]
_, ev, _, _, _ = Fe.dirac_orbitals(lambda s: Fuse * np.abs(s), 1.0, m, nlev=1, N=512, Lg=80, locmax=60)
print(f'F = {Fuse:.4f}: непрерывные КГ {kg:.4f} Дирак {ev[0]:.4f} (сдвиг {ev[0] - kg:+.4f}); кристалл: КГ {E[0]:.4f} уголок {E[1]:.4f} зеркальный {E[2]:.4f} сдвиг {0.5 * (E[1] + E[2]) - E[0]:+.4f}', flush=True)
