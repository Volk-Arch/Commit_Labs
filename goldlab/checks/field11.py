"""ПОЛЕ ЗАРЯДА ИЗ ПОРЯДКА (1+1). Неподвижный заряд — трубка событий |x| < a во всё время. Потенциал в событии z — запаздывающий
пропагатор света Джонстона, ½ за каждое событие трубки в прошлом z: φ(z) = (q/2)·N_past(z)/(ρ·2a). В непрерывном пределе
φ = (q/2)(t − t₀ − |x|): ЛИНЕЙНЫЙ по |x| с наклоном −q/2 — сила между зарядами в 1+1 из одного порядка, без «руками»."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
for L in (16, 24):
    w = Wd.gold11(4.0, L); X, t, x = w.X, w.t, w.x; a = 1.0; rho = 4.0
    tube = np.nonzero(np.abs(x) < a)[0]; C = Cs.causal(X)
    Np = C[tube].sum(0)                                     # число событий трубки в прошлом каждого события
    lam = len(tube) / (2 * L - 2 * a) / (2 * a)             # плотность трубки (на единицу времени и ширины) — подсчёт
    phi = 0.5 * Np / (lam * 2 * a)
    for tl in (0.0, L / 4):
        S = (np.abs(t - tl) < 0.5) & (np.abs(x) > a + 0.5) & (np.abs(x) < L - abs(tl) - 2)
        k, b = np.polyfit(np.abs(x[S]), phi[S] - (t[S] - (-L)), 1); r = phi[S] - (t[S] - (-L)) - (k * np.abs(x[S]) + b)
        print(f'L = {L}, t ≈ {tl:.0f}: φ − (t − t₀) = {k:+.4f}·|x| + {b:+.3f}  (континуум: −0.5·|x| + 0); разброс вокруг прямой {r.std():.3f}, точек {S.sum()}', flush=True)
