"""ИЗМЕРИТЕЛЬ БЕЗ ТЕОРИИ: «спектроскопическая проекция». Шаг 1 — удар x·e^{−x²/4.5}, пик ω*. Шаг 2 — из самого отклика берём
профиль на частоте ω* (Фурье по времени в каждой полоске x) — это оценка собственной функции уровня; ею же бьём и читаем.
Повтор. Возбуждённые уровни подавляются, частота остаётся измеренной (теория в фильтр не входит)."""
import os, sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, fields as Fl, observe as Ob
F = float(os.environ.get('F', 0.08)); kg = [e - 1 for e in Ob.kg_levels_linear(F, 1.0, 'odd', 2.6)[:2]]
for L in [float(a) for a in sys.argv[1:]]:
    w = Wd.gold11(4.0, L); t, x = w.t, w.x; ts = -L + 7; P, dU, U = Fl.vshape(F); out = []
    gfun = lambda xx: xx * np.exp(-xx * xx / 4.5)
    for it in range(4):
        g = gfun(x); phi = Pr.corner_respond(w.X, P, dU, 1.0, 4.0, np.where(np.abs(t - ts) < 0.45, g, 0))
        sel = (t > ts + 1.5) & (t < L - 5); k = np.round(t[sel], 6); tu, inv = np.unique(k, return_inverse=True); s = np.zeros(len(tu), complex); np.add.at(s, inv, (g * phi)[sel])
        om = np.linspace(0.6, 2.2, 1601); S = Ob.spectrum(tu, s, om); S /= S.max(); pk = Ob.peaks(om, S, 0.1); E = max(pk, key=lambda z: z[1])[0]; out.append(E - 1)
        # профиль на частоте E: u(x) = Σ_t окно·φ·e^{iEt} по полоскам x
        tt = t[sel]; u0 = (tt - tt.min()) / np.ptp(tt); win = np.sin(np.pi * u0) ** 2
        edges = np.arange(-10, 10.01, 0.5); cen = 0.5 * (edges[1:] + edges[:-1]); prof = np.zeros(len(cen), complex)
        b = np.digitize(x[sel], edges) - 1; ok = (b >= 0) & (b < len(cen)); np.add.at(prof, b[ok], (win * phi[sel] * np.exp(1j * E * tt))[ok])
        cnt = np.bincount(b[ok], minlength=len(cen)); prof = prof / np.maximum(cnt, 1)
        ph = np.angle(prof[np.argmax(np.abs(prof))]); pr = np.real(prof * np.exp(-1j * ph)); pr = 0.5 * (pr - pr[::-1])          # нечётная часть
        gfun = (lambda pr: (lambda xx: np.interp(xx, cen, pr, left=0, right=0)))(pr / np.abs(pr).max())
    print(f'L = {L:g}: проекции 0…3: ' + ' → '.join(f'{e:.3f}' for e in out) + f'   (КГ {kg[0]:.3f}; следующий {kg[1]:.3f})', flush=True)
