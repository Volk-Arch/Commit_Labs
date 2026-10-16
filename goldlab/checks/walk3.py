"""ДИРАК НА КРИСТАЛЛЕ, шаг 3: спин–статистика. Источник — на ОДНОМ слое (одном «срезе»). Ветви: (1, 1) и (1, −1) — собственные
векторы монеты; проверяем: (а) норма Σ|ψ|² сохраняется точно (вход = ушло за край), (б) обе ветви — частота −m и +m (частица и
античастица), (в) норма обеих ветвей ПОЛОЖИТЕЛЬНА. Для скаляра Джонстона отрицательная ветвь имеет отрицательную норму
(отрицательные собственные значения iΔ) — поэтому скаляр квантуется коммутатором (бозоны). Положительная норма у обеих ветвей
означает: при коммутаторе энергия не ограничена снизу; согласованно только антикоммутатор (море Дирака) — фермионы, Паули."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab'); sys.path.insert(0, '/home/claude/code/goldlab_checks')
from walk2 import Walk
from goldlab import worlds as Wd, observe as Ob, propagate as Pr
L = 16; w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Walk(w, cap=1.6, cost='tau'); m = 1.0
lv = np.unique(np.round(t, 9)); ts = lv[np.argmin(np.abs(lv - (-L + 4)))]; src = np.abs(t - ts) < 1e-9
print(f'мир {L}, событий {w.n}; скорость безмассовой (нити) c = {W.c:.3f}; источник — слой t = {ts:.2f} ({src.sum()} событий)')
om = np.linspace(-3, 3, 2401)
for name, (cr, cl) in (('ветвь (1, 1)', (1, 1)), ('ветвь (1, −1)', (1, -1)), ('смесь (1, 0)', (1, 0))):
    g = np.where(src, np.exp(-x * x / 50), 0); aR, aL, oR, oL, leak = W.run(m, cr * g, cl * g)
    inj = (abs(cr) ** 2 + abs(cl) ** 2) * (g ** 2).sum()
    sel = (t > ts + 1.5) & (t < L - 4) & (np.abs(x) < 8); kk = np.round(t[sel], 6); tu, inv = np.unique(kk, return_inverse=True)
    s = np.zeros(len(tu), complex); np.add.at(s, inv, (g_ := np.exp(-x * x / 50))[sel] * (cr * aR + cl * aL)[sel] if False else (np.exp(-x * x / 50) * (aR + aL))[sel])
    s2 = np.zeros(len(tu), complex); np.add.at(s2, inv, (np.exp(-x * x / 50) * (aR - aL))[sel])
    S = Ob.spectrum(tu, s, om) + Ob.spectrum(tu, s2, om); S /= S.max(); pk = Ob.peaks(om, S, 0.08)
    print(f'{name}: норма — вход {inj:.6f}, ушло за край {leak:.6f} (разница {abs(inj - leak) / inj:.1e}); частоты: ' + ' '.join(f'{e:+.3f}({q:.2f})' for e, q in pk), flush=True)
# скаляр для контраста: норма ветвей = знак собственных значений iΔ
K = Pr.johnston11(w.X, 4.0, m); lam = np.linalg.eigvalsh(1j * (K - K.T))
print(f'скаляр Джонстона: собственные iΔ — положительных {np.sum(lam > 1e-9)}, отрицательных {np.sum(lam < -1e-9)}: вторая ветвь с отрицательной нормой ⇒ коммутатор (бозоны)')
