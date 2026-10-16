#!/usr/bin/env python3
"""«НОГИ В ТЕНЬ», шаг 3: замкнутая система «мир + тень».

Звенья мира — звенья порядка с Δt ≤ DT (длинные почти-световые звенья, тянущиеся к краю мира, — именно они
дают «разбалансировку по времени»: без них корреляция (исх − вх) со временем 0.02 вместо −0.36).
Скрытые звенья — спаривание лишних ног по близости теней (shadowlegs.py). Незакрытые ноги — внешние (край).
Каждое событие — унитарная матрица D × D, D = вх = исх после добавления скрытых ног. Матрица — дискретное
Фурье по ногам, упорядоченным по быстроте (скрытые — в конце); множитель поворота i за смену «лево ↔ право»
здесь не вводим — проверяем только норму и интерференцию.
Меряем: (1) полная норма на выходе = 1 точно? (2) доля амплитуды в мире и в тени по срезам;
(3) две щели: видность полос в мире, когда часть амплитуды гуляет через тень."""
import sys, math, numpy as np
from scipy.optimize import linear_sum_assignment
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
import shadowlegs as SL
sys.argv = _a
DT = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); L = CH.links_of(R); X = w.X; n = w.n
I, J = np.nonzero(L); dt = X[J, 0] - X[I, 0]; keep = dt <= DT; I, J = I[keep], J[keep]
reg = np.abs(X).sum(1) < 14
ins, outs = np.bincount(J, minlength=n), np.bincount(I, minlength=n)
exo = np.maximum(ins - outs, 0) * reg; exi = np.maximum(outs - ins, 0) * reg
ha, hb, _, _ = SL.match(w, R, X, exo, exi, 'тень')
print(f'Δt ≤ {DT}: звеньев мира {len(I)}, скрытых {len(ha)}; лишних ног {exo.sum()} → незакрытых {exo.sum() - len(ha)} (уходят за край)')
# ноги: список (откуда, куда, вид); вид 0 — мир, 1 — тень
edges = [(int(a), int(b), 0) for a, b in zip(I, J) if reg[a] and reg[b]] + [(int(a), int(b), 1) for a, b in zip(ha, hb)]
E = len(edges); ein = {k: [] for k in range(n)}; eout = {k: [] for k in range(n)}
for e, (a, b, kind) in enumerate(edges): eout[a].append(e); ein[b].append(e)
order = [k for k in np.argsort(X[:, 0]) if reg[k]]
def rap(e):
    a, b, kind = edges[e]
    if kind: return 10.0 + e * 1e-6
    d = X[b] - X[a]; return math.atanh(d[1] / d[0])
def run(inj, phase_edges=None):
    """inj: {событие: амплитуда на «внешнем входе»}; возвращает амплитуды на ногах и вытекшее наружу"""
    amp = np.zeros(E, complex); leaked = 0.0
    for k in order:
        li = sorted(ein[k], key=rap); lo = sorted(eout[k], key=rap)
        D = max(len(li), len(lo)); ext_in = D - len(li); ext_out = D - len(lo)
        vin = np.zeros(D, complex); vin[:len(li)] = amp[li]
        if k in inj: vin[len(li) if ext_in else 0] += inj[k]
        if D == 0: continue
        F = np.exp(2j * math.pi * np.outer(np.arange(D), np.arange(D)) / D) / math.sqrt(D)
        vout = F @ vin
        amp[lo] = vout[:len(lo)]
        if phase_edges is not None:
            for e in lo:
                if e in phase_edges: amp[e] *= phase_edges[e]
        leaked += (np.abs(vout[len(lo):]) ** 2).sum()
    # ноги, уходящие из области (конец ноги вне области) не входят в edges: учтены как ext_out
    return amp, leaked
if __name__ == '__main__':
    src = int(np.argmin(((X - np.array([-9.0, 0.0])) ** 2).sum(1)))
    amp, leaked = run({src: 1.0})
    src_out = sum(abs(a) ** 2 for a in [])
    print(f'источник {np.round(X[src], 2)}: вытекло через край области {leaked:.6f}')
    for t in np.arange(-8, 10, 2.0):
        cw = [e for e, (a, b, kind) in enumerate(edges) if X[a, 0] < t <= X[b, 0] and kind == 0]
        ch = [e for e, (a, b, kind) in enumerate(edges) if X[a, 0] < t <= X[b, 0] and kind == 1]
        # вытекло к моменту t: считаем полную норму = в полёте + уже вытекло до t
        print(f'  срез t = {t:+.0f}: в мире {np.sum(np.abs(amp[cw]) ** 2):.4f}, в тени {np.sum(np.abs(amp[ch]) ** 2):.4f}')
