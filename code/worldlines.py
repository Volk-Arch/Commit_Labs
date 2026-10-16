#!/usr/bin/env python3
"""ЧАСТИЦА КАК СУПЕРПОЗИЦИЯ МИРОВЫХ ЛИНИЙ.

Каждая цепь событий — сама себе световод (вдоль неё вход = выход = 1). Амплитуда ставится на целую цепь:
  a(γ) = √p(γ) · i^{число поворотов},
где p(γ) — вероятность этой мировой линии у случайного блуждающего, который в каждом событии выбирает одно из
исходящих звеньев поровну (p = Π 1/исх), а поворот — смена «лево ↔ право» (метки киральности 0 ↔ 1; звенья
«и туда, и туда» поворота не делают). Тогда Σ_γ |a(γ)|² = 1 точно на любом срезе (это просто вероятность
классического блуждания) — нормировка из самой картины, без подгонки и без κ.
Квантовость — в том, что цепи, приходящие на одно звено, складываются амплитудами. Меряем:
  C(t) = Σ_{звенья через срез t} Σ_{цепи через звено} |a|²  — классическая (должна быть 1, кроме ушедших за край мира);
  N(t) = Σ_{звенья через срез t} |Σ_{цепи} a|²            — квантовая норма.
Если N(t) = C(t) на каждом срезе — интерференция перераспределяет, но не создаёт и не губит вероятность."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

def setup(rho=4.0, Lw=24.0):
    w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1)
    I, J, lab = LK['I'], LK['J'], LK['lab']
    out = np.bincount(I, minlength=w.n)
    # для каждого события: список входящих и исходящих звеньев (индексы в массивах LK)
    order_out = np.argsort(I, kind='stable'); so = np.searchsorted(I[order_out], np.arange(w.n)); eo = np.searchsorted(I[order_out], np.arange(w.n), side='right')
    return w, R, LK, out, order_out, so, eo

def turn(l_in, l_out, phase):
    """множитель за переход с входящего звена на исходящее"""
    t = np.ones(np.broadcast(l_in, l_out).shape, complex)
    t[(l_in == 0) & (l_out == 1) | (l_in == 1) & (l_out == 0)] = phase
    return t

def run(w, LK, out, order_out, so, eo, sources, phase=1j, field=None):
    """sources: {событие: амплитуда}; возвращает амплитуды a и вероятности p на звеньях"""
    I, J, lab, st, en = LK['I'], LK['J'], LK['lab'], LK['st'], LK['en']
    nl = len(I); A = np.zeros(nl, complex); Pc = np.zeros(nl)
    fl = np.ones(nl, complex) if field is None else field
    first = min(sources)
    for j in range(first, w.n):
        outs = order_out[so[j]:eo[j]]
        if len(outs) == 0: continue
        k = 1 / math.sqrt(len(outs))
        ins = np.arange(st[j], en[j])               # входящие звенья в j (в порядке LK)
        if j in sources:
            A[outs] += sources[j] * k * fl[outs]; Pc[outs] += abs(sources[j]) ** 2 / len(outs)
        if len(ins):
            # амплитуда на каждом исходящем: сумма по входящим с множителем поворота
            tm = turn(lab[ins][:, None], lab[outs][None, :], phase)
            A[outs] += k * (A[ins][:, None] * tm).sum(0) * fl[outs]
            Pc[outs] += Pc[ins].sum() / len(outs)
    return A, Pc

if __name__ == '__main__':
    w, R, LK, out, oo, so, eo = setup(); X = w.X; I, J = LK['I'], LK['J']
    print(f'мир ρ = 4, ромб 24: событий {w.n}, звеньев {len(I)}')
    src = int(np.argmin(((X - np.array([-14.0, 0.0])) ** 2).sum(1)))
    times = np.arange(-12, 13, 3.0)
    for name, ph in (('без фазы (классика)', 1.0), ('поворот = i', 1j), ('поворот = −1', -1.0), ('поворот = e^{iπ/3}', np.exp(1j * math.pi / 3))):
        A, Pc = run(w, LK, out, oo, so, eo, {src: 1.0}, phase=ph)
        rows = []
        for t in times:
            cr = (X[I, 0] < t) & (X[J, 0] >= t)
            rows.append((Pc[cr].sum(), (np.abs(A[cr]) ** 2).sum()))
        rows = np.array(rows)
        print(f'  {name}: C(t) = ' + ' '.join(f'{c:.3f}' for c in rows[:, 0]) + '\n' + ' ' * (len(name) + 4) + 'N(t) = ' + ' '.join(f'{v:.3f}' for v in rows[:, 1]))
        sys.stdout.flush()
