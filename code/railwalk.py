#!/usr/bin/env python3
"""БЛУЖДАНИЕ ПО РЕЛЬСАМ (волноводы) и Белл без отбора после подготовки.

Рельсы Алисы — две тонкие цепочки событий кристалла вдоль её оси (в её системе отсчёта x̃ = ±x̃_A),
начинаются в A₁ и A₂ = h(A₁); рельсы Боба — их отражение P. Оставляем только события рельсов и звенья
индуцированного порядка: «свои» (по рельсу) и «перескоки». Амплитуда живёт на звеньях. Событие с двумя
входами и двумя выходами — монета U(θ) = [[cos θ, i sin θ], [i sin θ, cos θ]] (свой, перескок) → (свой, перескок):
это то же «поворот = i», только поворот — это перескок между световодами. Один вход и два выхода —
изометрия; два входа и один выход — потеря (меряем). Пластинка — фаза на своём звене второго рельса до связи.
Детекторы — всё, что уходит за конец рельсов, по тому рельсу, куда приходит (каждое звено — своя запись).
Белл: пара готовится в кристалле (амплитуды в A₁, A₂ и B₁, B₂ — как в chiral.py), дальше — только рельсы;
«ничего не пришло» не выбрасывается (считается +1)."""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from beamsplit import smax
sys.argv = _a

K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
w = CH.sym_world(4.0, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)

def build_rails(A1, A2, length, delta=0.25):
    xa1, xa2, t0 = Xa[A1, 1], Xa[A2, 1], Xa[A1, 0]
    r1 = [k for k in range(n) if abs(Xa[k, 1] - xa1) < delta and t0 - 1e-9 <= Xa[k, 0] <= t0 + length]
    r1 = [k for k in r1 if CH.img(w, k, H, Hs) >= 0 and np.abs(X[k]).sum() < 23]
    r2 = [CH.img(w, k, H, Hs) for k in r1]
    assert A1 in r1 and A2 in r2
    ev = np.array(r1 + r2); rail = np.array([0] * len(r1) + [1] * len(r2))
    o = np.argsort(Xa[ev, 0], kind='stable'); ev, rail = ev[o], rail[o]
    Rs = R[np.ix_(ev, ev)]; Rf = Rs.astype(np.float32); L = Rs & ((Rf @ Rf) == 0)
    return ev, rail, L

def walk(ev, rail, L, inj, theta, plate=None):
    """inj: {индекс события: амплитуда на входе (свой рельс)}; возвращает амплитуды на выходах за концом,
    сгруппированные по рельсу прихода, и потерю нормы на событиях 2→1"""
    m = len(ev); amp = {}           # (i, j) -> амплитуда на звене
    outs = {0: [], 1: []}; lost = 0.0
    for j in range(m):
        ins = np.nonzero(L[:, j])[0]; os_ = np.nonzero(L[j, :])[0]
        own_in = [i for i in ins if rail[i] == rail[j]]; cr_in = [i for i in ins if rail[i] != rail[j]]
        own_out = [k for k in os_ if rail[k] == rail[j]]; cr_out = [k for k in os_ if rail[k] != rail[j]]
        a = sum(amp.get((i, j), 0) for i in own_in) + inj.get(j, 0); c = sum(amp.get((i, j), 0) for i in cr_in)
        th = theta(j)
        o_own, o_cr = math.cos(th) * a + 1j * math.sin(th) * c, 1j * math.sin(th) * a + math.cos(th) * c
        if not own_out and not cr_out:          # конец рельса: детектор своего рельса
            outs[rail[j]].append(o_own); outs[1 - rail[j]].append(o_cr); continue
        if own_out:
            amp[(j, own_out[0])] = o_own * (plate[1] if plate and rail[j] == 1 and plate[0](j) else 1)
        else: outs[rail[j]].append(o_own)          # нет своего выхода — уходит детектору
        if cr_out: amp[(j, cr_out[0])] = o_cr
        else: lost += abs(o_cr) ** 2 if not own_out else abs(o_cr) ** 2
    return outs, lost

LEN = 4.5
if __name__ == '__main__':
    for xa in (0.62, 1.00, 1.62):
        A1 = [k for k in range(n) if 2.4 <= Xa[k, 0] <= 2.9 and abs(Xa[k, 1] - xa) < 0.01]
        if not A1: continue
        A1 = A1[0]; A2 = CH.img(w, A1, H, Hs)
        ev, rail, L = build_rails(A1, A2, LEN)
        ins, outs_ = L.sum(0), L.sum(1)
        own_in = np.array([sum(rail[i] == rail[j] for i in np.nonzero(L[:, j])[0]) for j in range(len(ev))])
        combo = {}
        for a, b in zip(ins, outs_): combo[(int(a), int(b))] = combo.get((int(a), int(b)), 0) + 1
        i1, i2 = int(np.nonzero(ev == A1)[0][0]), int(np.nonzero(ev == A2)[0][0])
        tA = Xa[A1, 0]
        print(f'\nрельсы x̃ = ±{xa} (расстояние {2 * xa:.2f}), длина {LEN}: событий {len(ev) // 2}+{len(ev) // 2}; «свой вход» ≤ 1 у {np.mean(own_in <= 1):.0%}; (вх, исх): {dict(sorted(combo.items()))}')
        # 1) норма при сильной связи всюду
        for th in (0.0, math.pi / 8, math.pi / 4):
            o, lost = walk(ev, rail, L, {i1: 1.0}, lambda j: th)
            P = [sum(abs(z) ** 2 for z in o[r]) for r in (0, 1)]
            print(f'   связь θ = {math.degrees(th):.1f}° всюду: на детекторах {P[0]:.4f} + {P[1]:.4f} = {sum(P):.4f}; потеряно на 2→1 {lost:.4f}')
        # 2) Маха–Цендер: две зоны связи, пластинка между ними
        z1 = lambda j: tA + 0.4 <= Xa[ev[j], 0] <= tA + 1.4; z2 = lambda j: tA + 3.0 <= Xa[ev[j], 0] <= tA + 4.0
        mid = lambda j: tA + 1.8 <= Xa[ev[j], 0] <= tA + 2.6
        for thc in (math.pi / 8, math.pi / 4):
            th = lambda j: thc if (z1(j) or z2(j)) else 0.0
            vis = []
            for ph in np.linspace(0, 2 * math.pi, 9)[:-1]:
                o, lost = walk(ev, rail, L, {i1: 1.0}, th, plate=(mid, np.exp(1j * ph)))
                P = [sum(abs(z) ** 2 for z in o[r]) for r in (0, 1)]; vis.append((P[0], P[1], lost))
            p0 = np.array([v[0] for v in vis]); tot = np.array([v[0] + v[1] for v in vis])
            print(f'   Маха–Цендер θ = {math.degrees(thc):.1f}° в двух зонах: P(рельс 1) по фазе пластинки {np.round(p0, 3).tolist()}; '
                  f'полная вероятность {tot.min():.4f}…{tot.max():.4f}')
        sys.stdout.flush()
