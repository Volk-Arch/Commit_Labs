#!/usr/bin/env python3
"""ПЕРЕПИСЬ, дверь 1: выживает ли разделение против ШИРОКОГО набора.

census2.py (оба якоря воспроизведены) показал: полное распределение
размеров интервалов по лог-полосам РАЗДЕЛЯЕТ засев от слоистых, тогда
как восемь объёмных счётов не разделяют. Но оговорка 2 архива: любая
выполнимость предварительна относительно набора соперников, а там были
только слоистые.

Два новых семейства соперников, и второе — настоящая проверка:

  1. ПЕРКОЛЯЦИЯ при разных p. Энтропия взята щедро как M·H2(p),
     M = N(N-1)/2 — это завышение в пользу соперника, то есть
     безопасная сторона для утверждения о разделимости.

  2. ЗАСЕВЫ ДРУГИХ РАЗМЕРНОСТЕЙ d = 2, 3, 5, 6 с той же энтропией
     4N lnN (архив, оговорка 3: энтропия взята одна для всех d). Тогда
     требование сводится к чистому разделению счётов, без энтропии:
     theta·(f_4 - f_d) > 0.

ПОЧЕМУ ЭТО РЕШАЮЩЕЕ. Граничный счёт разделяет засев от слоистых, но с
ним разделимой становится КАЖДАЯ размерность — значит он выделяет
«похожесть на засев», а четвёрка задаётся коэффициентами, а не
выводится. Если распределение интервалов отделит d=4 от d=3 и d=5, это
первый набор счётов, который видит размерность. Если нет — дверь ведёт
туда же, куда граничный счёт.

ПРЕДСКАЗАНИЕ до прогона: отделит. Распределение размеров интервалов
стремится к Gamma(m+1/2)/m! в d=4 (`nm-doc.md`), а в других d у него
другая форма — то есть размерность сидит в самой форме распределения,
а не в её среднем.
"""
import math

import numpy as np

from census1 import EDGES, closure, feats, sprinkle_rc
from census2 import certify, make, solve, verdict
from sectors import sprinkle_d


def perc(n, p, rs):
    Rc = np.triu(rs.random((n, n)) < p, 1)
    Rc = closure(Rc)
    M = n * (n - 1) / 2
    h2 = (-p * math.log(p) - (1 - p) * math.log1p(-p)) if 0 < p < 1 else 0.
    o, ab, b, R = feats(Rc, n)
    return o, ab, b, M * h2, ('перк', round(p, 4), '-')


def sprink_d(n, d, rs, S4):
    P = sprinkle_d(n, d, rs).astype(np.float64)
    dt = P[None, :, 0] - P[:, None, 0]
    dd = P[None, :, 1:] - P[:, None, 1:]
    Rc = (dt > 0) & (dt * dt > (dd * dd).sum(-1))
    o, ab, b, R = feats(Rc, n)
    return o, ab, b, S4, ('засев', d, '-')


def main():
    print(__doc__)
    n, rs = 400, np.random.default_rng(777001)
    S4 = 4.0 * n * math.log(n)
    o4, ab4, b4, _ = feats(sprinkle_rc(n, rs), n)

    pool = []
    for k in (2, 3, 4, 6, 9, 14, 22, 34, 55, 90, 140, 220):
        for q in (0.03, 0.08, 0.2, 0.45, 0.75, 0.95):
            for kind in (0, 1, 2):
                m = make(n, k, q, kind, rs)
                if m:
                    pool.append(m)
    nl = len(pool)
    for p in (0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5):
        pool.append(perc(n, p, rs))
    npc = len(pool) - nl
    for d in (2, 3, 5, 6):
        for _ in range(3):
            pool.append(sprink_d(n, d, rs, S4))
    print(f'соперники: слоистых {nl}, перколяций {npc}, '
          f'засевов иных d {len(pool)-nl-npc}; всего {len(pool)}')

    tests = [
        ('восемь объёмных', lambda r: r[0], o4),
        ('объёмные + граничный', lambda r: np.concatenate([r[0], r[2]]),
         np.concatenate([o4, b4])),
        ('распределение по лог-полосам', lambda r: r[1], ab4),
        ('распределение + граничный',
         lambda r: np.concatenate([r[1], r[2]]),
         np.concatenate([ab4, b4])),
    ]
    for nm, pick, f4 in tests:
        Fj = np.array([pick(r) for r in pool], float)
        Sj = np.array([r[3] for r in pool])
        sc = np.maximum(np.abs(f4), 1.0)
        G = (f4[None, :] - Fj) / sc
        C = Sj - S4
        t, th = solve(G, C)
        print(f'\n{nm} ({G.shape[1]} координат): {verdict(t)}')
        if t is not None and t <= 0:
            c = certify(G, C)
            if c:
                res, sl, lam = c
                top = np.argsort(lam)[::-1][:3]
                print(f'  Фаркаш: |Gᵗλ|={res:.1e}, Σλ·C={sl:+.2f} => '
                      f'{"ДЕЙСТВИТЕЛЕН" if res<1e-6 and sl>0 else "нет"}')
                print('  смесь: ' + '; '.join(
                    f'λ={lam[i]:.3f} {pool[i][4]}'
                    for i in top if lam[i] > 1e-4))
        else:
            # кто ближе всех к нарушению и какие полосы несут разделение
            marg = G @ th - C
            o = np.argsort(marg)[:4]
            print('  ближайшие соперники: ' + '; '.join(
                f'{pool[i][4]} запас {marg[i]:+.3g}' for i in o))
            w = np.abs(th) * np.abs(G).mean(axis=0)
            iw = np.argsort(w)[::-1][:4]
            print('  несущие координаты (номер, вклад): ' + ', '.join(
                f'{int(i)}:{w[i]:.3g}' for i in iw))
            if nm.startswith('распределение по'):
                print('  полосы m: ' + ', '.join(
                    f'[{EDGES[int(i)]}..{EDGES[int(i)+1] if EDGES[int(i)+1]<10**8 else "∞"})'
                    for i in iw))
                zero = [j for j in range(G.shape[1])
                        if Fj[:, j].max() == 0 and f4[j] > 0]
                print(f'  полосы, где у ВСЕХ соперников ноль, а у засева '
                      f'нет: {zero if zero else "таких нет"}')


if __name__ == '__main__':
    main()
