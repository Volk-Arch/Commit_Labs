#!/usr/bin/env python3
"""КУДА ВЫГИБАЕТСЯ ЦЕПЬ: исправление моей же ошибки в знаке.

МОЯ ОШИБКА (delay-doc.md §4). Я написал, что дырка «отталкивает цепи, и
это неверный знак для массы», и объявил две половины тяготения
несовместимыми. Знак я взял из евклидовой интуиции.

ПРАВИЛЬНО. В ОТО геодезическая между двумя событиями на ОДНОЙ высоте,
разделёнными временем, выгибается ОТ массы. Это брошенный вверх мяч:
наверху часы идут быстрее, выигрыш гравитационного замедления
перевешивает проигрыш от скорости, и максимум собственного времени
достигается на параболе, уходящей ВВЕРХ. Геодезические отходят от
массы, а не прижимаются к ней. Притяжение — это то, что мяч
возвращается, а не то, что путь прижат.

ЗНАЧИТ ПРЕДСКАЗАНИЕ ПЕРЕВОРАЧИВАЕТСЯ:

    дырка = масса   =>  часы медленнее (померено, 4.5 сигмы)
                        И цепи выгибаются НАРУЖУ, от оси
    трубка          =>  часы быстрее (померено, +0.36)
                        И цепи выгибаются ВНУТРЬ, к оси

Если так, то дырка даёт обе половины ПРАВИЛЬНО, а трубка обе неправильно
и согласованно — и мой вывод о несовместимости снимается.

ЧТО МЕРИМ. Пары событий на ОДНОМ радиусе d от оси, в одном
направлении, разделённые временем. Максимальная цепь между ними; средний
радиус её внутренних событий минус d. Положительное = выгиб наружу.

КОНТРОЛЬ обязателен: в ромбе есть собственный выгиб (к середине, где
событий больше), поэтому сравнение только с миром, где удалено или
добавлено столько же событий случайно.

  python3 bow.py
  python3 bow.py --rad 0.24 --mass 240 --seeds 12
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel
from delay import chain_tube
from detour import longest_between


def bow_of(R, X, a, b):
    """Средний радиус внутренних событий максимальной цепи от a до b."""
    if not R[b, a]:
        return None
    mid = np.flatnonzero(R[:, a] & R[b, :])
    if len(mid) < 2:
        return None
    nodes = np.concatenate(([a], mid, [b]))
    sub = R[np.ix_(nodes, nodes)]
    n = len(nodes)
    order = np.argsort(sub.sum(1), kind='stable')
    L = np.full(n, -1, np.int64)
    prev = np.full(n, -1, np.int64)
    L[0] = 0
    for i in order:
        i = int(i)
        pre = np.flatnonzero(sub[i])
        if len(pre) == 0:
            continue
        j = pre[np.argmax(L[pre])]
        if L[j] >= 0:
            L[i], prev[i] = L[j] + 1, j
    end = n - 1
    if L[end] < 2:
        return None
    path, cur = [], end
    while cur > 0:
        path.append(int(nodes[cur]))
        cur = int(prev[cur])
        if cur < 0:
            return None
    path = path[1:]                     # без конечного события
    if not path:
        return None
    return float(np.linalg.norm(X[path][:, 1:], axis=1).mean())


def pairs_same_radius(X, T, d, tol, rs, npair=70):
    """Пары на одном радиусе d и в одном направлении, разнесённые во времени."""
    r = np.linalg.norm(X[:, 1:], axis=1)
    band = (np.abs(r - d) < tol) & (r > 1e-9)
    lo = np.flatnonzero(band & (X[:, 0] < 0.40 * T))
    hi = np.flatnonzero(band & (X[:, 0] > 0.60 * T))
    if len(lo) < 3 or len(hi) < 3:
        return []
    out = []
    for _ in range(npair * 16):
        if len(out) >= npair:
            break
        a, b = int(rs.choice(lo)), int(rs.choice(hi))
        ua = X[a, 1:] / np.linalg.norm(X[a, 1:])
        ub = X[b, 1:] / np.linalg.norm(X[b, 1:])
        if float(ua @ ub) < 0.85:            # одно направление
            continue
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description='Куда выгибается цепь')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rad', type=float, default=0.24)
    ap.add_argument('--mass', type=int, default=240)
    ap.add_argument('--tuberad', type=float, default=0.02)
    ap.add_argument('--seeds', type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}; пары на одном радиусе, разнесённые во времени')
    print('мерим средний радиус внутренних событий максимальной цепи')
    print('ПОЛОЖИТЕЛЬНОЕ = выгиб НАРУЖУ (от оси), как брошенный мяч от массы')
    print(f'дырка радиуса {a.rad}; трубка из {a.mass} событий')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'объект':>10}{'радиус пар':>12}{'объект':>11}{'контроль':>11}"
          f"{'выгиб':>10}{'сигм':>7}{'цепь об':>9}{'цепь кон':>10}")

    for kind in ('дырка', 'трубка'):
        for d in (0.34, 0.46, 0.58):
            got = {True: ([], []), False: ([], [])}
            for t in range(a.seeds):
                rs = np.random.default_rng(2501000 + 977 * t)
                X0 = diamond(a.t, rs)
                keep = np.linalg.norm(X0[:, 1:], axis=1) >= a.rad
                k = int((~keep).sum())
                for mk in (True, False):
                    if kind == 'дырка':
                        if mk:
                            X = X0[keep]
                        else:
                            ii = rs.choice(len(X0), len(X0) - k, replace=False)
                            X = X0[np.sort(ii)]
                    else:
                        ob = (chain_tube(a.mass, a.tuberad, a.t, rs) if mk
                              else diamond(a.t, rs, n_at_t1=a.mass / a.t ** 4))
                        X = np.vstack([ob, X0])
                    R = cone_rel(X)
                    bs, ls = [], []
                    for p, q in pairs_same_radius(X, a.t, d, 0.05, rs):
                        v = bow_of(R, X, p, q)
                        if v is not None:
                            bs.append(v - d)
                            ls.append(longest_between(R, p, q))
                    if bs:
                        got[mk][0].append(float(np.mean(bs)))
                        got[mk][1].append(float(np.mean(ls)))
            h, c = np.array(got[True][0]), np.array(got[False][0])
            hl, cl = np.array(got[True][1]), np.array(got[False][1])
            if len(h) < 2 or len(c) < 2:
                print(f'{kind:>10}{d:>12.2f}{"мало пар":>11}')
                continue
            dd = h.mean() - c.mean()
            er = math.hypot(h.std() / math.sqrt(len(h)),
                            c.std() / math.sqrt(len(c)))
            print(f'{kind:>10}{d:>12.2f}{h.mean():>+11.4f}{c.mean():>+11.4f}'
                  f'{dd:>+10.4f}{abs(dd)/max(er,1e-12):>7.1f}'
                  f'{hl.mean():>9.2f}{cl.mean():>10.2f}', flush=True)

    print('\n  дырка: выгиб наружу + часы медленнее => обе половины как у массы')
    print('  трубка: выгиб внутрь + часы быстрее   => обе наоборот')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
