#!/usr/bin/env python3
"""ПЕРЕПИСЬ, дверь 1: снимает ли невыполнимость ПОЛНОЕ распределение
размеров интервалов.

Постановка (`actionexist-doc.md`). Вес сектора ln W = S + theta·f, и
доминирование засева требует

    theta·(f_засев - f_j) > S_j - S_засев   для всех соперников j

Это линейное программирование. Архив доказал НЕВЫПОЛНИМОСТЬ по восьми
объёмным счётам (плюс N4..N8 и C5) сертификатом Фаркаша: счёты засева
суть выпуклая смесь счётов соперников. И оговорка 1 там же говорит:
«любой новый счёт её может снять — что и произошло с граничным».

ЗДЕСЬ новый счёт — это ПОЛНОЕ распределение размеров интервалов,
разложенное по ЛОГАРИФМИЧЕСКИМ полосам от m=0 до m~N. Смысл: смесь,
подобранная Фаркашем, есть блин (все интервалы пустые) плюс игла (все
огромные), то есть распределение ДВУГОРБОЕ, а у засева оно гладкое.
Линейная функция не различает распределение от смеси с тем же средним
— но если взять координатами сами полосы распределения, различение
становится линейным.

ПРЕДСКАЗАНИЕ до прогона. Соперники имеют свободными число слоёв и
плотность, то есть масштаб интервалов у них непрерывно настраивается.
Поэтому я ожидаю, что оболочка всё ещё содержит засев и ответ будет
«не разделяет». Если так — дверь 1 ведёт не к новым счётам, а к
ВОГНУТОСТИ (следующий прогон). Если нет — линейное действие
существует, и закрытие было артефактом малого набора счётов.

ДВА ЯКОРНЫХ КОНТРОЛЯ (без них стенду верить нельзя):
  1. старый набор восьми объёмных счётов обязан дать НЕ РАЗДЕЛЯЕТ;
  2. он же плюс граничный счёт обязан дать РАЗДЕЛЯЕТ.
Оба числа взяты из архива, а не из этого прогона.
"""
import math

import numpy as np
from scipy.optimize import linprog

from qcarrier import sprinkle4


def closure(Rc):
    cur = Rc.copy()
    for _ in range(64):
        nxt = cur | ((cur.astype(np.float32) @ cur.astype(np.float32)) > .5)
        if (nxt == cur).all():
            return cur
        cur = nxt
    return cur


def layered(n, sizes, q, rs):
    off = np.cumsum([0] + list(sizes))
    Rc = np.zeros((n, n), bool)
    for i in range(len(sizes) - 1):
        Rc[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
            rs.random((sizes[i], sizes[i + 1])) < q)
    S = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)) * (
        -q * math.log(q) - (1 - q) * math.log1p(-q) if 0 < q < 1 else 0.0)
    return closure(Rc), S


def sprinkle_rc(n, rs):
    P = sprinkle4(n, rs, 1.0).astype(np.float64)
    dt = P[None, :, 0] - P[:, None, 0]
    d = P[None, :, 1:] - P[:, None, 1:]
    return (dt > 0) & (dt * dt > (d * d).sum(-1))


EDGES = [0, 1, 2, 3, 4, 6, 9, 14, 21, 32, 48, 72, 108, 162, 243, 365,
         548, 822, 10 ** 9]


def feats(Rc, n):
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int64)
    rel = Rc
    R = int(rel.sum())
    v = Iv[rel]
    ab = np.array([int(((v >= EDGES[i]) & (v < EDGES[i + 1])).sum())
                   for i in range(len(EDGES) - 1)], float)
    C3 = float(Iv.sum())
    C4 = float(np.rint(F @ F @ F).sum())
    hgt = np.zeros(n, np.int64)
    for _ in range(3):
        for x in np.argsort(Rc.sum(1))[::-1]:
            pre = np.nonzero(Rc[:, x])[0]
            hgt[x] = (hgt[pre].max() + 1) if len(pre) else 0
    nmax = int((~Rc.any(axis=1)).sum())
    nmin = int((~Rc.any(axis=0)).sum())
    old = np.array([R, ab[0], ab[1], ab[2], ab[3], C3, C4,
                    float(hgt.max() + 1)])
    return old, ab, np.array([float(nmax), float(nmin)]), R


def lp(G, C):
    """max t при theta·g_j - t >= c_j; theta без границ."""
    k = G.shape[1]
    A = np.hstack([-G, np.ones((len(G), 1))])
    res = linprog(np.concatenate([np.zeros(k), [-1.0]]), A_ub=A, b_ub=-C,
                  bounds=[(None, None)] * (k + 1), method='highs')
    if res.status == 3:
        return math.inf, None
    if res.status != 0:
        return None, None
    return float(res.x[-1]), res.x[:k]


def farkas(G, C):
    m = len(G)
    res = linprog(-C, A_eq=G.T, b_eq=np.zeros(G.shape[1]),
                  A_ub=np.ones((1, m)), b_ub=[1.0],
                  bounds=[(0, None)] * m, method='highs')
    if res.status != 0:
        return None
    lam = res.x
    return float(G.T @ lam @ np.ones(1) if False else
                 np.abs(G.T @ lam).max()), float(C @ lam), lam


def main():
    print(__doc__)
    n = 400
    rs = np.random.default_rng(424242)
    S4 = 4.0 * n * math.log(n)
    print(f'N = {n}, энтропия засева взята щедро: 4 N lnN = {S4:.0f}')

    o4, ab4, b4, R4 = feats(sprinkle_rc(n, rs), n)
    rows = []
    for k in list(range(2, 13)) + [16, 20, 26, 34, 44, 57, 75, 100,
                                   130, 170, 220, 300]:
        base = n // k
        for q in (0.08, 0.14, 0.22, 0.32, 0.44, 0.56, 0.68, 0.80, 0.9, .97):
            for tw in (0, 1, 2):
                sz = [base] * k
                sz[-1] += n - base * k
                if tw and k >= 4 and base > 6:
                    thin = max(1, base // (4 * tw))
                    sz[0], sz[-1] = thin, thin
                    sz[k // 2] += 2 * (base - thin) - (n - base * k) * 0
                    if sum(sz) != n:
                        sz[k // 2] += n - sum(sz)
                if min(sz) < 1 or sum(sz) != n:
                    continue
                Rc, S = layered(n, sz, q, rs)
                o, ab, b, R = feats(Rc, n)
                rows.append((o, ab, b, S, k, q, tw))
    print(f'соперников: {len(rows)}')

    sets = {
        'ЯКОРЬ 1: восемь объёмных (архив: НЕ РАЗДЕЛЯЕТ)':
            ([r[0] for r in rows], o4),
        'ЯКОРЬ 2: они же + граничный счёт (архив: РАЗДЕЛЯЕТ)':
            ([np.concatenate([r[0], r[2]]) for r in rows],
             np.concatenate([o4, b4])),
        'НОВОЕ: полное распределение по лог-полосам':
            ([r[1] for r in rows], ab4),
        'НОВОЕ: распределение + граничный':
            ([np.concatenate([r[1], r[2]]) for r in rows],
             np.concatenate([ab4, b4])),
    }
    Sj = np.array([r[3] for r in rows])
    for name, (fj, f4) in sets.items():
        Fj = np.array(fj, float)
        sc = np.maximum(np.abs(f4), 1.0)
        G = (f4[None, :] - Fj) / sc
        C = Sj - S4
        t, th = lp(G, C)
        print(f'\n{name}')
        print(f'  координат: {G.shape[1]}')
        if t is None:
            print('  ЛП не решилась')
            continue
        if t == math.inf:
            print('  РАЗДЕЛЯЕТ (запас не ограничен)')
        elif t > 0:
            print(f'  РАЗДЕЛЯЕТ, запас {t:.3g}')
        else:
            print(f'  НЕ РАЗДЕЛЯЕТ (лучший запас {t:.3g})')
        fk = farkas(G, C)
        if fk and t is not None and t <= 0:
            res, sc2, lam = fk
            top = np.argsort(lam)[::-1][:3]
            print(f'  Фаркаш: |Gᵗλ| = {res:.2e}, Σλ·C = {sc2:+.1f} '
                  f'=> сертификат {"ДЕЙСТВИТЕЛЕН" if res < 1e-6 and sc2 > 0 else "нет"}')
            print('  смесь: ' + '; '.join(
                f'λ={lam[i]:.3f} k={rows[i][4]} q={rows[i][5]} тонк={rows[i][6]}'
                for i in top if lam[i] > 1e-4))


if __name__ == '__main__':
    main()
