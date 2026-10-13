#!/usr/bin/env python3
"""ЧЕТВЁРТАЯ ТОЧКА: n = 9, и проверка самой сходимости.

perm2iso.py: на классах изоморфизма меры расходятся, но расхождение по
доле связанных убывает 8.1% -> 6.7% -> 5.2% при n = 6,7,8, то есть
примерно как n^{-1.5}. Если так, асимптотически вес не нужен. Но средняя
кратность растёт (1.50 -> 2.73), максимальная быстро (3,4,6,12,24), и это
работает против сходимости. Трёх точек мало.

Здесь n=9 (362880 перестановок) с ускоренной канонизацией, и ГЛАВНОЕ -
показатель убывания считается по четырём точкам, а не по трём.

КОНТРОЛЬ: A:r обязано быть ровно 0.5 (точное свойство); число классов
при n=9 - это число порядков с dim <= 2 на 9 элементах, независимо
неизвестное, но оно обязано быть меньше полного числа порядков (183231).
"""
import math
from itertools import permutations, product

import numpy as np

KEYS = ('r', 'H', 'n0', 'n1', 'n2', 'n3')
N = 9
TRI = np.triu(np.ones((N, N), bool), 1)


def canon(R):
    n = N
    dn = [tuple(np.nonzero(R[:, i])[0]) for i in range(n)]
    upp = [tuple(np.nonzero(R[i, :])[0]) for i in range(n)]
    inv = [(len(dn[i]), len(upp[i])) for i in range(n)]
    for _ in range(n):
        new = [(inv[i], tuple(sorted(inv[j] for j in dn[i])),
                tuple(sorted(inv[j] for j in upp[i]))) for i in range(n)]
        enc = {v: k for k, v in enumerate(sorted(set(new)))}
        new = [enc[v] for v in new]
        if new == inv:
            break
        inv = new
    cells = {}
    for i, v in enumerate(inv):
        cells.setdefault(v, []).append(i)
    order = [cells[k] for k in sorted(cells)]
    if all(len(c) == 1 for c in order):
        perm = [c[0] for c in order]
        return R[np.ix_(perm, perm)].tobytes()
    tot = 1
    for c in order:
        tot *= math.factorial(len(c))
        if tot > 40000:
            return None
    best = None
    for combo in product(*[permutations(c) for c in order]):
        perm = [i for part in combo for i in part]
        b = R[np.ix_(perm, perm)].tobytes()
        if best is None or b < best:
            best = b
    return best


def stats(R, T):
    Iv = R.astype(np.int16) @ R.astype(np.int16)
    sz = Iv[R]
    best = np.zeros(N, int)
    for i in range(N):
        m = 0
        for j in range(i):
            if T[j] < T[i] and best[j] > m:
                m = best[j]
        best[i] = 1 + m
    return dict(r=R.sum() / (N * (N - 1) / 2), H=int(best.max()),
                n0=int((sz == 0).sum()), n1=int((sz == 1).sum()),
                n2=int((sz == 2).sum()), n3=int((sz == 3).sum()))


def main():
    print(__doc__)
    tally, acc, skip = {}, {}, 0
    for t in permutations(range(N)):
        T = np.array(t)
        R = TRI & (T[:, None] < T[None, :])
        k = canon(R)
        if k is None:
            skip += 1
            continue
        tally[k] = tally.get(k, 0) + 1
        if k not in acc:
            acc[k] = stats(R, t)
    tot, d = sum(tally.values()), len(tally)
    print(f'n={N}: перестановок {math.factorial(N)}, классов {d}, '
          f'пропущено {skip}')
    print(f'  кратность: макс {max(tally.values())}, '
          f'средняя {np.mean(list(tally.values())):.2f}')
    print(f"\n    {'величина':>10}{'A (засев)':>12}{'B′ (классы)':>13}"
          f"{'расхождение':>13}")
    out = {}
    for key in KEYS:
        a = sum(tally[x] * acc[x][key] for x in tally) / tot
        b = sum(acc[x][key] for x in tally) / d
        out[key] = (b - a) / max(abs(a), 1e-9) * 100
        print(f'    {key:>10}{a:>12.4f}{b:>13.4f}{out[key]:>12.1f}%')

    ns = np.array([6, 7, 8, 9], float)
    rs = np.array([8.1, 6.7, 5.2, out['r']])
    p = np.polyfit(np.log(ns), np.log(rs), 1)[0]
    print(f'\nрасхождение по r: ' + ' '.join(f'{v:.1f}%' for v in rs))
    print(f'показатель: расхождение ~ n^{p:+.2f}')
    print(f'  => при n=100 это {rs[-1] * (9 / 100.0) ** (-p):.3f}%'
          if p < 0 else '  => НЕ убывает')


if __name__ == '__main__':
    main()
