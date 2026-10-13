#!/usr/bin/env python3
"""ТО ЖЕ С ТОЧНОСТЬЮ ДО ИЗОМОРФИЗМА: метки нефизичны.

perm2d.py: отображение tau -> P(tau) взаимно однозначно (различных
помеченных порядков ровно n! при всех n, кратность всегда 1), поэтому
мера A (равномерно по перестановкам = засев) и мера B (равномерно по
помеченным порядкам) СОВПАДАЮТ до 0.0%. Контроль: A:r = 0.5000 ровно.

Но причинное множество - объект АБСТРАКТНЫЙ, метки нефизичны, и
суммировать надо по классам изоморфизма. Там кратности появляются: разные
перестановки дают изоморфные порядки (простейший пример при n=3 -
«одна связь плюс изолированный элемент» выходит из двух перестановок).
Если мера A, спущенная на классы, не равна равномерной по классам, то
кинематического ограничения МАЛО, и недостающий вес есть ровно число
реализаторов класса, то есть число способов вложить.

КОНТРОЛЬ, жёсткий и независимый: число классов обязано совпасть с
утренним замером числа порядков с dim <= 2 (`dimcount.py`): 16, 63, 315,
1956 при n = 4, 5, 6, 7. Если канонизация неверна, числа не сойдутся.
"""
import math
from itertools import permutations, product

import numpy as np

KNOWN_DIM2 = {4: 16, 5: 63, 6: 315, 7: 1956}
KEYS = ('r', 'H', 'n0', 'n1', 'n2', 'n3')


def build(t, n):
    T = np.array(t)
    return np.triu(np.ones((n, n), bool), 1) & (T[:, None] < T[None, :])


def canon(R, cap=40000):
    n = R.shape[0]
    down = [np.nonzero(R[:, i])[0] for i in range(n)]
    up = [np.nonzero(R[i, :])[0] for i in range(n)]
    inv = [(len(down[i]), len(up[i])) for i in range(n)]
    for _ in range(n):
        new = [(inv[i], tuple(sorted(inv[j] for j in down[i])),
                tuple(sorted(inv[j] for j in up[i]))) for i in range(n)]
        enc = {v: k for k, v in enumerate(sorted(set(new)))}
        new = [enc[v] for v in new]
        if new == inv:
            break
        inv = new
    cells = {}
    for i, v in enumerate(inv):
        cells.setdefault(v, []).append(i)
    order = [cells[k] for k in sorted(cells)]
    sizes = [math.factorial(len(c)) for c in order]
    total = 1
    for s in sizes:
        total *= s
        if total > cap:
            return None
    best = None
    for combo in product(*[permutations(c) for c in order]):
        perm = [i for part in combo for i in part]
        b = R[np.ix_(perm, perm)].tobytes()
        if best is None or b < best:
            best = b
    return best


def stats(R, t):
    n = R.shape[0]
    Iv = R.astype(np.int16) @ R.astype(np.int16)
    sz = Iv[R]
    T = np.array(t)
    best = np.zeros(n, int)
    for i in range(n):
        best[i] = 1 + max([best[j] for j in range(i) if T[j] < T[i]],
                          default=0)
    return dict(r=R.sum() / (n * (n - 1) / 2), H=int(best.max()),
                n0=int((sz == 0).sum()), n1=int((sz == 1).sum()),
                n2=int((sz == 2).sum()), n3=int((sz == 3).sum()))


def main():
    print(__doc__)
    print(f"{'n':>3}{'перестановок':>13}{'классов':>9}{'известно':>10}"
          f"{'макс кратн.':>13}{'ср. кратн.':>12}{'пропущено':>11}")
    data = {}
    for n in range(4, 9):
        tally, acc, skip = {}, {}, 0
        for t in permutations(range(n)):
            R = build(t, n)
            k = canon(R)
            if k is None:
                skip += 1
                continue
            tally[k] = tally.get(k, 0) + 1
            if k not in acc:
                acc[k] = stats(R, t)
        kn = KNOWN_DIM2.get(n, 0)
        ok = '' if (kn == 0 or len(tally) == kn) else '  <-- НЕ СОШЛОСЬ'
        data[n] = (tally, acc)
        print(f'{n:>3}{math.factorial(n):>13}{len(tally):>9}{kn:>10}'
              f'{max(tally.values()):>13}'
              f'{np.mean(list(tally.values())):>12.2f}{skip:>11}{ok}',
              flush=True)

    print('\nмера A (засев, по перестановкам) против B′ (равномерно по КЛАССАМ):')
    for n in (6, 7, 8):
        tally, acc = data[n]
        tot = sum(tally.values())
        d = len(tally)
        a = {k: sum(tally[x] * acc[x][k] for x in tally) / tot for k in KEYS}
        b = {k: sum(acc[x][k] for x in tally) / d for k in KEYS}
        print(f'  n={n}:  ' + '  '.join(
            f'{k}: {a[k]:.4f} / {b[k]:.4f} ({(b[k] - a[k]) / max(abs(a[k]), 1e-9) * 100:+.1f}%)'
            for k in ('r', 'H', 'n0')))
    n = 8
    tally, acc = data[n]
    tot, d = sum(tally.values()), len(tally)
    print(f'\n  подробно при n={n}:')
    print(f"    {'величина':>10}{'A (засев)':>12}{'B′ (классы)':>13}"
          f"{'расхождение':>13}")
    for k in KEYS:
        a = sum(tally[x] * acc[x][k] for x in tally) / tot
        b = sum(acc[x][k] for x in tally) / d
        print(f'    {k:>10}{a:>12.4f}{b:>13.4f}'
              f'{(b - a) / max(abs(a), 1e-9) * 100:>12.1f}%')


if __name__ == '__main__':
    main()
