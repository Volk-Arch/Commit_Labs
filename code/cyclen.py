import math
import sys

import numpy as np

sys.path.insert(0, '/home/claude/code')
from majority import rel_threshold, shortest_cycle

print('ДЛИНА КРАТЧАЙШЕГО ЦИКЛА:  предсказание  L = ceil(k/(k-m))')
print('(короткие циклы: они собираются уже в небольшом мире)')
print(f"{'k':>3}{'m':>4}{'предсказ.':>11}{'наблюдали':>11}{'сходится':>10}")
ok = tot = 0
for k in range(3, 13):
    for m in range(k - 1, k // 2, -1):
        pred = math.ceil(k / (k - m))
        if pred > 4:
            continue
        best = 99
        for s in range(3):
            X = np.random.default_rng(7 * s + 31 * k + m).random((700, k))
            L = shortest_cycle(rel_threshold(X, m), cap=pred + 2)
            if 0 < L < best:
                best = L
        obs = best if best < 99 else -1
        good = (obs == pred)
        ok += good
        tot += 1
        print(f'{k:>3}{m:>4}{pred:>11}'
              f'{(obs if obs > 0 else "не нашли"):>11}'
              f'{("да" if good else "НЕТ"):>10}')
print()
print(f'  подтвердилось: {ok} из {tot}')
print()
print('Длинные циклы (m=k-1, предсказание L=k) — проверено отдельно:')
print('  k=5 -> 5, k=6 -> 6, k=7 -> 7 при N выше критического')
