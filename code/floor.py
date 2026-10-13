#!/usr/bin/env python3
"""ПОЛ СЛУЧАЙНОГО БЛУЖДАНИЯ: исправление моей же утренней ошибки.

ЧТО БЫЛО НЕ ТАК. В `band.py` амплитуда многообразного сектора
оценивалась как `count · e^{−σ²/2ℏ²}` — это формула для СРЕДНЕГО
`⟨e^{iS/ℏ}⟩`, верная лишь пока среднее доминирует. Но у суммы `n`
слагаемых со случайными фазами есть ПОЛ: `|Σ| ≈ √n`. Правильно

    |Z| ≈ max( n·e^{−σ²/2ℏ²},  √n ),

и при `σ = 2.0N` экспоненциальный член ничтожен, значит работает пол.
Я взял подавленный член вместо пола и получил запрет, которого нет.

ЧТО ПРОВЕРЯЕТСЯ ЗДЕСЬ, замером а не рассуждением:

  1. ПОЛ. Для выборки посевов: `|Σe^{iS/ℏ}| / √n` обязано быть `O(1)`
     и НЕ зависеть от `n`, когда `σ/ℏ ≫ 1`. Если так — пол реален, и
     утреннее заключение отменяется.
  2. ТОЧНАЯ ФОРМУЛА для слоистых: прямым перебором всех `2^M`
     конфигураций при малом `M` сверить
     `|Σe^{iS/ℏ}| = |2cos(k_d/2ℏ)|^M`. Это проверяет и само
     разложение, и то, что `S = k_d(N − N₀)` в плотной области.
  3. СРАВНЕНИЕ полов: у слоистых точная сумма `|2cos|^M` против их
     собственного пола `2^{M/2}`. Ниже пола она может быть только за
     счёт СИСТЕМАТИЧЕСКОГО гашения — это и есть механизм.

  python3 floor.py
"""
import math
import time
import itertools

import numpy as np

from bd import action, layers
from sprinkle import sprinkle_d
from uphill import gen3

K4 = 4.0 / math.sqrt(6.0)
PARTS = (0.25, 0.5, 0.25)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 100)
    print('ПРОВЕРКА 1. Пол √n у посевов. N = 1200, действия собраны один раз')
    S = []
    for s in range(400):
        A = sprinkle_d(1200, 4, np.random.default_rng(60000 + 7919 * s))
        S.append(action(A, 4)[0])
        del A
    S = np.array(S, float)
    sd = S.std(ddof=1)
    print(f'  собрано {len(S)} действий, σ = {sd:.1f} = {sd/1200:.3f}·N'
          f'  ({time.time()-t0:.0f} с)')
    print()
    print(f"{'ℏ':>8}{'σ/ℏ':>10}{'n=50':>12}{'n=100':>12}{'n=200':>12}"
          f"{'n=400':>12}{'пол?':>8}")
    for hb in (0.4, 1.0, 4.0, 40.0, 400.0, 4000.0):
        cells, vals = [], []
        for n in (50, 100, 200, 400):
            z = np.abs(np.exp(1j * S[:n] / hb).sum()) / math.sqrt(n)
            cells.append(f'{z:>12.3f}')
            vals.append(z)
        flat = max(vals) / max(min(vals), 1e-12) < 4.0
        print(f'{hb:>8.3g}{sd/hb:>10.3g}' + ''.join(cells)
              + f'{("ДА" if flat else "нет"):>8}')
    print('  («ДА» = отношение к √n не растёт с n, то есть это пол, а не')
    print('   когерентная сумма; у когерентной росло бы как √n)')
    print()
    print('=' * 100)
    print('ПРОВЕРКА 2. Точная формула для слоистых: полный перебор')
    print(f"{'слои':>14}{'M':>5}{'ℏ':>8}{'|Σ| перебор':>16}"
          f"{'|2cos(k/2ℏ)|^M':>18}{'отн.':>9}")
    for n1, n2, n3 in ((2, 2, 2), (2, 3, 2), (3, 2, 3)):
        n = n1 + n2 + n3
        M = n1 * n2 + n2 * n3
        if M > 18:
            continue
        i1 = list(range(n1))
        i2 = list(range(n1, n1 + n2))
        i3 = list(range(n1 + n2, n))
        pairs = [(a, b) for a in i1 for b in i2] + \
                [(a, b) for a in i2 for b in i3]
        acts = []
        for bits in itertools.product((0, 1), repeat=M):
            A = np.zeros((n, n), dtype=bool)
            for (a, b), v in zip(pairs, bits):
                if v:
                    A[a, b] = True
            A |= (A.astype(np.float32) @ A.astype(np.float32)) > 0.5
            acts.append(action(A, 4)[0])
        acts = np.array(acts, float)
        for hb in (0.5, 1.0, 2.0):
            z = abs(np.exp(1j * acts / hb).sum())
            pred = abs(2.0 * math.cos(K4 / (2.0 * hb))) ** M
            print(f'{f"{n1},{n2},{n3}":>14}{M:>5}{hb:>8.2f}{z:>16.4g}'
                  f'{pred:>18.4g}{z/max(pred,1e-30):>9.3f}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print('  (отношение ≈ 1 ⟹ и разложение, и S = k_d(N−N₀) верны;')
    print('   при малых слоях ожидается отличие — там λ = q²n₂ ~ 1)')
    print()
    print('=' * 100)
    print('ПРОВЕРКА 3. Точная сумма слоистых против ИХ ЖЕ пола 2^(M/2)')
    print(f"{'ℏ':>8}{'|2cos(k/2ℏ)|':>15}{'ниже 1?':>10}{'ниже √2?':>11}"
          f"{'полоса многообр.':>19}")
    for hb in (0.2, 0.30, 0.39, 0.5, 0.65, 0.78, 0.9, 1.5, 3.0):
        c = abs(2.0 * math.cos(K4 / (2.0 * hb)))
        print(f'{hb:>8.2f}{c:>15.4f}{("да" if c < 1 else "нет"):>10}'
              f'{("да" if c < math.sqrt(2) else "нет"):>11}'
              f'{("ДА" if c < 1 else "нет"):>19}')
    print()
    print('  ПОЛОСА (|2cos(k₄/2ℏ)| < 1) аналитически:')
    for k in range(4):
        lo = K4 / (4.0 * math.pi / 3.0 + 2 * math.pi * k)
        hi = K4 / (2.0 * math.pi / 3.0 + 2 * math.pi * k)
        print(f'     ℏ ∈ [{lo:.4f}, {hi:.4f}]')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
