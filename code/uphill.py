#!/usr/bin/env python3
"""ПОДЪЁМ: вторая бесконечность действия, которую я вечером пропустил.

ПРОБЕЛ В МОЁМ РАСЧЁТЕ. В `suppress.py` свободная энергия слоистых
считалась только при `q = O(1)`: там пары через слой имеют много
промежуточных (`λ = q²a₂N → ∞`), поэтому `N₁…N₃` экспоненциально малы и
`S = −k_d N₀ < 0`. Отсюда вышел вывод «действие вознаграждает звенья».

Но есть вторая область, и я её не рассмотрел: если связь между слоями
РЕДКАЯ, так что `λ ≈ 1`, то промежуточных в среднем единица, работает
член `+9N₁` — он с ПЛЮСОМ, — а `N₀ ∝ N^{3/2}` становится подведущим.
Тогда действие большое ПОЛОЖИТЕЛЬНОЕ, тоже порядка `N²`.

ПРЕДСКАЗАНИЕ, закрытое и без свободных чисел. Для трёх слоёв с долями
`a₁, a₂, a₃` и `λ = q²a₂N`:

    N_m ≈ a₁a₃N² · e^{−λ}λ^m/m!   (число промежуточных ~ Poisson(λ))
    N₀  = q·a₂(a₁+a₃)N² ∝ N^{3/2}  — подведущий

    S/N² → (4/√6) · a₁a₃ · g(λ),   g(λ) = e^{−λ}(9λ − 8λ² + (4/3)λ³)

`g` максимальна при `λ ≈ 0.45`, где `g ≈ 1.63`.

ЧТО ЭТО РЕШАЕТ. Если подтвердится, действие не ограничено НИ СВЕРХУ, НИ
СНИЗУ на уровне `N²`:
    вниз — полные слоистые,  `S/N² → −1/√6 = −0.408`
    вверх — редкие слоистые, `S/N² → +(4/√6)a₁a₃g(λ)`
Тогда статистическая сумма расходится при ЛЮБОМ знаке `β`, и
многообразные порядки проигрывают в обе стороны. Это строже вечернего
вывода, где закрыт был только один знак.

ПРОВЕРКИ: (1) формула против замера по сетке `λ`; (2) `N₀/N² → 0`, то
есть область та; (3) `S/N²` не зависит от `N`.

  python3 uphill.py
"""
import math
import time

import numpy as np

from bd import action, layers
from suppress import gen_layered

K4 = 4.0 / math.sqrt(6.0)
K2 = 2.0      # исправлено: 2, не 4 (см. g2)


def g4(lam):
    return math.exp(-lam) * (9.0 * lam - 8.0 * lam ** 2
                             + (4.0 / 3.0) * lam ** 3)


def g2(lam):
    """S² = 2(N − 2N₀ + 4N₁ − 2N₂), вклад = 2·a₁a₃·e^{−λ}(4λ − λ²).
    ПОПРАВКА: множитель 2, не 4 — сперва стояло K2 = 4, вдвое больше."""
    return math.exp(-lam) * (4.0 * lam - 1.0 * lam ** 2)


def gen3(n, parts, q, rs):
    """Три слоя, замыкание В ОДИН ШАГ: A[L1,L3] = (A[L1,L2] @ A[L2,L3]) > 0.
    Общий цикл с булевым умножением был слишком медленным."""
    p = np.array(parts, float); p /= p.sum()
    sz = np.floor(p * n).astype(int); sz[-1] += n - sz.sum()
    i1 = np.arange(0, sz[0])
    i2 = np.arange(sz[0], sz[0] + sz[1])
    i3 = np.arange(sz[0] + sz[1], n)
    A = np.zeros((n, n), dtype=bool)
    M12 = rs.random((len(i1), len(i2))) < q
    M23 = rs.random((len(i2), len(i3))) < q
    A[np.ix_(i1, i2)] = M12
    A[np.ix_(i2, i3)] = M23
    A[np.ix_(i1, i3)] = (M12.astype(np.float32)
                         @ M23.astype(np.float32)) > 0.5
    return A, sz


def one(n, parts, lam, rs):
    p = np.array(parts, float); p /= p.sum()
    a2 = p[1]
    q = math.sqrt(lam / (a2 * n))
    if q > 1.0:
        return None
    A, sz = gen3(n, parts, q, rs)
    nn, L = layers(A, 4)
    s4, _, _ = action(A, 4)
    s2, _, _ = action(A, 2)
    del A
    return (s2 / nn ** 2, s4 / nn ** 2, L[0] / nn ** 2,
            L[1] / nn ** 2, L[2] / nn ** 2)


def arm(n, parts, lam, seeds, root):
    rows = [one(n, parts, lam, np.random.default_rng(root + 31 * s))
            for s in range(seeds)]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    P = np.array(rows, float)
    return P.mean(axis=0), (P.std(axis=0, ddof=1) / math.sqrt(len(P))
                            if len(P) > 1 else np.zeros(P.shape[1]))


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    N, PARTS = 1600, (0.45, 0.10, 0.45)
    p = np.array(PARTS, float); p /= p.sum()
    a1a3 = p[0] * p[2]
    print('=' * 100)
    print(f'СКАН ПО λ, доли {PARTS}, N = {N}, a₁a₃ = {a1a3:.4f}')
    print(f"{'λ':>7}{'S⁴/N² замер':>18}{'формула':>11}{'S²/N² замер':>18}"
          f"{'формула':>11}{'N₀/N²':>9}{'N₁/N²':>9}")
    bestm, bestl = -9e9, None
    for lam in (0.15, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5, 5.0, 12.0):
        r = arm(N, PARTS, lam, 5, 8000 + int(100 * lam))
        if r is None:
            continue
        m, e = r
        f4 = K4 * a1a3 * g4(lam)
        f2 = K2 * a1a3 * g2(lam)
        print(f'{lam:>7.2f}{m[1]:>11.4f}±{e[1]:<6.4f}{f4:>11.4f}'
              f'{m[0]:>11.4f}±{e[0]:<6.4f}{f2:>11.4f}'
              f'{m[2]:>9.4f}{m[3]:>9.4f}')
        if m[1] > bestm:
            bestm, bestl = m[1], lam
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 100)
    print(f'ЗАВИСИМОСТЬ ОТ РАЗМЕРА при λ = {bestl} (формула от N не зависит)')
    print(f"{'N':>7}{'S⁴/N²':>18}{'формула':>11}{'N₀/N² (→0?)':>14}"
          f"{'N₀/N^1.5':>12}")
    f4 = K4 * a1a3 * g4(bestl)
    for n in (600, 1200, 2400, 3600):
        r = arm(n, PARTS, bestl, 4, 9000 + n)
        if r is None:
            continue
        m, e = r
        print(f'{n:>7}{m[1]:>11.4f}±{e[1]:<6.4f}{f4:>11.4f}'
              f'{m[2]:>14.5f}{m[2]*n**0.5:>12.4f}')
        print(f'    ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 100)
    print('ДВЕ БЕСКОНЕЧНОСТИ, итог')
    print(f'''
   вниз (полные слоистые, q = 1):   S⁴/N² = −1/√6      = −0.4082
   вверх (редкие слоистые, λ≈0.45): S⁴/N² = {bestm:+.4f}  (замер)

Если оба предела подтверждены, действие БД не ограничено НИ СВЕРХУ, НИ
СНИЗУ на уровне `N²`. Тогда `Σ e^{{∓βS}}` расходится при любом знаке и
любом `β > 0`, а многообразные порядки (у них `S = O(1)`, то есть ровно
посередине) проигрывают в ОБЕ стороны.

Это закрывает подъём так же, как вечер закрыл спуск.
''')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
