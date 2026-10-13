#!/usr/bin/env python3
"""ПОДАВЛЕНИЕ: спасает ли вес e^{∓βS} многообразные порядки?

Мой же результат `S/N ≈ 8` на множествах из роста означал «такие
конфигурации должны гаситься», но это была оговорка, а не расчёт. Точное
значение действия на слоистых порядках позволяет расчёт довести, потому
что у них И энтропия, И действие берутся на бумаге.

СЛОИСТЫЙ ПОРЯДОК. Доли слоёв `a_1, a_2, a_3`, соседние слои связаны с
вероятностью `q`, замыкание вниз. Тогда:
  * все отношения СОСЕДНИХ слоёв — звенья (между ними ничего нет),
    поэтому `N_0 = q·a_2(a_1+a_3)·N²`;
  * отношения через слой имеют `Binomial(a_2 N, q²)` промежуточных, то
    есть в среднем `λ = q² a_2 N → ∞` при `q = O(1)`, поэтому
    `N_1, N_2, N_3` экспоненциально малы;
  * значит `S = −k_d · N_0` с `k_2 = 4`, `k_4 = 4/√6`, и при `q = ½`
    это даёт ровно измеренные `−1/2` и `−1/(2√6)`.

ЭНТРОПИЯ. Свободны две двудольные матрицы связей, то есть
`a_2(a_1+a_3)N²` пар по вероятности `q`: энтропия
`a_2(a_1+a_3)·H(q)·N²` нат.

СВОБОДНАЯ ЭНЕРГИЯ на `N²`, вес `e^{−βS}` (действие ВОЗНАГРАЖДАЕТ звенья,
так как `S < 0`):

    F = a_2(a_1+a_3) [ H(q) + β k_d q ]      → максимум при q = 1

а при весе `e^{+βS}`:

    F = a_2(a_1+a_3) [ H(q) − β k_d q ],
    максимум при q* = 1/(1 + e^{β k_d}),  F = a_2(a_1+a_3)·ln(1/(1−q*))

У многообразных порядков свободная энергия на `N²` равна НУЛЮ: они
задаются `d·N` вещественными координатами, значит их число растёт как
`e^{O(N ln N)}`, а не `e^{O(N²)}`.

ПРОВЕРКА: формула сверяется с прямым счётом при разных `q` и разных
долях слоёв.

  python3 suppress.py
"""
import math
import time

import numpy as np

from grow import mm_dim, gen_percolation
from csg import gen_csg
from bd import action, layers

K = {2: 4.0, 4: 4.0 / math.sqrt(6.0)}


def gen_layered(n, parts, q, rs):
    """Слоистый порядок: доли parts, соседние слои связаны с вер. q."""
    parts = np.array(parts, float)
    parts = parts / parts.sum()
    sz = np.floor(parts * n).astype(int)
    sz[-1] += n - sz.sum()
    lab = np.concatenate([np.full(s, i) for i, s in enumerate(sz)])
    A = np.zeros((n, n), dtype=bool)
    for i in range(len(sz) - 1):
        a = np.flatnonzero(lab == i)
        b = np.flatnonzero(lab == i + 1)
        if len(a) == 0 or len(b) == 0:
            continue
        M = rs.random((len(a), len(b))) < q
        A[np.ix_(a, b)] = M
    # транзитивное замыкание по слоям
    for _ in range(len(sz)):
        A |= A @ A
    return A, sz


def H(q):
    if q <= 0 or q >= 1:
        return 0.0
    return -q * math.log(q) - (1 - q) * math.log(1 - q)


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 96)
    print('ПРОВЕРКА формулы S = −k_d·N₀ на слоистых порядках, N = 1600')
    print(f"{'доли':>18}{'q':>7}{'N₀/N² изм':>12}{'N₀/N² форм':>12}"
          f"{'S²/N² изм':>12}{'форм':>9}{'S⁴/N² изм':>12}{'форм':>9}")
    N = 1600
    for parts in ((1, 2, 1), (1, 1, 1), (1, 1), (1, 3, 1)):
        p = np.array(parts, float); p /= p.sum()
        a2 = p[1] if len(p) >= 2 else 0.0
        rest = (p[0] + p[2]) if len(p) == 3 else p[0]
        for q in (0.25, 0.5, 0.75, 1.0):
            rs = np.random.default_rng(int(1000 * q) + 7 * len(parts))
            A, sz = gen_layered(N, parts, q, rs)
            nn, L = layers(A, 1)
            a2m, _, _ = action(A, 2)
            a4m, _, _ = action(A, 4)
            del A
            f0 = q * a2 * rest
            print(f'{str(parts):>18}{q:>7.2f}{L[0]/N**2:>12.4f}{f0:>12.4f}'
                  f'{a2m/N**2:>12.4f}{-K[2]*f0:>9.4f}'
                  f'{a4m/N**2:>12.4f}{-K[4]*f0:>9.4f}')
    print(f'  ({time.time()-t0:.0f} с)', flush=True)
    print()
    print('=' * 96)
    print('СВОБОДНАЯ ЭНЕРГИЯ на N². У многообразных порядков она равна НУЛЮ.')
    print()
    print('вес e^{−βS}: действие вознаграждает звенья, максимум при q = 1')
    print(f"{'β':>8}{'q*':>8}{'F на N², d=2':>16}{'F на N², d=4':>16}"
          f"{'подавление многообразных':>28}")
    for b in (0.1, 0.3, 1.0, 3.0, 10.0):
        f2 = 0.25 * (H(1.0) + b * K[2] * 1.0)
        f4 = 0.25 * (H(1.0) + b * K[4] * 1.0)
        print(f'{b:>8.2f}{1.0:>8.2f}{f2:>16.4f}{f4:>16.4f}'
              f'{"e^{-" + f"{f4:.3f}" + "N²}":>28}')
    print()
    print('вес e^{+βS}: действие карает звенья, есть внутренний максимум')
    print(f"{'β':>8}{'q* (d=4)':>11}{'F на N², d=2':>16}{'F на N², d=4':>16}"
          f"{'многообразные выигрывают?':>28}")
    for b in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        out = []
        for d in (2, 4):
            qs = 1.0 / (1.0 + math.exp(b * K[d]))
            out.append((qs, 0.25 * (-math.log(1.0 - qs))))
        print(f'{b:>8.2f}{out[1][0]:>11.2e}{out[0][1]:>16.3e}'
              f'{out[1][1]:>16.3e}{"нет, F > 0":>28}')
    print()
    print('=' * 96)
    print('ЭМПИРИЧЕСКИ: внутри меры роста малое |S| и большая d — вместе?')
    print(f"{'правило':>24}{'реализ':>8}{'⟨d⟩':>8}{'corr(|S⁴|,d)':>15}"
          f"{'d в нижней декили по |S⁴|':>28}{'макс d':>9}")
    NR, n = 240, 300
    for tag, gen, kw in (
        ('перколяция p=0.01', gen_percolation, dict(p=0.01)),
        ('перколяция p=0.05', gen_percolation, dict(p=0.05)),
        ('степенное a=0.3', gen_csg,
         dict(draw_m=lambda j, rs: max(1, int(round(j ** 0.3))))),
        ('постоянное m=3', gen_csg, dict(draw_m=lambda j, rs: 3)),
    ):
        D, S = [], []
        for s in range(NR):
            A = gen(n, rs=np.random.default_rng(31 * s + 5), **kw)
            a4, _, _ = action(A, 4)
            r = float(np.count_nonzero(A)) / (n * (n - 1) / 2.0)
            del A
            if not (0 < r < 1):
                continue
            dd = mm_dim(r / 2.0)
            if np.isfinite(dd):
                D.append(dd); S.append(abs(a4) / n)
        D, S = np.array(D), np.array(S)
        if len(D) < 20:
            continue
        c = float(np.corrcoef(S, D)[0, 1])
        lo = D[S <= np.quantile(S, 0.1)]
        print(f'{tag:>24}{len(D):>8}{D.mean():>8.2f}{c:>15.3f}'
              f'{lo.mean():>20.2f}±{lo.std(ddof=1)/math.sqrt(len(lo)):<7.2f}'
              f'{D.max():>9.2f}')
        print(f'      ({time.time()-t0:.0f} с)', flush=True)
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
