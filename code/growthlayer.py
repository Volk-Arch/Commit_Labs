#!/usr/bin/env python3
"""ВЕС СЛОИСТОГО СЕКТОРА ПОД МЕРОЙ РОСТА — замкнутая форма.

Фазовое погашение сектор не убивает (`cancel5-doc.md`). Остаётся мера.
У транзитивной перколяции она известна точно:

    Pr(P) = p^{L} (1-p)^{M-R},   L = число СВЯЗЕЙ, M = N(N-1)/2

Для слоистого сектора из `k` слоёв отношения между СОСЕДНИМИ слоями
свободны (матрицы A^{(i)}), остальные определены замыканием. Тогда
  * все соседние отношения суть СВЯЗИ (между соседними слоями ничего
    нет, интервал пуст), а несоседние связями быть не могут (путь
    обязан идти через промежуточный слой);
  * внутрислойные пары всегда НЕсвязаны (все отношения идут вверх).
Значит L = sum|A^{(i)}|, R = L + r, где r — число связанных
НЕсоседних пар.

КЛЮЧЕВОЙ ХОД. Сумма по всем матрицам с весом x^{|A|}, x = p/(1-p),
есть в точности среднее по НЕЗАВИСИМЫМ бернуллиевским матрицам с
вероятностью p, потому что x/(1+x) = p. Отсюда весь сектор берётся
целиком, без максимумов и без допущений о когерентности:

    W = (1-p)^{M - sum a_i a_{i+1}} * E[ z^{r} ],   z = 1/(1-p)

    M - sum a_i a_{i+1} = sum_i C(a_i,2) + sum_{|i-j|>=2} a_i a_j

СМЫСЛ: мера роста берёт плату (1-p) с КАЖДОЙ пары, обязанной быть
несвязанной. Внутри слоёв таких пар порядка N², и однородная мера за
это не берёт ничего.

Так как r <= sum_{|i-j|>=2} a_i a_j, выходит строгая верхняя граница

    ln W <= -lambda * sum_i C(a_i,2),   lambda = -ln(1-p)

то есть много тонких слоёв дешевле. Но счёт сектора 2^{sum a_i a_{i+1}}
при тонких слоях мал — отсюда настоящая развилка, и её считает код.

КОНТРОЛИ:
  1. L и R, посчитанные ОТ МАТРИЦЫ отношений, против L = sum|A|,
     R = L + r (не по формуле);
  2. W прямым перечислением всех A против замкнутой формы;
  3. sum_P Pr(P) = 1 на точном перечислении (ensemble.py) — проверка
     самого понимания меры.

  python3 growthlayer.py
"""
import itertools
import math

import numpy as np


def layered_poset(sizes, mats):
    """Матрица отношений слоистого порядка и её замыкание."""
    n = sum(sizes)
    off = np.cumsum([0] + list(sizes))
    R = np.zeros((n, n), dtype=bool)
    for i, A in enumerate(mats):
        for a in range(sizes[i]):
            for b in range(sizes[i + 1]):
                if A[a, b]:
                    R[off[i] + a, off[i + 1] + b] = True
    Rc = R.copy()
    for _ in range(len(sizes)):
        nxt = Rc | (Rc.astype(np.int8) @ Rc.astype(np.int8) > 0)
        if (nxt == Rc).all():
            break
        Rc = nxt
    return Rc


def links_and_rel(Rc):
    """L (связи: пустой интервал) и R прямо от матрицы."""
    n = Rc.shape[0]
    Rel = int(Rc.sum())
    L = 0
    for x in range(n):
        for y in range(n):
            if Rc[x, y] and not (Rc[x] & Rc[:, y]).any():
                L += 1
    return L, Rel


def w_brute(sizes, p):
    """КОНТРОЛЬ 2: W прямым перечислением всех A, Pr от матрицы."""
    n = sum(sizes)
    M = n * (n - 1) // 2
    slots = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    tot = 0.0
    nbits = [a * b for a, b in slots]
    for bits in itertools.product(*[range(1 << nb) for nb in nbits]):
        mats = []
        for (a, b), v in zip(slots, bits):
            A = np.array([[(v >> (i * b + j)) & 1 for j in range(b)]
                          for i in range(a)], dtype=bool)
            mats.append(A)
        Rc = layered_poset(sizes, mats)
        L, Rel = links_and_rel(Rc)
        # контроль 1 внутри перечисления
        Lf = sum(int(A.sum()) for A in mats)
        radj = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
        assert L == Lf, f'связи: от матрицы {L}, по формуле {Lf}'
        assert Rel >= Lf
        tot += p ** L * (1 - p) ** (M - Rel)
    return tot


def w_closed_exact(sizes, p):
    """Замкнутая форма с ТОЧНЫМ E[z^r] (перечислением A)."""
    n = sum(sizes)
    M = n * (n - 1) // 2
    radj = sum(sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1))
    z = 1.0 / (1.0 - p)
    slots = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    nbits = [a * b for a, b in slots]
    acc = 0.0
    for bits in itertools.product(*[range(1 << nb) for nb in nbits]):
        mats, pr = [], 1.0
        for (a, b), v in zip(slots, bits):
            A = np.array([[(v >> (i * b + j)) & 1 for j in range(b)]
                          for i in range(a)], dtype=bool)
            mats.append(A)
            k = int(A.sum())
            pr *= p ** k * (1 - p) ** (a * b - k)
        Rc = layered_poset(sizes, mats)
        _, Rel = links_and_rel(Rc)
        r = Rel - sum(int(A.sum()) for A in mats)
        acc += pr * z ** r
    return (1 - p) ** (M - radj) * acc


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('КОНТРОЛИ 1+2: связи от матрицы, и W перечислением против формы')
    for sizes in ((2, 2, 2), (2, 3, 2), (1, 2, 2, 1), (2, 2, 2, 1)):
        for p in (0.09, 0.3):
            wb = w_brute(sizes, p)
            wc = w_closed_exact(sizes, p)
            print(f'  слои {sizes}, p={p}: W перечислением {wb:.12e}, '
                  f'замкнуто {wc:.12e}, отн. '
                  f'{abs(wb-wc)/max(wb,1e-300):.2e}', flush=True)
    print('  (внутри перечисления проверено L == sum|A| для каждой'
          ' конфигурации)')
    print()
    print('КОНТРОЛЬ 3: sum_P Pr(P) = 1 на точном перечислении порядков')
    from ensemble import enumerate_orders
    for n in (6, 7):
        d = enumerate_orders(n)
        rel = d['rel'].astype(float)
        N0 = d['N0'].astype(float)
        M = n * (n - 1) // 2
        for p in (0.09, 0.3):
            lp = N0 * math.log(p) + (M - rel) * math.log1p(-p)
            print(f'  N={n}, p={p}: sum Pr = {np.exp(lp).sum():.10f}')


if __name__ == '__main__':
    main()
