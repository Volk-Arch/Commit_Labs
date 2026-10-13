#!/usr/bin/env python3
"""НОСИТЕЛЬ ВМЕСТО ПОРОГА: локальное правило со замыканием.

Почему именно это. У проекта есть симметрия перешкалирования
Diff^+(R)^k: ЛЮБОЕ монотонное преобразование любой шкалы ничего не
меняет. Значит «взять логарифм», «сменить масштаб», «возвести в степень»
- доказуемо пустые операции, сломать правило ими нельзя. Реальных ручек
ровно две: ПОРОГ (m из k) и НОСИТЕЛЬ (какие пары сравниваются). Порог
крутили (`majority-doc.md`, `orthant-doc.md`). Носитель - нет.

И носитель важнее. Сегодняшняя теорема о невозможности (порядок на k
шкалах имеет dim <= k, а у 4d Минковского размерность не ограничена,
`orthant-doc.md`) держится на том, что правило ВСЕПАРНОЕ И РАЗДЕЛЬНОЕ.
Локальное правило с транзитивным замыканием под неё НЕ попадает:
замыкание локальных связей не есть пересечение k линейных порядков.

Правило: i -> j, если i раньше хотя бы на m шкалах ИЗ k И пара ЛОКАЛЬНА,
то есть max_s |pos_s(i) - pos_s(j)| <= w. Затем транзитивное замыкание.

ПРЕДСКАЗАНИЯ до замера:
  * при m = k замыкание обязано вернуться В ОРТУ (она транзитивна, и
    подотношение замыкается внутрь) - это встроенный контроль;
  * при m < k локальность может подавить циклы, которых всепарное
    ослабление не избегало;
  * если найдётся (m, w) с ацикличным и более плотным порядком, у
    которого S_k находится при больших k, это НОВАЯ кинематика, не
    подпадающая под теорему.

КОНТРОЛИ: ацикличность - топологической сортировкой (не числом
3-циклов, эта ошибка уже делалась); при m=k и большом w результат обязан
совпасть с ортой.
"""
import numpy as np

from sectors import sprinkle_d
from waveop import causal, sprinkle_11
from orthant import scales, thresh_order, acyclic, closure, prof, region, rf
from ordim import best_sk

N = 400
K = 7


def counts(S):
    k, n = S.shape
    c = np.zeros((n, n), np.int16)
    g = np.zeros((n, n), np.int32)
    for s in S:
        pos = np.empty(n, int)
        pos[s] = np.arange(n)
        d = pos[None, :] - pos[:, None]
        c += (d > 0).astype(np.int16)
        g = np.maximum(g, np.abs(d).astype(np.int32))
    return c, g


def main():
    print(__doc__)
    rs = np.random.default_rng(112233)
    chi4, thr4 = region(lambda: sprinkle_d(N, 4, rs), 40, rs)
    chi2, thr2 = region(lambda: causal(sprinkle_11(N, rs)), 40, rs)
    print(f'пороги при N={N}: 4d {thr4:.1f}, 1+1 {thr2:.1f}')

    S = scales(N, K, rs)
    c, g = counts(S)
    orth = thresh_order(S, K)
    print(f'орта (m=k, все пары): доля {rf(orth):.4f}, '
          f'хи2 к 4d {float(chi4(prof(orth))):.1f}, '
          f'S_k = {best_sk(orth, rs)}')

    print(f'\n{"m":>3}{"w":>7}{"доля до":>10}{"ацикл.":>8}{"доля после":>12}'
          f'{"хи2 4d":>9}{"хи2 1+1":>10}{"S_k":>5}')
    for m in (K, K - 1, K - 2):
        for w in (N // 20, N // 8, N // 4, N // 2, N):
            R = (c >= m) & (g <= w)
            np.fill_diagonal(R, False)
            ok = acyclic(R)
            C = closure(R) if ok else R
            if not ok:
                print(f'{m:>3}{w:>7}{rf(R):>10.4f}{"НЕТ":>8}'
                      f'{"—":>12}{"—":>9}{"—":>10}{"—":>5}', flush=True)
                continue
            p = prof(C)
            print(f'{m:>3}{w:>7}{rf(R):>10.4f}{"да":>8}{rf(C):>12.4f}'
                  f'{float(chi4(p)):>9.1f}{float(chi2(p)):>10.1f}'
                  f'{best_sk(C, rs):>5}', flush=True)

    print('\nКОНТРОЛЬ: при m=k и w=N обязана выйти ровно орта')
    R = (c >= K) & (g <= N)
    np.fill_diagonal(R, False)
    same = bool((closure(R) == orth).all())
    print(f'  совпало: {same} => {"ОК" if same else "СБОЙ"}')


if __name__ == '__main__':
    main()
