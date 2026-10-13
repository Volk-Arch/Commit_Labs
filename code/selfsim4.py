#!/usr/bin/env python3
"""ПРАВИЛЬНЫЙ ВОПРОС: проходит ли условие сектор с ЭНТРОПИЕЙ выше мира.

Два огреха предыдущего прогона:
  1. иерархические секторы строились с N = b^глубина (243...256), а
     сравнивались с засевом на 600 — отношение зависит от размера, и
     сравнение было не при равном N. Здесь построение обобщено на
     произвольное N (рекурсивное деление на b почти равных блоков);
  2. вопрос «есть ли зазор» неверен. Жёсткие порядки (почти цепь,
     r -> 1) проходят любое условие самоподобия ДАРОМ — у них нет
     места для отклонения, — но они отсекаются ЭНТРОПИЕЙ и без всякого
     условия: у цепи ln N! ~ N lnN - N против 4N lnN у мира.

Значит условие полезно тогда и только тогда, когда оно отсекает
секторы с ЭНТРОПИЕЙ ВЫШЕ мировой — то есть ровно те, из-за которых
отбор и не работает. Вопрос:

    существует ли сектор с S > 4N lnN, проходящий условие
    не хуже засева?

Если да — условие бесполезно. Если нет — связка «условие + энтропия»
закрывает дело.

  python3 selfsim4.py
"""
import math

import numpy as np

from sectors import sprinkle_d, layered, s_layered, H2
from selfsim import dominance, percolation, chain
from selfsim2 import stats


def hier_n(n, b, q, rs):
    """Иерархический слоистый на РОВНО n элементов: рекурсивное деление
    на b почти равных блоков, плотность q между соседними. Возвращает
    матрицу и число свободных мест (для энтропии)."""
    slots = [0]

    def build(m):
        if m <= 1:
            return np.zeros((m, m), dtype=bool)
        k = min(b, m)
        base, extra = divmod(m, k)
        sizes = [base + (1 if i < extra else 0) for i in range(k)]
        subs = [build(s) for s in sizes]
        off = np.cumsum([0] + sizes)
        R = np.zeros((m, m), dtype=bool)
        for i, s in enumerate(subs):
            R[off[i]:off[i + 1], off[i]:off[i + 1]] = s
        for i in range(k - 1):
            a, c = sizes[i], sizes[i + 1]
            slots[0] += a * c
            R[off[i]:off[i + 1], off[i + 1]:off[i + 2]] = (
                rs.random((a, c)) < q)
        C = R.copy()
        while True:
            nxt = C | ((C.astype(np.float32) @ C.astype(np.float32)) > 0.5)
            if (nxt == C).all():
                break
            C = nxt
        return C

    M = build(n)
    return M, slots[0]


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(2718)
    N = 600
    S_world = 4.0 * N * math.log(N)
    print(f'N = {N}, энтропия мира (щедро, Милнор-Том) = {S_world:.0f}')
    print()
    sp = []
    for d in (2, 3, 4, 5, 6):
        rg, md, rt, cnt = stats(sprinkle_d(N, d, rs), 50)
        if rt is not None and cnt >= 10:
            sp.append(rt)
            print(f'  засев d={d}: отношение {rt:>6.2f} (r={rg:.4f}, '
                  f'прошлых {cnt})')
    thr = max(sp)
    print(f'  порог (максимум по засевам): {thr:.2f}')
    print()
    rows = []
    for k in (2, 3, 4, 6, 8, 12, 20, 30, 50, 80, 120):
        if N // k < 2:
            continue
        for q in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
            Rc, sizes = layered(N, k, q, rs)
            S = s_layered(N, sizes, q)
            rg, md, rt, cnt = stats(Rc, 50)
            if rt is None or cnt < 10:
                continue
            rows.append((f'слоистый k={k} q={q}', rt, S, rg, cnt))
    for b in (2, 3, 4, 6, 10, 24):
        for q in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9):
            Rc, slots = hier_n(N, b, q, rs)
            S = slots * H2(q) + math.lgamma(N + 1) / 2.0
            rg, md, rt, cnt = stats(Rc, 50)
            if rt is None or cnt < 10:
                continue
            rows.append((f'иерарх b={b} q={q}', rt, S, rg, cnt))
    passing_hi = [r for r in rows if r[1] <= thr and r[2] > S_world]
    rows.sort(key=lambda t: -t[2])
    print('СЕКТОРЫ С ЭНТРОПИЕЙ ВЫШЕ МИРОВОЙ, по убыванию энтропии (топ 12)')
    print(f"{'сектор':>24}{'S':>10}{'отношение':>11}{'r':>8}{'проходит?':>11}")
    for nm, rt, S, rg, cnt in [r for r in rows if r[2] > S_world][:12]:
        print(f'{nm:>24}{S:>10.0f}{rt:>11.2f}{rg:>8.4f}'
              f'{("ДА" if rt <= thr else "нет"):>11}')
    print()
    print(f'  всего секторов проверено: {len(rows)}')
    print(f'  из них с S > энтропии мира: '
          f'{sum(1 for r in rows if r[2] > S_world)}')
    print(f'  ПРОХОДЯТ условие И имеют S выше мировой: {len(passing_hi)}')
    if passing_hi:
        print()
        print('  ВЕРДИКТ: условие БЕСПОЛЕЗНО — вот секторы, которые'
              ' и богаче энтропией, и самоподобны не хуже мира:')
        for nm, rt, S, rg, cnt in sorted(passing_hi, key=lambda t: -t[2])[:6]:
            print(f'    {nm:>24}: S = {S:.0f} (мир {S_world:.0f}), '
                  f'отношение {rt:.2f} (порог {thr:.2f}), r = {rg:.4f}')
    else:
        print()
        print('  ВЕРДИКТ: ни один сектор не проходит условие, имея'
              ' энтропию выше мировой.')
        print('  Связка «условие самоподобия + энтропия» закрывает'
              ' проверенную сетку.')


if __name__ == '__main__':
    main()
