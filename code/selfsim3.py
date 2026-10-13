#!/usr/bin/env python3
"""ЕСТЬ ЛИ ЗАЗОР ПО ВСЕЙ СЕТКЕ, И УСТОЙЧИВ ЛИ ОН ПО N.

Сырое условие («доля отношений прошлого равна доле целого») ВЫВЕРНУТО:
иерархический слоистый сектор даёт отклонение 0.0024, а засев d=4 —
0.158, в пятьдесят раз хуже. Причина видна: у почти полного порядка
(r = 0.996) всякое прошлое тоже почти полное, и совпадение даётся
ЖЁСТКОСТЬЮ, а не похожестью на мир.

Починка: сравнивать наблюдённый разброс с ОЖИДАЕМЫМ (биномиальным по
размеру прошлого). Первый замер дал обнадёживающее разделение:
засевы и доминирование 1.8...7.7, слоистые и перколяция 12...96.

ЗДЕСЬ решается, настоящий ли это зазор: плотная сетка соперников
(слоистые по k и q, иерархические по ветвлению и глубине, перколяция по
p) против засевов d=2..6, при двух размерах N. Если минимум по
соперникам опускается к максимуму по засевам — зазора нет.

ОГОВОРКА, которую надо держать: нормировка биномиальная, а внутри
прошлого засева пары КОРРЕЛИРОВАНЫ транзитивностью. Поэтому отношение
2...8 у засева — это корреляции, а не провал самоподобия. Статистика
смешивает две причины, и порог по ней — эмпирический, а не выведенный.

  python3 selfsim3.py
"""
import math

import numpy as np

from sectors import sprinkle_d, layered
from selfsim import dominance, percolation, hier_layered, chain
from selfsim2 import stats


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(31415)
    for N in (300, 600):
        print('=' * 70)
        print(f'N = {N}')
        sp = []
        for d in (2, 3, 4, 5, 6):
            rg, md, rt, cnt = stats(sprinkle_d(N, d, rs), 50)
            if rt is not None:
                sp.append((f'засев d={d}', rt, rg, cnt))
        print('  ЗАСЕВЫ:')
        for nm, rt, rg, cnt in sp:
            print(f'    {nm:>20}: отношение {rt:>7.2f}  (r={rg:.4f}, '
                  f'прошлых {cnt})')
        dom = []
        for k in (2, 3, 4, 5):
            rg, md, rt, cnt = stats(dominance(N, k, rs), 50)
            if rt is not None:
                dom.append((f'доминирование k={k}', rt, rg, cnt))
        print('  ДОМИНИРОВАНИЕ:')
        for nm, rt, rg, cnt in dom:
            print(f'    {nm:>20}: отношение {rt:>7.2f}  (r={rg:.4f})')
        comp = []
        for k in (2, 3, 4, 6, 8, 12, 20, 30, 50, 80):
            if N // k < 2:
                continue
            for q in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
                rg, md, rt, cnt = stats(layered(N, k, q, rs)[0], 50)
                if rt is not None and cnt >= 20:
                    comp.append((f'слоистый k={k} q={q}', rt, rg, cnt))
        for b, dep in ((2, 8), (3, 5), (3, 6), (4, 4), (5, 4), (10, 2),
                       (24, 2)):
            if b ** dep > 1200 or b ** dep < N // 3:
                continue
            for q in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9):
                rg, md, rt, cnt = stats(hier_layered(b, dep, q, rs), 50)
                if rt is not None and cnt >= 20:
                    comp.append((f'иерарх b={b} d={dep} q={q}', rt, rg, cnt))
        for p in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3):
            rg, md, rt, cnt = stats(percolation(N, p, rs), 50)
            if rt is not None and cnt >= 20:
                comp.append((f'перколяция p={p}', rt, rg, cnt))
        comp.sort(key=lambda t: t[1])
        print(f'  СОПЕРНИКОВ: {len(comp)}; пять самых похожих '
              f'(наименьшее отношение):')
        for nm, rt, rg, cnt in comp[:5]:
            print(f'    {nm:>24}: отношение {rt:>7.2f}  (r={rg:.4f}, '
                  f'прошлых {cnt})')
        mx_sp = max(r for _, r, _, _ in sp)
        mx_dom = max(r for _, r, _, _ in dom)
        mn_c = comp[0][1] if comp else float('inf')
        print()
        print(f'  максимум по засевам      : {mx_sp:.2f}')
        print(f'  максимум по доминированию: {mx_dom:.2f}')
        print(f'  МИНИМУМ по соперникам    : {mn_c:.2f}')
        if mn_c > max(mx_sp, mx_dom):
            print(f'  -> ЗАЗОР ЕСТЬ, кратность {mn_c/max(mx_sp,mx_dom):.2f}')
        else:
            print('  -> ЗАЗОРА НЕТ: соперник похож на мир не хуже засева')
        print()


if __name__ == '__main__':
    main()
