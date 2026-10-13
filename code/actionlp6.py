#!/usr/bin/env python3
"""ЗАКРЫВАЕТСЯ ЛИ И ЭТА ДВЕРЬ: подделка счёта максимальных элементов.

Число максимальных и минимальных элементов вернуло разделимость. Но у
слоистого семейства свободны РАЗМЕРЫ слоёв, а минимальные элементы —
это ровно первый слой. Значит сектор с ТОНКИМИ крайними слоями
подделывает и этот счёт, почти не теряя энтропии (её держат толстые
слои в середине).

Если так — дверь закрывается, и утверждение усиливается: никакое
действие из пятнадцати счётов не выделяет пространство-время. А причина
становится видна: у слоистого семейства свободных параметров формы
порядка N/2, а у любого локального действия коэффициентов конечное
число, поэтому семейство всегда может подделать любой конечный набор
счётов, сохранив избыток энтропии.
"""
import math

import numpy as np

from actionlp5 import (counts2, build2, margin, cert, BASE, EXTRA, ALL,
                       KSD, QSD)
from sectors import s_layered


def layered_sizes(n, sizes, q, rs):
    k = len(sizes)
    off = np.cumsum([0] + list(sizes))
    Rc = np.zeros((n, n), dtype=bool)
    for i in range(k - 1):
        Rc[off[i]:off[i+1], off[i+1]:off[i+2]] = (
            rs.random((sizes[i], sizes[i+1])) < q)
    cur = Rc.copy()
    for _ in range(k):
        nxt = cur | ((cur.astype(np.float32) @ cur.astype(np.float32)) > .5)
        if (nxt == cur).all():
            break
        cur = nxt
    return cur


def main():
    print(__doc__.strip())
    print()
    N = 600
    rs = np.random.default_rng(321)
    secs = build2(N, rs, KSD, QSD)
    print('СЧЁТ КРАЙНИХ ЭЛЕМЕНТОВ: чем именно засев выделяется')
    for nm in ('засев d=2', 'засев d=4', 'засев d=6', 'слоистый k=2 q=0.5',
               'слоистый k=20 q=0.15', 'слоистый k=150 q=0.5', 'цепь'):
        s = [x for x in secs if x['name'] == nm]
        if not s:
            continue
        s = s[0]
        print(f"  {nm:>22}: макс+мин = {s['f']['MAXMIN']:>8.1f}, "
              f"R = {s['f']['R']:>9.0f}, H = {s['f']['H']:>5.0f}, "
              f"S = {s['S']:>9.0f}")
    print()
    m, _, _, _, _ = margin(secs, 'засев d=4', BASE + ('MAXMIN',))
    print(f'  до добавления подделок: m = {m:.4g}')
    print()
    print('ДОБАВЛЯЮ СЕКТОРЫ С ТОНКИМИ КРАЙНИМИ СЛОЯМИ')
    extra = []
    for k in (6, 10, 20, 40, 80):
        for q in (0.05, 0.15, 0.35, 0.5, 0.8):
            for s_end in (1, 2, 4, 8, 16):
                if k < 3:
                    continue
                mid = (N - 2 * s_end) // (k - 2)
                if mid < 2:
                    continue
                sizes = [s_end] + [mid] * (k - 2) + [s_end]
                sizes[-2] += N - sum(sizes)
                if min(sizes) < 1:
                    continue
                Rc = layered_sizes(N, sizes, q, rs)
                extra.append({
                    'name': f'тонкие края k={k} q={q} s={s_end}',
                    'f': counts2(Rc), 'S': s_layered(N, sizes, q)})
    print(f'  добавлено {len(extra)} секторов')
    secs2 = secs + extra
    for feats, lab in ((BASE + ('MAXMIN',), 'BASE + MAXMIN'),
                       (ALL, 'все 15 счётов')):
        m, _, nm, G, C = margin(secs2, 'засев d=4', feats)
        v = 'ВЫПОЛНИМО' if m > 1e-9 else 'НЕВЫПОЛНИМО'
        print(f'  {lab:>16}: m = {m:>12.4g}   {v}')
        if m <= 1e-9:
            lam, val, res = cert(G, C)
            if lam is not None:
                idx = np.argsort(-lam)[:5]
                print(f'      сертификат: sum lambda*C = {val:+.5g}, '
                      f'остаток {res:.1e}')
                print('      смесь: ' + ', '.join(
                    f'{nm[i]} ({lam[i]:.3f})' for i in idx if lam[i] > 1e-6))


if __name__ == '__main__':
    main()
