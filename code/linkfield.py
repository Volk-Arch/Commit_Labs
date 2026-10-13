#!/usr/bin/env python3
"""Доля прямых связей против материи внутри мира.

Это соединение двух половин, которые в проекте не встречались:

  наблюдаемая   доля прямых связей в интервале — величина, которая
                кривизну РАЗЛИЧАЕТ (z = 9.4 в bigworld-doc, z = 17.0
                в gravity-doc), в отличие от объёма против высоты
  источник      материя ВНУТРИ мира (добавленные события), а не внешняя
                ручка засева warp, на которой держались прежние отклики

Механизм измерения чистый: отношения между фоновыми событиями от
добавления массы не меняются вовсе (координаты фона те же). Меняются
только ПРЯМЫЕ связи — событие массы, попавшее между двумя фоновыми,
их прямую связь разрывает. Знаменатель фиксирован, числитель меряет
действие массы на окружающий фон.

Скан по радиусу при постоянном M — контроль на связность, который убил
три прежние попытки (matter-doc.md): внутренняя высота массы от радиуса
не зависит (масштабная инвариантность), центральность тоже, меняется
одна плотность.

  python3 linkfield.py --k 4
  python3 linkfield.py --k 4 --n 50000 --pairs 8
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import rel_of
from chaingrav import chain_len

CENTER = 0.5
RADII = (0.02, 0.04, 0.07, 0.11, 0.16, 0.22)


def ball(M, k, r, rs):
    v = rs.normal(size=(M, k))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rad = r * rs.random(M) ** (1.0 / k)
    return (CENTER + v * rad[:, None]).astype(np.float32)


def pick_pairs(Xb, r_max, n_pairs, rs):
    """Пары, накрывающие шар, но прилегающие к нему: интервал минимален."""
    ia = np.flatnonzero((Xb < CENTER - r_max).all(1))
    ib = np.flatnonzero((Xb > CENTER + r_max).all(1))
    if len(ia) == 0 or len(ib) == 0:
        return []
    ia = ia[np.argsort(-Xb[ia].max(1))]
    ib = ib[np.argsort(Xb[ib].min(1))]
    pool = max(n_pairs * 3, n_pairs)
    ia, ib = ia[:pool], ib[:pool]
    na = min(n_pairs, len(ia), len(ib))
    return list(zip(rs.choice(ia, na, replace=False),
                    rs.choice(ib, na, replace=False)))


def links_among(Xsub, nb):
    """Прямые связи среди первых nb событий (фон), в полном множестве.

    Прямая связь = отношение с пустым интервалом. Масса участвует в
    проверке пустоты, но не в подсчёте: так меряется действие массы на
    геометрию ФОНА, а не её собственная структура.
    """
    R = rel_of(Xsub)
    A = np.asarray(R, dtype=np.float32)
    empty = (A @ A) == 0
    L = R & empty
    return int(L[:nb, :nb].sum()), int(R[:nb, :nb].sum()), R


def run(N, M, k, n_pairs, seed, vol_cap, reals):
    rs = np.random.default_rng(seed)
    Xb = rs.random((N, k)).astype(np.float32)
    rb = np.random.default_rng(seed + 77)
    balls = {r: [ball(M, k, r, rb) for _ in range(reals)] for r in RADII}
    pairs = pick_pairs(Xb, max(RADII), n_pairs,
                       np.random.default_rng(seed + 1))

    acc = {r: [] for r in RADII}
    base = []
    for a, b in pairs:
        xa, xb = Xb[a], Xb[b]
        m = (Xb > xa).all(1) & (Xb < xb).all(1)
        m[a] = m[b] = False
        ib = np.flatnonzero(m)
        if len(ib) > vol_cap or len(ib) < 200:
            continue
        Xbg = Xb[ib]
        nb = len(ib)
        L0, R0, Rbase = links_among(
            np.vstack([Xbg, xa[None], xb[None]]), nb)
        tau0 = chain_len(Rbase) - 1
        base.append((nb, L0, R0, tau0))
        for r in RADII:
            for B in balls[r]:
                L1, R1, Rfull = links_among(
                    np.vstack([Xbg, B, xa[None], xb[None]]), nb)
                assert R1 == R0, 'отношения фона изменились — ошибка схемы'
                sl = slice(nb, nb + M)
                h_add = chain_len(Rfull[sl, sl])
                # фон, вмешанный в массу: масса и в прошлом, и в будущем
                bg = slice(0, nb)
                mixed = int((Rfull[bg, sl].any(1) &
                             Rfull[sl, bg].any(0)).sum())
                # вторая наблюдаемая на тех же данных: задержка
                dtau = (chain_len(Rfull) - 1) - tau0
                acc[r].append((L0 - L1, L0, R0, h_add, mixed, dtau))
    return acc, base


def main():
    ap = argparse.ArgumentParser(
        description='Доля прямых связей против материи внутри мира')
    ap.add_argument('--n', type=int, default=32000)
    ap.add_argument('--m', type=int, default=120)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--pairs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=1800)
    ap.add_argument('--reals', type=int, default=2)
    a = ap.parse_args()

    t0 = time.time()
    acc, base = run(a.n, a.m, a.k, a.pairs, a.seed, a.volcap, a.reals)
    if not base:
        print('пар не набралось: поднимите --volcap или --n')
        return
    B = np.array(base, dtype=float)
    print(f'N={a.n}, масса M={a.m}, k={a.k}, пар {len(B)}, '
          f'V_bg {B[:, 0].mean():.0f}, M/V_bg {a.m/B[:, 0].mean():.3f}')
    print(f'фон без массы: прямых связей {B[:, 1].mean():.0f}, '
          f'отношений {B[:, 2].mean():.0f}, '
          f'доля прямых {(B[:, 1]/B[:, 2]).mean():.4f}')
    print()
    print('ДВЕ НАБЛЮДАЕМЫЕ НА ОДНИХ И ТЕХ ЖЕ ПАРАХ И МАССАХ')
    print(f"{'радиус':>8}{'h_масса':>10}{'фона внутри':>13}"
          f"{'доля порванных связей':>24}{'задержка Δτ':>20}")
    rows = []
    for r in RADII:
        A = np.array(acc[r], dtype=float)
        if len(A) == 0:
            continue
        broke, L0, ha, mx, dt = A[:, 0], A[:, 1], A[:, 3], A[:, 4], A[:, 5]
        frac = broke / L0
        se = frac.std() / np.sqrt(len(frac))
        dse = dt.std() / np.sqrt(len(dt))
        print(f'{r:>8.2f}{ha.mean():>10.2f}{mx.mean():>13.1f}'
              f'{f"{frac.mean():.5f} ± {se:.5f}":>24}'
              f'{f"{dt.mean():.2f} ± {dse:.2f}":>20}')
        rows.append((r, frac.mean(), se, dt.mean(), dse))
    if len(rows) >= 2:
        (r1, f1, s1, d1, e1) = rows[0]
        (r2, f2, s2, d2, e2) = rows[-1]
        print()
        print(f'  плотная ({r1}) против рыхлой ({r2}) при одной и той же массе:')
        for nm, a1, b1, a2, b2 in (
                ('доля порванных связей', f1, s1, f2, s2),
                ('задержка Δτ', d1, e1, d2, e2)):
            d = a1 - a2
            sg = np.hypot(b1, b2)
            ratio = a1 / a2 if a2 else float('nan')
            print(f'    {nm:>22}: {d:+.5f} ± {sg:.5f}'
                  f'  ({abs(d)/sg if sg > 0 else float("nan"):.1f}σ),'
                  f'  отношение {ratio:.2f}x')
        print()
        print('  знаки противоположны -> нет одного числа «масса»:')
        print('  те же события тяжелы для одной наблюдаемой и легки для другой')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
