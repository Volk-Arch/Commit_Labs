#!/usr/bin/env python3
"""Задержка от массы на настоящем причинном порядке.

Пересчёт chaingrav-doc.md на конусном порядке. Там было установлено:
задержка собственного времени от массы существует, локальность точна,
а величина зависит от плотности массы, не от её протяжённости. Но всё
это мерилось на покомпонентном порядке, который, как выяснилось, не
причинный порядок Минковского (not-minkowski-doc.md).

Схема та же: три мира на одном фоне, сравнение при одинаковом потолке
h_add (высоте массы внутри интервала), скан по радиусу как контроль на
связность. Отличие одно — порядок настоящий.

  python3 delaycone.py
  python3 delaycone.py --n 6000 --pairs 40
"""
import argparse
import math
import time

import numpy as np

from coneworld import sprinkle, ball4, cone_rel, between, interval_order
from chaingrav import chain_len

CENTRE = np.zeros(4)
RADII = (0.04, 0.08, 0.15, 0.25, 0.40)


def covering_pairs(X, r, n_pairs, rs, margin=0.05):
    """Пары, чей причинный интервал накрывает шар радиуса r в начале.

    Условие: световой конус a содержит весь шар, и шар содержится в
    прошлом конусе b.
    """
    t, x = X[:, 0], np.linalg.norm(X[:, 1:], axis=1)
    before = np.flatnonzero((-t - x > r + margin) & (t < 0))
    after = np.flatnonzero((t - x > r + margin) & (t > 0))
    if len(before) == 0 or len(after) == 0:
        return []
    n = min(n_pairs, len(before), len(after))
    return list(zip(rs.choice(before, n, replace=False),
                    rs.choice(after, n, replace=False)))


def measure(X, ib, a, b, Xadd):
    R, nb = interval_order(X, ib, a, b, Xadd)
    tau = chain_len(R) - 1
    if Xadd is None or len(Xadd) == 0:
        return tau, 0
    sl = slice(1 + nb, 1 + nb + len(Xadd))
    return tau, chain_len(R[sl, sl])


def run(N, M, n_pairs, seed, vol_cap):
    rs = np.random.default_rng(seed)
    X = sprinkle(N, 4, rs)
    rb = np.random.default_rng(seed + 77)
    balls = {r: ball4(M, 4, r, CENTRE, rb) for r in RADII}
    spread = sprinkle(M, 4, rb)                    # контроль: рассыпано
    pairs = covering_pairs(X, max(RADII), n_pairs,
                           np.random.default_rng(seed + 1))

    acc = {r: [] for r in RADII}
    acc['рассыпано'] = []
    vols, far = [], []
    for a, b in pairs:
        ib = np.flatnonzero(between(X, a, b))
        if len(ib) > vol_cap or len(ib) < 60:
            continue
        t0, _ = measure(X, ib, a, b, None)
        vols.append(len(ib))
        for key, A in list(balls.items()) + [('рассыпано', spread)]:
            inside = np.flatnonzero(
                (A[:, 0] - X[a, 0] > np.linalg.norm(A[:, 1:] - X[a, 1:], axis=1)) &
                (X[b, 0] - A[:, 0] > np.linalg.norm(X[b, 1:] - A[:, 1:], axis=1)))
            t1, h = measure(X, ib, a, b, A[inside])
            acc[key].append((t1 - t0, h, len(inside), t0))

    # Локальность НЕ меряется: она следует из определения цепи. Любое
    # событие цепи от a к b причинно лежит между a и b, поэтому масса вне
    # интервала не может изменить tau ни при каком порядке.
    #
    # Попытка померить её прямо дала ложный результат: в той ветке я
    # передавал в подмножество ВЕСЬ шар, а не его часть внутри интервала,
    # а chain_len возвращает длиннейшую цепь во всём подмножестве, не от
    # a к b. Для пар, чей интервал массу не накрывает, a перестаёт быть
    # единственным минимумом, и цепь считается не та. Основной путь
    # фильтр применяет (A[inside]), поэтому он верен.
    return acc, np.array(vols, float), np.array(far, float)


def main():
    ap = argparse.ArgumentParser(description='Задержка от массы на конусе')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--m', type=int, default=200)
    ap.add_argument('--pairs', type=int, default=30)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--volcap', type=int, default=2500)
    a = ap.parse_args()

    t0 = time.time()
    acc, vols, far = run(a.n, a.m, a.pairs, a.seed, a.volcap)
    if len(vols) == 0:
        print('пар не набралось: поднимите --n или --volcap')
        return
    print(f'N={a.n}, масса M={a.m}, пар {len(vols)}, '
          f'V_bg {vols.mean():.0f}, M/V_bg {a.m/vols.mean():.3f}')
    print()
    print('ЛОКАЛЬНОСТЬ не меряется: следует из определения цепи')
    print('  (масса вне интервала не может изменить tau при любом порядке)')
    print()
    print('ЗАДЕРЖКА ПРОТИВ ПЛОТНОСТИ МАССЫ (масса и её высота постоянны)')
    print(f"{'радиус':>12}{'внутри':>9}{'h_масса':>10}"
          f"{'задержка Δτ':>18}{'Δτ / h':>12}")
    rows = []
    for key in list(RADII) + ['рассыпано']:
        A = np.array(acc[key], dtype=float)
        if len(A) == 0:
            continue
        dt, h, nin = A[:, 0], A[:, 1], A[:, 2]
        ok = h > 0
        ratio = (dt[ok] / h[ok]) if ok.any() else np.array([np.nan])
        lbl = f'{key:.2f}' if isinstance(key, float) else key
        print(f'{lbl:>12}{nin.mean():>9.0f}{h.mean():>10.2f}'
              f'{f"{dt.mean():.2f} ± {dt.std()/math.sqrt(len(dt)):.2f}":>18}'
              f'{ratio.mean():>12.3f}')
        rows.append((lbl, dt.mean(), dt.std() / math.sqrt(len(dt))))
    if len(rows) >= 2:
        (l1, m1, s1), (l2, m2, s2) = rows[0], rows[-2]
        d = m1 - m2
        sg = math.hypot(s1, s2)
        print()
        print(f'  плотная ({l1}) против рыхлой ({l2}): {d:+.2f} ± {sg:.2f} '
              f'({abs(d)/sg if sg else float("nan"):.1f}σ), '
              f'отношение {m1/m2 if m2 else float("nan"):.2f}x')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
