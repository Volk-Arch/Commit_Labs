#!/usr/bin/env python3
"""ПЛАТО — СВОЙСТВО ОЦЕНЩИКА ИЛИ ПОРЯДКА?

`qcarrier3.py` с честной метрикой (пол перемешивания 0.9993) дал: SNR
оценщика по общему прошлому = 1.77, доля объяснённой дисперсии 68 %, и
показатель роста РОВНО НОЛЬ (0.0033 +- 0.0043) при размахе N в 35 раз.

Но причина может быть не в порядке, а в оценщике: проект уже отмечал,
что «разброс ПОЗИЦИОННЫЙ, а не статистический» — пары с одинаковым x в
разных местах и с разной ориентацией внутри диаманта обрезаются краем
по-разному, и это плотностью не снимается.

Проверка прямая: дать оценщику ещё и положение. Если SNR вырастет И
начнёт расти с N — плато было моим, а не порядка. Если нет — плато
принадлежит порядку.

Предикторы: ln V_cp, ln V_a, ln V_b, ln N_both, возраст пары, и
радиальное положение (расстояние от оси диаманта) — последнее есть
«внешнее» знание о краевом обрезании, и его надо указать отдельно.

КОНТРОЛЬ: перемешивание x обязано давать SNR = 1.00 при любом числе
предикторов (иначе подгонка просто ест степени свободы).

  python3 qcarrier4.py
"""
import math

import numpy as np

from qcarrier import sprinkle4


def harvest2(n, rs, npair, xlo, xhi, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    t, X = P[:, 0], P[:, 1:]
    rad = np.sqrt((X.astype(np.float64) ** 2).sum(1))
    tau = np.sqrt(np.maximum(t.astype(np.float64) ** 2 - rad ** 2, 0.0))
    mid = (tau > 0.35 * T) & (tau < 0.65 * T)
    order = np.nonzero(mid)[0]
    order = order[np.argsort(tau[order])]
    ta = tau[order]
    rows = []
    tries, lim = 0, npair * 400
    while len(rows) < npair and tries < lim:
        tries += 1
        a = int(order[int(rs.integers(len(order)))])
        lo = np.searchsorted(ta, tau[a] * (1 - agew))
        hi = np.searchsorted(ta, tau[a] * (1 + agew))
        if hi - lo < 2:
            continue
        b = int(order[int(rs.integers(lo, hi))])
        if a == b:
            continue
        dt = float(t[b]) - float(t[a])
        dx2 = float(((X[b] - X[a]) ** 2).sum())
        if dt * dt > dx2:
            continue
        xv = math.sqrt(dx2 - dt * dt) / (0.5 * (tau[a] + tau[b]))
        if not (xlo <= xv < xhi):
            continue
        dta, dtb = t[a] - t, t[b] - t
        da = ((X - X[a]) ** 2).sum(1)
        db = ((X - X[b]) ** 2).sum(1)
        pa = (dta > 0) & (dta * dta > da)
        pb = (dtb > 0) & (dtb * dtb > db)
        fa = (dta < 0) & (dta * dta > da)
        fb = (dtb < 0) & (dtb * dtb > db)
        rows.append((xv, int((pa & pb).sum()), int(pa.sum()),
                     int(pb.sum()), int((~pa & ~fa & ~pb & ~fb).sum()),
                     0.5 * (tau[a] + tau[b]),
                     0.5 * (rad[a] + rad[b])))
    return np.array(rows, dtype=np.float64)


def snr_multi(D, cols, deg=3, extra=()):
    """SNR многомерной подгонки. cols — лог-предикторы, extra — линейные."""
    half = len(D) // 2
    cal, ev = D[:half], D[half:]

    def feats(A):
        F = [np.ones(len(A))]
        for c in cols:
            v = np.log(np.maximum(A[:, c], 1.0))
            v = v - v.mean()
            for p in range(1, deg + 1):
                F.append(v ** p)
        for c in extra:
            v = A[:, c] - A[:, c].mean()
            for p in range(1, deg + 1):
                F.append(v ** p)
        return np.vstack(F).T

    Fc, Fe = feats(cal), feats(ev)
    beta, *_ = np.linalg.lstsq(Fc, cal[:, 0], rcond=None)
    r = Fe @ beta - ev[:, 0]
    sd = float(np.std(r, ddof=1))
    return float(np.std(ev[:, 0], ddof=1)) / sd, len(ev)


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(3141592)
    plan = [(20000, 12000), (70000, 10000), (250000, 7000), (700000, 4500)]
    data = {}
    for n, npr in plan:
        data[n] = harvest2(n, rs, npr, 0.10, 1.60)
        print(f'  N = {n:>7}: пар {len(data[n]):>6}', flush=True)
    sets = (
        ('только V_cp', (1,), ()),
        ('+ V_a, V_b', (1, 2, 3), ()),
        ('+ N_both', (1, 2, 3, 4), ()),
        ('+ возраст и радиус', (1, 2, 3, 4), (5, 6)),
    )
    print()
    print('КОНТРОЛЬ: перемешать x — SNR обязан дать 1.00 при любом наборе')
    D = data[70000].copy()
    D[:, 0] = rs.permutation(D[:, 0])
    for nm, cols, ex in sets:
        s, m = snr_multi(D, cols, extra=ex)
        print(f'  {nm:>22}: {s:.4f}')
    print()
    print(f"{'набор предикторов':>22}" + ''.join(
        f'{n:>11}' for n in sorted(data)) + f"{'показатель':>14}")
    for nm, cols, ex in sets:
        vals, rows = [], []
        for n in sorted(data):
            s, m = snr_multi(data[n], cols, extra=ex)
            vals.append(s)
            rows.append((n, s, m))
        R = np.array(rows)
        err = R[:, 1] / np.sqrt(2 * (R[:, 2] - 1))
        A = np.vstack([np.log(R[:, 0]), np.ones(len(R))]).T
        W = np.diag(1.0 / (err / R[:, 1]) ** 2)
        cov = np.linalg.inv(A.T @ W @ A)
        be = cov @ (A.T @ W @ np.log(R[:, 1]))
        se = np.sqrt(np.diag(cov))
        print(f'{nm:>22}' + ''.join(f'{v:>11.3f}' for v in vals)
              + f'{be[0]:>+9.4f}±{se[0]:.4f}')
    print()
    print('ЧТЕНИЕ. Если с добавлением положения SNR растёт И показатель')
    print('становится положительным — плато было свойством оценщика.')
    print('Если показатель остаётся нулём при любом наборе — плато')
    print('принадлежит порядку.')


if __name__ == '__main__':
    main()
