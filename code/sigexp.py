#!/usr/bin/env python3
"""ПОКАЗАТЕЛЬ sigma(S): закрыть конфликт 0.799 против 0.967.

Архив (`phase-doc.md`) даёт sigma(S) ~ N^{0.799}; вчерашний замер на
условном ансамбле (`qcond-doc.md`) дал 0.9672 +- 0.0226 — расхождение
7.4 сигмы, и число в работе (через него считался порог видимости
квантового слоя).

ПРОГНОЗ ЛЕСТНИЦЫ (`carrier-doc.md`). Действие
S = k4 (N - N0 + 9N1 - 16N2 + 8N3) есть сумма по парам с МАЛЫМ
интервалом, то есть по СВЯЗЯМ и почти-связям. Связей ~ N^{2-2/d} =
N^{3/2} при d = 4 (замер: 0.40, 0.51, 0.57 от sqrt(N) — растёт к
плато). Сумма N^{3/2} слабо зависимых членов даёт разброс N^{3/4},
значит

    sigma(S) ~ N^{0.75}

Если же разброс держат ВСЕ пары (~N^2), вышло бы N^{1.0}.

ЗДЕСЬ оба ансамбля мерятся ОДНИМ кодом:
  (i)  безусловный засев диаманта (как в архиве);
  (ii) интервал с закреплёнными вершинами (как вчера).

КОНТРОЛЬ: разложение дисперсии по вкладам N0..N3 — какой счёт её держит.
Если N0 (связи) — прогноз 3/4 обоснован; если вклады сравнимы и
коррелированы, надо смотреть сумму.

  python3 sigexp.py
"""
import math

import numpy as np

from qcarrier import sprinkle4

K4 = 4.0 / math.sqrt(6.0)


def counts4(P):
    t, X = P[:, 0], P[:, 1:]
    dt = t[None, :] - t[:, None]
    d = X[None, :, :] - X[:, None, :]
    Rc = (dt > 0) & (dt * dt > (d * d).sum(-1))
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    n = len(P)
    N0 = int((Rc & (Iv == 0)).sum())
    N1 = int((Rc & (Iv == 1)).sum())
    N2 = int((Rc & (Iv == 2)).sum())
    N3 = int((Rc & (Iv == 3)).sum())
    return n, N0, N1, N2, N3


def ens(n, nsamp, rs, fixed_ends):
    rows = []
    for _ in range(nsamp):
        P = sprinkle4(n, rs, 1.0)
        if fixed_ends:
            P = np.vstack([np.zeros((1, 4), dtype=P.dtype), P,
                           np.array([[1.0, 0, 0, 0]], dtype=P.dtype)])
        rows.append(counts4(P))
    A = np.array(rows, dtype=float)
    S = K4 * (A[:, 0] - A[:, 1] + 9 * A[:, 2] - 16 * A[:, 3]
              + 8 * A[:, 4])
    return S, A


def fit(Ns, sd, nsamp):
    err = sd / np.sqrt(2 * (np.array(nsamp) - 1))
    A = np.vstack([np.log(Ns), np.ones(len(Ns))]).T
    W = np.diag(1.0 / (err / sd) ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    be = cov @ (A.T @ W @ np.log(sd))
    se = np.sqrt(np.diag(cov))
    rr = (np.log(sd) - A @ be) / (err / sd)
    return be[0], se[0], rr


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(161803)
    plan = [(100, 500), (200, 400), (400, 300), (800, 200), (1600, 90)]
    for fixed in (False, True):
        lab = 'интервал с концами' if fixed else 'безусловный засев'
        print(f'=== {lab} ===')
        print(f"{'N':>7}{'засевов':>9}{'<S>':>12}{'sigma(S)':>11}"
              f"{'<N0>':>9}{'sigma(N0)':>11}{'sigma(N1)':>11}")
        Ns, sd, ns, sd0 = [], [], [], []
        for n, k in plan:
            S, A = ens(n, k, rs, fixed)
            Ns.append(n + (2 if fixed else 0))
            sd.append(float(S.std(ddof=1)))
            ns.append(k)
            sd0.append(float(A[:, 1].std(ddof=1)))
            print(f'{n:>7}{k:>9}{S.mean():>12.1f}{S.std(ddof=1):>11.1f}'
                  f'{A[:,1].mean():>9.0f}{A[:,1].std(ddof=1):>11.1f}'
                  f'{A[:,2].std(ddof=1):>11.1f}', flush=True)
        a, e, rr = fit(np.array(Ns), np.array(sd), ns)
        print(f'  sigma(S) ~ N^{a:.4f} ± {e:.4f}  | '
              f'3/4: {abs(a-0.75)/e:.1f}с, 0.799: {abs(a-0.799)/e:.1f}с, '
              f'1.0: {abs(a-1.0)/e:.1f}с')
        print('  остатки: ' + ', '.join(f'{v:+.1f}' for v in rr))
        a0, e0, _ = fit(np.array(Ns), np.array(sd0), ns)
        print(f'  для сравнения: sigma(N0) ~ N^{a0:.4f} ± {e0:.4f} '
              f'(связей ~ N^1.5, значит прогноз 0.75)')
        print()
    print('КОНТРОЛЬ: кто держит дисперсию действия')
    S, A = ens(800, 200, rs, False)
    names = ('N', 'N0', 'N1', 'N2', 'N3')
    coef = (1.0, -1.0, 9.0, -16.0, 8.0)
    print(f"{'счёт':>6}{'<>':>11}{'sigma':>10}{'вклад k*sigma':>15}"
          f"{'корр. с S':>11}")
    for i, (nm, c) in enumerate(zip(names, coef)):
        s = A[:, i].std(ddof=1)
        cr = np.corrcoef(A[:, i], S)[0, 1]
        print(f'{nm:>6}{A[:,i].mean():>11.0f}{s:>10.1f}'
              f'{abs(c)*s*K4:>15.1f}{cr:>+11.3f}')
    print(f'  sigma(S) = {S.std(ddof=1):.1f} — сравнить с вкладами выше')


if __name__ == '__main__':
    main()
