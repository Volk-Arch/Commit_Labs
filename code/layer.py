#!/usr/bin/env python3
"""Кластеры частичного согласия — СЛОИ или комки? Проверка графена.

ОТКУДА. partial.py: кластеры из частичного согласия существуют —
модульность 0.439 на конусе против 0.019 на ящике из независимых шкал,
контроль решающий. Но они не компактны (протяжённость 0.559 против
случайной 0.591), а размерность внутри 8-10, то есть связанных пар мало.
Это подпись СЛОЯ, не сгустка.

Причина видна из формулы шапки m/k = (1 + dt/|dx|)/2:
    m = k-1   ->  dt/|dx| = 1 - 2/k   ->  почти СВЕТОПОДОБНО
    m = k/2   ->  dt = 0              ->  ОДНОВРЕМЕННО
Первый граф склеивает световые конусы, второй — одновременные события.

ЧТО МЕРИМ (его вопрос про графен: 2D слой, который бывает многослойным).
  1. Протяжённость кластера ПО ВРЕМЕНИ против ПО ПРОСТРАНСТВУ.
     Слой => одно сильно меньше другого.
  2. КОРРЕЛЯЦИОННУЮ РАЗМЕРНОСТЬ точек кластера в 3 пространственных
     координатах: C(r) ~ r^D. Лист дал бы D около 2, объём около 3.
  3. То же в 4 координатах: прижат ли кластер к световому конусу.
  4. Сколько слоёв: расслаивается ли кластер по времени на несколько
     тонких, как многослойный графен.
  5. КОНТРОЛЬ всюду: случайное подмножество того же размера.

  python3 layer.py
  python3 layer.py --n 400 --k 64 --nc 6
"""
import argparse
import math
import time

import numpy as np

from partial import diamond, scales_cone, m_matrix, spectral, modularity


def corr_dim(P, lo=0.08, hi=0.45):
    """Корреляционная размерность: C(r) ~ r^D по наклону в логах."""
    n = len(P)
    if n < 12:
        return float('nan')
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    d = d[np.triu_indices(n, 1)]
    sc = np.median(d)
    if sc <= 0:
        return float('nan')
    rs_ = np.geomspace(lo * sc, hi * sc, 10)
    C = np.array([(d < r).mean() for r in rs_])
    ok = (C > 2.0 / len(d)) & (C < 0.6)
    if ok.sum() < 4:
        return float('nan')
    return float(np.polyfit(np.log(rs_[ok]), np.log(C[ok]), 1)[0])


def layers(ts, nmax=4):
    """На сколько тонких слоёв расслаивается набор времён.

    Сравниваем разброс внутри слоёв с полным: если расслоение настоящее,
    отношение резко падает при нужном числе слоёв.
    """
    ts = np.sort(ts)
    tot = ts.var()
    if tot <= 0:
        return 1, 0.0
    best, bg = 1, 0.0
    for L in range(2, nmax + 1):
        # наилучшее разбиение отсортированного набора на L отрезков
        # (динамика по 1D k-means, точно)
        n = len(ts)
        cs = np.concatenate([[0], np.cumsum(ts)])
        cs2 = np.concatenate([[0], np.cumsum(ts ** 2)])

        def cost(i, j):
            m = j - i
            if m <= 0:
                return 0.0
            s = cs[j] - cs[i]
            return (cs2[j] - cs2[i]) - s * s / m

        D = np.full((L + 1, n + 1), np.inf)
        D[0, 0] = 0.0
        for l in range(1, L + 1):
            for j in range(1, n + 1):
                for i in range(l - 1, j):
                    v = D[l - 1, i] + cost(i, j)
                    if v < D[l, j]:
                        D[l, j] = v
        within = D[L, n] / n
        gain = 1.0 - within / tot
        if gain > bg + 0.12:            # требуем заметного выигрыша
            best, bg = L, gain
    return best, bg


def main():
    ap = argparse.ArgumentParser(description='Слои из частичного согласия')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--nc', type=int, default=6)
    ap.add_argument('--seeds', type=int, default=6)
    a = ap.parse_args()

    t0 = time.time()
    print(f'N={a.n}, k={a.k} шкал, кластеров {a.nc}, засевов {a.seeds}')
    print('формула шапки: m/k = (1 + dt/|dx|)/2, подтверждена')
    print()
    print(f"{'граф':>16}{'модуль':>9}{'по времени':>12}{'по простр.':>12}"
          f"{'отнош.':>9}{'разм.3D':>10}{'случ.3D':>10}{'слоёв':>8}")

    for gname in ('почти (m=k-1)', 'около (dt=0)', 'случайный граф'):
        Q, TT, XX, D3, D30, LL = [], [], [], [], [], []
        for t in range(a.seeds):
            rs = np.random.default_rng(1701000 + 977 * t)
            X = diamond(a.n, a.t, rs)
            S = scales_cone(X, a.k, rs)
            M = m_matrix(S)
            k = a.k
            if gname.startswith('почти'):
                A = ((M >= k - max(1, k // 16)) & (M < k)).astype(float)
                A = A + A.T
            elif gname.startswith('около'):
                ang = np.abs(M / k - 0.5)
                A = np.where((M < k) & (M.T < k),
                             np.exp(-((ang / 0.10) ** 2)), 0.0)
            else:
                A = (rs.random((a.n, a.n)) < 0.08).astype(float)
                A = np.maximum(A, A.T)
            np.fill_diagonal(A, 0.0)
            if A.sum() < 20:
                continue
            lab = spectral(A, a.nc, rs)
            Q.append(modularity(A, lab))
            for c in np.unique(lab):
                s = np.flatnonzero(lab == c)
                if len(s) < 14:
                    continue
                P = X[s]
                et = float(P[:, 0].std())
                ex = float(np.linalg.norm(
                    P[:, 1:] - P[:, 1:].mean(0), axis=1).std())
                TT.append(et)
                XX.append(ex)
                D3.append(corr_dim(P[:, 1:]))
                rnd = rs.choice(a.n, len(s), replace=False)
                D30.append(corr_dim(X[rnd][:, 1:]))
                LL.append(layers(P[:, 0])[0])
        if not Q:
            continue
        print(f'{gname:>16}{np.mean(Q):>9.3f}{np.mean(TT):>12.4f}'
              f'{np.mean(XX):>12.4f}{np.mean(TT)/max(np.mean(XX),1e-9):>9.2f}'
              f'{np.nanmean(D3):>10.2f}{np.nanmean(D30):>10.2f}'
              f'{np.mean(LL):>8.2f}', flush=True)

    print('\n  отношение время/простр. много меньше 1 => СЛОЙ по времени')
    print('  разм.3D заметно ниже случайной => лист, а не объём')
    print('  слоёв > 1 => расслоение, как многослойный графен')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
