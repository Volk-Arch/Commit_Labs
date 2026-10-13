#!/usr/bin/env python3
"""КАРАНДАШ ОРИЕНТАЦИЙ: весь R_mu_nu из счёта, и скаляр R — свёрткой.

ЗАМЫСЕЛ. В `chains-doc.md` измерено: три счёта интервала дают
`R_mu_nu u^mu u^nu` вдоль его собственного направления и к скаляру `R`
не чувствительны (вклад `R` снижает остаток на 0.002 %). Но
симметричный тензор полностью определён своей квадратичной формой на
времениподобных направлениях. Значит нужен ПУЧОК интервалов разных
наклонов через одно место — его мысль про карандаш цепей.

В конформно-плоской метрике с пространственной изотропией

    R_uu = [ R_etaeta + R_xx v² ] / ( a² (1 - v²) )

то есть зависимость от наклона `v` разводит две компоненты. А дальше

    R = ( -R_etaeta + 3 R_xx ) / a²

ПОЧЕМУ ЭТО ЖЁСТКАЯ ПРОВЕРКА. У линейного семейства `R_xx/R_etaeta` равно
ровно `+1/3`, у квадратичного ровно `-1/3`. **Знаки разные**, поэтому
различать надо знак, а не доли процента. И `R = 0` у линейного
возникает ТОЧНЫМ СОКРАЩЕНИЕМ двух ненулевых измеренных чисел
(`R_etaeta = 3h²`, `R_xx = h²`), а не тем, что мерить нечего.

ОТНОШЕНИЕ НЕ ТРЕБУЕТ КАЛИБРОВКИ. Постоянная `v_2 = -0.0201` входит общим
множителем в обе компоненты и из отношения `R_xx/R_etaeta` выпадает. Она
же выпадает из `R/R_etaeta`. Абсолютные значения приводятся отдельно, с
её ошибкой.

УСТОЙЧИВОСТЬ К МЕСТУ. Вдоль интервала `a` меняется, но у линейного обе
компоненты идут как `1/a²`, поэтому ОТНОШЕНИЕ ровно `1/3` всюду; у
квадратичного оно плывёт на 7 % на наших размерах. Поэтому headline —
именно отношение.

ДВЕ ЛОВУШКИ С МЕШАЮЩИМИ ЧЛЕНАМИ, обе пойманы.

  ПЕРВАЯ. Если подгонять по `W = X_2 (1-v²)/sqrt(C_1)`, то мешающая
  постоянная `d` в `X_2` даёт вклад `d/sqrt(C_1) - d v²/sqrt(C_1)`, то
  есть ЛОЖНЫЙ член с `v²`, и при `d = 0.005` он того же размера, что
  искомый.

  ВТОРАЯ. Подгонять мешающие члены вместе с сигналом тоже нельзя:
  столбцы `sqrt(C_1)/(1-v²)` и `sqrt(C_1)v²/(1-v²)` отличаются РОВНО на
  `sqrt(C_1)`, а над нашим диапазоном `C_1` (вдвое по корню) степени
  `sqrt(C_1)`, `1`, `1/sqrt(C_1)` почти коллинеарны. Число
  обусловленности вышло 6e8, и отношение «измерилось» как ровно `-1.00`
  в обоих семействах — подпись вырождения, а не замера.

  ПОЧИНКА: мешающие члены не подгоняются, а ВЫЧИТАЮТСЯ парно. Плоское
  плечо проходит тот же отбор и разбивается на те же ячейки по
  `(v², C_1)`; из каждой ячейки кривого вычитается та же ячейка
  плоского. Что бы ни было источником помехи — неточность `c_2` и `c_3`,
  отбор по вложению в ящик, конечность `C_1`, — оно одинаково в обоих
  плечах и уходит нацело. Остаётся план из ДВУХ столбцов.

ЧТО СЧИТАЕТСЯ ВХОДОМ. Только порядок — и НАКЛОН интервала `v`. Наклон
есть геометрия пучка, а не метрика: относительная скорость двух цепей в
причинном множестве измеряется из порядка (у нас это сделано радарным
кадром, `A = gamma²`). Кривизна на вход не подаётся нигде.

КОНТРОЛИ:
  1. ПЛОСКОЕ: `alpha` и `beta` обязаны быть нулями;
  2. ЗНАК отношения обязан смениться между семействами;
  3. `R` из свёртки обязан дать НУЛЬ у линейного и `6c` у квадратичного;
  4. обусловленность плана печатается — если столбцы почти коллинеарны,
     отношение не определено и об этом надо знать, а не подгонять.

  python3 pencil.py [--seeds N] [--per M]
"""
import argparse
import math
import pickle
import time

import numpy as np

from chains import c_k, counts, fits

C2C, C3C = c_k(2), c_k(3)
# ПОСТОЯННОЕ хранилище. Прежде путь шёл в сессионную папку
# /tmp/claude-0/-home-claude/<НОМЕР-СЕССИИ>/scratchpad, которая
# умирает вместе с сессией, а номер в пути меняется — то есть весь
# набор пропадал бы при каждом новом запуске. Переставлено 8 окт.
SCRATCH = '/home/claude/data'
KFAC = math.sqrt(24.0 / math.pi)
# ВНИМАНИЕ. Эта калибровка ЗАМЕРЕНА, а не выведена, и погрешность
# 14% — систематика, которая не падает ни от какого набора. Хуже:
# сама модель, в которой она применялась (X_2 ~ R_uu), ОПРОВЕРГНУТА
# точным интегралом 8 окт — промах 9.5 сигмы на плече, где
# предсказание есть точный нуль. Правильный план: `pwfix.py`,
# выведенные профили A(v²), B(v²); разбор в `design-doc.md`.
# Здесь оставлено только для сравнения со старыми прогонами.
V2CAL, V2ERR = -0.0201, 0.0028
V3CAL, V3ERR = -0.0452, 0.0059


def a_of(eta, fam, par, eta0):
    e = np.asarray(eta, dtype=float)
    if fam == 'flat':
        return np.ones_like(e)
    if fam == 'lin':
        return 1.0 + par * (e - eta0)
    if fam == 'root':
        # ЧЕТВЁРТОЕ СЕМЕЙСТВО, добавлено 8 октября. Условие R_xx = u + v = 0,
        # то есть a''a = -a'^2, равносильно (a^2)'' = 0, откуда
        # a = sqrt(1 + c(eta - eta0)). Тогда R_xx обращается в
        # ТОЖДЕСТВЕННЫЙ нуль, а R_etaeta = 3c^2/2 в центре. Проверено
        # символьно от метрики, расхождение 0.
        # Отклик R_uu = R_etaeta/(a^2(1-v^2)) имеет МАКСИМУМ при v = 0 —
        # форма, ПРОТИВОПОЛОЖНАЯ экспоненте, у которой там нуль. Два
        # семейства вместе дают обусловленный план по построению.
        return np.sqrt(1.0 + par * (e - eta0))
    if fam == 'exp':
        # ТРЕТЬЕ СЕМЕЙСТВО, добавлено 8 октября. Условие u = v, то есть
        # a''a = a'^2, решается как a = e^{k eta}, и тогда R_etaeta
        # обращается в ТОЖДЕСТВЕННЫЙ нуль, а R_xx = 2k^2. Проверено
        # символьно от метрики (символы Кристоффеля -> Риччи),
        # расхождение 0. Это ломает вырождение плана по ПОСТРОЕНИЮ:
        # отклик R_uu = R_xx v^2 / (a^2 (1-v^2)) обращается в нуль при
        # v = 0, тогда как у линейного семейства он там максимален.
        return np.exp(par * (e - eta0))
    return 1.0 + par * (e - eta0) ** 2 / 2.0


def truth(fam, par):
    """R_etaeta, R_xx, R в центре — только для СРАВНЕНИЯ, не для подгонки."""
    if fam == 'flat':
        return 0.0, 0.0, 0.0
    if fam == 'lin':
        return 3 * par ** 2, par ** 2, 0.0
    if fam == 'exp':
        return 0.0, 2 * par ** 2, 6 * par ** 2
    if fam == 'root':
        # a = sqrt(1 + c(eta-eta0)):  R_xx = 0 ТОЖДЕСТВЕННО,
        # R_etaeta = 3c^2/2,  R = -3c^2/2  в центре. Сверено символьно.
        return 1.5 * par ** 2, 0.0, -1.5 * par ** 2
    return -3 * par, par, 6 * par


def sprinkle(fam, par, L, T, eta0, rs):
    e = np.linspace(0.0, T, 40000)
    aa = a_of(e, fam, par, eta0)
    if (aa <= 0).any():
        raise ValueError('a ушло в нуль')
    w = aa ** 4
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(e))])
    n = rs.poisson(L ** 3 * cdf[-1])
    X = np.empty((n, 4))
    X[:, 0] = np.interp(rs.uniform(0, cdf[-1], n), cdf, e)
    X[:, 1:] = rs.uniform(0, L, (n, 3))
    return X[np.argsort(X[:, 0])]


VB = (0.0, 0.15, 0.30, 0.45, 0.60, 0.72, 0.85)   # границы полос по v²


def harvest(fam, par, cfg, rs, want):
    """Интервалы из одного сприклинга, С РАВНОМЕРНЫМ НАБОРОМ ПО НАКЛОНУ.

    Без выравнивания по полосам `v²` почти весь набор садится на малые
    наклоны (у быстрых интервалов больше шансов не влезть в ящик), и
    рычаг, которым только и разводятся две компоненты, пропадает.
    """
    L, T, eta0, half, lo, hi = cfg
    X = sprinkle(fam, par, L, T, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    tlo = (24.0 * lo / math.pi) ** 0.25
    thi = (24.0 * hi / math.pi) ** 0.25
    nb = len(VB) - 1
    quota = want // nb + 1
    got = [0] * nb
    out = []
    guard = 0
    while len(out) < want and guard < want * 25:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - half - thi),
                             np.searchsorted(t, eta0 + half)))
        p0, ps = t[ip], S[ip]
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 2.2)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0:
            continue
        v2 = dx2[ok] / (de[ok] ** 2)
        bi = np.digitize(v2, VB) - 1
        order = np.argsort([got[b] if 0 <= b < nb else 10 ** 9 for b in bi])
        for q in order[:8]:
            if len(out) >= want:
                break
            b = bi[q]
            if not (0 <= b < nb) or got[b] >= quota:
                continue
            jj = int(ok[q])
            f0, fs = t[j0 + jj], S[j0 + jj]
            mid = 0.5 * (p0 + f0)
            if abs(mid - eta0) > half:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, T):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo:
                continue
            Sl, tl = S[k0:k1], t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo <= a <= hi):
                continue
            _, bb, cc = counts(np.column_stack([tl[m], Sl[m]]))
            out.append((a,
                        bb / (C2C * a * (a - 1.0)) - 1.0,
                        cc / (C3C * a * (a - 1.0) * (a - 2.0)) - 1.0,
                        float(v2[q]), mid))
            got[b] += 1
    del X, t, S
    return out, got


# столбцы: C_1, X_2, X_3, v², mid
IC, IX2, IX3, IV, IM = 0, 1, 2, 3, 4


def design(A):
    """Два столбца кривизны плюс два мешающих."""
    s = np.sqrt(A[:, IC])
    g = 1.0 / (1.0 - A[:, IV])
    return np.column_stack([s * g, s * A[:, IV] * g,
                            np.ones(len(A)), 1.0 / s])


def solve(A, col, mask=None):
    D = design(A)
    w = A[:, IC]
    y = A[:, col]
    if mask is not None:
        D, w, y = D[mask], w[mask], y[mask]
    Dw = D * w[:, None]
    return np.linalg.solve(D.T @ Dw, Dw.T @ y)


def fit(A, sd, col):
    full = solve(A, col)
    us = np.unique(sd)
    jk = np.array([solve(A, col, sd != u) for u in us])
    k = len(us)
    cov = (k - 1) / k * ((jk - jk.mean(0)).T @ (jk - jk.mean(0)))
    return full, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=26)
    ap.add_argument('--per', type=int, default=420)
    args = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    cfg = (26.0, 20.0, 10.0, 1.6, 200, 900)
    fams = (('flat', 0.0, 'ПЛОСКОЕ — контроль'),
            ('lin', 0.035, 'линейное: R_xx/R_etaeta = +1/3, R = 0'),
            ('quad', 0.0012, 'квадратичное: R_xx/R_etaeta = -1/3, R = 6c'))
    print('=' * 78)
    print(f'ящик {cfg[0]:.0f}x{cfg[1]:.0f}, полоса |mid-eta0| <= {cfg[3]}, '
          f'C_1 от {cfg[4]} до {cfg[5]}')
    print(f'засевов {args.seeds} по {args.per}, полосы по v²: {VB}')
    print()
    res = {}
    for fam, par, tag in fams:
        rows, sdi, tot = [], [], None
        for s in range(args.seeds):
            rs = np.random.default_rng(310000 + 1931 * s + int(1e7 * par))
            r, got = harvest(fam, par, cfg, rs, args.per)
            rows += r
            sdi += [s] * len(r)
            tot = got if tot is None else [x + y for x, y in zip(tot, got)]
        A = np.array(rows)
        res[(fam, par)] = (A, np.array(sdi))
        print(f'--- {tag}: {len(A)} интервалов, по полосам v² {tot} ---')
        print(f"{'v²':>8}{'шт':>7}{'C_1':>7}{'X_2':>10}{'ст.ош':>9}"
              f"{'X_3':>10}{'ст.ош':>9}")
        for i in range(len(VB) - 1):
            m = (A[:, IV] >= VB[i]) & (A[:, IV] < VB[i + 1])
            if m.sum() < 20:
                continue
            B, s_ = A[m], np.array(sdi)[m]
            us = np.unique(s_)
            k = len(us)
            j2 = np.array([B[s_ != u][:, IX2].mean() for u in us])
            j3 = np.array([B[s_ != u][:, IX3].mean() for u in us])
            e2 = math.sqrt((k - 1) / k * ((j2 - j2.mean()) ** 2).sum())
            e3 = math.sqrt((k - 1) / k * ((j3 - j3.mean()) ** 2).sum())
            print(f'{B[:,IV].mean():>8.3f}{m.sum():>7}{B[:,IC].mean():>7.0f}'
                  f'{B[:,IX2].mean():>+10.4f}{e2:>9.4f}'
                  f'{B[:,IX3].mean():>+10.4f}{e3:>9.4f}')
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
        print()
    with open(f'{SCRATCH}/pencil.pkl', 'wb') as fh:
        pickle.dump(res, fh)

    print(f'  набор сохранён, время {time.time()-t0:.0f} с')
    print('  разбор: python3 pfit.py')


if __name__ == '__main__':
    main()
