#!/usr/bin/env python3
"""КРИВИЗНА ИЗ ТРЁХ СЧЁТОВ. Ни часов, ни координат, ни плотности.

ЗАМЫСЕЛ. Интервал между ДВУМЯ НАСТОЯЩИМИ событиями `p < f`. С него
снимаются три числа, и все три — чистый порядок:

    C_1 = сколько событий между ними
    C_2 = сколько пар   x < y   между ними
    C_3 = сколько троек x < y < z между ними

В плоском они связаны универсальными постоянными (проверено замером в
`chains.py`): при `m` событиях упорядоченных пар в среднем ровно
`c_2 m(m-1)`, троек `c_3 m(m-1)(m-2)`, где `c_2 = 1/20`, `c_3 = 1/2100`.
Поэтому

    X_2 = C_2/(c_2 m(m-1)) - 1,    X_3 = C_3/(c_3 m(m-1)(m-2)) - 1

в плоском — ТОЧНЫЕ нули: при данном `m` события внутри интервала
независимы и ровно распределены, значит оценка несмещена при любом `m`,
любой плотности и в любой системе отсчёта.

В кривом оба линейны по `T²R` и `T²R_00`. Собственное время `T` в
причинном множестве не дано, но его и НЕ НУЖНО ЗНАТЬ: оно выражается
через сам счёт, `T⁴ = 24 C_1/(pi rho)`, то есть

    T²R = sqrt(24 C_1/pi) * (R l²),     l = rho^(-1/4)

и значит

    X_2 = sqrt(C_1) * ( p_2 * R l² + q_2 * R_00 l² )
    X_3 = sqrt(C_1) * ( p_3 * R l² + q_3 * R_00 l² )

Три счёта на входе — две кривизны в единицах шага дискретности на
выходе. Больше не нужно НИЧЕГО: ни вложения, ни посаженных часов, ни
плотности — она сама сидит в `C_1`.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПРЕЖНЕГО. Раньше (`twoterm.py`, `coef.py`) остриё
`p`, остриё `f` и `tau` между ними сажались по известной метрике, и
причинное множество давало лишь счёт. Тогда «счёт воспроизводит объём»
есть ТОЖДЕСТВО: сприклинг плотности `rho` в область объёма `V` даёт
`<N> = rho V` по определению пуассоновского процесса. Здесь же посажено
НЕ ЧТО.

ЧТО ПРОВЕРЯЕТСЯ, по убыванию строгости:

  1. КОНТРОЛЬ ПЛОСКОГО: `X_2` и `X_3` — нули во всех полосах по размеру.
     Это же и контроль на снос с размером, который иначе подменял бы
     собой кривизну.
  2. ЗАКОН sqrt(C_1) — БЕЗ СВОБОДНЫХ ПАРАМЕТРОВ: при одной кривизне
     отклонение обязано расти как корень из числа событий.
  3. РАЗДЕЛЕНИЕ: линейное семейство имеет `R = 0` тождественно, значит
     отзывается только `R_00`; квадратичное даёт `R > 0` и `R_00 < 0`.
  4. ПЕРЕНОС: матрица 2x2 калибруется на одних кривизнах и проверяется
     предсказанием на других, ОТЛОЖЕННЫХ.

ПРАВИЛО РАЗМЕРА, добытое дорого: держать `T²R <= 0.3`. Параметр
разложения есть `(T/L_кривизны)²`, и при `T²R = 0.3` он равен 0.025.
Прежние замеры шли при `T²R` до 2.3, где обрезание ряда даёт до 13 %, —
это я и гонял как «промах коэффициентов».

БЮДЖЕТ. Отношение сигнала к шуму на интервал есть `alpha*sqrt(C_1)/1.7`,
а стоимость интервала растёт как `C_1²` (матрица отношений). Значит при
данном времени выгодны МНОГО интервалов среднего размера, а не мало
больших: суммарная значимость идёт как `alpha*sqrt(бюджет/C_1)`.

  python3 curve.py [--seeds N]
"""
import argparse
import math
import pickle
import time

import numpy as np

from chains import c_k, counts, fits

C2, C3 = c_k(2), c_k(3)
SCRATCH = ('/tmp/claude-0/-home-claude/'
           '8f20417e-eecf-5054-8071-09c440210416/scratchpad')


def a_of(eta, fam, par, eta0):
    e = np.asarray(eta, dtype=float)
    if fam == 'flat':
        return np.ones_like(e)
    if fam == 'lin':
        return 1.0 + par * (e - eta0)
    return 1.0 + par * (e - eta0) ** 2 / 2.0


def curvature(eta, fam, par, eta0):
    """R, R_etaeta, R_xx для ds² = a²(-deta²+dx²). Формулы сверены
    символьно в `twoterm.py`: 0 расхождений из 27 точек на семейство."""
    a = float(a_of(eta, fam, par, eta0))
    if fam == 'flat':
        return 0.0, 0.0, 0.0
    if fam == 'lin':
        return 0.0, 3.0 * par ** 2 / a ** 2, par ** 2 / a ** 2
    s = par * (eta - eta0) ** 2
    return (6.0 * par / a ** 3,
            6.0 * par * (s - 2) / (s + 2) ** 2,
            2.0 * par * (3 * s + 2) / (s + 2) ** 2)


def R_pair(mid, de, dx2, fam, par, eta0):
    """R и R_uu для НАПРАВЛЕНИЯ самого интервала.

    Направление даёт пара: u^mu ∝ (deta, dx), нормировка по g = a²eta,
    откуда R_uu = [R_ee deta² + R_xx |dx|²] / (a²(deta² - |dx|²)).
    При dx = 0 сводится к R_ee/a², как и обязано.

    Это и есть карандаш цепей: `R` от направления не зависит, `R_uu`
    зависит, и разные ориентации интервала дают разные сочетания тех же
    двух чисел — потому их и удаётся развести.
    """
    a = float(a_of(mid, fam, par, eta0))
    R, Ree, Rxx = curvature(mid, fam, par, eta0)
    return R, (Ree * de ** 2 + Rxx * dx2) / (a ** 2 * (de ** 2 - dx2))


def sprinkle(fam, par, L, T, eta0, rs):
    """Постоянная ФИЗИЧЕСКАЯ плотность 1: по eta плотность идёт как a⁴,
    по пространству ровно. Тогда шаг дискретности l = 1 всюду, и
    кривизна сразу выходит в его единицах."""
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


def harvest(fam, par, cfg, rs, want):
    """Интервалы из ОДНОГО сприклинга. Возвращает строки
    (C_1, X_2, X_3, R, R_00, T²R, T²R_00)."""
    L, T, eta0, half, lo, hi = cfg
    X = sprinkle(fam, par, L, T, eta0, rs)
    t = np.ascontiguousarray(X[:, 0])
    S = np.ascontiguousarray(X[:, 1:])
    # tau, отвечающее нужному размеру: C_1 = (pi/24) tau⁴
    tlo = (24.0 * lo / math.pi) ** 0.25
    thi = (24.0 * hi / math.pi) ** 0.25
    out = []
    guard = 0
    while len(out) < want and guard < want * 12:
        guard += 1
        ip = int(rs.integers(np.searchsorted(t, eta0 - half),
                             np.searchsorted(t, eta0 + half * 0.2)))
        p0, ps = t[ip], S[ip]
        # кандидаты в f: непрерывный срез по времени (массив отсортирован)
        j0 = np.searchsorted(t, p0 + tlo)
        j1 = np.searchsorted(t, p0 + thi * 1.8)
        if j1 - j0 < 40:
            continue
        de = t[j0:j1] - p0
        d = S[j0:j1] - ps
        dx2 = np.einsum('ij,ij->i', d, d)
        tau2 = de * de - dx2
        ok = np.flatnonzero((tau2 > tlo * tlo) & (tau2 < thi * thi))
        if len(ok) == 0:
            continue
        rs.shuffle(ok)
        for jj in ok[:6]:
            if len(out) >= want:
                break
            f0 = t[j0 + jj]
            fs = S[j0 + jj]
            mid = 0.5 * (p0 + f0)
            if abs(mid - eta0) > half * 0.45:
                continue
            if not fits(np.concatenate([[p0], ps]),
                        np.concatenate([[f0], fs]), L, T):
                continue
            k0 = np.searchsorted(t, p0, 'right')
            k1 = np.searchsorted(t, f0, 'left')
            if k1 - k0 < lo:
                continue
            Sl = S[k0:k1]
            tl = t[k0:k1]
            u = Sl - ps
            w = fs - Sl
            m = (((tl - p0) ** 2 > np.einsum('ij,ij->i', u, u))
                 & ((f0 - tl) ** 2 > np.einsum('ij,ij->i', w, w)))
            a = int(m.sum())
            if not (lo <= a <= hi):
                continue
            I = np.column_stack([tl[m], Sl[m]])
            _, b, c = counts(I)
            R, Ruu = R_pair(mid, f0 - p0,
                            float(((fs - ps) ** 2).sum()), fam, par, eta0)
            tt = math.sqrt(24.0 * a / math.pi)
            out.append((a,
                        b / (C2 * a * (a - 1.0)) - 1.0,
                        c / (C3 * a * (a - 1.0) * (a - 2.0)) - 1.0,
                        R, Ruu, tt * R, tt * Ruu))
            del I
    del X, t, S
    return out


def bands(A, nb=3):
    q = np.quantile(A[:, 0], np.linspace(0, 1, nb + 1))
    for i in range(nb):
        m = (A[:, 0] >= q[i]) & (A[:, 0] <= q[i + 1])
        if m.sum() < 10:
            continue
        B = A[m]
        n = len(B)
        yield (B[:, 0].mean(), n,
               B[:, 1].mean(), B[:, 1].std(ddof=1) / math.sqrt(n),
               B[:, 2].mean(), B[:, 2].std(ddof=1) / math.sqrt(n),
               B[:, 5].mean(), B[:, 6].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=16)
    ap.add_argument('--per', type=int, default=220)
    args = ap.parse_args()
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    cfg = (26.0, 24.0, 12.0, 8.0, 160, 900)
    fams = (('flat', 0.0, 'ПЛОСКОЕ — контроль, обязаны быть нули'),
            ('lin', 0.025, 'линейное слабое: R = 0, отзыв только от R_00'),
            ('lin', 0.040, 'линейное сильное (верхние полосы вне окна)'),
            ('quad', 0.00070, 'квадратичное слабое: R > 0, R_00 < 0'),
            ('quad', 0.00140, 'квадратичное сильное'))
    print('=' * 78)
    print(f'ящик {cfg[0]:.0f}x{cfg[1]:.0f}, C_1 от {cfg[4]} до {cfg[5]}, '
          f'засевов {args.seeds} по {args.per} интервалов')
    print()
    res = {}
    for fam, par, tag in fams:
        rows, sdi = [], []
        for s in range(args.seeds):
            rs = np.random.default_rng(60000 + 977 * s + int(1e7 * par))
            r = harvest(fam, par, cfg, rs, args.per)
            rows += r
            sdi += [s] * len(r)
        A = np.array(rows)
        res[(fam, par)] = (A, np.array(sdi))
        print(f'--- {tag}   (интервалов {len(A)}) ---')
        print(f"{'C_1':>7}{'шт':>7}{'X_2':>10}{'ст.ош':>9}{'сигм':>6}"
              f"{'X_3':>10}{'ст.ош':>9}{'сигм':>6}{'T²R':>8}{'T²R_00':>9}")
        for c1, n, x2, s2, x3, s3, aR, aU in bands(A):
            print(f'{c1:>7.0f}{n:>7}{x2:>+10.4f}{s2:>9.4f}'
                  f'{abs(x2)/max(s2,1e-12):>6.1f}'
                  f'{x3:>+10.4f}{s3:>9.4f}{abs(x3)/max(s3,1e-12):>6.1f}'
                  f'{aR:>8.3f}{aU:>9.3f}')
        print(f'  ({time.time()-t0:.0f} с)', flush=True)
        print()
    with open(f'{SCRATCH}/curve.pkl', 'wb') as fh:
        pickle.dump(res, fh)
    print('=' * 78)
    print('ЧТЕНИЕ. Плоское обязано дать нули во всех полосах. У кривых')
    print('X_2 и X_3 обязаны расти с C_1 как корень, и у линейного отзыв')
    print('идёт целиком от R_00, потому что R там тождественный нуль.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
