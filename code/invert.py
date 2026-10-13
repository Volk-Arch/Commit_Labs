#!/usr/bin/env python3
"""ОБРАЩЕНИЕ: из трёх счётов — две кривизны. Разбор набора `curve.py`.

ЧТО ЗДЕСЬ ДЕЛАЕТСЯ.

  1. КОНТРОЛЬ ПЛОСКОГО. На плоском наборе подгонка обязана дать нули.
     Это же контроль на снос с размером интервала.

  2. ОТКЛИК. По всем кривым семействам сразу:

         X_2 = u_2 * (T²R) + v_2 * (T²R_00)
         X_3 = u_3 * (T²R) + v_3 * (T²R_00)

     Линейное семейство имеет `R = 0` тождественно и потому даёт строки
     с чистым `R_00`; квадратичное — смешанные. Вместе они разводят
     столбцы, и ковариация выходит из подгонки сама.

     Заметим: `T²R = sqrt(24 C_1/pi) * R l²` уже содержит `sqrt(C_1)`,
     то есть закон корня ВСТРОЕН в план. Если он неверен, это вылезет
     несогласием подгонок по полосам размера — пункт 3.

  3. ЗАКОН sqrt(C_1). Те же коэффициенты, подогнанные ОТДЕЛЬНО в полосах
     по `C_1`. Обязаны совпасть: закон корня предсказан без единого
     свободного параметра, и расхождение полос его опровергает.

  4. ПЕРЕНОС. Матрица калибруется на СЛАБЫХ кривизнах, а затем ею
     предсказываются кривизны СИЛЬНЫХ, отложенных. Подгонка, проверенная
     на своих же данных, не стоит ничего.

  5. ОДИН ИНТЕРВАЛ. Главный вопрос проекта: хватает ли одного интервала,
     чтобы померить кривизну в его месте. Если да, то интервал И ЕСТЬ
     та окрестность, которой «не существует», и затор снят замером, а
     не рассуждением.

О ВЕСАХ, и это важно. Вес берётся от `C_1`. Так МОЖНО: `X_2` несмещено
ПРИ ДАННОМ `C_1` (при `m` событиях внутри они независимы и ровны), то
есть вес есть функция обуславливающей величины, а не самого шума. Ровно
та ошибка, что стоила `coef.py` ложной систематики в 4.2 сигмы — там
весом было НАБЛЮДЁННОЕ число событий, оно же измеряемая величина, — тут
не возникает.

  python3 invert.py
"""
import math
import pickle

import numpy as np

SCRATCH = ('/tmp/claude-0/-home-claude/'
           '8f20417e-eecf-5054-8071-09c440210416/scratchpad')
# столбцы A: C_1, X_2, X_3, R, R_00, T²R, T²R_00
C1, X2, X3, AR, AU = 0, 1, 2, 5, 6


def wls(D, y, w):
    Dw = D * w[:, None]
    return np.linalg.solve(D.T @ Dw, Dw.T @ y)


def design(A, nuis=True):
    """План подгонки.

    Два первых столбца — кривизна. Дальше, если нужно, МЕШАЮЩИЕ члены:

        постоянная   вбирает неточность самих c_2 и c_3 (они известны
                     формулой, но проверены замером лишь до процента);
        1/sqrt(C_1)  вбирает любой размерный артефакт отбора.

    Эти формы НЕ совпадают с искомой: отклик кривизны идёт как
    `sqrt(C_1)`, то есть РАСТЁТ с размером, а мешающие — постоянны или
    УБЫВАЮТ. Закрепляются они плоским набором, где кривизна равна нулю
    тождественно, поэтому кривизну они у себя не отнимают.
    """
    cols = [A[:, AR], A[:, AU]]
    if nuis:
        cols += [np.ones(len(A)), 1.0 / np.sqrt(A[:, C1])]
    return np.column_stack(cols)


def fit(A, sd, col, nuis=True):
    """Коэффициенты для X_col и ошибка складным ножом по засевам."""
    D = design(A, nuis)
    y = A[:, col]
    w = A[:, C1]
    full = wls(D, y, w)
    seeds = np.unique(sd)
    jk = np.array([wls(D[sd != s], y[sd != s], w[sd != s]) for s in seeds])
    n = len(seeds)
    cov = (n - 1) / n * ((jk - jk.mean(0)).T @ (jk - jk.mean(0)))
    return full, cov


def pool(res, keys):
    A = np.vstack([res[k][0] for k in keys])
    sd = np.concatenate([res[k][1] + 1000 * i for i, k in enumerate(keys)])
    return A, sd


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    with open(f'{SCRATCH}/curve.pkl', 'rb') as fh:
        res = pickle.load(fh)
    keys = list(res)
    flat = [k for k in keys if k[0] == 'flat']
    curv = [k for k in keys if k[0] != 'flat']
    weak = [k for k in curv if k[1] in (0.025, 0.00070)]
    strong = [k for k in curv if k not in weak]
    print('набор:', {f'{k[0]} {k[1]}': len(res[k][0]) for k in keys})
    print()

    print('=' * 74)
    print('1. КОНТРОЛЬ ПЛОСКОГО — подгонка обязана дать нули')
    Af, sf = pool(res, flat)
    # у плоского T²R = T²R_00 = 0, поэтому план берём от кривых семейств:
    # подставляем плоские X в план сильного квадратичного той же длины
    Ac, _ = pool(res, curv)
    idx = np.random.default_rng(0).integers(len(Ac), size=len(Af))
    Afake = Af.copy()
    Afake[:, [AR, AU]] = Ac[idx][:, [AR, AU]]
    for col, nm in ((X2, 'X_2'), (X3, 'X_3')):
        c, cv = fit(Afake, sf, col)
        e = np.sqrt(np.diag(cv))
        print(f'  {nm}:  u = {c[0]:+.4f} +- {e[0]:.4f} '
              f'({abs(c[0])/e[0]:.1f} сигм),   '
              f'v = {c[1]:+.4f} +- {e[1]:.4f} ({abs(c[1])/e[1]:.1f} сигм)')
    print()

    print('2. ОТКЛИК по всем семействам сразу (плоское закрепляет мешающие)')
    Aa, sa = pool(res, curv + flat)
    M, E = {}, {}
    for col, nm in ((X2, 'X_2'), (X3, 'X_3')):
        c, cv = fit(Aa, sa, col)
        e = np.sqrt(np.diag(cv))
        M[nm], E[nm] = c, e
        print(f'  {nm}:  u = {c[0]:+.4f} +- {e[0]:.4f} '
              f'({abs(c[0])/e[0]:.1f} сигм),   '
              f'v = {c[1]:+.4f} +- {e[1]:.4f} ({abs(c[1])/e[1]:.1f} сигм)')
    print()

    print('3. ЗАКОН sqrt(C_1): те же коэффициенты по полосам размера')
    q = np.quantile(Aa[:, C1], [0, 1 / 3, 2 / 3, 1.0])
    print(f"{'C_1':>8}{'шт':>7}{'u(X_2)':>11}{'ст.ош':>9}"
          f"{'v(X_2)':>11}{'ст.ош':>9}{'v(X_3)':>11}{'ст.ош':>9}")
    for i in range(3):
        m = (Aa[:, C1] >= q[i]) & (Aa[:, C1] <= q[i + 1])
        if m.sum() < 200:
            continue
        c2_, cv2 = fit(Aa[m], sa[m], X2)
        c3_, cv3 = fit(Aa[m], sa[m], X3)
        e2, e3 = np.sqrt(np.diag(cv2)), np.sqrt(np.diag(cv3))
        print(f'{Aa[m][:,C1].mean():>8.0f}{m.sum():>7}{c2_[0]:>+11.4f}'
              f'{e2[0]:>9.4f}{c2_[1]:>+11.4f}{e2[1]:>9.4f}'
              f'{c3_[1]:>+11.4f}{e3[1]:>9.4f}')
    print()

    print('4. ПЕРЕНОС: калибровка на СЛАБЫХ, предсказание на СИЛЬНЫХ')
    Aw, sw = pool(res, weak + flat)
    f2, f3 = fit(Aw, sw, X2)[0], fit(Aw, sw, X3)[0]
    nu = np.vstack([f2[2:], f3[2:]])
    cal = np.vstack([f2[:2], f3[:2]])
    print(f'  матрица по слабым: [[{cal[0,0]:+.4f} {cal[0,1]:+.4f}]'
          f' [{cal[1,0]:+.4f} {cal[1,1]:+.4f}]]')
    print(f'  мешающие (вычитаются из отложенных): '
          f'пост. {nu[0,0]:+.5f}/{nu[1,0]:+.5f}, '
          f'1/sqrt(C_1) {nu[0,1]:+.4f}/{nu[1,1]:+.4f}')
    det = np.linalg.det(cal)
    print(f'  определитель {det:+.6f}, '
          f'число обусловленности {np.linalg.cond(cal):.1f}')
    if abs(det) < 1e-8:
        print('  => матрица вырождена: два счёта дают одно и то же, '
              'развести кривизны нельзя')
        return
    inv = np.linalg.inv(cal)
    print()
    print(f"{'отложенное':>22}{'шт':>7}{'R l² замер':>14}{'ст.ош':>10}"
          f"{'R l² истина':>14}{'R_00 l² замер':>16}{'ст.ош':>10}"
          f"{'R_00 l² истина':>16}")
    for k in strong:
        A, sd = res[k]
        seeds = np.unique(sd)
        est = []
        for s in seeds:
            B = A[sd == s]
            w = B[:, C1]
            tt = np.sqrt(24.0 * B[:, C1] / math.pi)
            # мешающие члены, закреплённые на слабых, вычитаются
            g = np.column_stack([np.ones(len(B)), 1.0 / np.sqrt(B[:, C1])])
            y = np.array([np.average(B[:, X2] - g @ nu[0], weights=w),
                          np.average(B[:, X3] - g @ nu[1], weights=w)])
            est.append((inv @ y) / np.average(tt, weights=w))
        est = np.array(est)
        m = est.mean(0)
        e = est.std(0, ddof=1) / math.sqrt(len(seeds))
        print(f'{k[0]+" "+str(k[1]):>22}{len(A):>7}{m[0]:>+14.5f}{e[0]:>10.5f}'
              f'{A[:,AR].mean():>+14.5f}{m[1]:>+16.5f}{e[1]:>10.5f}'
              f'{A[:,AU].mean():>+16.5f}')
    print()

    print('5. ОДИН ИНТЕРВАЛ: какой размер нужен, чтобы он сам мерил кривизну')
    s2 = Aa[:, X2].std(ddof=1) * math.sqrt(Aa[:, C1].mean())
    s3 = Aa[:, X3].std(ddof=1) * math.sqrt(Aa[:, C1].mean())
    print(f'  разброс на интервал: X_2 ~ {s2:.2f}/sqrt(C_1), '
          f'X_3 ~ {s3:.2f}/sqrt(C_1)')
    print(f"{'C_1':>9}{'ст.ош T²R':>14}{'ст.ош T²R_00':>16}")
    for c1 in (300, 1000, 3000, 10000, 30000):
        S = np.diag([s2 ** 2 / c1, s3 ** 2 / c1])
        cov = inv @ S @ inv.T
        print(f'{c1:>9}{math.sqrt(cov[0,0]):>14.3f}'
              f'{math.sqrt(cov[1,1]):>16.3f}')
    print()
    print('  ЧТЕНИЕ: при T²R порядка 0.3 нужен такой C_1, чтобы ст.ош')
    print('  была заметно меньше 0.3 — тогда ОДИН интервал меряет')
    print('  кривизну в своём месте, и окрестность в причинном множестве')
    print('  существует как рабочий объект, а не как пожелание.')


if __name__ == '__main__':
    main()
