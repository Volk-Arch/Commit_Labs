#!/usr/bin/env python3
"""ПРОВЕРКА ФОРМУЛЫ ОБЪЁМА МАЛОГО РОМБА, а затем её счёт в порядке.

ЗАЧЕМ. Его мысль: окрестностей нет, выживает только двустороннее. Тогда
уравнение поля надо искать не как `δS/δ(точка)`, а как закон на паре:
«связь между двумя событиями задана тем, что между ними». А между ними мы
умеем считать двумя способами сразу:

    объём интервала     |I(x,y)|            — счёт событий
    собственное время   длина долгой цепи   — счёт звеньев

В непрерывной теории объём малого причинного ромба отклоняется от `τ⁴`
именно на кривизну. Формула (Гиббонс и Солодухин, hep-th/0703098):

    V = (π/24) τ⁴ [ 1 + a τ²R + b R_μν T^μ T^ν + ... ],
    a = −1/180,   b = +1/30,   T^μ = τ u^μ

ЭТАП 1 (этот файл). Соглашения о знаках на слово не берутся. Формула
проверяется НАПРЯМУЮ в пространстве, где кривизна известна точно:
конформно-плоская метрика `ds² = a(η)²(−dη² + dx²)`.

Почему именно она удобна: конформные преобразования сохраняют световые
конусы, значит причинный ромб между двумя событиями на оси — это РОВНО
плоский ромб, и объём берётся прямым интегралом

    V = ∫_ромб a(η)⁴ dη d³x,      τ = ∫ a(η) dη

а `R` и `R_μν u^μu^ν` считаются символьно (sympy), а не из памяти.
Коэффициенты `a` и `b` ВЫТАСКИВАЮТСЯ ИЗ ДАННЫХ и сравниваются с
опубликованными — не подгоняются под них.

Проверка соглашений пределом: при `a ∝ η` выходит `a'' = 0` и `R = 0`,
что верно для FRW с излучением (след тензора энергии нуль).

  python3 diamond.py
"""
import math

import numpy as np


def curvature_symbolic():
    """R и R_μν u^μ u^ν для ds² = a(η)²(−dη² + dx²), символьно."""
    import sympy as sp
    eta, c, eta0 = sp.symbols('eta c eta0', real=True)
    a = 1 + c * (eta - eta0) ** 2 / 2
    x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
    X = [eta, x1, x2, x3]
    g = sp.diag(-a ** 2, a ** 2, a ** 2, a ** 2)
    ginv = g.inv()
    n = 4
    # связность
    Gam = [[[0] * n for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for m in range(n):
            for k in range(n):
                s = 0
                for s_ in range(n):
                    s += ginv[l, s_] * (sp.diff(g[s_, m], X[k])
                                        + sp.diff(g[s_, k], X[m])
                                        - sp.diff(g[m, k], X[s_]))
                Gam[l][m][k] = sp.simplify(s / 2)
    # Риччи
    Ric = sp.zeros(n)
    for m in range(n):
        for k in range(n):
            s = 0
            for l in range(n):
                s += sp.diff(Gam[l][m][k], X[l]) - sp.diff(Gam[l][m][l], X[k])
                for p in range(n):
                    s += (Gam[l][l][p] * Gam[p][m][k]
                          - Gam[l][k][p] * Gam[p][m][l])
            Ric[m, k] = sp.simplify(s)
    R = sp.simplify(sum(ginv[m, k] * Ric[m, k]
                        for m in range(n) for k in range(n)))
    # u — единичный времениподобный вдоль оси: u^mu = (1/a, 0,0,0)
    R00 = sp.simplify(Ric[0, 0] / a ** 2)
    return sp.lambdify((eta, c, eta0), R, 'numpy'), \
        sp.lambdify((eta, c, eta0), R00, 'numpy'), sp.simplify(R), \
        sp.simplify(R00)


def volume_and_tau(c, eta0, dh, n=120):
    """Точный объём ромба и собственное время между его остриями.

    Ромб: между (eta0 − dh/2, 0) и (eta0 + dh/2, 0) по КООРДИНАТАМ.
    Конусы плоские (конформная инвариантность), поэтому ромб плоский:
    |x| <= min(eta − e1, e2 − eta).

    ПОЙМАННАЯ ОШИБКА СЧЁТА. Сначала интеграл брался трапецией по всему
    отрезку, а подынтегральная функция имеет ИЗЛОМ в центре ромба
    (радиус дольки есть минимум двух линейных функций). Трапеция на
    изломе даёт ошибку порядка 10⁻⁵, то есть сравнимую с самим
    измеряемым отклонением: в столбце dev/tau² числа при одном `c`
    расходились на 4...16 %, хотя обязаны совпадать, и согласие
    улучшалось с ростом dh вместо обратного.

    Починка: интеграл разбит в точке излома, на каждой половине —
    квадратура Гаусса–Лежандра. Контроль при c = 0 ниже обязан дать
    нуль машинной точности.
    """
    e1, e2 = eta0 - dh / 2.0, eta0 + dh / 2.0
    mid = eta0
    xg, wg = np.polynomial.legendre.leggauss(n)
    V = 0.0
    tau = 0.0
    for lo, hi in ((e1, mid), (mid, e2)):
        et = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        a = 1 + c * (et - eta0) ** 2 / 2
        rad = np.minimum(et - e1, e2 - et)
        V += float(((4.0 / 3.0) * math.pi * rad ** 3 * a ** 4 * w).sum())
        tau += float((a * w).sum())
    return V, tau


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 70)
    Rf, R00f, Rsym, R00sym = curvature_symbolic()
    print('СИМВОЛЬНО, для a = 1 + c(eta−eta0)²/2:')
    print(f'  R        = {Rsym}')
    print(f'  R_uu     = {R00sym}')
    print()
    print('ПРОВЕРКА СОГЛАШЕНИЙ ПРЕДЕЛОМ: при a ∝ eta (то есть a\'\'=0)')
    print('  должно быть R = 0 — верно для FRW с излучением.')
    import sympy as sp
    eta, k = sp.symbols('eta k', positive=True)
    # отдельная проверка: a = k*eta
    a2 = k * eta
    g = sp.diag(-a2 ** 2, a2 ** 2, a2 ** 2, a2 ** 2)
    X = [eta] + list(sp.symbols('y1 y2 y3', real=True))
    ginv = g.inv()
    Gam = [[[0] * 4 for _ in range(4)] for _ in range(4)]
    for l in range(4):
        for m in range(4):
            for kk in range(4):
                s = 0
                for s_ in range(4):
                    s += ginv[l, s_] * (sp.diff(g[s_, m], X[kk])
                                        + sp.diff(g[s_, kk], X[m])
                                        - sp.diff(g[m, kk], X[s_]))
                Gam[l][m][kk] = sp.simplify(s / 2)
    Ric = sp.zeros(4)
    for m in range(4):
        for kk in range(4):
            s = 0
            for l in range(4):
                s += sp.diff(Gam[l][m][kk], X[l]) - sp.diff(Gam[l][m][l], X[kk])
                for p in range(4):
                    s += (Gam[l][l][p] * Gam[p][m][kk]
                          - Gam[l][kk][p] * Gam[p][m][l])
            Ric[m, kk] = sp.simplify(s)
    R_rad = sp.simplify(sum(ginv[m, kk] * Ric[m, kk]
                            for m in range(4) for kk in range(4)))
    print(f'  R при a = k*eta: {R_rad}   '
          f'{"СХОДИТСЯ" if R_rad == 0 else "НЕ СХОДИТСЯ"}')
    print()

    print('=' * 70)
    print('ВЫТАСКИВАЕМ КОЭФФИЦИЕНТЫ ИЗ ДАННЫХ')
    print('В конформно-плоском семействе R и R_uu оба пропорциональны c,')
    print('поэтому по одному семейству разделить a и b нельзя — измеряется')
    print('их КОМБИНАЦИЯ. Её и сравниваем с опубликованной.')
    print()
    eta0 = 3.0
    # КОНТРОЛЬ: при c = 0 пространство плоское, отклонение обязано быть
    # нулём машинной точности. Без этого контроля ошибка интегрирования
    # маскируется под физику.
    print('КОНТРОЛЬ ПЛОСКОГО (c = 0): dev обязан быть нулём машинным')
    for dh in (0.5, 1.0, 2.0):
        V0, t0_ = volume_and_tau(0.0, eta0, dh)
        print(f'  dh = {dh:.1f}: tau = {t0_:.12f} (ожидание {dh:.1f}), '
              f'dev = {V0/(math.pi/24*t0_**4)-1:+.2e}')
    print()
    print(f"{'c':>9}{'dh':>7}{'tau':>10}{'R':>12}{'R_uu':>12}"
          f"{'(V/Vпл −1)':>13}{'/(tau²)':>12}")
    rows = []
    for c in (0.004, 0.008, 0.016):
        for dh in (0.5, 1.0):
            V, tau = volume_and_tau(c, eta0, dh)
            Vflat = math.pi / 24.0 * tau ** 4
            R = float(Rf(eta0, c, eta0))
            R00 = float(R00f(eta0, c, eta0))
            dev = V / Vflat - 1.0
            rows.append((c, dh, tau, R, R00, dev))
            print(f'{c:>9.4f}{dh:>7.2f}{tau:>10.4f}{R:>12.5f}{R00:>12.5f}'
                  f'{dev:>+13.6f}{dev/tau**2:>+12.6f}')

    print()
    print('ПОДГОНКА: dev = tau² (a·R + b·R_uu), по всем строкам сразу')
    A = np.array([[r[2] ** 2 * r[3], r[2] ** 2 * r[4]] for r in rows])
    y = np.array([r[5] for r in rows])
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    print(f'  из данных:      a = {coef[0]:+.6f}, b = {coef[1]:+.6f}, '
          f'R² = {ss:.6f}')
    print(f'  опубликовано:   a = {-1/180:+.6f}, b = {1/30:+.6f}')
    print()
    # проверка опубликованных коэффициентов напрямую
    pub = A @ np.array([-1 / 180.0, 1 / 30.0])
    print('СРАВНЕНИЕ ПОСТРОЧНО (опубликованные коэффициенты, без подгонки)')
    print(f"{'c':>9}{'dh':>7}{'замер dev':>14}{'формула':>14}{'отнош':>9}")
    for r, p in zip(rows, pub):
        print(f'{r[0]:>9.4f}{r[1]:>7.2f}{r[5]:>+14.6f}{p:>+14.6f}'
              f'{r[5]/p if abs(p)>1e-15 else float("nan"):>9.3f}')
    rel = np.abs((y - pub) / np.maximum(np.abs(y), 1e-15))
    print(f'  среднее расхождение с опубликованной формулой: '
          f'{rel.mean()*100:.1f} %')


if __name__ == '__main__':
    main()
