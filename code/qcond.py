#!/usr/bin/env python3
"""УСЛОВНАЯ СУММА: есть ли фаза у пропагатора. Пересчёт с концами.

6 октября и сегодня утром считалась сумма по ВСЕМУ ансамблю, без
закрепления концов, и мерился МОДУЛЬ. Для гауссова действия модуль есть
exp(-sigma^2/2 hbar^2) — чистое затухание, структуры нет. Но пропагатор
есть сумма УСЛОВНАЯ, между заданными концами, и у неё

    K(tau) = <exp(iS/hbar)>_tau = exp(i mu(tau)/hbar) * exp(-sigma^2/2hbar^2)

то есть ФАЗА есть, и она идёт по mu(tau)/hbar. Гауссовость убивает
структуру в модуле безусловной суммы, но не в фазе условной.

КРИТЕРИЙ (та же форма, что 6 октября, но разность по КРАЕВЫМ ДАННЫМ, а
не по семействам): чтобы модуль не задавило (hbar >~ sigma) и фаза
успела провернуться (Delta mu / hbar >~ 2 pi), нужно

    Delta mu  >~  2 pi sigma  ~  6.3 sigma

Шестого октября эту роль играл разрыв между семействами, ΔS/σ = 292.
Здесь вопрос: даёт ли изменение КРАЕВЫХ ДАННЫХ столько само по себе.

ПОСТАНОВКА. Концы закреплены: порядок есть интервал I(p,q) с вершинами
p и q, засеянный с плотностью 1. Тогда собственное время между концами
tau = (24 N / pi)^{1/4} при d = 4 (объём диаманта (pi/24) tau^4).
Краевые данные меняются через tau, то есть через N.

РАЗЛИЧАТЬ ДВЕ ВЕЩИ, иначе ошибка:
  * статистическая разрешимость mu: нужна Delta mu >> sigma/sqrt(n) —
    это легко и ничего не значит;
  * ФИЗИЧЕСКИЙ критерий интерференции: Delta mu >~ 2 pi sigma, то есть
    против СОБСТВЕННОГО разброса ансамбля, а не против ошибки среднего.

КОНТРОЛИ:
  1. при фиксированном tau повторная выборка обязана дать Delta mu ~ 0
     (нет ложного сноса);
  2. |K| обязан совпасть с exp(-sigma^2/2 hbar^2) — проверка гауссовой
     формы на условном ансамбле;
  3. arg K обязан совпасть с mu/hbar по модулю 2 pi;
  4. асимметрия и эксцесс условного действия — гауссово ли оно вообще.

  python3 qcond.py
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)


def bd_action(Rc):
    n = Rc.shape[0]
    F = Rc.astype(np.float32)
    Iv = np.rint(F @ F).astype(np.int32)
    N0 = int((Rc & (Iv == 0)).sum())
    N1 = int((Rc & (Iv == 1)).sum())
    N2 = int((Rc & (Iv == 2)).sum())
    N3 = int((Rc & (Iv == 3)).sum())
    return K4 * (n - N0 + 9 * N1 - 16 * N2 + 8 * N3)


def interval_sprinkle(nint, rs):
    """Интервал I(p,q) в 4D: nint точек внутри диаманта + две вершины."""
    T = 1.0
    pts = []
    need = nint
    while len(pts) < need:
        m = int((need - len(pts)) * 5) + 32
        t = rs.random(m) * T
        v = rs.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / 3.0)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    P = np.array(pts[:need])
    P = np.vstack([np.array([[0.0, 0, 0, 0]]), P,
                   np.array([[T, 0, 0, 0]])])
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2)


def ens(nint, nsamp, rs):
    S = np.array([bd_action(interval_sprinkle(nint, rs))
                  for _ in range(nsamp)], dtype=float)
    mu, sd = float(S.mean()), float(S.std(ddof=1))
    g1 = float(((S - mu) ** 3).mean() / sd ** 3)
    g2 = float(((S - mu) ** 4).mean() / sd ** 4 - 3.0)
    return S, mu, sd, g1, g2


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    plan = [(50, 400), (100, 350), (200, 250), (300, 200),
            (450, 140), (600, 100)]
    print('УСЛОВНЫЙ АНСАМБЛЬ: концы закреплены, краевое данное = tau')
    print(f"{'N внутр':>9}{'tau':>8}{'истор':>7}{'mu':>12}{'sigma':>10}"
          f"{'mu/sigma':>10}{'асимм':>9}{'эксц':>9}")
    res = []
    for nint, nsamp in plan:
        S, mu, sd, g1, g2 = ens(nint, nsamp, rs)
        tau = (24.0 * (nint + 2) / math.pi) ** 0.25
        res.append((nint, tau, mu, sd, nsamp))
        print(f'{nint:>9}{tau:>8.3f}{nsamp:>7}{mu:>12.1f}{sd:>10.1f}'
              f'{mu/sd:>10.2f}{g1:>+9.3f}{g2:>+9.3f}', flush=True)
    print()
    print('КОНТРОЛЬ 1: при фиксированном tau повторная выборка -> нет сноса')
    a = ens(200, 250, rs)
    b = ens(200, 250, rs)
    dm = abs(a[1] - b[1])
    se = math.hypot(a[2] / math.sqrt(250), b[2] / math.sqrt(250))
    print(f'  mu = {a[1]:.1f} и {b[1]:.1f}, разница {dm:.1f} '
          f'при ошибке {se:.1f} -> {dm/se:.2f} сигмы')
    print()
    print('КОНТРОЛИ 2-3: |K| и arg K против гауссовой формы (N внутр = 300)')
    S, mu, sd, _, _ = ens(300, 200, rs)
    print(f"      {'hbar/s':>8}{'|K| замер':>12}{'exp(-s^2/2h^2)':>16}"
          f"{'arg K':>10}{'mu/h mod 2pi':>14}")
    for ratio in (4.0, 2.0, 1.2, 0.8):
        hb = ratio * sd
        K = np.exp(1j * S / hb).mean()
        pred = math.exp(-sd ** 2 / (2 * hb ** 2))
        ph = math.atan2(K.imag, K.real) % (2 * math.pi)
        php = (mu / hb) % (2 * math.pi)
        print(f'      {ratio:>8.2f}{abs(K):>12.5f}{pred:>16.5f}'
              f'{ph:>10.3f}{php:>14.3f}')
    print()
    print('=' * 72)
    print('РЕШАЮЩЕЕ: хватает ли Delta mu по КРАЕВЫМ ДАННЫМ')
    print(f"{'переход tau':>22}{'Delta mu':>12}{'sigma':>10}"
          f"{'Delta mu/sigma':>16}{'нужно 6.3':>11}")
    ok = []
    for i in range(len(res) - 1):
        n1, t1, m1, s1, _ = res[i]
        n2, t2, m2, s2, _ = res[i + 1]
        sbar = 0.5 * (s1 + s2)
        r = abs(m2 - m1) / sbar
        ok.append(r >= 2 * math.pi)
        print(f'{f"{t1:.2f} -> {t2:.2f}":>22}{abs(m2-m1):>12.1f}'
              f'{sbar:>10.1f}{r:>16.2f}'
              f'{("ДА" if r >= 2*math.pi else "нет"):>11}')
    n1, t1, m1, s1, _ = res[0]
    n2, t2, m2, s2, _ = res[-1]
    sbar = 0.5 * (s1 + s2)
    print(f'{f"весь охват {t1:.2f}->{t2:.2f}":>22}{abs(m2-m1):>12.1f}'
          f'{sbar:>10.1f}{abs(m2-m1)/sbar:>16.2f}'
          f'{("ДА" if abs(m2-m1)/sbar >= 2*math.pi else "нет"):>11}')
    print()
    if any(ok):
        print('ФАЗА ПРОВОРАЧИВАЕТСЯ: изменение краевых данных даёт сдвиг')
        print('среднего действия больше 2 pi sigma, значит у условной')
        print('амплитуды есть осцилляция, переживающая затухание —')
        print('и смесь семейств для этого НЕ нужна.')
    else:
        print('НЕ ХВАТАЕТ: сдвиг среднего по краевым данным меньше')
        print('2 pi sigma на каждом шаге. Смотреть на полный охват и на')
        print('зависимость от N.')


if __name__ == '__main__':
    main()
