#!/usr/bin/env python3
"""БЮДЖЕТ ПРОСТРАНСТВЕННОЙ ФАЗЫ: что может нести линейка порядка.

Волна есть exp(i(omega t - k x)). Частоту мы получили ИЗ СЧЁТА
(`qclock-doc.md`: бюджет 4.94 sqrt(N) радиан). Волновое число брать
неоткуда: порядок есть совершенные часы (+0.9577) и слепая линейка
(-0.1055). Здесь спрашиваем у линейки то, чего не спрашивали: сколько
радиан она может нести.

ОЦЕНЩИК. Для несравнимой пары (a,b) наблюдаемое есть общее прошлое.
В непрерывном случае пересечение двух прошлых конусов от точек одного
возраста tau с пространственным разносом d есть конус возраста
tau - d/2, поэтому

    u = V_общ / sqrt(V_a V_b) = (1 - x/2)^4,    x = d / tau

КОНТРОЛЬ БЕЗ СВОБОДНЫХ ПАРАМЕТРОВ: замеренное u против (1-x/2)^4.
Отсюда оценщик x = 2(1 - u^{1/4}).

БЮДЖЕТ. Фаза есть k*d, ошибка фазы k*delta_d, значит число доступных
радиан равно (размах d)/(абсолютное разрешение delta_d) ~ tau/delta_d.
Из пуассоновского шума V_общ выходит прогноз delta_x = u^{1/4}/(2 sqrt
V_общ), то есть

    бюджет ~ 2 sqrt(V_общ)

против 4.94 sqrt(N) у часов. Прогноз ПОЛОЖИТЕЛЬНЫЙ, поэтому контроли
строже обычного.

КОНТРОЛИ:
  1. u против (1-x/2)^4 — геометрия, без подгонки;
  2. часы на тех же данных обязаны дать 0.286/sqrt(N);
  3. перемешивание расстояний обязано убить связь (нуль);
  4. пол: разрешение не может быть лучше пуассоновского 1/sqrt(V_общ);
  5. пределы: d -> 0 даёт u -> 1, большое d даёт u -> 0.

  python3 qruler.py
"""
import math

import numpy as np


def sprinkle4(n, rs, T=1.0):
    pts = []
    while len(pts) < n:
        m = int((n - len(pts)) * 5) + 64
        t = rs.random(m) * T
        v = rs.normal(size=(m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        rad = (rs.random(m) ** (1.0 / 3.0)) * (T / 2)
        x = v * rad[:, None]
        ok = rad < np.minimum(t, T - t)
        for i in np.nonzero(ok)[0]:
            pts.append(np.concatenate([[t[i]], x[i]]))
    return np.array(pts[:n])


def rel(P):
    dt = P[:, 0][None, :] - P[:, 0][:, None]
    dx2 = ((P[:, 1:][None, :, :] - P[:, 1:][:, None, :]) ** 2).sum(-1)
    return (dt > 0) & (dt ** 2 > dx2), dt, dx2


def collect(n, rs, npair, T=1.0):
    P = sprinkle4(n, rs, T)
    Rc, dt, dx2 = rel(P)
    past = Rc.copy()                       # past[:, a] = прошлое a
    Vp = past.sum(axis=0)
    # возраст от нижней вершины: tau = sqrt(t^2 - |x|^2)
    tau = np.sqrt(np.maximum(P[:, 0] ** 2 -
                             (P[:, 1:] ** 2).sum(1), 0.0))
    # несравнимые пары
    un = ~(Rc | Rc.T)
    np.fill_diagonal(un, False)
    xs, ys = np.nonzero(np.triu(un, 1))
    if len(xs) == 0:
        return None
    pick = rs.choice(len(xs), size=min(npair, len(xs)), replace=False)
    rows = []
    for i in pick:
        a, b = int(xs[i]), int(ys[i])
        if Vp[a] < 25 or Vp[b] < 25:
            continue
        if not (0.8 <= tau[a] / max(tau[b], 1e-9) <= 1.25):
            continue
        vcp = int((past[:, a] & past[:, b]).sum())
        sep = math.sqrt(max(dx2[a, b] - dt[a, b] ** 2, 0.0))
        tb = math.sqrt(tau[a] * tau[b])
        u = vcp / math.sqrt(Vp[a] * Vp[b])
        rows.append((sep, tb, u, vcp, Vp[a], Vp[b], sep / tb))
    return np.array(rows) if rows else None


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(10102026)
    print('КОНТРОЛЬ 1: u против (1 - x/2)^4 — геометрия без подгонки')
    D = collect(1400, rs, 30000)
    x, u = D[:, 6], D[:, 2]
    print(f"      {'x = d/tau':>12}{'u замер':>11}{'(1-x/2)^4':>12}"
          f"{'пар':>7}")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5),
                   (0.5, 0.7), (0.7, 1.0), (1.0, 1.5)):
        k = (x >= lo) & (x < hi)
        if k.sum() < 40:
            continue
        xm = float(x[k].mean())
        print(f'      {f"{lo}-{hi}":>12}{float(u[k].mean()):>11.4f}'
              f'{(1 - xm / 2) ** 4:>12.4f}{int(k.sum()):>7}')
    print()
    print('КОНТРОЛЬ 5: пределы')
    k = x < 0.05
    print(f'  x < 0.05: u = {float(u[k].mean()):.4f} (ожидание -> 1)')
    k = x > 1.2
    if k.sum() > 20:
        print(f'  x > 1.2 : u = {float(u[k].mean()):.4f} (ожидание -> 0)')
    print()
    print('РАЗРЕШЕНИЕ ЛИНЕЙКИ И БЮДЖЕТ ФАЗЫ')
    print(f"{'V_общ':>12}{'пар':>7}{'delta_x':>11}{'пуассон':>11}"
          f"{'бюджет tau/delta_d':>20}{'прогноз 2vV':>13}")
    rows = []
    for lo, hi in ((25, 60), (60, 120), (120, 240), (240, 480),
                   (480, 1000)):
        k = (D[:, 3] >= lo) & (D[:, 3] < hi)
        if k.sum() < 60:
            continue
        uu = D[k, 2]
        xt = D[k, 6]
        xe = 2.0 * (1.0 - np.maximum(uu, 1e-12) ** 0.25)
        dx = float(np.std(xt - xe, ddof=1))
        vbar = float(D[k, 3].mean())
        pois = float(np.mean(np.maximum(uu, 1e-12) ** 0.25
                             / (2 * np.sqrt(D[k, 3]))))
        budget = 1.0 / dx
        rows.append((vbar, dx, budget, int(k.sum())))
        print(f'{f"{lo}-{hi}":>12}{int(k.sum()):>7}{dx:>11.5f}'
              f'{pois:>11.5f}{budget:>20.1f}{2*math.sqrt(vbar):>13.1f}')
    print()
    print('КОНТРОЛЬ 3: перемешать расстояния — связь обязана исчезнуть')
    perm = rs.permutation(len(x))
    xe = 2.0 * (1.0 - np.maximum(u, 1e-12) ** 0.25)
    print(f'  ст.откл. (x - x_оценка)      = {np.std(x - xe, ddof=1):.5f}')
    print(f'  то же при перемешанных x     = '
          f'{np.std(x[perm] - xe, ddof=1):.5f}')
    print()
    print('КОНТРОЛЬ 2: часы на тех же данных')
    rho = 1400 / (math.pi / 24)
    th = (24.0 * D[:, 4] / (math.pi * rho)) ** 0.25
    # D[:,1] = sqrt(tau_a tau_b); сверяем с оценкой по V_a
    r = D[:, 1] / th
    print(f'  отн. разброс часов = {np.std(r, ddof=1)/np.mean(r):.5f}, '
          f'прогноз 0.286/sqrt(V) при V~{D[:,4].mean():.0f} = '
          f'{0.286/math.sqrt(D[:,4].mean()):.5f}')
    print()
    R = np.array([(r[0], r[2]) for r in rows])
    a, b = np.polyfit(np.log(R[:, 0]), np.log(R[:, 1]), 1)
    print(f'ЗАКОН БЮДЖЕТА: ~ V^{a:.3f}, коэффициент {math.exp(b):.2f} '
          f'(прогноз V^0.5, коэффициент 2)')
    print(f'  максимальный замеренный бюджет: {R[:,1].max():.1f} радиан '
          f'= {R[:,1].max()/(2*math.pi):.2f} оборотов')
    print(f'  порог 2pi = {2*math.pi:.2f} -> '
          f'{"ПРОХОДИТ" if R[:,1].max() > 2*math.pi else "НЕ проходит"}')


if __name__ == '__main__':
    main()
