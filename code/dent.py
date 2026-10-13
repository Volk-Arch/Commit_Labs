#!/usr/bin/env python3
"""Вмятина: локальный объект, его масса и его движение.

ПОВОД. weightmass.py: плотная мировая трубка оставляет во взвешенной
сумме ЛОКАЛЬНУЮ вмятину, и её длина не зависит от размера мира
(при L=1.4 и 1.8 первая корзина дала -0.5756 и -0.5759).

Цену изгиба здесь не мерим: она требует веса на РАССТАНОВКАХ exp(-S),
которого в конструкции нет (no-dynamics-doc.md). Без него изгибу неоткуда
стоить, и погоня за ним статистикой — тупик.

Зато вмятина позволяет проверить два свойства, которых у проекта не было.

ОПЫТ A: ЧИТАЕТСЯ ЛИ МАССА. Глубина вмятины против числа событий в
трубке при неизменном радиусе. masstest-doc.md: скалярной массы нет,
наблюдаемые спорят, что ею считать. Если глубина монотонна по M, то
появляется ОДНА наблюдаемая, присваивающая локальному объекту число.

ОПЫТ B: СЛЕДУЕТ ЛИ ВМЯТИНА ЗА ОБЪЕКТОМ. Трубка движется со скоростью v.
Значения взвешенной суммы порядковые, значит от буста не зависят;
поэтому профиль мерится в системе покоя трубки (бустим координаты, не
величины). Если профиль движущейся трубки совпадает с покоящейся, вмятина
к объекту привязана, а не к системе отсчёта.

  python3 dent.py
  python3 dent.py --seeds 20
"""
import argparse
import math
import time

import numpy as np

from weightmass import box_sprinkle, tube, per_ev, T_MAX, RHO

NEAR, FAR = 0.45, (1.0, 1.8)


def n_for(L):
    return int(round(RHO * T_MAX * (2 * L) ** 3))


def moving_tube(M, rad, v, rs):
    X = tube(M, rad, rs)
    X[:, 1] += v * (X[:, 0] - T_MAX / 2)
    return X


def boost(X, v):
    if v == 0.0:
        return X
    g = 1.0 / math.sqrt(1.0 - v * v)
    Y = X.copy()
    t, x = X[:, 0] - T_MAX / 2, X[:, 1]
    Y[:, 0] = g * (t - v * x) + T_MAX / 2
    Y[:, 1] = g * (x - v * t)
    return Y


def depth(X, s, n_obj, v, L):
    """Глубина вмятины: около оси минус далёкое поле, в системе трубки."""
    Y = boost(X, v)[n_obj:]
    sv = s[n_obj:]
    keep = (np.abs(Y[:, 0] - T_MAX / 2) < 1.2) & (np.abs(Y[:, 1]) < L)
    r = np.linalg.norm(Y[:, 1:], axis=1)
    near = keep & (r < NEAR)
    far = keep & (r >= FAR[0]) & (r < FAR[1])
    if near.sum() < 5 or far.sum() < 5:
        return float('nan')
    return float(sv[near].mean() - sv[far].mean())


def run(M, v, L, rad, eps, seeds, localized):
    N, out = n_for(L), []
    for t in range(seeds):
        rs = np.random.default_rng(500 + 17 * t)
        bg = box_sprinkle(N, L, rs)
        ob = (moving_tube(M, rad, v, rs) if localized
              else box_sprinkle(M, L, rs))
        X = np.vstack([ob, bg])
        out.append(depth(X, per_ev(X, eps), M, v, L))
        del X
    a = np.array(out, float)
    a = a[~np.isnan(a)]
    return float(a.mean()), float(a.std() / math.sqrt(max(len(a), 1)))


def main():
    ap = argparse.ArgumentParser(description='Вмятина: масса и движение')
    ap.add_argument('--rad', type=float, default=0.05)
    ap.add_argument('--eps', type=float, default=0.2)
    ap.add_argument('--seeds', type=int, default=12)
    a = ap.parse_args()

    t0 = time.time()
    print(f'плотность {RHO}, радиус трубки {a.rad}, eps={a.eps}, '
          f'засевов {a.seeds}')
    print(f'глубина = <взвеш.сумма при r<{NEAR}> - <при {FAR[0]}<=r<{FAR[1]}>')
    print()

    L = 1.4
    print(f'ОПЫТ A: ЧИТАЕТСЯ ЛИ МАССА (L={L}, N={n_for(L)}, трубка покоится)')
    print(f"{'M':>6}{'трубка':>22}{'контроль':>22}{'разница':>22}{'знач.':>8}")
    ms, ds, es = [], [], []
    for M in (50, 100, 200, 400, 800):
        mt, et = run(M, 0.0, L, a.rad, a.eps, a.seeds, True)
        mc, ec = run(M, 0.0, L, a.rad, a.eps, a.seeds, False)
        d, e = mt - mc, math.hypot(et, ec)
        ms.append(M); ds.append(d); es.append(e)
        print(f'{M:>6}{mt:>14.4f} ±{et:<7.4f}{mc:>14.4f} ±{ec:<7.4f}'
              f'{d:>14.4f} ±{e:<7.4f}{abs(d)/max(e,1e-12):>8.1f}')
    p = np.polyfit(np.log(ms), np.log(np.abs(ds)), 1)[0]
    print(f'  показатель глубины по M: {p:.3f}   (1.0 = пропорционально массе)')

    L = 1.6
    print()
    print(f'ОПЫТ B: СЛЕДУЕТ ЛИ ВМЯТИНА ЗА ОБЪЕКТОМ '
          f'(L={L}, N={n_for(L)}, M=400)')
    print('профиль мерится в системе покоя трубки')
    print(f"{'v':>6}{'трубка':>22}{'контроль':>22}{'разница':>22}{'знач.':>8}")
    for v in (0.0, 0.3, 0.5):
        mt, et = run(400, v, L, a.rad, a.eps, a.seeds, True)
        mc, ec = run(400, 0.0, L, a.rad, a.eps, a.seeds, False)
        d, e = mt - mc, math.hypot(et, ec)
        print(f'{v:>6.1f}{mt:>14.4f} ±{et:<7.4f}{mc:>14.4f} ±{ec:<7.4f}'
              f'{d:>14.4f} ±{e:<7.4f}{abs(d)/max(e,1e-12):>8.1f}')
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
