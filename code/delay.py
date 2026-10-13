#!/usr/bin/env python3
"""МАССА ЕСТЬ ЗАДЕРЖКА: замедление хода часов, померенное одним порядком.

ИДЕЯ (его). «То, что растёт медленнее, то массивнее. Вещь — это не то,
что добавлено, а место, где причинный порядок вынужден сделать лишний
шаг, чтобы остаться самим собой.»

ПОЧЕМУ ЭТО НЕ ПОПАДАЕТ ПОД ОЦЕНКУ preclusion-doc.md. Та оценка
ограничивает ВЕС — функцию расстановки через один скаляр. Задержка не
вес, а ТЕМП: отношение двух счётов. И ровно поэтому она допустима по
теореме о перешкалировании, которая запрещает размерные величины, но
не отношения.

ЧТО МЕРИМ. Два счёта на каждом событии, оба порядковые:

    h = длина длиннейшей цепи, кончающейся в событии   (своё время)
    V = |прошлое события|                              (объём)

В засеве размерности 4 цепь растёт как h ~ m4 * V^(1/4), значит
h / V^(1/4) есть ХОД ЧАСОВ. Замедление у массы означает: при ОДИНАКОВОМ
объёме прошлого цепь короче.

Сравнение на фиксированном V обязательно: события у трубки имеют её
события в прошлом, отчего V механически больше. Это бухгалтерия, а не
физика, и снимается только привязкой к V.

Расстояние до объекта тоже порядковое — окно несравнимости из
partial-doc.md:

    W = число событий объекта, несравнимых с событием,  W = 2*lam*r

Итого ВСЯ величина, ВСЁ расстояние и ВЕСЬ контроль — из одного порядка,
ни одной координаты. Первое такое измерение силы в проекте.

ПРЕДСКАЗАНИЕ. При фиксированном V цепь короче у трубки (малое W) и
такая же вдали; у рассыпанного контроля разницы по W нет.

  python3 delay.py
  python3 delay.py --masses 120 240 480 --seeds 8
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel


def chain_tube(M, rad, T, rs):
    u = rs.normal(size=(M, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    r = rad * rs.random(M) ** (1.0 / 3.0)
    X = np.empty((M, 4))
    X[:, 0] = np.sort(rs.uniform(rad, T - rad, M))
    X[:, 1:] = u * r[:, None]
    return X


def heights(R):
    """h[i] = длиннейшая цепь, кончающаяся в i. R[i,j] = j раньше i."""
    N = len(R)
    deg = R.sum(1)
    order = np.argsort(deg, kind='stable')
    h = np.zeros(N, np.int32)
    for i in order:
        i = int(i)
        pre = np.flatnonzero(R[i])
        if len(pre):
            h[i] = h[pre].max() + 1
    return h


def window(R, M):
    rel = R[M:, :M] | R[:M, M:].T
    return (~rel).sum(1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description='Масса как задержка')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--masses', type=int, nargs='+', default=[120, 240, 480])
    ap.add_argument('--rad', type=float, default=0.02)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    wed = np.array([0.0, 12.0, 24.0, 40.0, 64.0, 1e9])
    lbl = ['W<12', 'W<24', 'W<40', 'W<64', 'W дальше']
    print(f'ромб T={a.t}, трубка — цепь на оси радиуса {a.rad}, '
          f'засевов {a.seeds}')
    print('h = длиннейшая цепь до события, V = |прошлое|; '
          'ход часов = h / V^(1/4)')
    print('сравнение НА ФИКСИРОВАННОМ V: иначе события трубки в прошлом '
          'раздувают V сами')
    print('W — окно несравнимости с объектом: малое W = близко')
    print('ОТРИЦАТЕЛЬНЫЙ остаток h при малом W = ЗАДЕРЖКА = масса')
    print('ни одной координаты в измеряемых величинах')

    for M in a.masses:
        lam = M / (a.t - 2 * a.rad)
        print(f'\nМАССА M = {M}  (lam = {lam:.1f}, '
              f'ожидаемое W = {2*lam:.0f}*r)')
        print(f"{'мир':>12}" + ''.join(f'{s:>12}' for s in lbl))
        res = {}
        for mk in (True, False):
            acc = [[] for _ in range(len(wed) - 1)]
            for t in range(a.seeds):
                rs = np.random.default_rng(2001000 + 977 * t)
                bg = diamond(a.t, rs)
                ob = (chain_tube(M, a.rad, a.t, rs) if mk
                      else diamond(a.t, rs, n_at_t1=M / a.t ** 4))
                X = np.vstack([ob, bg])
                R = cone_rel(X)
                h = heights(R).astype(np.float64)
                V = R.sum(1).astype(np.float64)
                W = window(R, len(ob))
                hb, Vb = h[len(ob):], V[len(ob):]
                # берём середину мира и непустое прошлое
                keep = (Vb >= 40) & (Vb <= 600)
                if keep.sum() < 50:
                    continue
                # ОСТАТОК h после снятия зависимости от V:
                # регрессия log h на log V, затем остаток
                lv, lh = np.log(Vb[keep]), np.log(hb[keep])
                p = np.polyfit(lv, lh, 1)
                resid = lh - np.polyval(p, lv)
                ww = W[keep]
                for b in range(len(acc)):
                    m = (ww >= wed[b]) & (ww < wed[b + 1])
                    if m.sum() > 5:
                        acc[b].append(float(resid[m].mean()))
            m = np.array([np.mean(v) if len(v) > 1 else np.nan for v in acc])
            e = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1
                          else np.nan for v in acc])
            res[mk] = (m, e)
            name = 'трубка' if mk else 'контроль'
            print(f'{name:>12}' + ''.join(f'{x:>+12.5f}' for x in m))
            print(f"{'±':>12}" + ''.join(f'{x:>12.5f}' for x in e))
        d = res[True][0] - res[False][0]
        er = np.hypot(res[True][1], res[False][1])
        print(f"{'задержка':>12}" + ''.join(f'{x:>+12.5f}' for x in d))
        print(f"{'значим.':>12}"
              + ''.join(f'{(abs(x)/y if y > 1e-12 else float("nan")):>12.1f}'
                        for x, y in zip(d, er)), flush=True)
    print(f'\n  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
