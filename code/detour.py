#!/usr/bin/env python3
"""ЛИШНИЙ ШАГ, померенный буквально: цена обхода дырки.

ЕГО ФРАЗА. «Вещь — это место, где порядок вынужден сделать лишний шаг,
чтобы остаться самим собой.»

ЧТО УЖЕ ИЗМЕРЕНО (hole.py). Дырка в засеве ЗАДЕРЖИВАЕТ часы: остаток
log h после снятия log V даёт −0.070 (4.5 сигмы) при радиусе 0.16 и
−0.040 (4.3 сигмы) при 0.24, спадает с расстоянием, контроль с тем же
числом удалённых событий вчетверо слабее. Силы при этом нет (максимум
1.5 сигмы на градиенте действия).

ТЕПЕРЬ БУКВАЛЬНО. Берём пары событий по РАЗНЫЕ стороны оси: прямой путь
между ними идёт через дырку, значит максимальная цепь обязана идти в
обход. Цена обхода — укорочение цепи, то есть потеря собственного
времени. Это и есть «лишний шаг», и это ОТНОШЕНИЕ ДВУХ СЧЁТОВ, то есть
безразмерная величина, допустимая по теореме о перешкалировании.

КОНТРОЛЬ обязателен и двойной:
  1. удалено столько же событий СЛУЧАЙНО по всему ромбу — отделяет
     форму от простого уменьшения числа событий;
  2. пары по ОДНУ сторону оси (прямой путь дырку не задевает) — там
     укорочения быть не должно.

ПРЕДСКАЗАНИЕ. Укорочение растёт с радиусом дырки, у поперечных пар есть,
у продольных нет, и у контроля много меньше.

  python3 detour.py
  python3 detour.py --rads 0.16 0.24 0.32 --seeds 10
"""
import argparse
import math
import time

import numpy as np

from weight import diamond, cone_rel


def longest_between(R, a, b):
    """Длина максимальной цепи от a до b (в звеньях), -1 если нет пути."""
    if not R[b, a]:
        return -1
    mid = np.flatnonzero(R[:, a] & R[b, :])
    nodes = np.concatenate(([a], mid, [b]))
    sub = R[np.ix_(nodes, nodes)]
    n = len(nodes)
    deg = sub.sum(1)
    order = np.argsort(deg, kind='stable')
    L = np.full(n, -1, np.int64)
    L[0] = 0
    pos = {int(v): i for i, v in enumerate(nodes)}
    for i in order:
        i = int(i)
        pre = np.flatnonzero(sub[i])
        if len(pre):
            m = L[pre].max()
            if m >= 0:
                L[i] = max(L[i], m + 1)
    return int(L[pos[b]])


def pick_pairs(X, rad, T, rs, across, npair=60):
    """Пары (a,b), a раньше b, по разные (across) или одну сторону оси."""
    r = np.linalg.norm(X[:, 1:], axis=1)
    lo = (X[:, 0] < 0.42 * T) & (r > rad) & (r < rad + 0.22)
    hi = (X[:, 0] > 0.58 * T) & (r > rad) & (r < rad + 0.22)
    A, B = np.flatnonzero(lo), np.flatnonzero(hi)
    if len(A) < 3 or len(B) < 3:
        return []
    out = []
    for _ in range(npair * 14):
        if len(out) >= npair:
            break
        a = int(rs.choice(A)); b = int(rs.choice(B))
        ca = X[a, 1:] / max(np.linalg.norm(X[a, 1:]), 1e-12)
        cb = X[b, 1:] / max(np.linalg.norm(X[b, 1:]), 1e-12)
        cos = float(ca @ cb)
        if across and cos > -0.75:
            continue
        if (not across) and cos < 0.75:
            continue
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description='Цена обхода дырки')
    ap.add_argument('--t', type=float, default=2.0)
    ap.add_argument('--rads', type=float, nargs='+',
                    default=[0.16, 0.24, 0.32])
    ap.add_argument('--seeds', type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    print(f'ромб T={a.t}; максимальная цепь между парами событий')
    print('ПОПЕРЁК = по разные стороны оси (прямой путь через дырку)')
    print('ВДОЛЬ  = по одну сторону (прямой путь дырку не задевает)')
    print('контроль: удалено столько же событий случайно по всему ромбу')
    print('укорочение цепи = ЦЕНА ОБХОДА = лишний шаг')
    print(f'засевов {a.seeds}')
    print()
    print(f"{'радиус':>8}{'удалено':>9}" + ''.join(
        f'{s:>13}' for s in ['поперёк дырка', 'поперёк контр',
                             'разница', 'сигм',
                             'вдоль дырка', 'вдоль контр', 'разница',
                             'сигм']))

    for rad in a.rads:
        got = {('across', True): [], ('across', False): [],
               ('along', True): [], ('along', False): []}
        nrem = []
        for t in range(a.seeds):
            rs = np.random.default_rng(2301000 + 977 * t)
            X0 = diamond(a.t, rs)
            r0 = np.linalg.norm(X0[:, 1:], axis=1)
            keep = r0 >= rad
            k = int((~keep).sum())
            nrem.append(k)
            for mk in (True, False):
                if mk:
                    X = X0[keep]
                else:
                    idx = rs.choice(len(X0), len(X0) - k, replace=False)
                    X = X0[np.sort(idx)]
                R = cone_rel(X)
                for nm, acr in (('across', True), ('along', False)):
                    ps = pick_pairs(X, rad, a.t, rs, acr)
                    vals = [longest_between(R, p, q) for p, q in ps]
                    vals = [v for v in vals if v > 0]
                    if vals:
                        got[(nm, mk)].append(float(np.mean(vals)))
        row = [f'{rad:>8.2f}{np.mean(nrem):>9.0f}']
        for nm in ('across', 'along'):
            h = np.array(got[(nm, True)]); c = np.array(got[(nm, False)])
            if len(h) < 2 or len(c) < 2:
                row.append(f'{"нет":>13}' * 4)
                continue
            d = h.mean() - c.mean()
            er = math.hypot(h.std() / math.sqrt(len(h)),
                            c.std() / math.sqrt(len(c)))
            row.append(f'{h.mean():>13.2f}{c.mean():>13.2f}'
                       f'{d:>+13.2f}{abs(d)/max(er,1e-12):>13.1f}')
        print(''.join(row), flush=True)

    print('\n  отрицательная разница ПОПЕРЁК при нулевой ВДОЛЬ => '
          'цена именно обхода')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
