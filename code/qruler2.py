#!/usr/bin/env python3
"""ЛИНЕЙКА, ВЕРСИЯ 2: без формулы, через условный разброс.

Версия 1 провалила три контроля из пяти, и все три — мои ошибки:
  * пересечение двух прошлых конусов с разнесёнными вершинами НЕ есть
    конус (это линза), поэтому (1-x/2)^4 — нижняя оценка объёма, а не
    объём. Замер шёл выше формулы, что это и подтвердило;
  * окно по возрастам [0.8,1.25] слишком широко: при разнице возрастов
    в 25 % знаменатель sqrt(V_a V_b) завышен в 1.55 раза, отсюда
    u -> 0.79 вместо 1 при нулевом разносе;
  * контроль часов сравнивал sqrt(tau_a tau_b) с оценкой по одному V_a.

Здесь формула не нужна вовсе. Меряется ПРЯМО: насколько точно порядок
определяет пространственный разнос, то есть условный разброс истинного
x = d/tau при заданных порядковых наблюдаемых. Бюджет фазы есть число
РАЗЛИЧИМЫХ значений разноса:

    бюджет = (полный размах x) / (условный разброс x)

КОНТРОЛИ:
  1. при нулевом разносе u обязано идти к 1 (теперь с узким окном
     возрастов [0.97,1.03]);
  2. ЧАСЫ на тех же данных и в той же форме: оценка tau по V_a против
     истинного tau_a — обязано дать 0.286/sqrt(V);
  3. перемешивание расстояний обязано поднять разброс до полного
     размаха (информации нет);
  4. пуассоновский пол: условный разброс не может быть лучше, чем
     даёт дробовой шум V_общ;
  5. монотонность: u обязано убывать по x во всех корзинах.

  python3 qruler2.py
"""
import math

import numpy as np

from qruler import sprinkle4, rel


def collect(n, rs, npair, agew=0.03, T=1.0):
    P = sprinkle4(n, rs, T)
    Rc, dt, dx2 = rel(P)
    past = Rc
    Vp = past.sum(axis=0)
    tau = np.sqrt(np.maximum(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1), 0.0))
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
        ra = tau[a] / max(tau[b], 1e-9)
        if abs(ra - 1.0) > agew:
            continue
        vcp = int((past[:, a] & past[:, b]).sum())
        sep = math.sqrt(max(dx2[a, b] - dt[a, b] ** 2, 0.0))
        tb = 0.5 * (tau[a] + tau[b])
        u = vcp / (0.5 * (Vp[a] + Vp[b]))
        rows.append((sep / tb, u, vcp, 0.5 * (Vp[a] + Vp[b]),
                     tau[a], tau[b]))
    return np.array(rows) if rows else None


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    rs = np.random.default_rng(31337)
    chunks = [collect(1500, rs, 60000) for _ in range(4)]
    D = np.vstack([c for c in chunks if c is not None])
    x, u, vcp, vbar = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    print(f'пар после отбора (окно возрастов ±3 %): {len(D)}')
    print()
    print('КОНТРОЛИ 1 и 5: u по корзинам x (монотонность и предел)')
    print(f"      {'x = d/tau':>12}{'u':>9}{'пар':>7}")
    prev = None
    mono = True
    for lo, hi in ((0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5),
                   (0.5, 0.8), (0.8, 1.2), (1.2, 2.0)):
        k = (x >= lo) & (x < hi)
        if k.sum() < 30:
            continue
        um = float(u[k].mean())
        print(f'      {f"{lo}-{hi}":>12}{um:>9.4f}{int(k.sum()):>7}')
        if prev is not None and um > prev:
            mono = False
        prev = um
    print(f'  монотонно: {"ВЕРНО" if mono else "НЕВЕРНО"}')
    k = x < 0.05
    print(f'  предел x->0: u = {float(u[k].mean()):.4f} '
          f'(ожидание -> 1)')
    print()
    print('КОНТРОЛЬ 2: часы в той же форме (tau по V против истинного)')
    rho = 1500 / (math.pi / 24)
    th = (24.0 * vbar / (math.pi * rho)) ** 0.25
    tt = 0.5 * (D[:, 4] + D[:, 5])
    r = tt / th
    print(f'  отн. разброс = {np.std(r, ddof=1)/np.mean(r):.5f}, '
          f'прогноз 0.286/sqrt(V) при V~{vbar.mean():.0f} = '
          f'{0.286/math.sqrt(vbar.mean()):.5f}')
    print()
    print('РАЗРЕШЕНИЕ БЕЗ ФОРМУЛЫ: условный разброс x при заданном u')
    print('(калибровка — медиана x в узкой корзине по u; разброс внутри)')
    print(f"{'V_общ':>12}{'пар':>7}{'размах x':>11}{'усл. разброс':>14}"
          f"{'пуассон. пол':>14}{'бюджет':>9}")
    rows = []
    for lo, hi in ((25, 60), (60, 120), (120, 260), (260, 600)):
        s = (vcp >= lo) & (vcp < hi)
        if s.sum() < 100:
            continue
        xs_, us_ = x[s], u[s]
        # узкие корзины по u, разброс внутри
        qs = np.quantile(us_, np.linspace(0, 1, 9))
        resid = []
        for j in range(len(qs) - 1):
            m = (us_ >= qs[j]) & (us_ <= qs[j + 1])
            if m.sum() < 12:
                continue
            resid.append(xs_[m] - np.median(xs_[m]))
        if not resid:
            continue
        rr = np.concatenate(resid)
        cond = float(np.std(rr, ddof=1))
        rng = float(np.std(xs_, ddof=1))
        pois = float(np.mean(1.0 / np.sqrt(vcp[s])))
        rows.append((float(vcp[s].mean()), cond, rng / cond, int(s.sum())))
        print(f'{f"{lo}-{hi}":>12}{int(s.sum()):>7}{rng:>11.4f}'
              f'{cond:>14.4f}{pois:>14.4f}{rng/cond:>9.2f}')
    print()
    print('КОНТРОЛЬ 3: перемешать x — бюджет обязан упасть до 1')
    perm = rs.permutation(len(x))
    xs_, us_ = x[perm], u
    qs = np.quantile(us_, np.linspace(0, 1, 9))
    resid = []
    for j in range(len(qs) - 1):
        m = (us_ >= qs[j]) & (us_ <= qs[j + 1])
        if m.sum() < 12:
            continue
        resid.append(xs_[m] - np.median(xs_[m]))
    rr = np.concatenate(resid)
    print(f'  перемешанный бюджет = '
          f'{np.std(x, ddof=1)/np.std(rr, ddof=1):.3f} (ожидание ~1)')
    print()
    if len(rows) >= 3:
        R = np.array([(r[0], r[2]) for r in rows])
        a, b = np.polyfit(np.log(R[:, 0]), np.log(R[:, 1]), 1)
        print(f'ЗАКОН БЮДЖЕТА: ~ V^{a:.3f}, коэффициент {math.exp(b):.2f}')
        print(f'  максимум замеренного бюджета: {R[:,1].max():.2f} '
              f'различимых значений разноса')
        print(f'  в радианах фазы это и есть {R[:,1].max():.2f}; '
              f'порог 2pi = 6.28 -> '
              f'{"ПРОХОДИТ" if R[:,1].max() > 2*math.pi else "НЕ ПРОХОДИТ"}')
        print()
        print('СРАВНЕНИЕ С ЧАСАМИ: у часов 4.94 sqrt(N) радиан, '
              'то есть при V ~ 300 это 86 радиан.')


if __name__ == '__main__':
    main()
