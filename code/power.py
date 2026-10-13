#!/usr/bin/env python3
"""СИЛА ЗАМЕРА СТАЦИОНАРНОСТИ: какого размера бывает производная вообще.

ЗАЧЕМ. Его вопрос: «плоское есть классическое решение, то есть нашли
таки?» Нет, и я написал слишком сильно. `dyn.py` дал
`alpha = −32.5 ± 62.4` (0.5 сигмы) по ОДНОМУ направлению — радиальному
профилю плотности. Это нуль без силы, пока неизвестно, какого размера
бывает `alpha`: граница `|alpha| < 125` может означать не
«плоское стационарно», а «мы не умеем мерить alpha».

ПРОВЕРКА СИЛЫ, и она теперь возможна точно. У нас есть направление с
ИЗВЕСТНЫМ градиентом: слоение. Измерено (`quantum-doc.md`), что
действие на сприклинге ⟨S⟩ ≈ +218, а на слоистом порядке того же N
⟨S⟩ ≈ −19600, причём на слоистых оно равно `N − (число звеньев)`
тождественно. Значит вдоль деформации «сприклинг -> слоистое»
производная обязана быть огромной.

ДЕФОРМАЦИЯ. Время сжимается к трём центрам: внутри каждой трети
отрезка события стягиваются к её середине с коэффициентом `(1−A)`.

    A = 0  — ровный сприклинг
    A -> 1 — все события в трёх мгновениях, то есть трёхслойный порядок

Это непрерывный путь в пространстве конфигураций с известными концами.

ЧТО СРАВНИВАЕМ:
    |alpha| вдоль СЛОЕНИЯ      — известно, что велико
    |alpha| вдоль РАДИАЛЬНОГО  — измерено, |alpha| < 125 при 2 сигмах

    если первое на порядки больше второго, радиальный нуль осмыслен:
    плоское стационарно вдоль радиального направления в долях
    естественного масштаба;
    если они одного порядка — замер слеп, и нуль ничего не значит.

КОНТРОЛИ:
  1. A = 0 обязано дать ровно исходное действие (та же конфигурация);
  2. при A -> 1 действие обязано подойти к слоистому значению,
     посчитанному отдельно, — это проверка самой деформации;
  3. высота при A -> 1 обязана упасть к 3;
  4. производная берётся парно по засеву, как в `dyn.py`.

  python3 power.py
"""
import argparse
import math
import time

import numpy as np

from dyn import relation, bd_density
from entropy import kr_order, height_of


def base(N, L, T, rs):
    X = np.empty((N, 4))
    X[:, 0] = rs.uniform(0, T, N)
    X[:, 1:] = rs.uniform(0, L, (N, 3))
    return X


def layerize(X, T, A, k=3):
    """Сжатие времени к k центрам: A = 0 — ровно, A -> 1 — k мгновений."""
    Y = X.copy()
    t = Y[:, 0]
    w = T / k
    which = np.clip((t / w).astype(int), 0, k - 1)
    cen = (which + 0.5) * w
    Y[:, 0] = cen + (1.0 - A) * (t - cen)
    return Y[np.argsort(Y[:, 0])]


def act(X, L):
    R = relation(X, L)
    v = float(bd_density(R).sum())
    h = height_of(R)
    del R
    return v, h


def main():
    ap = argparse.ArgumentParser(description='Сила замера стационарности')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--lell', type=float, default=5.0)
    ap.add_argument('--tell', type=float, default=6.0)
    ap.add_argument('--seeds', type=int, default=30)
    a = ap.parse_args()

    t0 = time.time()
    L, T = a.lell, a.tell
    As = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
    print('ВОПРОС: какого размера бывает alpha = dS/dA вообще?')
    print('Направление слоения имеет ИЗВЕСТНЫЙ градиент, значит по нему')
    print('можно узнать естественный масштаб и взвесить радиальный нуль.')
    print()
    print(f'N = {a.n}, ящик {L}x{T} ell, засевов {a.seeds}')
    print()

    M = np.empty((a.seeds, len(As)))
    Hh = np.empty((a.seeds, len(As)))
    for s in range(a.seeds):
        X0 = base(a.n, L, T, np.random.default_rng(3300 + 19 * s))
        for j, A in enumerate(As):
            M[s, j], Hh[s, j] = act(layerize(X0, T, A), L)
        del X0
        if s == 0 or (s + 1) % 10 == 0:
            print(f'  засев {s} ({time.time()-t0:.0f} с)', flush=True)

    print()
    print('КОНТРОЛЬ №1: при A = 0 это та же конфигурация')
    X0 = base(a.n, L, T, np.random.default_rng(3300))
    v_direct, _ = act(X0[np.argsort(X0[:, 0])], L)
    print(f'  прямо: {v_direct:+.1f}, через деформацию: {M[0,0]:+.1f}, '
          f'разница {abs(v_direct-M[0,0]):.1e}')

    print()
    print('КОНТРОЛЬ №2 и №3: подходит ли деформация к слоистому порядку')
    Skr = []
    for s in range(10):
        R, _ = kr_order(a.n, np.random.default_rng(777 + s))
        Skr.append(float(bd_density(R).sum()))
        del R
    print(f'  слоистый порядок того же N: ⟨S⟩ = {np.mean(Skr):+.1f}')
    print(f'  деформация при A = 0.99:    ⟨S⟩ = {M[:,-1].mean():+.1f}, '
          f'высота {Hh[:,-1].mean():.1f} (ожидание 3)')

    print()
    print('ДЕЙСТВИЕ ВДОЛЬ ПУТИ (парно по засеву)')
    print(f"{'A':>7}{'⟨S⟩':>12}{'ст.ош':>10}{'⟨S⟩−S(0)':>12}"
          f"{'высота':>9}")
    for j, A in enumerate(As):
        d = M[:, j] - M[:, 0]
        print(f'{A:>7.2f}{M[:,j].mean():>12.1f}'
              f'{M[:,j].std(ddof=1)/math.sqrt(a.seeds):>10.1f}'
              f'{d.mean():>+12.1f}{Hh[:,j].mean():>9.1f}')

    print()
    print('ПРОИЗВОДНАЯ В НУЛЕ вдоль слоения')
    # наклон по первым точкам, парно; ошибка складным ножом по засевам
    use = [j for j, A in enumerate(As) if A <= 0.3]
    AA = np.array([As[j] for j in use])
    def slope(Mx):
        y = (Mx[:, use] - Mx[:, 0][:, None]).mean(0)
        return np.polyfit(AA, y, 1)[0]
    al = slope(M)
    jk = []
    for s in range(a.seeds):
        keep = [i for i in range(a.seeds) if i != s]
        jk.append(slope(M[keep]))
    jk = np.array(jk)
    se = math.sqrt((a.seeds - 1) / a.seeds * ((jk - jk.mean()) ** 2).sum())
    print(f'  alpha(слоение) = {al:+.1f} +- {se:.1f} '
          f'({abs(al)/max(se,1e-12):.1f} сигм)')

    print()
    print('ВЗВЕШИВАНИЕ РАДИАЛЬНОГО НУЛЯ')
    rad_bound = 125.0      # |alpha| < 125 при 2 сигмах (dyn.py, 220 засевов)
    print(f'  |alpha| вдоль СЛОЕНИЯ:     {abs(al):.0f}')
    print(f'  |alpha| вдоль РАДИАЛЬНОГО: < {rad_bound:.0f} (2 сигмы)')
    ratio = abs(al) / rad_bound
    print(f'  отношение: {ratio:.1f}')
    print()
    if ratio > 10:
        print(f'  => замер ИМЕЕТ СИЛУ: естественный масштаб производной в')
        print(f'     {ratio:.0f} раз больше нашей границы. Значит радиальный')
        print(f'     нуль осмыслен: плоское стационарно вдоль радиального')
        print(f'     направления с точностью 1/{ratio:.0f} естественного')
        print(f'     масштаба. Но это ОДНО направление, а не все.')
    else:
        print('  => замер СЛЕП: естественный масштаб производной того же')
        print('     порядка, что наша граница, и радиальный нуль ничего')
        print('     не значит.')
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
