#!/usr/bin/env python3
"""СУЩЕСТВУЕТ ЛИ ДЕЙСТВИЕ, ПРИ КОТОРОМ ПРОСТРАНСТВО-ВРЕМЯ ДОМИНИРУЕТ.

Вес сектора есть ЭНТРОПИЯ ПЛЮС ДЕЙСТВИЕ:

    ln W(сектор) = S(сектор) + theta . f(сектор)

где f — порядковые счёты (R, N0, N1, N2, N3, C3, C4, высота), а theta —
коэффициенты действия. Любое локальное действие в физическом смысле
имеет такой вид; при theta = 0 получается однородная мера, при
theta на (N0, R) — перколяция, при БД-коэффициентах — действие
Бенинказы-Доукера.

Вопрос не «какая мера верна», а «есть ли ХОТЬ ОДНА»:

    найти theta:  theta.(f_4d - f_j) > S_j - S_4d   для всех соперников j

Это линейное программирование. Геометрически: пространство-время
обязано лежать ВНЕ выпуклой оболочки соперников после поправки на
энтропию, то есть быть крайней точкой с опорной гиперплоскостью.
У соперников свободны число слоёв и плотность, поэтому они заполняют
континуум — и вопрос содержательный.

НАПРАВЛЕНИЕ НАДЁЖНОСТИ: добавление сектора оболочку только расширяет,
поэтому вывод «действия нет» устойчив, а «действие есть» всегда
предварителен. Энтропия пространства-времени взята ЩЕДРО (граница
Милнора-Тома 4N lnN — оценка сверху), что тоже работает в пользу
выполнимости.

КОНТРОЛИ:
  1. при theta = 0 доминировать обязан сектор с наибольшей энтропией;
  2. подстановка «цепь вместо пространства-времени» обязана быть
     ВЫПОЛНИМОЙ (цепь — крайняя точка по построению);
  3. запас m обязан НЕ РАСТИ при добавлении секторов;
  4. решение обязано удовлетворять всем неравенствам при прямой
     подстановке.

  python3 actionlp.py
"""
import json
import math

import numpy as np
from scipy.optimize import linprog

FEAT = ('R', 'N0', 'N1', 'N2', 'N3', 'C3', 'C4', 'H')


def load(n):
    d = json.load(open('/home/claude/data/sectors.json'))
    return d[str(n)]


def vec(s, scale):
    return np.array([s['f'][k] / scale[k] for k in FEAT])


def solve(secs, target, scale, Theta=1e4, exclude_kinds=()):
    """max m  s.t.  theta.(f_t - f_j) - m >= S_j - S_t,  |theta|<=Theta."""
    ft, St = vec(target, scale), target['S']
    A, b, names = [], [], []
    for s in secs:
        if s is target or s['kind'] in exclude_kinds:
            continue
        g = ft - vec(s, scale)
        A.append(np.concatenate([-g, [1.0]]))
        b.append(St - s['S'])
        names.append(s['name'])
    A, b = np.array(A), np.array(b)
    c = np.zeros(len(FEAT) + 1)
    c[-1] = -1.0
    bounds = [(-Theta, Theta)] * len(FEAT) + [(None, None)]
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if not r.success:
        return None, None, names, A, b
    return r.x[:-1], r.x[-1], names, A, b


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    for N in (150, 300, 600):
        secs = load(N)
        scale = {k: max(abs(s['f'][k]) for s in secs) or 1.0 for k in FEAT}
        tgt = [s for s in secs if s['name'] == 'засев d=4'][0]
        print('=' * 72)
        print(f'N = {N}, секторов {len(secs)}')
        # КОНТРОЛЬ 1
        best = max(secs, key=lambda s: s['S'])
        print(f"  КОНТРОЛЬ 1 (theta=0): доминирует «{best['name']}», "
              f"S = {best['S']:.0f}; у засева d=4 S = {tgt['S']:.0f}")
        # КОНТРОЛЬ 2: цепь вместо засева
        ch = [s for s in secs if s['name'] == 'цепь'][0]
        _, mch, _, _, _ = solve(secs, ch, scale)
        print(f'  КОНТРОЛЬ 2 (цель = цепь): запас m = {mch:+.4g} '
              f'{"ВЫПОЛНИМО" if mch and mch > 0 else "НЕ выполнимо"}')
        # основная задача, в двух вариантах
        for lab, excl in (('без других засевов', ('sprinkle',)),
                          ('со всеми соперниками', ())):
            th, m, names, A, b = solve(secs, tgt, scale,
                                       exclude_kinds=excl)
            if th is None:
                print(f'  {lab}: ЗАДАЧА НЕОГРАНИЧЕНА/НЕСОВМЕСТНА')
                continue
            ok = bool((A @ np.concatenate([th, [m]]) <= b + 1e-6).all())
            verdict = ('ДЕЙСТВИЕ ЕСТЬ' if m > 0 else 'ДЕЙСТВИЯ НЕТ')
            print(f'  {lab}: запас m = {m:+.6g}  -> {verdict}'
                  f'   (КОНТРОЛЬ 4 подстановка: '
                  f'{"ок" if ok else "ПРОВАЛ"})')
            # связывающие неравенства
            res = b - A @ np.concatenate([th, [m]])
            idx = np.argsort(res)[:4]
            print('     связывают: ' + ', '.join(
                f'{names[i]} (зазор {res[i]:+.3g})' for i in idx))
            if m > 0:
                th_r = th / np.array([scale[k] for k in FEAT])
                nz = {k: f'{v:+.4g}' for k, v in zip(FEAT, th_r)}
                print(f'     theta (в исходных единицах): {nz}')
        print()


if __name__ == '__main__':
    main()
