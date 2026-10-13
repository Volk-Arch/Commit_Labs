#!/usr/bin/env python3
"""ДОСТИЖИМО ЛИ d = 3 в семействе роста: сравнение ЗНАЧЕНИЙ.

`csg-doc.md` вывел замок: показатель L = 1 + beta, показатель H = beta,
значит L - H = 1 тождественно при любой последовательности t_k. А у
d-мерного засева L - H = 2 - 3/d, что равно 1 РОВНО при d = 3. И оба
условия (на связи и на ширину) дают там одно и то же a = 2.

Но прямая проверка в csg4.py провалилась (связи 1.2752 против 1.5215,
10.6 сигмы) — и диагноз был: показатель связей у ЗАСЕВА не сходится до
N ~ 10^7 (`nm-doc.md`), то есть сравнивалось сошедшееся с несошедшимся.

Здесь метод, который только что сработал на критичности: сравнивать
ЗНАЧЕНИЯ при одном и том же N, а не показатели. Это снимает вопрос
сходимости целиком.

ПОСТАНОВКА РАВНОПРАВНАЯ. При каждом N постоянная x подбирается так,
чтобы ДОЛЯ СВЯЗАННЫХ роста совпала с долей связанных засева d=3 при том
же N. Тогда одна наблюдаемая согласована по построению, и вопрос
честный: совпадут ли ОСТАЛЬНЫЕ ТРИ (высота, ширина, связи). Так
проверяют равносильность двух ансамблей.

ПРЕДРЕГИСТРАЦИЯ: если a=2 есть то же, что d=3, совпасть должны все
четыре при каждом N, а не только в асимптотике. Для сравнения рядом
стоят a=1 и a=3 — они совпасть не должны.

КОНТРОЛЬ: засевы мерятся тем же кодом в том же прогоне.
"""
import math

import numpy as np

from csg2 import grow_rc, observables
from sectors import sprinkle_d


def r_of(n, a, lx, rs, reps=2):
    return float(np.mean([observables(grow_rc(n, a, lx, rs))['r']
                          for _ in range(reps)]))


def tune_r(n, a, target, rs, iters=12):
    lo, hi = -30.0, 12.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if r_of(n, a, mid, rs, reps=1) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    print(__doc__)
    rs = np.random.default_rng(303303)
    for n in (150, 350, 700):
        ref = [observables(sprinkle_d(n, 3, rs)) for _ in range(12)]
        tgt = float(np.mean([v['r'] for v in ref]))
        rH = float(np.mean([v['H'] for v in ref]))
        rW = float(np.mean([v['W'] for v in ref]))
        rL = float(np.mean([v['L'] for v in ref]))
        sH = float(np.std([v['H'] for v in ref])) / math.sqrt(12)
        sW = float(np.std([v['W'] for v in ref])) / math.sqrt(12)
        sL = float(np.std([v['L'] for v in ref])) / math.sqrt(12)
        print(f'\n=== N = {n} ===')
        print(f'{"засев d=3":>14}{tgt:>12.4f}{rH:>10.2f}{rW:>10.2f}'
              f'{rL:>11.1f}   (эталон, тот же код)')
        print(f'{"":>14}{"доля связ.":>12}{"высота":>10}{"ширина":>10}'
              f'{"связи":>11}')
        for a in (1, 2, 3):
            lx = tune_r(n, a, tgt, rs)
            vals = [observables(grow_rc(n, a, lx, rs)) for _ in range(6)]
            gr = float(np.mean([v['r'] for v in vals]))
            gH = float(np.mean([v['H'] for v in vals]))
            gW = float(np.mean([v['W'] for v in vals]))
            gL = float(np.mean([v['L'] for v in vals]))
            dH = abs(gH - rH) / max(sH, 0.05)
            dW = abs(gW - rW) / max(sW, 0.05)
            dL = abs(gL - rL) / max(sL, 0.5)
            mark = '  <-- ждём совпадения' if a == 2 else ''
            print(f'{f"рост a={a}":>14}{gr:>12.4f}{gH:>10.2f}'
                  f'{gW:>10.2f}{gL:>11.1f}{mark}')
            print(f'{"":>14}{"":>12}{dH:>9.1f}с{dW:>9.1f}с{dL:>10.1f}с',
                  flush=True)
    print('\nЧТЕНИЕ: если при a=2 все три расхождения малы при КАЖДОМ N,')
    print('семейство роста действительно рождает трёхмерные множества, и')
    print('это первый положительный результат о законной динамике.')
    print('Если нет — замок L-H=1 выполняется, а равносильности с d=3')
    print('всё равно нет, и семейство не даёт многообразий вообще.')


if __name__ == '__main__':
    main()
