#!/usr/bin/env python3
"""ТЕПЛОЁМКОСТЬ: правильный признак перехода вместо гистограммы.

crit5.py показал, что гистограмма S при N = 6, 7 даёт 9 и 11 пиков —
это ГРЕБЁНКА допустимых значений действия (S/k4 есть целочисленная
комбинация, при малых N таких значений мало), а не сосуществование фаз.
Значит и мой тест на двугорбость негоден, И ТО ЖЕ ВОЗРАЖЕНИЕ относится
к архивному «двугорбость опровергнута при N <= 7»: оно опиралось на
такую же гистограмму. Маршрут при малых N не закрыт, он там
НЕПРОВЕРЯЕМ этим способом.

Правильный признак: рост пика ТЕПЛОЁМКОСТИ C = beta^2 * Var(S) с N.
У перехода первого рода пик растёт как объём системы, у плавного
кроссовера — медленнее или выходит на плато.

Исправления к MCMC: лестница СГУЩЕНА в области перехода (точные числа
дают переход при beta ≈ 0.05…0.10) и проходов вдесятеро больше.

КОНТРОЛЬ: при N = 6 отжиг обязан воспроизвести точные <S> и C на
КАЖДОМ beta лестницы. Без этого выводов нет.
"""
import math

import numpy as np

from crit import enumerate_exact
from crit4 import Chain, temper

BET = ([0.0, 0.01, 0.02, 0.03] + list(np.round(np.arange(0.04, 0.131,
                                                         0.01), 3))
       + [0.15, 0.18, 0.22, 0.3, 0.45])


def exact_curve(S, betas):
    out = []
    for b in betas:
        w = np.exp(-b * (S - S.min()))
        w /= w.sum()
        m = float((S * w).sum())
        v = float((S ** 2 * w).sum() - m ** 2)
        out.append((m, b * b * v))
    return out


def main():
    print(__doc__)
    rs = np.random.default_rng(13579)
    print(f'лестница из {len(BET)} точек, сгущена в 0.04…0.13')

    print('\nКОНТРОЛЬ: N = 6, отжиг против точного')
    S6 = enumerate_exact(6)
    ex = exact_curve(S6, BET)
    sm, accr, swr, bad = temper(6, BET, 2500, rs)
    print(f'  обменов {swr:.3f}, сбоев приращений {bad}, '
          f'доля принятых {min(accr):.3f}…{max(accr):.3f}')
    print(f"{'beta':>7}{'<S> точно':>11}{'<S> отжиг':>11}{'расх.':>8}"
          f"{'C точно':>10}{'C отжиг':>10}")
    worst = 0.0
    for k, b in enumerate(BET):
        m = sm[k].mean()
        # ПОЛ ОШИБКИ: при высоких beta цепь правильно сидит в
        # единственном основном состоянии, её разброс НОЛЬ, и прежняя
        # метрика делила на ноль, выдавая 10^7 сигм. Пол берётся по
        # точному разбросу ансамбля плюс шаг дискретности действия.
        exsd = math.sqrt(max(ex[k][1], 0.0)) / max(b, 1e-3)
        e = max(sm[k].std(), exsd) / math.sqrt(max(len(sm[k]), 1))
        e = max(e, 0.05)
        dv = abs(m - ex[k][0]) / e
        worst = max(worst, dv)
        c = b * b * sm[k].var()
        if k % 2 == 0:
            print(f'{b:>7.3f}{ex[k][0]:>11.3f}{m:>11.3f}{dv:>7.1f}с'
                  f'{ex[k][1]:>10.3f}{c:>10.3f}')
    print(f'  наибольшее расхождение {worst:.1f} сигм => контроль '
          f'{"ПРОЙДЕН" if worst < 6 else "ПРОВАЛЕН"}')
    if worst >= 6:
        print('  выводов по скану не делать')
        return

    print('\nПИК ТЕПЛОЁМКОСТИ по N')
    print(f"{'N':>4}{'beta пика':>11}{'C пика':>10}{'<S> в пике':>12}"
          f"{'обменов':>9}{'мин.принят.':>12}")
    rows = []
    for n in (6, 9, 12):
        sm, accr, swr, bad = temper(n, BET, 1100, rs)
        cs = [b * b * sm[k].var() for k, b in enumerate(BET)]
        k = int(np.argmax(cs))
        rows.append((n, BET[k], cs[k]))
        print(f'{n:>4}{BET[k]:>11.3f}{cs[k]:>10.3f}'
              f'{sm[k].mean():>12.2f}{swr:>9.3f}{min(accr):>12.4f}',
              flush=True)
    if len(rows) >= 3:
        ns = np.array([r[0] for r in rows], float)
        cm = np.array([r[2] for r in rows])
        p = np.polyfit(np.log(ns), np.log(np.maximum(cm, 1e-9)), 1)[0]
        print(f'\n  пик теплоёмкости ~ N^{p:.3f}')
        print('  beta пика по N: ' + ', '.join(
            f'N={int(r[0])}: {r[1]:.3f}' for r in rows))
        print('  ЧТЕНИЕ: показатель около 1 и выше отвечает переходу')
        print('  первого рода; заметно меньше — плавному кроссоверу.')
        print('  Четыре точки — не вывод, но направление.')


if __name__ == '__main__':
    main()
