#!/usr/bin/env python3
"""ДВУГОРБОСТЬ: точно при N = 6, 7 и отжигом выше.

Исправление собственных ошибок по порядку.

1. crit.py замерзал при beta > 0 (все выборки одинаковы, ± 0.0000).
2. crit2.py добавил параллельный отжиг — контроль всё равно провалился
   на 45...97 сигм.
3. Я диагностировал нарушение детального баланса (добавление
   транзитивно вынужденной пары) и починил. Патч исполнялся (защита
   срабатывала 17723 раза), а числа вышли БАЙТ В БАЙТ те же — значит
   диагноз НЕВЕРЕН: старый код уже отвергал такие пары через проверку
   на удаление, и ход был обратим.
4. Настоящая причина видна в точных числах: <S> прыгает с -22.55 при
   beta=0.05 до -95.92 при beta=0.10. Переход РЕЗКИЙ, а в этом
   промежутке у лестницы была одна точка. Плюс минимум действия лежит
   на 6 сигм ниже среднего при beta=0, то есть горячая цепь попадает в
   него раз на пять тысяч состояний.

Здесь: (а) ТОЧНЫЙ разбор при N=6,7 — двугорбость проверяется прямым
взвешиванием перечисленного распределения, без MCMC вовсе; (б) отжиг с
ПЛОТНОЙ лестницей в области перехода и большим числом проходов, с тем
же контролем.

Двугорбость определяется строго: взвешенная гистограмма S имеет два
локальных максимума, разделённых провалом ниже 40% меньшего из них.
"""
import math

import numpy as np

from crit import enumerate_exact


def weighted_hist(S, beta, bins=40):
    w = np.exp(-beta * (S - S.min()))
    w = w / w.sum()
    h, ed = np.histogram(S, bins=bins, weights=w)
    return h, ed


def bimodal(h, ed):
    pk = [i for i in range(1, len(h) - 1)
          if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 1e-4]
    if len(pk) < 2:
        return False, 0.0, len(pk)
    lo = h[pk[0]:pk[-1] + 1].min()
    if lo < 0.4 * min(h[pk[0]], h[pk[-1]]):
        return True, float(ed[pk[-1]] - ed[pk[0]]), len(pk)
    return False, 0.0, len(pk)


def main():
    print(__doc__)
    for n in (6, 7):
        S = enumerate_exact(n)
        print(f'\n=== ТОЧНО, N = {n}: {len(S)} транзитивных матриц ===')
        print(f'  S от {S.min():.2f} до {S.max():.2f}, '
              f'<S> при beta=0 = {S.mean():.3f}')
        print(f"{'beta':>7}{'<S>':>11}{'теплоёмк.':>12}{'пиков':>7}"
              f"{'двугорб.':>10}{'разрыв':>9}")
        betas = np.concatenate([np.linspace(0.0, 0.2, 21),
                                np.array([0.3, 0.5, 0.8])])
        best = None
        for b in betas:
            w = np.exp(-b * (S - S.min()))
            w /= w.sum()
            m = float((S * w).sum())
            v = float((S ** 2 * w).sum() - m ** 2)
            h, ed = weighted_hist(S, b)
            bi, gap, npk = bimodal(h, ed)
            if best is None or v > best[1]:
                best = (b, v)
            if abs(b * 100 - round(b * 100)) < 1e-9 and (
                    round(b * 100) % 2 == 0 or b > 0.2):
                print(f'{b:>7.3f}{m:>11.3f}{v*b*b:>12.3f}{npk:>7}'
                      f'{("ДА" if bi else "нет"):>10}{gap:>9.2f}')
        print(f'  максимум теплоёмкости (разброс S) при beta = '
              f'{best[0]:.3f}')
        h, ed = weighted_hist(S, best[0])
        bi, gap, npk = bimodal(h, ed)
        print(f'  в этой точке: пиков {npk}, двугорбость '
              f'{"ДА" if bi else "НЕТ"}, разрыв {gap:.2f}')
        nz = h > 1e-3
        print(f'  форма (доли по 40 полосам, где > 0.001): '
              + ' '.join(f'{v:.2f}' for v in h[nz][:14]))


if __name__ == '__main__':
    main()
