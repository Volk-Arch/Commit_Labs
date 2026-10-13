#!/usr/bin/env python3
"""ЕСЛИ ДЕФИЦИТ - ЭТО КРИВИЗНА, ОН ОБЯЗАН ВЕСТИ СЕБЯ КАК КРИВИЗНА.

Вторая половина догадки: провалы в спектре интервалов - по форме ровно
то, из чего в программе Соркина читается кривизна, ведь действие БД есть
взвешенная недостача:
    S = k4 (N - N_0 + 9 N_1 - 16 N_2 + 8 N_3)
и у плоского засева она гаснет (точное тождество
sum_m w_m Gamma(m+1/2)/m! = 0, `nm-doc.md`), а у искривлённого даёт R·V.

Тогда проверка не по аналогии, а по поведению. Считаем удельную
величину s = (N - N_0 + 9 N_1 - 16 N_2 + 8 N_3)/N и смотрим:
  * у ЗАСЕВА s обязано быть около нуля в среднем (плоское);
  * у ФРАКТАЛА, если его дефицит есть кривизна, s должно быть
    СРАВНИМО с разбросом засева и МЕНЯТЬСЯ С МАСШТАБОМ связно -
    кривизна есть малая поправка в разложении по производным;
  * если же s у фрактала на порядки больше разброса засева при ВСЕХ
    масштабах - это не кривизна, а другой класс геометрии: кривизна,
    огромная на каждом масштабе, континуального предела не имеет.

КОНТРОЛЬ: у засева s в среднем ноль в пределах своей ошибки. Это
независимая проверка тождества весов - если не ноль, сломан счётчик.
"""
import math

import numpy as np

from sectors import sprinkle_d
from ratios import nm
from fracscale import thin
from fracbig import build
from compose2 import posets

W = np.array([-1.0, 9.0, -16.0, 8.0])
B = 6


def dens(Rc):
    v, _ = nm(Rc)
    n = Rc.shape[0]
    return float(n + (W * v[:4]).sum()) / n


def main():
    print(__doc__)
    rs = np.random.default_rng(99887766)

    print('КОНТРОЛЬ: удельное s у засева d=4 (ждём ноль)')
    for n in (300, 600, 1200):
        vs = [dens(sprinkle_d(n, 4, rs)) for _ in range(25)]
        m, e = float(np.mean(vs)), float(np.std(vs) / math.sqrt(len(vs)))
        print(f'  N={n:>5}: s = {m:+9.2f} ± {e:>7.2f}   '
              f'разброс по конфигурациям {np.std(vs):>8.2f}  '
              f'=> {(m / e):+.1f}σ от нуля')
        if n == 1200:
            spread = float(np.std(vs))

    seeds = [(int(RB.sum()), RB.copy()) for RB in posets(B)
             if RB.shape[0] == B and 0 < int(RB.sum()) < 15]
    seeds.sort(key=lambda t: t[0])
    pick = [RB for nr, RB in seeds if nr in (2, 3)][:2]
    print(f'\nфрактал N={B ** 5}, удельное s при разных масштабах')
    print(f"{'p':>6}{'N′':>7}" + ''.join(f'{f"фрактал {i + 1}":>16}'
                                         for i in range(len(pick)))
          + f"{'засев (контроль)':>18}")
    big = [build(RB, 5) for RB in pick]
    for p in (1.0, 0.3, 0.12, 0.05):
        cells = []
        for R in big:
            T = thin(R, p, rs) if p < 1.0 else R
            if T.shape[0] > 2000:
                T = thin(R, 2000 / R.shape[0], rs)
            cells.append(dens(T))
        S = sprinkle_d(1200, 4, rs)
        cs = dens(thin(S, p, rs) if p < 1.0 else S)
        n2 = int(B ** 5 * p)
        print(f'{p:>6.2f}{n2:>7}' + ''.join(f'{v:>16.1f}' for v in cells)
              + f'{cs:>18.1f}', flush=True)

    print(f'\nразброс s у засева по конфигурациям при N=1200: '
          f'{spread:.1f}')
    print('ЧТЕНИЕ: если s фрактала на порядки превышает этот разброс при')
    print('ВСЕХ масштабах — дефицит не кривизна, а другой класс геометрии.')


if __name__ == '__main__':
    main()
