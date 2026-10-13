#!/usr/bin/env python3
"""ЕСТЬ ЛИ hbar, ПРИ КОТОРОМ РАЗРЕЖЕННЫЙ СЕКТОР ГИБНЕТ. Финал пункта 4.

Установлено точно (`sparsescale.py`): подавление сектора идёт как
`q^{a_3}`, где `q = max_{M12} |G|/2^{a_2}` падает экспоненциально с
`a_2` — наклон `ln(max q)` по `a_2` равен −0.099 при `hbar = 0.55` и
−0.120 при 0.62. Значит суммарное подавление `e^{-s a_2 a_3}`, то есть
КВАДРАТИЧНО по `N`: форма правильная.

Но решает КОЭФФИЦИЕНТ. Число конфигураций сектора растёт как
`2^{a_2 a_3} = e^{0.693 a_2 a_3}`, поэтому сектор гибнет только если

    s = -ln(max q)/a_2  >=  ln 2 = 0.693

Лучшее наблюдённое `s` = 0.114, то есть в шесть раз мало. ЗДЕСЬ `hbar`
сканируется свободно: прежнее окно [0.390, 0.780] выводилось из
ПЛОТНОГО сектора через `|2cos(k_d/2 hbar)|`, и у разреженного оптимум
может стоять в другом месте.

Если максимум `s` по всем `hbar` окажется меньше `ln 2` — вечерняя
архитектура 8 октября опровергнута ЧИСЛОМ: регулярные сектора
действительно гасятся, но недостаточно, чтобы уступить
пространству-времени.

КОНТРОЛЬ: резонансы. При `k_4/hbar`, кратном `2pi/m` для малого `m`,
целочисленная фаза принимает всего `m` значений и часть конфигураций
выравнивается — `s` там проваливается к нулю не по физике, а по
арифметике. Такие `hbar` помечаются.

  python3 hbarscan.py
"""
import math

import numpy as np

K4 = 4.0 / math.sqrt(6.0)
W = {0: -1.0, 1: 9.0, 2: -16.0, 3: 8.0}
WV = np.vectorize(lambda m: W.get(int(m), 0.0))


def maxq(a1, a2, hbar):
    sub = np.arange(1 << a2)
    pc = np.array([bin(int(t)).count('1') for t in sub], dtype=float)
    best = 0.0
    for m12 in range(1 << (a1 * a2)):
        rows = [(m12 >> (i * a2)) & ((1 << a2) - 1) for i in range(a1)]
        ph = -pc.copy()
        for s in rows:
            ph += WV(np.array([bin(s & int(t)).count('1') for t in sub]))
        g = abs(np.exp(1j * K4 * ph / hbar).sum()) / (1 << a2)
        if g > best:
            best = g
    return best


def resonant(hbar, mmax=8, tol=0.02):
    x = K4 / hbar / (2 * math.pi)
    for m in range(1, mmax + 1):
        for j in range(1, m):
            if math.gcd(j, m) == 1 and abs(x % 1.0 - j / m) < tol:
                return m
    return 0


def main():
    print(__doc__.split('  python3')[0].rstrip())
    print()
    print('=' * 76)
    print('СКАН ПО hbar: s = -ln(max q)/a_2 при a_1 = 2, a_2 = 6')
    print(f"{'hbar':>7}{'max q':>11}{'s':>10}{'s/ln2':>9}"
          f"{'резонанс':>11}{'итог':>24}")
    best = None
    rows = []
    for hb in np.arange(0.20, 2.01, 0.06):
        q = maxq(2, 6, float(hb))
        s = -math.log(max(q, 1e-300)) / 6.0
        rz = resonant(float(hb))
        rows.append((float(hb), q, s, rz))
        if rz == 0 and (best is None or s > best[1]):
            best = (float(hb), s)
        print(f'{hb:>7.2f}{q:>11.6f}{s:>10.4f}{s/math.log(2):>9.3f}'
              f'{(str(rz) if rz else "—"):>11}'
              f'{("ГИБНЕТ" if s >= math.log(2) else "выживает"):>24}',
              flush=True)
    print()
    print('=' * 76)
    print(f'лучшее нерезонансное s = {best[1]:.4f} при hbar = {best[0]:.2f}')
    print(f'нужно для гибели: ln 2 = {math.log(2):.4f}')
    print(f'не хватает множителя {math.log(2)/best[1]:.1f}')
    print()
    print('РОСТ s С РАЗМЕРОМ: может ли он дойти до ln 2 при больших a_2')
    hb = best[0]
    print(f"{'a_2':>5}{'max q':>11}{'s':>10}{'s/ln2':>9}")
    sv = []
    for a2 in (3, 4, 5, 6, 7, 8):
        q = maxq(2, a2, hb)
        s = -math.log(max(q, 1e-300)) / a2
        sv.append((a2, s))
        print(f'{a2:>5}{q:>11.6f}{s:>10.4f}{s/math.log(2):>9.3f}',
              flush=True)
    S = np.array(sv)
    sl = np.polyfit(S[:, 0], S[:, 1], 1)
    need = (math.log(2) - sl[1]) / sl[0] if sl[0] > 1e-9 else float('inf')
    print(f'  s растёт по a_2 с наклоном {sl[0]:+.5f}; '
          f'до ln 2 дошло бы при a_2 = {need:.0f}'
          if sl[0] > 1e-9 else '  s НЕ растёт с a_2')
    print()
    print('''ЧТЕНИЕ. Если `s` нигде не достигает `ln 2` и не растёт к нему с
размером — регулярные сектора гасятся, но НЕДОСТАТОЧНО, и отбор
пространства-времени фазовым погашением не работает. Это опровержение
числом, и оно закрывает единственный оставшийся живым маршрут отбора.''')


if __name__ == '__main__':
    main()
