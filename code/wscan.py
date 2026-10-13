#!/usr/bin/env python3
"""СВОБОДА ВЕСОВ в фазовой сумме: есть ли вес, при котором сектор гибнет.

Неосмотренное место. Фаркаш закрыл разделение по ЗНАЧЕНИЮ (никакая
линейная комбинация счётов не разводит секторы неравенством). Скан
`cancel5` закрыл гашение по ФАЗЕ — но он варьировал только hbar, m, d
при ФИКСИРОВАННЫХ весах Бенинказы–Доукера. Свободу самих весов в
фазовой сумме не трогали ни разу.

А сегодняшний двухточечный интеграл (`nm-doc.md`) дал недостающую
деталь — условие, при котором действие остаётся ГЕОМЕТРИЧЕСКИМ:

    sum_m w_m Gamma(m+1/2)/m! = 0

то есть ведущий член плоского пространства гасится весами, и остаётся
кривизна. Для БД: -1 + 9/2 - 6 + 5/2 = 0. В явном виде гиперплоскость:

    w0 + w1/2 + 3 w2/8 + 5 w3/16 = 0

Вердикт сектора (`cancel5-doc.md`): ln|Z| ~ c N², где
c(m,d) = ln|g_d(m)|/(4(m+d)), и сектор ГИБНЕТ если max по (m,d) от c
не больше нуля. При весах БД минимум по нерезонансным hbar равен
0.1167 — сектор выживает всюду.

Обобщение: u = exp(i*th*w0), phi(t) = exp(i*th*w_t), и

    g = (1+u)^m + sum_{t=1..3} C(m,t) u^t (phi(t)^d - 1)

РЕЗОНАНСЫ отсеиваются без старого правила: настоящее решение обязано
держаться на ОКНЕ по th, а резонанс есть одиночный шип. Поэтому
условие c <= 0 требуется на окне, а не в точке.

ЯКОРЬ: при w = БД код обязан воспроизвести 0.1167 (с точностью сетки).
Без якоря остальное не считается.
"""
import math

import numpy as np

GAM = np.array([math.gamma(m + 0.5) for m in range(4)]) / np.array(
    [1.0, 1.0, 2.0, 6.0])
BD = np.array([-1.0, 9.0, -16.0, 8.0])
MS = np.arange(1, 41)
DS = np.arange(1, 41)
CMT = {t: np.array([math.comb(int(m), t) if m >= t else 0
                    for m in MS], float) for t in (1, 2, 3)}


def maxc(w, th):
    """max по (m,d) от c(m,d) при весах w и фазе th."""
    u = np.exp(1j * th * w[0])
    base = (1.0 + u) ** MS                       # (m,)
    best = -np.inf
    for d in DS:
        tot = base.copy()
        for t in (1, 2, 3):
            tot = tot + CMT[t] * u ** t * (
                np.exp(1j * th * w[t] * d) - 1.0)
        v = np.abs(tot)
        c = np.log(np.maximum(v, 1e-300)) / (4.0 * (MS + d))
        best = max(best, float(c.max()))
    return best


def scan_th(w, ths):
    return np.array([maxc(w, t) for t in ths])


def window_min(w, ths, half=3):
    """Минимум по th от СКОЛЬЗЯЩЕГО максимума — отсев одиночных шипов."""
    c = scan_th(w, ths)
    k = 2 * half + 1
    if len(c) < k:
        return float(c.min()), float(ths[int(c.argmin())])
    sl = np.lib.stride_tricks.sliding_window_view(c, k).max(axis=1)
    i = int(sl.argmin())
    return float(sl[i]), float(ths[i + half])


def main():
    print(__doc__)
    ths = np.linspace(0.05, 3.0, 400)

    print('ЯКОРЬ: веса БД')
    raw = scan_th(BD, ths)
    wm, wt = window_min(BD, ths)
    print(f'  геометрическое условие: '
          f'{float(BD @ GAM):.3e}  (обязан быть 0)')
    print(f'  минимум по точкам:   {raw.min():.4f} при th={ths[int(raw.argmin())]:.4f}')
    print(f'  минимум по ОКНУ:     {wm:.4f} при th={wt:.4f}')
    print(f'  архив (нерезонансный минимум): 0.1167')
    print(f'  => якорь {"сошёлся" if abs(wm-0.1167) < 0.03 else "НЕ сошёлся"}')

    print('\nСКАН по гиперплоскости (w3 нормирован, w0 вынужден)')
    best = (np.inf, None, None)
    rows = []
    grid = np.linspace(-20, 20, 17)
    for w1 in grid:
        for w2 in grid:
            for w3 in (-16.0, -8.0, -4.0, 4.0, 8.0, 16.0):
                w0 = -(w1 / 2 + 3 * w2 / 8 + 5 * w3 / 16)
                w = np.array([w0, w1, w2, w3])
                s = np.abs(w).max()
                if s < 1e-9:
                    continue
                w = w * (16.0 / s)
                wm, wt = window_min(w, ths)
                rows.append((wm, w.copy(), wt))
                if wm < best[0]:
                    best = (wm, w.copy(), wt)
    rows.sort(key=lambda z: z[0])
    print(f'  точек на гиперплоскости: {len(rows)}')
    print(f"{'c по окну':>11}{'th':>8}   веса (w0,w1,w2,w3)")
    for wm, w, wt in rows[:8]:
        print(f'{wm:>11.4f}{wt:>8.3f}   '
              + ', '.join(f'{v:+.2f}' for v in w))
    neg = [r for r in rows if r[0] <= 0]
    print(f'\n  точек, где сектор ГИБНЕТ (c <= 0 на окне): '
          f'{len(neg)} из {len(rows)}')
    print(f'  наименьшее c = {best[0]:.4f} при весах '
          + ', '.join(f'{v:+.2f}' for v in best[1])
          + f', th = {best[2]:.3f}')
    print(f'  (для сравнения: БД даёт {wm if False else 0.1167:.4f})')
    if neg:
        print('\n  ЕСТЬ КАНДИДАТЫ — требуется проверка независимым счётом')
    else:
        print('\n  КАНДИДАТОВ НЕТ: ни один геометрический вес не гасит '
              'сектор')


if __name__ == '__main__':
    main()
