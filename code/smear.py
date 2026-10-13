#!/usr/bin/env python3
"""РАЗМАЗАННОЕ ДЕЙСТВИЕ: гасит ли оно слоистый сектор.

Мотив, а не список. Замер 11 октября (`horizon.py`) показал: связи в
каузальном множестве ДЛИННЫЕ — через срединный срез проходит 70-80%
всех связей. Эта нелокальность и есть причина всех наших закрытий:
разброс действия растёт как N (`sigma-doc.md`), локальное действие
бессильно, слоистый сектор не гаснет. А размазанное действие у Соркина
вводится ровно для её укрощения.

ПОСТРОЕНИЕ, из принципа, а не по памяти о коэффициентах. Размазка есть
усреднение резкого действия по случайному ПРОРЕЖИВАНИЮ множества с
плотностью eps: пара выживает с вероятностью eps^2, а размер её
интервала становится биномиальным Bin(m, eps). Отсюда эффективный вес

    W_m(eps) = eps^2 * sum_{j=0..3} w_j * C(m,j) eps^j (1-eps)^{m-j}

отличный от нуля при ВСЕХ m, с затуханием (1-eps)^m — это и есть
нелокальность, с промежуточным масштабом eps^{-1/4} в длинах
дискретности.

КОНТРОЛИ, встроенные:
  A. при eps = 1 конструкция обязана вернуть веса БД (-1,9,-16,8) и
     ноль при m >= 4;
  B. геометрическое тождество sum_m W_m Gamma(m+1/2)/m! — печатается
     как функция eps. Если оно не ноль, значит размазка в таком виде
     ломает непрерывный предел, и это надо знать ДО вердикта;
  C. вердикт c(m,d) считается тем же кодом, что в wscan, с полной
     суммой по t (прежняя версия обрывала на t=3, что для размазанных
     весов неверно);
  D. при eps = 1 вердикт обязан дать архивное 0.1083...0.1167.
"""
import math

import numpy as np

W_BD = np.array([-1.0, 9.0, -16.0, 8.0])
MMAX = 60
GAMN = np.array([math.gamma(m + 0.5) / math.factorial(m)
                 for m in range(MMAX + 1)])
MS = np.arange(1, 31)
DS = np.arange(1, 31)
MD = 4.0 * (MS[:, None] + DS[None, :])


def smeared_w(eps):
    """W_m(eps) по прореживанию, m = 0..MMAX."""
    m = np.arange(MMAX + 1)
    W = np.zeros(MMAX + 1)
    for j in range(4):
        cm = np.array([math.comb(int(mm), j) if mm >= j else 0
                       for mm in m], float)
        W += W_BD[j] * cm * eps ** j * (1 - eps) ** np.maximum(m - j, 0)
    return eps ** 2 * W


def curve(W, ths):
    """max по (m,d) от c(m,d) при весах W, для каждого th."""
    th = ths[:, None, None]
    u = np.exp(1j * ths * W[0])
    tot = np.repeat(((1.0 + u)[:, None] ** MS[None, :])[:, :, None],
                    len(DS), axis=2)
    for t in range(1, MS.max() + 1):
        if abs(W[t]) < 1e-14:
            continue
        cm = np.array([math.comb(int(mm), t) if mm >= t else 0
                       for mm in MS], float)
        ph = np.exp(1j * th * W[t] * DS[None, None, :]) - 1.0
        tot = tot + cm[None, :, None] * (u ** t)[:, None, None] * ph
    c = np.log(np.maximum(np.abs(tot), 1e-300)) / MD[None, :, :]
    return c.reshape(len(ths), -1).max(axis=1)


def winmin(W, ths, half=3):
    c = curve(W, ths)
    k = 2 * half + 1
    sl = np.lib.stride_tricks.sliding_window_view(c, k).max(axis=1)
    i = int(sl.argmin())
    return float(sl[i]), float(ths[i + half])


def main():
    print(__doc__)
    print('КОНТРОЛЬ A: eps = 1 обязан вернуть веса БД')
    W1 = smeared_w(1.0)
    print('  W_0..W_5 = ' + ', '.join(f'{v:+.4f}' for v in W1[:6])
          + '   (ждём -1, +9, -16, +8, 0, 0)')
    ok = (np.allclose(W1[:4], W_BD) and np.allclose(W1[4:], 0))
    print(f'  => {"совпало" if ok else "НЕ совпало"}')

    print('\nКОНТРОЛЬ B: геометрическое тождество по eps')
    print(f"{'eps':>7}{'sum W_m Г/m!':>16}{'масштаб 1/eps^(1/4)':>22}"
          f"{'W_0':>10}{'хвост при m=20':>16}")
    for eps in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05):
        W = smeared_w(eps)
        ident = float(W @ GAMN)
        print(f'{eps:>7.2f}{ident:>16.3e}{eps**-0.25:>22.2f}'
              f'{W[0]:>10.4f}{W[20]:>16.3e}')

    print('\nКОНТРОЛЬ D и ВЕРДИКТ: гаснет ли сектор')
    ths = np.linspace(0.02, 3.0, 260)
    print(f"{'eps':>7}{'c по окну':>12}{'th':>8}   вердикт")
    rows = []
    for eps in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15,
                0.1, 0.07, 0.05, 0.03):
        W = smeared_w(eps)
        cm, th = winmin(W, ths)
        rows.append((eps, cm, th))
        v = 'ГАСНЕТ' if cm <= 0 else 'выживает'
        print(f'{eps:>7.2f}{cm:>12.4f}{th:>8.3f}   {v}', flush=True)
    best = min(rows, key=lambda r: r[1])
    print(f'\n  якорь eps=1: {rows[0][1]:.4f} (архив 0.1083…0.1167) '
          f'=> {"сошёлся" if abs(rows[0][1]-0.112) < 0.02 else "НЕ сошёлся"}')
    print(f'  наименьшее c = {best[1]:.4f} при eps = {best[0]:.2f}')
    neg = [r for r in rows if r[1] <= 0]
    print(f'  точек с гашением: {len(neg)} из {len(rows)}')


if __name__ == '__main__':
    main()
