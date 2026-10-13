#!/usr/bin/env python3
"""ПОИСК ПРАВИЛА С ТЕЧЕНИЕМ ОТ 2 К 4.

`dimflow-doc.md`: рост с фиксированным числом родителей даёт течение
1 -> 2 -> 3 (доля внутри интервала 0.89 -> 0.50 -> 0.37), прибор
проверен плоским контролем на засевах. Нужно течение от 2 к 4, то есть
0.50 на малых масштабах и 0.10 на больших.

Единственный параметр, уже сдвинувший дело в нужную сторону: при шести
родителях крупномасштабное значение 0.264 против 0.290 при трёх. Больше
родителей - площе интервалы - выше размерность. Сканирую число родителей
и окно локальности.

ЦЕЛЬ: 0.50 на бине 3-6 и 0.10 на бине 50-120.
КОНТРОЛЬ: засев 4d в тех же бинах даёт 0.094 … 0.101 (плоско).
"""
import numpy as np

from sectors import sprinkle_d
from dimflow import spectrum, BINS, show

N = 1500


def grow(n, rs, nlink, win=None, anti=False):
    R = np.zeros((n, n), bool)
    deg = np.zeros(n)
    for i in range(1, n):
        lo = 0 if win is None else max(0, i - win)
        cand = np.arange(lo, i)
        k = min(nlink, len(cand))
        if anti:
            w = 1.0 / (deg[cand] + 1.0)
            w /= w.sum()
            par = rs.choice(cand, size=k, replace=False, p=w)
        else:
            par = rs.choice(cand, size=k, replace=False)
        anc = np.zeros(n, bool)
        for p in par:
            anc[p] = True
            anc |= R[:, p]
        R[:, i] = anc
        deg[par] += 1
    return R


def main():
    print(__doc__)
    rs = np.random.default_rng(2718281)
    print(f'N={N}')
    print(f'{"правило":>26}' + ''.join(
        f'{f"{a}–{b}" if b < 1e8 else f">{a}":>9}' for a, b in BINS)
        + f'{"доля св.":>10}')
    show('КОНТРОЛЬ засев 4d', spectrum(sprinkle_d(N, 4, rs), rs))
    for nl in (3, 6, 12, 25, 50, 100):
        R = grow(N, rs, nl)
        r = float(R.sum()) / (N * (N - 1) / 2)
        show(f'родителей {nl}', spectrum(R, rs), f'{r:.3f}')
    print('  с окном локальности:')
    for nl, w in ((6, 100), (12, 100), (25, 200), (50, 400)):
        R = grow(N, rs, nl, win=w)
        r = float(R.sum()) / (N * (N - 1) / 2)
        show(f'родителей {nl}, окно {w}', spectrum(R, rs), f'{r:.3f}')
    print('  обратное предпочтение (к малосвязным):')
    for nl in (12, 50):
        R = grow(N, rs, nl, anti=True)
        r = float(R.sum()) / (N * (N - 1) / 2)
        show(f'родителей {nl}, обратн.', spectrum(R, rs), f'{r:.3f}')


if __name__ == '__main__':
    main()
