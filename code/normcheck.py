#!/usr/bin/env python3
"""ДИАГНОСТИКА НОРМИРОВКИ сглаженного оператора.

ОБЯЗАТЕЛЬНОЕ ТОЖДЕСТВО. У сглаженного д'Аламбертиана на постоянной
функции в ПЛОСКОМ пространстве результат обязан быть нулём при ЛЮБОМ `ε`
(там `R = 0`), то есть

    −1 + ε·⟨Σ_{y≺x} f_d(ε·n(y,x))⟩ = 0   ⟹   Q ≡ ε·Σf/N = 1.

Это проверка нормировки, не требующая ни литературы, ни подгонки: если
`Q` зависит от `ε`, степени `ε` в моей записи неверны.

ПОЧЕМУ СЛОИСТЫЕ ЭТОГО НЕ ЛОВИЛИ. У слоистого порядка `n` либо нуль
(соседние слои), либо огромно (через слой), поэтому `f(εn)` равно 1 либо
0, и тонкое плоское сокращение НЕ ЗАДЕЙСТВУЕТСЯ. Контроль был
нечувствителен к ошибке, которую должен был ловить.

  python3 normcheck.py
"""
import math
import time

import numpy as np

from sprinkle import sprinkle_d
from smear import fpoly


def Q(n, dim, eps, seeds, root):
    v = []
    for s in range(seeds):
        A = sprinkle_d(n, dim, np.random.default_rng(root + 17 * s))
        F = A.astype(np.float32)
        B = np.rint(F @ F).astype(np.float64)
        z = eps * B[A]
        v.append(eps * float(fpoly(z, dim).sum()) / n)
        del A, B
    v = np.array(v, float)
    return v.mean(), v.std(ddof=1) / math.sqrt(len(v))


def main():
    t0 = time.time()
    print(__doc__.split('  python3')[0].rstrip())
    print()
    for dim in (4, 2):
        print('=' * 88)
        print(f'Q = ε·Σf/N в ПЛОСКОМ {dim}-мерном ромбе. Обязано быть 1.000')
        print(f"{'ε':>8}{'N=600':>16}{'N=1200':>16}{'N=2400':>16}"
              f"{'N=4800':>16}")
        for eps in (0.5, 0.3, 0.2, 0.1, 0.05, 0.03):
            cells = []
            for n in (600, 1200, 2400, 4800):
                m, e = Q(n, dim, eps, 8, 500 + n + int(1000 * eps))
                cells.append(f'{m:>9.3f}±{e:<5.3f}')
            print(f'{eps:>8.2f}' + ''.join(f'{c:>16}' for c in cells))
            print(f'    ({time.time()-t0:.0f} с)', flush=True)
        print()
    print('=' * 88)
    print('ВЫВОД читается так: если Q зависит от ε — степени ε неверны;')
    print('если Q зависит от N — оператор не имеет предела в этом виде;')
    print('если Q = 1 всюду — нормировка верна и дело в другом.')
    print()
    print(f'  время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
