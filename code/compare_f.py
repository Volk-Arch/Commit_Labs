import numpy as np
from math import gamma

def f_minkowski(d):
    """Доля связанных пар для засева в d-мерном Минковском (Майрхейм–Мейер)."""
    return gamma(d+1)*gamma(d/2)/(2*gamma(1.5*d))

def f_korder(k):
    """Доля связанных пар для пересечения k случайных линейных порядков."""
    return 2.0**-(k-1)

print('ДВЕ РАЗНЫЕ ФОРМУЛЫ ДЛЯ ОДНОЙ ВЕЛИЧИНЫ')
print(f"{'d=k':>5}{'Минковский':>14}{'k порядков':>14}{'отношение':>12}{'расхождение':>14}")
for d in range(2, 8):
    a, b = f_minkowski(d), f_korder(d)
    print(f'{d:>5}{a:>14.5f}{b:>14.5f}{b/a:>12.3f}{(b-a)/a:>13.1%}')
