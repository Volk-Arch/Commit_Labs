"""Ниже насыщения: читается ли масса из глубины вмятины."""
import math
import numpy as np
from dent import run, n_for

L, EPS, RAD, S = 1.4, 0.2, 0.05, 16
print(f'L={L}, N={n_for(L)}, радиус {RAD}, eps={EPS}, засевов {S}')
print('плотность фона 25 -> в объёме трубки ожидается 0.05 фоновых событий,')
print('то есть насыщение наступает очень рано; ищем порог')
print()
print(f"{'M':>5}{'трубка':>21}{'разница с контролем':>26}{'знач.':>7}")
mc, ec = run(50, 0.0, L, RAD, EPS, S, False)
print(f"контроль (M=50, рассыпан): {mc:.4f} ±{ec:.4f}")
ms, ds = [], []
for M in (3, 6, 12, 25, 50, 100):
    mt, et = run(M, 0.0, L, RAD, EPS, S, True)
    d, e = mt - mc, math.hypot(et, ec)
    ms.append(M); ds.append(d)
    print(f'{M:>5}{mt:>13.4f} ±{et:<7.4f}{d:>17.4f} ±{e:<7.4f}{abs(d)/e:>7.1f}')
p = np.polyfit(np.log(ms[:4]), np.log(np.abs(ds[:4])), 1)[0]
print(f'  показатель по M на малых массах (3..25): {p:.3f}')
