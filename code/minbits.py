import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'spin_loop.py'), encoding='utf-8').read())

def quantized_input(N, k, levels, rs):
    """Вход с ограниченной точностью: каждая шкала имеет только `levels` значений.
    levels=2 — один бит на шкалу."""
    if levels is None:
        return rs.random((N, k))
    return rs.integers(0, levels, (N, k)).astype(float)/levels

def bits_per_event(k, levels):
    return k*np.log2(levels) if levels else float('inf')
