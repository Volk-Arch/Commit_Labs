import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/spin_loop.py').read())

def quantized_input(N, k, levels, rs):
    """Вход с ограниченной точностью: каждая шкала имеет только `levels` значений.
    levels=2 — один бит на шкалу."""
    if levels is None:
        return rs.random((N, k))
    return rs.integers(0, levels, (N, k)).astype(float)/levels

def bits_per_event(k, levels):
    return k*np.log2(levels) if levels else float('inf')
