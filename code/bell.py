import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from double_slit import build_causet, direct_links, depth, topo_order, propagate

def chsh(A0, A1, B0, B1):
    """Величина CHSH: |<A0B0> + <A0B1> + <A1B0> - <A1B1>|.
    Классический предел 2, квантовый 2*sqrt(2) ~ 2.828."""
    def E(a, b): return float(np.mean(a*b))
    return abs(E(A0, B0) + E(A0, B1) + E(A1, B0) - E(A1, B1))
