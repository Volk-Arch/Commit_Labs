import numpy as np
import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
from cliff import clifford

def rotor(g, i, j, theta):
    """Непрерывный поворот в плоскости (i,j) алгебры Клиффорда."""
    B = g[i] @ g[j]
    return np.cos(theta/2)*np.eye(len(B), dtype=complex) + np.sin(theta/2)*B

def continuous_lift(P, g, k, scale=0.15):
    """Непрерывный подъём вдоль пути: на каждом шаге поворот в плоскости
    той пары шкал, которая сближается. Угол зависит от РАЗНОСТИ фаз."""
    d = g[0].shape[0]
    M = np.eye(d, dtype=complex)
    for t in range(1, len(P)):
        prev, cur = P[t-1], P[t]
        for i in range(k):
            for j in range(i+1, k):
                a0, a1 = prev[i]-prev[j], cur[i]-cur[j]
                if a0*a1 < 0:                      # разность сменила знак — пересекли
                    M = M @ rotor(g, i, j, np.pi)
    return M
