import numpy as np

def interval_counts(R, kmax=4):
    """N_k = число пар с ровно (k-1) элементами строго между.
    N1 = связи (0 между), N2 = 1 между, и так далее."""
    C = np.asarray(R, dtype=np.float32)
    btw = (C @ C).astype(np.int32)          # btw[i,j] = сколько между
    Rb = np.asarray(R, dtype=bool)
    out = []
    for m in range(kmax):
        out.append(int(((btw == m) & Rb).sum()))
    return out

def bd_action(R, d):
    """Действие Бенинказы–Дауке. Пропорционально альтернированной сумме
    числа интервалов. Плоское пространство должно давать ~0."""
    N = len(R)
    if d == 2:
        n = interval_counts(R, 3)
        # S ∝ N - 2*(N1 - 2*N2 + N3)
        return N - 2*(n[0] - 2*n[1] + n[2])
    if d == 4:
        n = interval_counts(R, 4)
        # S ∝ N - N1 + 9*N2 - 16*N3 + 8*N4
        return N - n[0] + 9*n[1] - 16*n[2] + 8*n[3]
    return None
