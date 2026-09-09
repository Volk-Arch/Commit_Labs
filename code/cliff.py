import numpy as np

def clifford(k):
    """Генераторы алгебры Клиффорда: e_i^2 = -1, e_i e_j = -e_j e_i."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    n = (k+1)//2
    def kron(mats):
        M = mats[0]
        for A in mats[1:]: M = np.kron(M, A)
        return M
    gens = []
    for i in range(k):
        q, r = divmod(i, 2)
        mats = [Z]*q + [X if r == 0 else Y] + [I2]*(n-q-1)
        gens.append(1j*kron(mats))       # умножаем на i, чтобы e^2 = -1
    return gens

def transposition_op(gens, i, j):
    """t_ij = (e_i - e_j)/sqrt(2) — образ транспозиции в двойном накрытии."""
    return (gens[i] - gens[j])/np.sqrt(2)
