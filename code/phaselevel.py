import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'symfun.py'), encoding='utf-8').read())

def phase_quantities(U):
    """Величины, доступные ПРЯМО ИЗ ФАЗ — без требования выжить в порядке.
    Делим на симметричные (переживают перестановку шкал) и антисимметричные."""
    N, k = U.shape
    m = U.mean(1, keepdims=True)
    dev = U - m
    srt = np.sort(U, 1)
    Q = {}
    # симметричные
    Q['S1 сумма'] = U.sum(1)
    Q['S2 дисперсия'] = U.var(1)
    Q['S3 асимметрия'] = (dev**3).mean(1)
    Q['S4 эксцесс'] = (dev**4).mean(1)
    # антисимметричные и порядково-зависимые
    Q['A1 какая шкала max'] = U.argmax(1).astype(float)
    Q['A2 какая шкала min'] = U.argmin(1).astype(float)
    Q['A3 чётность ранжировки'] = np.array([
        np.sign(np.linalg.det(np.vander(row, increasing=True)[:, :k])) for row in U])
    Q['A4 знак (u0-u1)(u1-u2)'] = np.sign((U[:, 0]-U[:, 1])*(U[:, 1]-U[:, 2]))
    Q['A5 u0 - u3'] = U[:, 0]-U[:, 3] if k > 3 else np.zeros(N)
    return Q

def independence_matrix(Q):
    names = list(Q.keys())
    Y = np.column_stack([np.asarray(Q[n], float) for n in names])
    Z = (Y - Y.mean(0))/np.maximum(Y.std(0), 1e-9)
    return names, (Z.T @ Z)/len(Z)
