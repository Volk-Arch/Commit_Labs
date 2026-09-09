import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'reduce.py'), encoding='utf-8').read())

def gen_input(N, k, mode, params=None, rs=None):
    """События порождаются ПРАВИЛОМ из малого числа параметров,
    а не задаются по одному."""
    r = rs if rs is not None else np.random.default_rng(0)
    n = np.arange(N, dtype=float)
    if mode == 'random':                      # контроль: по 2 числа на событие
        S = r.random(N); V = r.random(N)
    elif mode == 'irrational':                # два несоизмеримых шага
        a, b = params if params else (np.sqrt(2)-1, np.sqrt(3)-1)
        S = np.mod(n*a, 1.0); V = np.mod(n*b, 1.0)
    elif mode == 'logistic':                  # хаотическое отображение
        S = np.zeros(N); V = np.zeros(N)
        x, y = 0.31, 0.67
        for i in range(N):
            x = 3.99*x*(1-x); y = 3.97*y*(1-y)
            S[i] = x; V[i] = y
    elif mode == 'digits':                    # цифры иррационального числа
        S = np.mod(n*np.pi, 1.0); V = np.mod(n*np.e, 1.0)
    else:
        return None
    shape = np.array([np.sin(2*np.pi*m/k + 0.7) for m in range(k)])
    shape -= shape.mean(); shape /= max(shape.std(), 1e-9)
    return np.mod(S[:, None] + 5.0*V[:, None]*shape[None, :], 1.0)
