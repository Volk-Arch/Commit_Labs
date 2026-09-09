import sys, os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import numpy as np
exec(open(os.path.join(_HERE, 'charge_weight.py'), encoding='utf-8').read())

def charge_census(X):
    """Перепись зарядов: сколько событий каждого типа и суммарный «заряд»."""
    lab, margin = labels_of(X)
    k = X.shape[1]
    counts = np.array([int((lab == m).sum()) for m in range(k)] + [int((lab < 0).sum())])
    # «векторный» заряд: сумма единичных векторов по метке, взвешенная отрывом
    vec = np.zeros(k)
    for m in range(k):
        sel = lab == m
        vec[m] = float(margin[sel].sum())
    return counts, vec, lab, margin
