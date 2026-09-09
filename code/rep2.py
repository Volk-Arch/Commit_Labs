import numpy as np
from itertools import permutations

def klein_class(p):
    """Двумерное представление S_4 идёт через фактор S_4/V, где V — четверная
    группа Клейна. Фактор изоморфен S_3, и представление — его стандартное двумерное.
    Класс перестановки в факторе определяется тем, как она действует на ТРИ пары."""
    # три способа разбить {0,1,2,3} на две пары
    pairs = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]
    def img(pr):
        a = tuple(sorted((p[pr[0][0]], p[pr[0][1]])))
        b = tuple(sorted((p[pr[1][0]], p[pr[1][1]])))
        return tuple(sorted([a, b]))
    canon = [tuple(sorted([tuple(sorted(x)) for x in pr])) for pr in pairs]
    out = []
    for pr in pairs:
        out.append(canon.index(img(pr)))
    return tuple(out)

def rep2_matrix(sigma):
    """Двумерное представление S_3 (перестановки трёх объектов)."""
    w = np.exp(2j*np.pi/3)
    # представление через действие на (1, w, w^2)
    P = np.zeros((3,3))
    for a,b in enumerate(sigma): P[a,b]=1
    # проецируем на двумерное подпространство, ортогональное (1,1,1)
    B = np.array([[1,-1,0],[1,1,-2]],dtype=float)
    B[0]/=np.linalg.norm(B[0]); B[1]/=np.linalg.norm(B[1])
    return B @ P @ B.T
