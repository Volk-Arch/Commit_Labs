import numpy as np

def orientation(U):
    """Псевдоскаляр: знак произведения попарных разностей."""
    N, k = U.shape
    s = np.ones(N)
    for i in range(k):
        for j in range(i+1, k):
            s = s*np.sign(U[:, i]-U[:, j])
    return s

def frame_of(P):
    """Локальный репер из точки фазового пространства: ортогонализуем
    базис, привязанный к порядку координат. Даёт элемент группы вращений."""
    k = P.shape[0]
    V = np.eye(k)
    order = np.argsort(P)              # перестановка координат по величине
    V = V[order]
    # приводим к SO(k): если определитель отрицателен, меняем знак строки
    if np.linalg.det(V) < 0: V[0] = -V[0]
    return V

def lift_sign(path_frames):
    """Подъём пути на двулистное накрытие: следим за непрерывностью.
    Возвращает знак после обхода — +1 (стянулся) или -1 (не стянулся)."""
    s = 1.0
    prev = path_frames[0]
    for F in path_frames[1:]:
        # скалярное произведение соседних реперов; смена знака = переход листа
        d = np.trace(prev.T @ F)
        if d < 0: s = -s
        prev = F
    return s
