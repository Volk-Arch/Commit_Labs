import numpy as np

def antichain_slice(R, D, level):
    """Срез 'сейчас': события одной глубины. Все несравнимы между собой."""
    return np.flatnonzero(D == level)

def slice_structure(R, idx):
    """Что есть на срезе, если порядка там нет?
    Общее прошлое — вот что связывает несравнимые события."""
    n = len(idx)
    if n < 3: return None
    Rb = np.asarray(R, dtype=bool)
    past = Rb[idx]                    # past[a] — прошлое события idx[a]
    shared = (past.astype(np.float32) @ past.astype(np.float32).T)
    sizes = past.sum(axis=1).astype(float)
    # мера близости: доля общего прошлого
    denom = np.minimum(sizes[:, None], sizes[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        prox = np.where(denom > 0, shared/np.maximum(denom, 1), 0.0)
    np.fill_diagonal(prox, 0)
    return prox, sizes
