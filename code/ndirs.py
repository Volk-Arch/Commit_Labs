import numpy as np

def anisotropy(ndir, d, samples=400000, seed=7):
    """Насколько 'конус' из ndir направлений отличается от круглого.
    Меряем максимальное отклонение радиуса по направлениям."""
    rs = np.random.default_rng(seed)
    V = rs.normal(size=(ndir, d-1)); V /= np.linalg.norm(V, axis=1, keepdims=True)
    # пробные направления
    U = rs.normal(size=(samples, d-1)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    # радиус области {x: |x·v| <= 1 для всех v} в направлении u
    proj = np.abs(U @ V.T)            # samples x ndir
    r = 1.0/np.maximum(proj.max(axis=1), 1e-12)
    return float(r.max()/r.min() - 1), float(r.std()/r.mean())
