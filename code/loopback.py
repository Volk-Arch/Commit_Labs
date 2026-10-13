import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/feedback.py').read())

def loop_step(X, alpha, rs=None):
    """Видимое влияет на невидимое: порядок слегка сдвигает фазы.
    alpha — сила обратного влияния. alpha=0 — фазы независимы (как раньше)."""
    r = rs if rs is not None else np.random.default_rng(0)
    N, k = X.shape
    R = rel_of(X)
    if alpha <= 0: return X, R
    Rf = np.asarray(R, dtype=np.float32)
    dep = proper_time(Rf).astype(float)
    dep = (dep - dep.mean())/max(dep.std(), 1e-9)
    past = Rf.sum(axis=1); past = (past-past.mean())/max(past.std(), 1e-9)
    # обратное влияние: фазы подтягиваются к тому, что уже сложилось в порядке
    pull = np.column_stack([dep, past, -dep, -past][:k])
    Xn = np.mod(X + alpha*0.01*pull, 1.0)
    return Xn, R
