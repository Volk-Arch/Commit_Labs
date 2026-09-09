import numpy as np
from interval_orient import rel_of, interval, interval_frame
from defects import axis_field

def loop_flip(X, F, keys, start, radius=0.13, steps=40, rs=None, turns=1):
    """Замкнутый путь: уходим и возвращаемся в ту же точку.
    Двулистность: за 1 оборот знак меняется, за 2 — возвращается."""
    r = rs if rs is not None else np.random.default_rng(0)
    cur = start; prev = F[start]; vis = [start]
    for _ in range(steps*turns):
        cand = [j for j in keys if np.linalg.norm(X[j]-X[cur]) < radius and j != cur]
        if not cand: return None
        # уходим от старта первую половину, возвращаемся вторую
        frac = len(vis)/(steps*turns)
        want_far = (frac % (1.0/turns)) < (0.5/turns)
        d = np.array([np.linalg.norm(X[j]-X[start]) for j in cand])
        pick = cand[int(np.argmax(d))] if want_far else cand[int(np.argmin(d))]
        if r.random() < 0.3: pick = int(r.choice(cand))
        cur = pick; vis.append(cur)
        v = F[cur]
        if float(prev @ v) < 0: v = -v
        prev = v
    # вернулись ли в старт
    if np.linalg.norm(X[cur]-X[start]) > 0.06: return None
    return float(prev @ F[start]) < 0, len(vis)
