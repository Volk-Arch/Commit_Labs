import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/spinlabel.py').read())

def defect_radius(N, pp, seed=317, samples=300000):
    """Как далеко от киральных ПАР распространяется нарушение транзитивности.
    Расстояние тройки до дефекта: 0 — хотя бы одна пара тройки помечена,
    1 — не помечена, но событие тройки участвует в помеченной паре,
    2 — и этого нет."""
    r = np.random.default_rng(seed)
    U = np.array([r.permutation(N) for _ in range(3)], dtype=np.int64)
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    par = np.sign(a-b)*np.sign(b-c)*np.sign(a-c)
    s = np.where(np.random.default_rng(seed+1).random(N) < pp, 1, -1)
    both = (s[None, :] > 0) & (s[:, None] > 0)
    R0 = base_rel(U)
    R = R0 & (~both | (par > 0))
    marked_pair = both & R0                       # пары, реально затронутые
    touched = marked_pair.any(axis=0) | marked_pair.any(axis=1)   # события у дефекта
    Rb = np.ascontiguousarray(R, dtype=bool)
    rr = np.random.default_rng(seed+2)
    cnt = {0: [0, 0], 1: [0, 0], 2: [0, 0]}       # [всего троек, нарушено]
    for _ in range(samples):
        i, j, k = (int(v) for v in rr.integers(0, N, 3))
        if not (Rb[j, i] and Rb[k, j]): continue
        if marked_pair[j, i] or marked_pair[k, j] or marked_pair[k, i]: d = 0
        elif touched[i] or touched[j] or touched[k]: d = 1
        else: d = 2
        cnt[d][0] += 1
        if not Rb[k, i]: cnt[d][1] += 1
    return cnt, float(touched.mean())
