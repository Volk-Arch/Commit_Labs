# целая решётка с анизотропной формой 7t² − x² − y² − z²: ищем буст с g00 = 2 (перебор малых целых)
import numpy as np, itertools
G = np.diag([7, -1, -1, -1])
col0 = []
for c in itertools.product(range(-5, 6), repeat=3):
    if sum(x * x for x in c) == 21: col0.append(c)
print('первых столбцов с g00=2:', len(col0))
# ищем полную матрицу: столбцы v_i с v_i^T G v_j = G_ij
R = range(-6, 7)
vecs = {k: [] for k in (1,)}
cand = [np.array(v) for v in itertools.product(R, repeat=4) if (lambda v: 7*v[0]**2 - v[1]**2 - v[2]**2 - v[3]**2)(v) == -1]
print('векторов с Q = −1:', len(cand))
c0 = np.array((2,) + col0[0]); found = None
C = np.array(cand); dots0 = C @ G @ c0
C1 = C[dots0 == 0]
for i, a in enumerate(C1):
    Ca = C1[(C1 @ G @ a) == 0]
    for b in Ca:
        Cb = Ca[(Ca @ G @ b) == 0]
        if len(Cb):
            found = np.column_stack([c0, a, b, Cb[0]]); break
    if found is not None: break
print(found); print('проверка gᵀGg = G:', np.array_equal(found.T @ G @ found, G))
