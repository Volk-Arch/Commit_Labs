"""Выбирает ли самосогласованность золото сама? 1+1, τ на входе нет.
Мир — решётка в световых координатах (u, v), интервал = u·v. Требование
«надёжного порядка»: ни одна пара событий не почти светоподобна, т.е.
m(L) = min |u·v| по ненулевым векторам / кообъём  — как можно больше.
Ищем максимумы m из случайных стартов (Нелдер–Мид), смотрим, где они."""
import numpy as np, math
from scipy.optimize import minimize
TAU = (1 + 5 ** .5) / 2
N = 25
A, Bb = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1)); A = A.ravel(); Bb = Bb.ravel()
nz = (A != 0) | (Bb != 0); A, Bb = A[nz], Bb[nz]

def gauss(e1, e2):
    """редукция Гаусса (евклидова в плоскости u,v): короткий почти ортогональный базис"""
    for _ in range(200):
        if e1 @ e1 > e2 @ e2: e1, e2 = e2, e1
        k = round((e1 @ e2) / (e1 @ e1))
        if k == 0: break
        e2 = e2 - k * e1
    return e1, e2

def m_of(p, AA=None, BB=None):
    # базис: e1 = (1, 1), e2 = (p0, p1); бусты и масштаб не меняют m
    e1 = np.array([1.0, 1.0]); e2 = np.array([p[0], p[1]])
    det = abs(e1[0] * e2[1] - e1[1] * e2[0])
    if det < 1e-3: return 0.0
    e1, e2 = gauss(e1 / det ** .5, e2 / det ** .5)
    AA = A if AA is None else AA; BB = Bb if BB is None else BB
    u = AA * e1[0] + BB * e2[0]; v = AA * e1[1] + BB * e2[1]
    return np.min(np.abs(u * v))

def classify(p):
    """форма Q(a,b) = (a e1u + b e2u)(a e1v + b e2v) → приводим к целой, если можем"""
    e2 = p; Aq = 1.0; Bq = e2[0] + e2[1]; Cq = e2[0] * e2[1]
    disc = Bq ** 2 - 4 * Aq * Cq
    # нормируем на минимум |Q| и смотрим, целые ли коэффициенты
    return disc, (Aq, Bq, Cq)

if __name__ == '__main__':
    rng = np.random.default_rng(1)
    found = []
    for trial in range(400):
        p0 = rng.uniform(-3, 3, 2)
        res = minimize(lambda p: -m_of(p), p0, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 4000})
        res = minimize(lambda p: -m_of(p), res.x, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 4000})
        found.append((-res.fun, res.x))
    # проверка на большом окне (до 150): усечённый перебор мог завысить
    N2 = 150
    A2, B2 = np.meshgrid(np.arange(-N2, N2 + 1), np.arange(-N2, N2 + 1)); A2 = A2.ravel(); B2 = B2.ravel()
    nz2 = (A2 != 0) | (B2 != 0); A2, B2 = A2[nz2], B2[nz2]
    found = [(m_of(x, A2, B2), x) for _, x in found]
    vals = np.array([f[0] for f in found])
    print('Теория Маркова: 1/√5 = %.4f (золото), 1/√8 = %.4f (серебро), 5/√221 = %.4f, 13/√1517 = %.4f'
          % (1 / 5 ** .5, 1 / 8 ** .5, 5 / 221 ** .5, 13 / 1517 ** .5))
    for lo, hi in ((0.44, 0.46), (0.35, 0.36), (0.333, 0.34), (0.32, 0.3335)):
        sel = (vals > lo) & (vals < hi)
        print(f'  найдено максимумов в [{lo}, {hi}]: {sel.sum()}  (значения {np.round(np.unique(np.round(vals[sel], 4)), 4)})')
    print('  ниже 0.32:', np.sum(vals < 0.32), '; лучший:', vals.max().round(5))
    # лучший: проверим точную симметрию бустом
    best = found[int(vals.argmax())][1]
    e1 = np.array([1.0, 1.0]); e2 = best
    M = np.column_stack([e1, e2])
    pts = (M @ np.vstack([A, Bb])).T
    # ищем буст λ, переводящий решётку в себя: λ = (u-растяжение). Проверяем кандидатов из отношений u-координат коротких векторов
    cands = []
    for lam in np.linspace(1.05, 8, 140000):
        pass
    # проще: собственные числа целочисленного автоморфа. Берём буст B=diag(λ,1/λ), ищем λ, при котором M^{-1} B M целочисленна.
    Minv = np.linalg.inv(M)
    lams = np.linspace(1.01, 10, 900001)
    X = np.array([[Minv @ np.diag([l, 1 / l]) @ M] for l in lams[::50]])[:, 0]
    err = np.abs(X - np.round(X)).max(axis=(1, 2))
    i = err.argmin(); l0 = lams[::50][i]
    # уточняем
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda l: np.abs(Minv @ np.diag([l, 1 / l]) @ M - np.round(Minv @ np.diag([l, 1 / l]) @ M)).max(), bounds=(l0 - 0.01, l0 + 0.01), method='bounded')
    K = np.round(Minv @ np.diag([r.x, 1 / r.x]) @ M).astype(int)
    print(f'лучшая решётка: точный буст λ = {r.x:.6f} (τ² = {TAU**2:.6f}), целая матрица буста {K.tolist()}, след {np.trace(K)}, невязка {r.fun:.1e}')
