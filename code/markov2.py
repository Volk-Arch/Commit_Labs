"""То же, что markov.py, но полным перебором формы решётки (сетка 2000×2000),
затем уточнение пиков и проверка на большом окне."""
import numpy as np, math, sys
from scipy.optimize import minimize
from markov import m_of, gauss, TAU
def vecs(N):
    a, b = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1)); a = a.ravel(); b = b.ravel()
    k = (a != 0) | (b != 0); return a[k], b[k]
a8, b8 = vecs(8); a25, b25 = vecs(25); a150, b150 = vecs(150)
G = 2000; ps = np.linspace(-2.5, 2.5, G)
P0, P1 = np.meshgrid(ps, ps); P0 = P0.ravel(); P1 = P1.ravel()
# векторизованная редукция Гаусса
e1 = np.stack([np.ones_like(P0), np.ones_like(P0)], 1); e2 = np.stack([P0, P1], 1)
det = np.abs(e2[:, 1] - e2[:, 0]); ok = det > 1e-3
e1 = e1[ok] / np.sqrt(det[ok])[:, None]; e2 = e2[ok] / np.sqrt(det[ok])[:, None]
for _ in range(60):
    sw = (e1 * e1).sum(1) > (e2 * e2).sum(1)
    e1[sw], e2[sw] = e2[sw].copy(), e1[sw].copy()
    k = np.round((e1 * e2).sum(1) / (e1 * e1).sum(1))
    if not k.any(): break
    e2 = e2 - k[:, None] * e1
M = np.zeros(len(e1))
for c in range(0, len(e1), 20000):
    u = e1[c:c+20000, 0:1] * a8 + e2[c:c+20000, 0:1] * b8
    v = e1[c:c+20000, 1:2] * a8 + e2[c:c+20000, 1:2] * b8
    M[c:c+20000] = np.abs(u * v).min(1)
full = np.zeros(G * G); full[ok] = M; F = full.reshape(G, G)
print(f'сетка: доля формы с m > 0.40: {np.mean(F > 0.40)*100:.3f} %, > 0.34: {np.mean(F > 0.34)*100:.2f} %, > 0.30: {np.mean(F > 0.30)*100:.1f} %')
# пики: топ-300 точек сетки → уточнение
idx = np.argsort(F.ravel())[::-1][:300]
peaks = []
for i in idx:
    x0 = np.array([P0[i], P1[i]])
    r = minimize(lambda p: -m_of(p, a25, b25), x0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 3000})
    peaks.append(m_of(r.x, a150, b150))
peaks = np.array(peaks)
u, cnt = np.unique(np.round(peaks, 4), return_counts=True)
print('уточнённые пики (значение: сколько из 300 лучших точек сетки к нему пришли):')
for val, c in sorted(zip(u, cnt), reverse=True)[:8]: print(f'   {val:.4f}: {c}')
print('Марков: 1/√5 = %.4f, 1/√8 = %.4f, 5/√221 = %.4f' % (1 / 5 ** .5, 1 / 8 ** .5, 5 / 221 ** .5))
np.save('/home/claude/data/markov_grid.npy', F.astype(np.float32))
