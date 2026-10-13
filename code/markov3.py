"""Петля «почини самую светоподобную пару» с растущим горизонтом:
мир сначала видит только ближних соседей (окно 6), улучшает форму решётки,
затем горизонт растёт (12, 25, 50) и улучшение продолжается. 2000 случайных стартов.
Куда приходят? Проверка на окне 150."""
import numpy as np, math, sys
from scipy.optimize import minimize
from markov import m_of, TAU
def vecs(N):
    a, b = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1)); a = a.ravel(); b = b.ravel()
    k = (a != 0) | (b != 0); return a[k], b[k]
W = [vecs(n) for n in (6, 12, 25, 50)]; a150, b150 = vecs(150)
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
out = []
for s in range(NS):
    x = rng.uniform(-2.5, 2.5, 2)
    for a, b in W:
        x = minimize(lambda p: -m_of(p, a, b), x, method='Nelder-Mead',
                     options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 3000}).x
    out.append((m_of(x, a150, b150), m_of(x, *W[-1]), x[0], x[1]))
out = np.array(out)
v = np.round(out[:, 0], 4)
print(f'стартов {NS}; итог на окне 150:')
for val in (0.4472, 0.3536, 0.3363, 0.3338):
    print(f'   {val:.4f}: {np.sum(np.abs(v - val) < 2e-4)} ({np.mean(np.abs(v - val) < 2e-4)*100:.1f} %)')
print(f'   около 1/3 (0.330…0.3337): {np.sum((v > 0.33) & (v < 0.3337))};  ниже 0.33: {np.sum(v < 0.33)} '
      f'(из них на окне 50 казались ≥ 0.33, но на 150 упали: {np.sum((v < 0.33) & (out[:, 1] >= 0.33))})')
np.save(f'/home/claude/data/markov3_{NS}.npy', out)

# точный буст: ищем целую матрицу K с K·(e1,e2) = (e1,e2)·diag(λ,1/λ) перебором малых целых
def exact_boost(x, lim=40):
    M = np.column_stack([[1.0, 1.0], x]); Mi = np.linalg.inv(M)
    best = None
    for l in np.exp(np.linspace(0.05, 4.5, 20000)):
        K = Mi @ np.diag([l, 1 / l]) @ M
        e = np.abs(K - np.round(K)).max()
        if e < 1e-6 and np.abs(np.round(K)).max() <= 10 ** 6:
            return l, np.round(K).astype(int)
    return None
from scipy.optimize import minimize_scalar
def exact_boost2(x):
    M = np.column_stack([[1.0, 1.0], x]); Mi = np.linalg.inv(M)
    f = lambda l: np.abs(Mi @ np.diag([l, 1 / l]) @ M - np.round(Mi @ np.diag([l, 1 / l]) @ M)).max()
    ls = np.exp(np.linspace(0.05, 6, 60000)); fs = np.array([f(l) for l in ls])
    for i in np.argsort(fs)[:20]:
        r = minimize_scalar(f, bounds=(ls[max(i-1,0)], ls[min(i+1,len(ls)-1)]), method='bounded', options={'xatol': 1e-13})
        if r.fun < 1e-7: return r.x, np.round(Mi @ np.diag([r.x, 1 / r.x]) @ M).astype(int)
    return None
print('точный буст у найденных пиков (первые по одному на значение):')
seen = set()
for m150, m50, x0, x1 in sorted(out.tolist(), reverse=True):
    key = round(m150, 4)
    if key in seen or m150 < 0.33: continue
    seen.add(key)
    eb = exact_boost2(np.array([x0, x1]))
    print(f'   m = {m150:.4f}: ' + (f'буст λ = {eb[0]:.5f} (быстрота {math.log(eb[0]):.3f}), след {np.trace(eb[1])}' if eb else 'точного буста не найдено'))
