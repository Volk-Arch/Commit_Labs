"""Скрытая фаза с конечным разрешением: N×N клеток тора (по Ханнею–Берри ħ = 1/(2πN)).
Кошка — целочисленная перестановка: (x, y) → (2x + y + f(x), x + y + f(x)) mod N,
f(x) = round(εN/2π · sin 2πx/N); сдвиг меток — на (round(N/τ), round(N/τ²)).
Событие есть ⟺ x < P(s)·N. Вопрос: сколько клеток нужно, чтобы незнание наблюдателя
было как у монеток (дисперсия/среднее = Σ P(1−P)/ΣP)?"""
import sys, math, time, numpy as np
sys.path.insert(0, '.')
import gold4test as G
from hidden import points_lab, prob, r_for
from gold4 import TAU
t0 = time.time(); EPS = 0.6
def phase_int(labels, seeds, N):
    S, n = len(seeds), len(labels)
    om = np.array([round(N / TAU) % N, round(N / TAU ** 2) % N], np.int64)
    th = np.broadcast_to(seeds[:, None, :], (S, n, 2)).astype(np.int64).copy()
    L = labels.astype(np.int64) % N
    for rep in range(2):
        for k in range(L.shape[1]):
            th = (th + L[None, :, k, None] * om) % N
            f = np.round(EPS * N / (2 * math.pi) * np.sin(2 * math.pi * th[..., 0] / N)).astype(np.int64)
            th = np.stack([(2 * th[..., 0] + th[..., 1] + f) % N, (th[..., 0] + th[..., 1] + f) % N], -1)
    return th
w = 0.3; r = r_for(w); rmax = r * (1 + w)
rng = np.random.default_rng(5); nv = np.array([.48, -.71, .52]); nv /= np.linalg.norm(nv); eta = 1.5
nexp = 3000; tau0 = (24 * nexp / math.pi) ** 0.25; G.tau0 = tau0
a = rng.uniform(0, 500, 4); b = a + tau0 * G.boost_vec(eta, nv)
Pp, s, Lb = points_lab(G.interval_box(a, b, eta, nv), rmax)
ins = G.inside(Pp, a, b); s, Lb = s[ins], Lb[ins]; pr = prob(s, r, w)
bern = (pr * (1 - pr)).sum() / pr.sum()
print(f'наблюдатель в общей системе, n ≈ {nexp}; у независимых монеток дисперсия/среднее = {bern:.3f}')
for N in (2, 3, 5, 8, 13, 21, 34, 89, 233, 1597, 10946):
    allseeds = N * N
    if allseeds <= 400: seeds = np.array([(i, j) for i in range(N) for j in range(N)])
    else: seeds = rng.integers(0, N, (200, 2))
    cnt = np.zeros(len(seeds))
    for c in range(0, len(seeds), 25):
        th = phase_int(Lb, seeds[c:c + 25], N); cnt[c:c + 25] = (th[..., 0] < pr[None] * N).sum(1)
    m = cnt.mean()
    print(f'N = {N:6d} (состояний фазы {N*N:>9d}): среднее/ожид. {m/pr.sum():.3f}, дисперсия/среднее {cnt.var(ddof=1)/m:.3f}  [{time.time()-t0:.0f} с]')
    sys.stdout.flush()
