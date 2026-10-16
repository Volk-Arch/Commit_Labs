"""Сравнение нормировок амплитуды для пары из «вспышки накачки»: идеальный CHSH и запутанность."""
import sys, math, numpy as np
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of, pathsum
from phaseplate import amp_sum
from beamsplit import smax, nearest
def run(make, seed, k=math.pi):
    w = make(4.0, 16.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    c0 = np.array([-7.0, 0.0]); tau = 7.0
    src = [j for j in range(w.n) if abs(X[j, 0] - c0[0]) <= 0.45 and abs(X[j, 1]) <= 3.0]
    A = [nearest(X, c0 + tau * np.array([math.cosh(e), math.sinh(e)])) for e in (0.25, 0.75)]
    B = [nearest(X, c0 + tau * np.array([math.cosh(e), -math.sinh(e)])) for e in (0.25, 0.75)]
    rows = {m: ([], []) for m in ('сумма (N⁰)', '√N (совет другой ИИ)', 'средняя фаза (N¹)', 'делёж вперёд', 'среднее прошлого')}
    for r in src:
        Z, N = pathsum(w, L, r, ph); Zs = amp_sum(w, L, r, ph, 'split'); Zp = amp_sum(w, L, r, ph, 'past')
        for m, f in (('сумма (N⁰)', lambda i: Z[i]), ('√N (совет другой ИИ)', lambda i: Z[i] / math.sqrt(max(N[i], 1))),
                     ('средняя фаза (N¹)', lambda i: Z[i] / max(N[i], 1)), ('делёж вперёд', lambda i: Zs[i]), ('среднее прошлого', lambda i: Zp[i])):
            rows[m][0].append([f(a) for a in A]); rows[m][1].append([f(b) for b in B])
    out = {}
    for m, (za, zb) in rows.items():
        za, zb = np.array(za), np.array(zb); M = np.einsum('ri,rj->ij', za, zb)
        psi = M.reshape(4) / np.linalg.norm(M); out[m] = (2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum(), smax(np.outer(psi, psi.conj())))
    return out
for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
    res = [run(make, s) for s in range(31, 43)]
    for m in res[0]:
        C = np.array([r[m][0] for r in res]); S = np.array([r[m][1] for r in res])
        print(f'{name:9s} {m:22s}: C средн. {C.mean():.3f}, S идеальный средн. {S.mean():.3f}, макс {S.max():.3f}, миров с S > 2.05: {np.sum(S > 2.05)} из {len(S)}')
    sys.stdout.flush()
