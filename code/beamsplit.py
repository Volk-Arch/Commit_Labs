#!/usr/bin/env python3
"""НОРМИРОВКА И СВЕТОДЕЛИТЕЛЬ ИЗ САМОГО КРИСТАЛЛА.

Нормировка (местное правило, без подгонки): каждое событие раздаёт то, что получило, поровну
своим прямым потомкам — вес звена e^{iφ}/(число исходящих связей). Это сохраняет поток: сумма
модулей весов по исходящим связям равна 1, как у вероятности случайного блуждания, а фаза
φ = π·σ² берётся из тени. Амплитуда Z(A→B) — сумма по путям таких весов.

Светоделитель: у частицы Алисы два пути (события A₁, A₂). Порты — пара поздних событий (p₊, p₋),
до которых доходят оба пути. Матрица T[p, i] = Z(A_i → p). Настройка детектора — выбор пары
портов (разные пары дают разные T). Качество делителя u = |det T|/(½‖T‖²) (1 — идеальный).
Вероятности P(a,b|x,y) ∝ |Σ T^A_x[a,i] T^B_y[b,j] M[i,j]|² с отбором «сработал один из двух
портов»; честно мерим и CHSH, и сигнал (зависимость маргиналов Алисы от настройки Боба).
"""
import sys, math, itertools, numpy as np
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of

PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
def smax(rho):
    T = np.array([[np.trace(rho @ np.kron(a, b)).real for b in PAULI] for a in PAULI])
    s = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]; return 2 * math.sqrt(s[0] ** 2 + s[1] ** 2)

def walk_sum(w, L, src, phase, outdeg):
    n = w.n; Z = np.zeros(n, complex); Z[src] = 1
    I, J = np.nonzero(L); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    wt = np.exp(1j * phase(I, J)) / outdeg[I]
    st = np.searchsorted(J, np.arange(n)); en = np.searchsorted(J, np.arange(n), side='right')
    for j in range(src + 1, n):
        a, b = st[j], en[j]
        if a < b: Z[j] = (Z[I[a:b]] * wt[a:b]).sum()
    return Z

def nearest(X, p): return int(np.argmin(((X - p) ** 2).sum(1)))

def experiment(make, seed, k=math.pi, src_half=3.0, sig_max=0.03, rule_walk=True, strip=False):
    w = make(4.0, 16.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    outdeg = np.maximum(L.sum(1), 1).astype(float)
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    c0 = np.array([-7.0, 0.0]); tau = 7.0
    if strip: src = [j for j in range(w.n) if abs(X[j, 0] - c0[0]) <= 0.45 and abs(X[j, 1]) <= src_half]   # «вспышка накачки» в один момент
    else: src = [j for j in range(w.n) if abs(X[j, 0] - c0[0]) + abs(X[j, 1]) <= src_half]
    A = [nearest(X, c0 + tau * np.array([math.cosh(e), math.sinh(e)])) for e in (0.25, 0.75)]
    B = [nearest(X, c0 + tau * np.array([math.cosh(e), -math.sinh(e)])) for e in (0.25, 0.75)]
    Zs = {}
    for r in src:
        Zs[r] = walk_sum(w, L, r, ph, outdeg) if rule_walk else None
    za = np.array([[Zs[r][a] for a in A] for r in src]); zb = np.array([[Zs[r][b] for b in B] for r in src])
    M = np.einsum('ri,rj->ij', za, zb)
    psi = M.reshape(4) / np.linalg.norm(M); S_ideal = smax(np.outer(psi, psi.conj()))
    C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
    rho_mix = sum(np.outer(np.kron(a, b), np.kron(a, b).conj()) for a, b in zip(za, zb)); rho_mix /= np.trace(rho_mix).real
    S_mix = smax(rho_mix)
    # порты: события через ~5 после путей, в будущем обоих путей своей стороны
    ZA = [walk_sum(w, L, a, ph, outdeg) for a in A]; ZB = [walk_sum(w, L, b, ph, outdeg) for b in B]
    def ports(side, Zp, paths):
        cand = [j for j in range(w.n) if 5.0 <= X[j, 0] <= 7.0 and side * X[j, 1] > 1.0 and all(R[p, j] for p in paths)]
        pairs = []
        for p, q in itertools.combinations(cand, 2):
            T = np.array([[Zp[0][p], Zp[1][p]], [Zp[0][q], Zp[1][q]]])
            nrm = (np.abs(T) ** 2).sum()
            if nrm == 0: continue
            pairs.append((2 * abs(np.linalg.det(T)) / nrm, T))
        pairs.sort(key=lambda t: -t[0]); return pairs[:25]
    PA, PB = ports(+1, ZA, A), ports(-1, ZB, B)
    def probs(TA, TB):
        amp = TA @ M @ TB.T; P = np.abs(amp) ** 2; return P / P.sum()
    best = (0, None, None)
    for (ua0, TA0), (ua1, TA1) in itertools.combinations(PA, 2):
        for (ub0, TB0), (ub1, TB1) in itertools.combinations(PB, 2):
            P = [[probs(TA0, TB0), probs(TA0, TB1)], [probs(TA1, TB0), probs(TA1, TB1)]]
            sig = max(abs(P[0][0].sum(1) - P[0][1].sum(1)).max(), abs(P[1][0].sum(1) - P[1][1].sum(1)).max(),
                      abs(P[0][0].sum(0) - P[1][0].sum(0)).max(), abs(P[0][1].sum(0) - P[1][1].sum(0)).max())
            E = lambda p: p[0, 0] - p[0, 1] - p[1, 0] + p[1, 1]
            S = abs(E(P[0][0]) + E(P[0][1]) + E(P[1][0]) - E(P[1][1]))
            if sig <= sig_max and S > best[0]: best = (S, sig, (ua0, ua1, ub0, ub1))
    return dict(n_src=len(src), C=C, S_ideal=S_ideal, S_mix=S_mix, S_ports=best[0], sig=best[1], u=best[2],
                u_best=(PA[0][0] if PA else None, PB[0][0] if PB else None), npA=len(PA), npB=len(PB))

STRIP = '--strip' in sys.argv
if __name__ == '__main__':
    for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
        rows = []
        for seed in (31, 32, 33, 34, 35, 36):
            r = experiment(make, seed, strip=STRIP); rows.append(r)
            u = r['u']
            print(f'{name}, мир {seed}: источник {r["n_src"]}, C = {r["C"]:.3f}, S идеальный {r["S_ideal"]:.3f}, место записано {r["S_mix"]:.3f}; '
                  f'порты из кристалла: S = {r["S_ports"]:.3f} при сигнале {r["sig"] if r["sig"] is None else round(r["sig"], 4)}, '
                  f'качество делителей {None if u is None else np.round(u, 2).tolist()} (лучший {np.round(r["u_best"], 2).tolist()})'); sys.stdout.flush()
        S = np.array([r['S_ports'] for r in rows]); Si = np.array([r['S_ideal'] for r in rows])
        print(f'== {name}: S из портов кристалла {S.mean():.3f} (макс {S.max():.3f}); S идеальный {Si.mean():.3f} (макс {Si.max():.3f})\n'); sys.stdout.flush()
