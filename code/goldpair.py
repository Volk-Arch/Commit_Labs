#!/usr/bin/env python3
"""ПАРА ИЗ ОБЩЕГО ПРОШЛОГО: рождает ли кристалл запутанные пары?

Пара рождается в неизвестном событии r малой области-источника. Одна частица уходит вправо (Алиса),
другая влево (Боб). У каждой два пути — события A₁, A₂ (B₁, B₂) на одном собственном расстоянии от
источника, но под разными углами. Амплитуда: M[i, j] = Σ_r Z(r→A_i)·Z(r→B_j), Z — сумма по путям
с фазой звена e^{ik·σ²} (тень). Если место рождения не записано, суммируем по r КОГЕРЕНТНО
(как в квантовой механике); если записано — смесь по r (классика).
Мера: наибольший CHSH при идеальных измерениях — формула Хородецких S = 2√(t₁² + t₂²).
Чистое состояние: S = 2√(1 + C²), C — запутанность (concurrence). Смесь произведений: S ≤ 2 всегда.
"""
import sys, math, numpy as np
from universe import golden_flat, poisson_flat, build_order
from phasesum import links_of, pathsum
NORM = 0.0   # 1.0 — амплитуда на один путь (средняя фаза), без экспоненты числа путей
PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
def smax(rho):
    T = np.array([[np.trace(rho @ np.kron(a, b)).real for b in PAULI] for a in PAULI])
    s = np.sort(np.linalg.svd(T, compute_uv=False))[::-1]; return 2 * math.sqrt(s[0] ** 2 + s[1] ** 2)
def nearest(X, p): return int(np.argmin(((X - p) ** 2).sum(1)))

def run(make, k, seed, tau=7.0, etas=(0.25, 0.75), src_half=1.2):
    w = make(4.0, 14.0, np.random.default_rng(seed)); R = build_order(w, 'proj'); L = links_of(R); X = w.X
    ph = lambda I, J: k * ((w.W[J] - w.W[I]) ** 2).sum(1)
    src = [j for j in range(w.n) if abs(X[j, 0] + 6) + abs(X[j, 1]) <= src_half]
    c0 = np.array([-6.0, 0.0])
    A = [nearest(X, c0 + tau * np.array([math.cosh(e), math.sinh(e)])) for e in etas]
    B = [nearest(X, c0 + tau * np.array([math.cosh(e), -math.sinh(e)])) for e in etas]
    za, zb = [], []
    for r in src:
        Z, N = pathsum(w, L, r, ph)
        za.append([Z[a] / max(N[a], 1) ** NORM for a in A]); zb.append([Z[b] / max(N[b], 1) ** NORM for b in B])
    za, zb = np.array(za), np.array(zb)                       # (r, 2)
    # нормируем каждый путь по его средней силе, чтобы пути были на равных
    na = np.sqrt((np.abs(za) ** 2).mean(0)); nb = np.sqrt((np.abs(zb) ** 2).mean(0))
    za, zb = za / na, zb / nb
    M = np.einsum('ri,rj->ij', za, zb); psi = M.reshape(4) / np.linalg.norm(M)
    rho_coh = np.outer(psi, psi.conj())
    C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
    rho_mix = sum(np.outer(np.kron(a, b), np.kron(a, b).conj()) for a, b in zip(za, zb)); rho_mix /= np.trace(rho_mix).real
    return len(src), C, smax(rho_coh), smax(rho_mix)

if __name__ == '__main__':
    for name, make in (('кристалл', golden_flat), ('засев', poisson_flat)):
        for k in (math.pi, 2 * math.pi * 1.618034, 0.0):
            res = [run(make, k, s) for s in (21, 22, 23, 24)]
            ns = [r[0] for r in res]
            print(f'{name}, k = {k:.3f}: источник {ns} событий; запутанность C = {np.round([r[1] for r in res], 3).tolist()}; '
                  f'CHSH когерентно {np.round([r[2] for r in res], 3).tolist()}; место рождения записано (смесь) {np.round([r[3] for r in res], 3).tolist()}')
            sys.stdout.flush()
