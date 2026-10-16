#!/usr/bin/env python3
"""Делитель из точного зеркала h_k = P·g^k (ось — мировая линия с быстротой −kη₀/2).
Алиса на оси h_k, Боб — её отражение P (ось +kη₀/2). Требуем: установки Алисы и Боба не связаны
причинно ни одним событием (пути, порты). Для каждой пары путей перебираем порты, считаем Белла
с пластинками (сигнал ≤ 0.5 %) и «предел конуса» — S с ближайшим унитарным делителем.
Аргументы: нормировка, k, размер области L, начало окна портов (по собственному времени оси)."""
import sys, math, numpy as np
from boostsplit import grid, amp, img, axis_frame, bell, G, Gs, P, TAU, smax, build_order, links_of
_a = sys.argv; sys.argv = sys.argv[:1]
from mirrorsplit import sym_world
sys.argv = _a

def unitary(T):
    U, s, Vh = np.linalg.svd(T); return U @ Vh

NORM = sys.argv[1] if len(sys.argv) > 1 else 'past'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
Lr = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
PLO = float(sys.argv[4]) if len(sys.argv) > 4 else 4.5
Gk, Gks = np.linalg.matrix_power(G, K), np.linalg.matrix_power(Gs, K)
H, Hs = P @ Gk, P @ Gks
assert np.allclose(H @ H, np.eye(2)) and np.allclose(Hs @ Hs, np.eye(2))
beta = K * math.acosh(TAU) / 2
RHO = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
w = sym_world(RHO, Lr); R = build_order(w, 'proj'); w._R = R; Lk = links_of(R); X = w.X; n = w.n
ph = lambda I, J: math.pi * ((w.W[J] - w.W[I]) ** 2).sum(1)
ok = np.abs(X @ H.T).sum(1) <= Lr - 1e-6
print(f'{NORM}, k = {K} (ось {math.tanh(beta):.3f} c), L = {Lr}: событий {n}; без двойника при h: '
      f'{sum(img(w, k, H, Hs) < 0 for k in np.nonzero(ok)[0])} из {ok.sum()}'); sys.stdout.flush()
src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
Zsrc = [amp(w, Lk, s, ph, NORM) for s in src]
Xa = axis_frame(X, -beta)
candA = [k for k in range(n) if 2.0 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.0]
summary = []
for A1 in candA:
    A2 = img(w, A1, H, Hs)
    if A2 < 0: continue
    B1, B2 = img(w, A1, P, P), img(w, A2, P, P)
    if any(R[a, b] or R[b, a] for a in (A1, A2) for b in (B1, B2)): continue
    za = np.array([[Z[A1], Z[A2]] for Z in Zsrc]); zb = np.array([[Z[B1], Z[B2]] for Z in Zsrc])
    M = np.einsum('ri,rj->ij', za, zb)
    if np.linalg.norm(M) == 0: continue
    psi = M.reshape(4) / np.linalg.norm(M); S_id = smax(np.outer(psi, psi.conj()))
    C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
    Z1, Z2 = amp(w, Lk, A1, ph, NORM), amp(w, Lk, A2, ph, NORM)
    ZB1, ZB2 = amp(w, Lk, B1, ph, NORM), amp(w, Lk, B2, ph, NORM)
    cands = []
    for p in range(n):
        if not (PLO <= Xa[p, 0] <= PLO + 2.5 and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
        q = img(w, p, H, Hs)
        if q < 0: continue
        Pp, Pq = img(w, p, P, P), img(w, q, P, P)
        if any(R[a, b] or R[b, a] for a in (A1, A2, p, q) for b in (B1, B2, Pp, Pq)): continue
        T = np.array([[Z1[p], Z2[p]], [Z1[q], Z2[q]]]); nrm = (np.abs(T) ** 2).sum()
        if nrm == 0: continue
        TB = np.array([[ZB1[Pp], ZB2[Pp]], [ZB1[Pq], ZB2[Pq]]])
        u = 2 * abs(np.linalg.det(T)) / nrm
        bal = 2 * abs(T[0, 0]) * abs(T[0, 1]) / (abs(T[0, 0]) ** 2 + abs(T[0, 1]) ** 2)
        se = (abs(T[0, 0] - T[1, 1]) + abs(T[0, 1] - T[1, 0])) / math.sqrt(nrm)
        cands.append((u, bal, se, T, TB))
    if not cands: continue
    cands.sort(key=lambda c: -(c[0] - 2 * max(0, 0.5 - c[1])))
    best = None
    for (u, bal, se, T, TB) in cands[:6]:
        g = grid(T, TB, M); gu = grid(unitary(T), unitary(TB), M, tols=(1e-9,))
        if best is None or g[0.001][0] > best[0]: best = (g[0.001][0], g[0.005][0], gu[1e-9][0], u, bal, se)
    S1, S5, Sc, u, bal, se = best
    summary.append((S1, S_id, Sc, S5))
    print(f'A1 ({Xa[A1,0]:.2f},{Xa[A1,1]:+.2f}): C {C:.3f}, S ид. {S_id:.3f}; портов {len(cands)}; лучший делитель: качество {u:.5f}, '
          f'баланс {bal:.2f}, ошибка h {se:.0e} → S {S1:.3f} при сигнале ≤0.1 % ({S5:.3f} при ≤0.5 %); с унит. делителем {Sc:.3f}')
    sys.stdout.flush()
if summary:
    s = np.array(summary); print(f'== S (сигнал ≤0.1 %): макс {s[:,0].max():.3f}, число S > 2: {(s[:,0] > 2.0).sum()} из {len(s)}, из них S ≤ S ид.: {((s[:,0] > 2) & (s[:,0] <= s[:,1] + 1e-6)).sum()}; S ид. макс {s[:,1].max():.3f}; с унит. делителем макс {s[:,2].max():.3f}')
