#!/usr/bin/env python3
"""ДЕЛИТЕЛЬ ИЗ ТОЧНОЙ СИММЕТРИИ: ОТРАЖЕНИЕ × БУСТ.

Кристалл с окном в центре точно симметричен относительно P: x → −x и буста g (быстрота η₀,
cosh η₀ = τ). Буст сам не зеркало, а h = P·g — зеркало (h² = 1): отражение относительно мировой
линии, движущейся со скоростью −tanh(η₀/2) = −0.486. В тени h — отражение круга, окно на месте.

Алиса стоит на оси h (наклон −η₀/2), Боб — её точное отражение P (ось +η₀/2).
Делитель Алисы: пути A₁, A₂ = h(A₁), порты p, q = h(p) ⇒ T = [[a, b], [b, a]].
Делитель Боба = P(делитель Алисы) ⇒ T_Б = T_А точно. Источник — полоса при t = −7, симметрична P.
Без потерь [[a, b], [b, a]] ⇔ фазы a и b отличаются на 90° (|a| = |b| не нужно).

Нормировки: raw (только фаза; точна, т.к. все пути A→p лежат внутри интервала) и past
(среднее по прямым предкам; прямые предки могут лежать далеко вдоль светового конуса,
у края области — там h не точна, ошибку меряем).
"""
import sys, math, numpy as np
from scipy.optimize import minimize
from bentcut import TAU
from universe import build_order
from phasesum import links_of
from beamsplit import smax
_a = sys.argv; sys.argv = sys.argv[:1]
from mirrorsplit import sym_world, find
sys.argv = _a

sT = math.sqrt(TAU)
G = np.array([[TAU, sT], [sT, TAU]])                      # буст в мире (T, x)
Gs = np.array([[-1 / TAU, 1 / sT], [-1 / sT, -1 / TAU]])  # он же в тени: поворот на 128.17°
P = np.diag([1.0, -1.0])
H, Hs = P @ G, P @ Gs                                     # зеркало h = P·g
assert np.allclose(H @ H, np.eye(2)) and np.allclose(Hs @ Hs, np.eye(2))
BETA = math.acosh(TAU) / 2                                 # ось Алисы: быстрота −β

def amp(w, L, src, phase, norm):
    n = w.n; Z = np.zeros(n, complex); Z[src] = 1
    I, J = np.nonzero(L); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    wt = np.exp(1j * phase(I, J))
    if norm.startswith('jsum'):   # сумма историй: κ·e^{i m s}, s — собственная длина звена; norm = 'jsum:m:κ'
        _, m_, k_ = norm.split(':'); d = w.X[J] - w.X[I]
        wt = float(k_) * np.exp(1j * float(m_) * np.sqrt(d[:, 0] ** 2 - d[:, 1] ** 2))
    if norm == 'past': wt = wt / np.maximum(L.sum(0), 1)[J]
    if norm == 'lit':   # среднее по тем прямым предкам, до которых дошла история из src (внутри интервала)
        R_ = getattr(w, '_R', None); lit = (I == src) | R_[src, I]
        wt = np.where(lit, wt / np.maximum(np.bincount(J[lit], minlength=n), 1)[J], 0)
    st = np.searchsorted(J, np.arange(n)); en = np.searchsorted(J, np.arange(n), side='right')
    for j in range(src + 1, n):
        a, b = st[j], en[j]
        if a < b: Z[j] = (Z[I[a:b]] * wt[a:b]).sum()
    return Z

def img(w, k, M, Ms): return find(w, M @ w.X[k], Ms @ w.W[k])
def axis_frame(X, b):      # координаты в системе, где ось с быстротой b вертикальна
    c, s = math.cosh(b), math.sinh(b)
    return np.stack([c * X[:, 0] - s * X[:, 1], -s * X[:, 0] + c * X[:, 1]], 1)

def bell(TA, TB, M, restarts=40, tol=0.005):
    def Pr(al, be):
        amp_ = (TA @ np.diag([1, np.exp(1j * al)])) @ M @ (TB @ np.diag([1, np.exp(1j * be)])).T
        pp = np.abs(amp_) ** 2; return pp / pp.sum()
    E = lambda pp: pp[0, 0] - pp[0, 1] - pp[1, 0] + pp[1, 1]
    def score(x):
        Ps = [[Pr(x[0], x[2]), Pr(x[0], x[3])], [Pr(x[1], x[2]), Pr(x[1], x[3])]]
        sig = max(abs(Ps[0][0].sum(1) - Ps[0][1].sum(1)).max(), abs(Ps[1][0].sum(1) - Ps[1][1].sum(1)).max(),
                  abs(Ps[0][0].sum(0) - Ps[1][0].sum(0)).max(), abs(Ps[0][1].sum(0) - Ps[1][1].sum(0)).max())
        return abs(E(Ps[0][0]) + E(Ps[0][1]) + E(Ps[1][0]) - E(Ps[1][1])), sig
    rs = np.random.default_rng(1); best = (0, 1)
    for _ in range(restarts):
        res = minimize(lambda x: -score(x)[0] + 60 * max(0, score(x)[1] - tol), rs.uniform(0, 2 * math.pi, 4),
                       method='Nelder-Mead', options=dict(maxiter=3000))
        S, sig = score(res.x)
        if sig <= tol and S > best[0]: best = (S, sig)
    return best

def grid(TA, TB, M, N=90, tols=(0.005, 0.001)):
    """CHSH перебором углов пластинок на сетке; лучший S при сигнале ≤ tol"""
    a=np.linspace(0,2*np.pi,N,endpoint=False)
    DA=np.einsum('ij,nj->nij',TA,np.stack([np.ones(N),np.exp(1j*a)],1))   # TA·diag(1,e^{ia})
    amp_=np.einsum('nij,jk,mlk->nmil',DA,M,DA if TB is TA else np.einsum('ij,nj->nij',TB,np.stack([np.ones(N),np.exp(1j*a)],1)))
    Pp=np.abs(amp_)**2; Pp/=Pp.sum((2,3),keepdims=True)
    E=Pp[...,0,0]-Pp[...,0,1]-Pp[...,1,0]+Pp[...,1,1]; mA=Pp[...,0,:].sum(-1); mB=Pp[...,:,0].sum(-1)
    out={}
    for tol in tols:
        best=(0,None)
        for i0 in range(N):
            # S = E[i0,j0]+E[i0,j1]+E[i1,j0]-E[i1,j1]
            S=E[i0][None,:,None]+E[i0][None,None,:]+E[:,:,None]-E[:,None,:]          # [i1,j0,j1]
            sig=np.maximum(np.maximum(np.abs(mA[i0][None,:,None]-mA[i0][None,None,:]), np.abs(mA[:,:,None]-mA[:,None,:])),
                np.maximum(np.abs(mB[i0][None,:,None]-mB[:,:,None]), np.abs(mB[i0][None,None,:]-mB[:,None,:])))
            Sm=np.where(sig<=tol,np.abs(S),0); k=Sm.argmax()
            if Sm.flat[k]>best[0]: best=(Sm.flat[k],sig.flat[k])
        out[tol]=best
    return out

if __name__ == '__main__':
    NORM = sys.argv[1] if len(sys.argv) > 1 else 'past'
    BAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    PLO = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0; PHI = PLO + 2.5
    rho, Lr = 4.0, float(sys.argv[3]) if len(sys.argv) > 3 else 18.0
    w = sym_world(rho, Lr); R = build_order(w, 'proj'); Lk = links_of(R); X = w.X; n = w.n
    ph = lambda I, J: math.pi * ((w.W[J] - w.W[I]) ** 2).sum(1)
    # 1. точность симметрий
    inside = np.abs(X).sum(1) <= Lr - 1e-9
    hx = X @ H.T; ok = np.abs(hx).sum(1) <= Lr - 1e-6
    missP = sum(img(w, k, P, P) < 0 for k in range(n))
    missH = sum(img(w, k, H, Hs) < 0 for k in np.nonzero(ok)[0])
    print(f'мир {NORM}: событий {n}; без двойника при P: {missP}; при h = P·g (образ внутри области, {ok.sum()} соб.): {missH}')
    src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
    Zsrc = [amp(w, Lk, s, ph, NORM) for s in src]
    Xa = axis_frame(X, -BETA)
    candA = [k for k in range(n) if 1.5 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.2]
    rows = []
    for A1 in candA:
        A2 = img(w, A1, H, Hs)
        if A2 < 0 or not any(R[s, A1] for s in src) or not any(R[s, A2] for s in src): continue
        B1, B2 = img(w, A1, P, P), img(w, A2, P, P)
        za = np.array([[Z[A1], Z[A2]] for Z in Zsrc]); zb = np.array([[Z[B1], Z[B2]] for Z in Zsrc])
        M = np.einsum('ri,rj->ij', za, zb)
        if np.linalg.norm(M) == 0: continue
        psi = M.reshape(4) / np.linalg.norm(M); S_id = smax(np.outer(psi, psi.conj()))
        C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
        Z1, Z2 = amp(w, Lk, A1, ph, NORM), amp(w, Lk, A2, ph, NORM)
        best = None
        for p in range(n):
            if not (PLO <= Xa[p, 0] <= PHI and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
            q = img(w, p, H, Hs)
            if q < 0: continue
            T = np.array([[Z1[p], Z2[p]], [Z1[q], Z2[q]]]); nrm = (np.abs(T) ** 2).sum()
            if nrm == 0: continue
            se = (abs(T[0, 0] - T[1, 1]) + abs(T[0, 1] - T[1, 0])) / math.sqrt(nrm)
            u = 2 * abs(np.linalg.det(T)) / nrm
            bal = 2 * abs(T[0, 0]) * abs(T[0, 1]) / (abs(T[0, 0]) ** 2 + abs(T[0, 1]) ** 2)
            key = u * (bal >= BAL) + 1e-3 * bal
            if best is None or key > best[0]: best = (key, u, bal, se, T, p, q)
        if best is None: continue
        _, u, bal, se, TA, p, q = best
        Pp, Pq = img(w, p, P, P), img(w, q, P, P)
        TB = np.array([[Z1[Pp] * 0, 0], [0, 0]])  # заполним ниже
        ZB1, ZB2 = amp(w, Lk, B1, ph, NORM), amp(w, Lk, B2, ph, NORM)
        TB = np.array([[ZB1[Pp], ZB2[Pp]], [ZB1[Pq], ZB2[Pq]]])
        dPB = np.abs(TB - TA).sum() / np.abs(TA).sum()
        A_ev, B_ev = [A1, A2, p, q], [B1, B2, Pp, Pq]
        caus = sum(R[a, b] or R[b, a] for a in A_ev for b in B_ev)
        rows.append((u, bal, se, dPB, caus, S_id, C, TA, TB, M, A1, A2))
    rows.sort(key=lambda r: -(r[0] + 0.3 * r[1] - (r[4] > 0)))
    print(f'пар путей Алисы: {len(rows)}')
    for (u, bal, se, dPB, caus, S_id, C, TA, TB, M, A1, A2) in rows[:8]:
        S, sig = bell(TA, TB, M)
        dph = math.degrees(abs(np.angle(TA[0, 0] * np.conj(TA[0, 1]))))
        print(f'A1 ({Xa[A1,0]:.2f},{Xa[A1,1]:+.2f}): C {C:.3f}, S ид. {S_id:.3f} | делитель: качество {u:.5f}, баланс {bal:.3f}, '
              f'разность фаз {dph:.1f}°, ошибка h {se:.1e}, Боб≠Алиса {dPB:.1e}, причинных связей А–Б {caus} | S {S:.3f} (сигнал {sig:.4f})')
        sys.stdout.flush()
