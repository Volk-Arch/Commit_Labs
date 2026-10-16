#!/usr/bin/env python3
"""СЕТЬ RENOU НА КРИСТАЛЛЕ: два независимых источника, три стороны.

Схема (Renou et al., Nature 600, 625 (2021), прил. E): источник 1 даёт пару Алисе и Бобу, источник 2 —
Бобу и Чарли. Боб делает измерение в базисе Белла (4 исхода b = b₁b₂), Алиса — 3 настройки
(в оптимуме Z, X, Y), Чарли — 6 ((Z±X)/√2, (Z±Y)/√2, (X±Y)/√2).
  T_b = (−1)^{b₂}(S11+S12) + (−1)^{b₁}(S21−S22) + (−1)^{b₂}(S13+S14) − (−1)^{b₁+b₂}(S33−S34)
        + (−1)^{b₁}(S25+S26) − (−1)^{b₁+b₂}(S35−S36),    T = Σ_b T_b.
Комплексная КМ: 6√2 = 8.485. Вещественная КМ (любая размерность): T ≤ 7.6605. Опыт 2022: 7.83 ± 0.03.

Правило «поворот = i» имеет точную вещественную структуру J (сопряжение × знак второй компоненты):
все амплитуды, делители и источники в базисе f = (e₁, −i·e₂) вещественны. Пластинки — внешние.
Считаем:
  A. проверка формулы: комплексный оптимум;
  B. вещественные кубиты — лучший T перебором;
  C. кристалл: источники и делители из правила, Боб — идеальный Белл;
     C1 «только правило» (пластинки 0 или π), C2 с внешними пластинками (любой угол).
"""
import sys, math, itertools, os, numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex)
r2 = math.sqrt(2)
BELL = {'f+': np.array([[1, 0], [0, 1]]) / r2, 'f-': np.array([[1, 0], [0, -1]]) / r2,
        'p+': np.array([[0, 1], [1, 0]]) / r2, 'p-': np.array([[0, 1], [-1, 0]]) / r2}

def S_b(psi, phi, beta, A, C):
    chi = psi @ beta.conj() @ phi
    return np.einsum('ac,ab,cd,bd->', chi.conj(), A, C, chi).real

def score(psi, phi, bobs, As, Cs):
    psi = psi / np.linalg.norm(psi); phi = phi / np.linalg.norm(phi); T = 0
    for (b1, b2), beta in zip(((0, 0), (0, 1), (1, 0), (1, 1)), bobs):
        S = lambda x, z: S_b(psi, phi, beta, As[x - 1], Cs[z - 1])
        T += ((-1) ** b2 * (S(1, 1) + S(1, 2)) + (-1) ** b1 * (S(2, 1) - S(2, 2)) + (-1) ** b2 * (S(1, 3) + S(1, 4))
              - (-1) ** (b1 + b2) * (S(3, 3) - S(3, 4)) + (-1) ** b1 * (S(2, 5) + S(2, 6)) - (-1) ** (b1 + b2) * (S(3, 5) - S(3, 6)))
    return T

def coeffs(psi, phi, bobs):
    """T = Σ_{x,z} tr-форма; возвращаем функцию T(As, Cs) и «линейные» вклады для перебора"""
    pass

A_opt = [Z, X, Y]
C_opt = [(Z + X) / r2, (Z - X) / r2, (Z + Y) / r2, (Z - Y) / r2, (X + Y) / r2, (X - Y) / r2]
phip = BELL['f+']

def best_labeling(psi, phi, As, Cs):
    best = None
    for perm in itertools.permutations(BELL.keys()):
        t = score(psi, phi, [BELL[k] for k in perm], As, Cs)
        if best is None or t > best[0]: best = (t, perm)
    return best

def local_search(psi, phi, bobs, Aset, Cset, restarts=30, seed=0):
    """T линеен по каждой наблюдаемой: подъём по координатам на конечных наборах Aset, Cset"""
    rs = np.random.default_rng(seed); best = (-1e9, None, None)
    # предвычислим S_b для всех пар (A, C) из наборов
    psi = psi / np.linalg.norm(psi); phi = phi / np.linalg.norm(phi)
    chis = [psi @ beta.conj() @ phi for beta in bobs]
    Aarr, Carr = np.array(Aset), np.array(Cset)
    SB = np.array([np.einsum('ac,iab,jcd,bd->ij', chi.conj(), Aarr, Carr, chi, optimize=True).real for chi in chis])  # [b, iA, iC]
    sg = [((-1) ** b2, (-1) ** b1, (-1) ** (b1 + b2)) for b1, b2 in ((0, 0), (0, 1), (1, 0), (1, 1))]
    # T = Σ_b Σ_terms w * SB[b, a_x, c_z]
    terms = []   # (x, z, весовой вектор по b)
    for x, z, k, sgn in ((1, 1, 0, 1), (1, 2, 0, 1), (2, 1, 1, 1), (2, 2, 1, -1), (1, 3, 0, 1), (1, 4, 0, 1),
                         (3, 3, 2, -1), (3, 4, 2, 1), (2, 5, 1, 1), (2, 6, 1, 1), (3, 5, 2, -1), (3, 6, 2, 1)):
        terms.append((x, z, np.array([sgn * s[k] for s in sg])))
    W = {}
    for x, z, wb in terms: W[(x, z)] = W.get((x, z), 0) + wb
    Wt = {k: np.einsum('b,bac->ac', v, SB) for k, v in W.items()}   # матрица вклада [iA, iC]
    def total(ia, ic): return sum(Wt[(x, z)][ia[x - 1], ic[z - 1]] for (x, z) in Wt)
    for _ in range(restarts):
        ia = list(rs.integers(0, len(Aset), 3)); ic = list(rs.integers(0, len(Cset), 6))
        cur = total(ia, ic)
        for _ in range(50):
            old = cur
            for x in (1, 2, 3):
                v = sum(Wt[(xx, z)][:, ic[z - 1]] for (xx, z) in Wt if xx == x); ia[x - 1] = int(np.argmax(v))
            for z in range(1, 7):
                v = sum(Wt[(x, zz)][ia[x - 1], :] for (x, zz) in Wt if zz == z); ic[z - 1] = int(np.argmax(v))
            cur = total(ia, ic)
            if cur <= old + 1e-12: break
        if cur > best[0]: best = (cur, ia[:], ic[:])
    return best

def real_qubit_opt(restarts=60, seed=0):
    """лучший T для вещественных кубитов: состояния R⁴, наблюдаемые в плоскости x–z, Боб — любой ортонормированный базис R⁴"""
    rs = np.random.default_rng(seed); best = -1e9
    def unpack(v):
        psi = v[0:4].reshape(2, 2); phi = v[4:8].reshape(2, 2)
        As = [math.cos(t) * Z + math.sin(t) * X for t in v[8:11]]; Cs = [math.cos(t) * Z + math.sin(t) * X for t in v[11:17]]
        Kq = np.zeros((4, 4)); Kq[np.triu_indices(4, 1)] = v[17:23]; O = expm(Kq - Kq.T)
        bobs = [O[i].reshape(2, 2).astype(complex) for i in range(4)]
        return psi.astype(complex), phi.astype(complex), bobs, As, Cs
    for r in range(restarts):
        v0 = rs.normal(size=23)
        res = minimize(lambda v: -score(*unpack(v)), v0, method='BFGS', options=dict(maxiter=2000))
        best = max(best, -res.fun)
    return best

def observables(Tlist, plates):
    """наблюдаемые Алисы/Чарли из делителей: исход +1 — порт p, −1 — порт q; пластинка α на втором пути"""
    obs = [Z]
    for T in Tlist:
        U = T / math.sqrt((np.abs(T[:, 0]) ** 2).sum())
        for a in plates:
            V = U @ np.diag([1, np.exp(1j * a)]); obs.append(V.conj().T @ Z @ V)
    return obs

if __name__ == '__main__':
    t, perm = best_labeling(phip, phip, A_opt, C_opt)
    print(f'A. комплексная КМ, φ⁺⊗φ⁺, Z X Y: T = {t:.4f} (6√2 = {6 * r2:.4f}); метки Белла b = 00,01,10,11 → {perm}')
    bobs = [BELL[k] for k in perm]
    t_real = score(phip, phip, bobs, [Z, X, Z], [(Z + X) / r2, (Z - X) / r2, Z, Z, X, X])
    print(f'   та же схема, но Y заменён вещественными (Z вместо Y): T = {t_real:.4f}')
    sys.stdout.flush()
    print(f'B. вещественные кубиты, полный перебор (BFGS, 60 стартов): T макс = {real_qubit_opt():.4f} (граница Renou 7.6605)'); sys.stdout.flush()
    # C. кристалл
    fn = '/home/claude/data/renou_crystal.npz'
    if not os.path.exists(fn):
        _a = sys.argv; sys.argv = sys.argv[:1]
        import chiral as CH
        sys.argv = _a
        K, LABK, rho = 2, 1, 4.0
        H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
        w = CH.sym_world(rho, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), LABK); Xw = w.X; n = w.n
        Xa = CH.axis_frame(Xw, -K * CH.ETA0 / 2)
        src = [j for j in range(n) if abs(Xw[j, 0] + 7) <= 0.45 and abs(Xw[j, 1]) <= 3.0]
        PS = {(s, m): CH.chiral(w, LK, R, s, m) for s in src for m in (0, 1)}
        Ms, Ts = [], []
        for A1 in [k for k in range(n) if 2.0 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.0]:
            A2 = CH.img(w, A1, H, Hs)
            if A2 < 0: continue
            B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
            for m0 in (0, 1):
                za = np.array([[PS[s, m][A1, m0], PS[s, m][A2, 1 - m0]] for s in src for m in (0, 1)])
                zb = np.array([[PS[s, m][B1, 1 - m0], PS[s, m][B2, m0]] for s in src for m in (0, 1)])
                Ms.append(za.T @ zb)
                ZA = {(x, mm): CH.chiral(w, LK, R, x, mm) for x, mm in ((A1, m0), (A2, 1 - m0))}
                for m1 in (0, 1):
                    for p in range(n):
                        if not (4.0 <= Xa[p, 0] <= 6.5 and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
                        q = CH.img(w, p, H, Hs)
                        if q < 0: continue
                        T = np.array([[ZA[A1, m0][p, m1], ZA[A2, 1 - m0][p, m1]], [ZA[A1, m0][q, 1 - m1], ZA[A2, 1 - m0][q, 1 - m1]]])
                        if (np.abs(T) ** 2).sum() > 1e-24: Ts.append(T)
        np.savez(fn, Ms=np.array(Ms), Ts=np.array(Ts))
    d = np.load(fn); Ms, Ts = d['Ms'], d['Ts']
    # в базис f = (e₁, −i e₂): всё вещественно с точностью до общей фазы
    D = np.diag([1, -1j])
    def realify(Mx):
        Mf = D @ Mx @ D.T; k = np.argmax(np.abs(Mf)); Mf = Mf * np.exp(-1j * np.angle(Mf.flat[k])); return Mf
    Mf = [realify(M) for M in Ms]; Tf = [realify(T) for T in Ts]
    im = max(max(np.abs(M.imag).max() / np.abs(M).max() for M in Mf), max(np.abs(T.imag).max() / np.abs(T).max() for T in Tf))
    print(f'C. кристалл ρ = 4: состояний {len(Mf)}, делителей {len(Tf)}; в базисе f мнимая часть ≤ {im:.1e}')
    conc = np.array([2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum() for M in Mf])
    # уникальные делители (по углу поворота)
    ang = np.array([math.atan2(T[1, 0].real, T[0, 0].real) for T in Tf])
    print(f'   углы делителей (вещественный поворот), градусы: {np.round(np.unique(np.round(np.degrees(ang) % 180, 1)), 1).tolist()[:40]}')
    Tuniq = [Tf[i] for i in np.unique(np.round(np.degrees(ang) % 180 / 3), return_index=True)[1]]   # по одному на каждые 3°
    print(f'   берём по одному делителю на 3° угла: {len(Tuniq)}')
    bobs_r = [BELL[k] for k in perm]
    obs_rule = observables(Tuniq, [0, math.pi])
    obs_cplx = observables(Tuniq, np.linspace(0, 2 * math.pi, 24, endpoint=False))
    print(f'   наблюдаемых: по правилу {len(obs_rule)}, с пластинками {len(obs_cplx)}'); sys.stdout.flush()
    order = np.argsort(-conc)
    rows = []
    for i in list(order[:6]) + [None]:
        M = phip if i is None else Mf[i]; name = 'идеальная φ⁺' if i is None else f'C = {conc[i]:.3f}'
        r_rule = max(local_search(M, M, [BELL[k] for k in pm], obs_rule, obs_rule, restarts=10)[0] for pm in itertools.permutations(BELL))
        r_cplx = max(local_search(M, M, [BELL[k] for k in pm], obs_cplx, obs_cplx, restarts=10)[0] for pm in itertools.permutations(BELL))
        # Боб — вещественный базис Белла с перестановкой меток (лучшая)
        print(f'   источник {name}: T только по правилу (пластинки 0/π) = {r_rule:.4f}; с внешними пластинками = {r_cplx:.4f}')
        sys.stdout.flush()
