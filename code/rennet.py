#!/usr/bin/env python3
"""RENOU С ДВУМЯ НАСТОЯЩИМИ ИСТОЧНИКАМИ (вместо «второй копии руками»).

Две независимые нечётные накачки на оси в разное время (t ≈ −11 и t ≈ −8), среда — весь мир, вершина — бозон.
Источник k даёт оператор пары K_k = ½ Σ G_k[i,j] a_i† a_j† по модам захвата (G_k = Σ_s p_k(s) βᵀ(J − I)β).
Сеть: Алиса — сторона A четвёрки q₁, Боб — стороны B четвёрок q₁ и q₂, Чарли — сторона A четвёрки q₂.
Отбор «по частице у Алисы, у Чарли и в каждом кубите Боба» (захват до настроек). Четыре частицы приходят:
  • по паре от каждого источника — все 6 способов разложить четвёрку на пары (включая «не те» пары) — когерентно;
  • две пары от одного источника (ψ₁₁, ψ₂₂) — некогерентно с первым (фазы независимых накачек) → примесь.
ρ = |ψ₁₂⟩⟨ψ₁₂| + |ψ₁₁⟩⟨ψ₁₁| + |ψ₂₂⟩⟨ψ₂₂|. Боб — идеальный Белл (руками), приборы Алисы и Чарли — события кристалла.
Сравниваем с «копией руками» (только задуманный член)."""
import sys, math, itertools, numpy as np
_argv = sys.argv[1:]
src = open('/home/claude/code/rensym.py').read().split('\nfor B in ')[0]
sys.argv = ['rensym.py']; exec(src)
from renou import BELL

def odd_beam(t0): return {j: np.sign(X[j, 1]) for j in range(n) if abs(X[j, 0] - t0) <= 0.45 and abs(X[j, 1]) <= 3.0 and X[j, 1] != 0}
def Gmat(p, beta):
    G = np.zeros((beta.shape[1],) * 2, complex)
    for s in range(n):
        lo = S.legs[s][1]
        if len(lo) < 2 or p[s] == 0: continue
        b = beta[lo]; u = b.sum(0); G += p[s] * (np.outer(u, u) - b.T @ b)
    return G

def T_rho(rho, As, Cs, restarts=8, seed=0):
    """Renou T для общего ρ на (Алиса, Боб₁, Боб₂, Чарли); Боб — базис Белла, перебор разметок, подъём по координатам"""
    rs = np.random.default_rng(seed); r4 = rho.reshape([2] * 8)
    Aarr, Carr = np.array(As), np.array(Cs)
    best = -1e9
    vecs = {k: v.reshape(4) for k, v in BELL.items()}
    PB = {k: np.outer(v, v.conj()).reshape(2, 2, 2, 2) for k, v in vecs.items()}
    # SB[b][iA, iC] = Tr ρ (A ⊗ Π_b ⊗ C)
    SBk = {k: np.einsum('abcdefgh,iea,fgbc,jhd->ij', r4, Aarr, PB[k], Carr, optimize=True).real for k in BELL}
    terms = ((1, 1, 0, 1), (1, 2, 0, 1), (2, 1, 1, 1), (2, 2, 1, -1), (1, 3, 0, 1), (1, 4, 0, 1),
             (3, 3, 2, -1), (3, 4, 2, 1), (2, 5, 1, 1), (2, 6, 1, 1), (3, 5, 2, -1), (3, 6, 2, 1))
    for pm in itertools.permutations(BELL):
        sg = [((-1) ** b2, (-1) ** b1, (-1) ** (b1 + b2)) for b1, b2 in ((0, 0), (0, 1), (1, 0), (1, 1))]
        W = {}
        for x, z, k, sgn in terms: W[(x, z)] = W.get((x, z), 0) + np.array([sgn * s_[k] for s_ in sg])
        Wt = {key: sum(v[i] * SBk[pm[i]] for i in range(4)) for key, v in W.items()}
        def total(ia, ic): return sum(Wt[(x, z)][ia[x - 1], ic[z - 1]] for (x, z) in Wt)
        for _ in range(restarts):
            ia = list(rs.integers(0, len(As), 3)); ic = list(rs.integers(0, len(Cs), 6)); cur = total(ia, ic)
            for _ in range(50):
                old = cur
                for x in (1, 2, 3): ia[x - 1] = int(np.argmax(sum(Wt[(xx, z)][:, ic[z - 1]] for (xx, z) in Wt if xx == x)))
                for z in range(1, 7): ic[z - 1] = int(np.argmax(sum(Wt[(x, zz)][ia[x - 1], :] for (x, zz) in Wt if zz == z)))
                cur = total(ia, ic)
                if cur <= old + 1e-12: break
            best = max(best, cur)
    return best

for B in [float(b) for b in (_argv or ['0', '0.3'])]:
    S.set_field(2 * B)
    pk = []
    for t0 in (-11.0, -8.0):
        bm = odd_beam(t0); _, _, _, r = S.run({k: np.array([v / math.sqrt(len(bm))]) for k, v in bm.items()}, record=set(range(n)))
        pk.append(np.array([r[k][0] for k in range(n)]))
    S.set_field(B)
    beta, emit, col = S.backward(rec)
    G1, G2 = Gmat(pk[0], beta), Gmat(pk[1], beta)
    def state(q1, q2):
        modes = [[col[q1[0]], col[q1[1]]], [col[q1[2]], col[q1[3]]], [col[q2[2]], col[q2[3]]], [col[q2[0]], col[q2[1]]]]
        psi12 = np.zeros((2,) * 4, complex); psi11 = np.zeros_like(psi12); psi22 = np.zeros_like(psi12); want = np.zeros_like(psi12)
        match = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
        for idx in itertools.product((0, 1), repeat=4):
            x = [modes[k][idx[k]] for k in range(4)]
            for (a, b), (c, d) in match:
                psi12[idx] += G1[x[a], x[b]] * G2[x[c], x[d]] + G2[x[a], x[b]] * G1[x[c], x[d]]
                psi11[idx] += G1[x[a], x[b]] * G1[x[c], x[d]]; psi22[idx] += G2[x[a], x[b]] * G2[x[c], x[d]]
            want[idx] = G1[x[0], x[1]] * G2[x[2], x[3]]
        v12, v11, v22 = psi12.reshape(16), psi11.reshape(16), psi22.reshape(16)
        rho = np.outer(v12, v12.conj()) + np.outer(v11, v11.conj()) + np.outer(v22, v22.conj())
        tr = np.trace(rho).real; wv = want.reshape(16)
        fid = abs(np.vdot(wv / np.linalg.norm(wv), rho @ (wv / np.linalg.norm(wv)))) / tr
        return rho / tr, np.outer(wv, wv.conj()) / np.vdot(wv, wv).real, fid, np.vdot(v12, v12).real / tr
    cands = []
    for q1 in quads:
        for q2 in quads:
            if len({*q1, *q2}) < 8: continue
            rho, rid, fid, w12 = state(q1, q2)
            cands.append((fid, w12, q1, q2, rho, rid))
    cands.sort(key=lambda c: -c[0])
    print(f'B = {B}: пар четвёрок {len(cands)}; доля «задуманного» состояния (верность): лучшая {cands[0][0]:.3f}, медиана {np.median([c[0] for c in cands]):.3f}')
    done = 0
    for fid, w12, q1, q2, rho, rid in cands:
        if done >= 3: break
        As, _ = devices2(q1[0], q1[1], B, nmax=NMAX); Cs, _ = devices2(q2[0], q2[1], B, nmax=NMAX)
        if len(As) < 3 or len(Cs) < 3: continue
        Tr_ = T_rho(rho, As, Cs); Ti = T_rho(rid, As, Cs); done += 1
        print(f'  q₁ A₁ ({Xa[q1[0], 0]:.2f}, {Xa[q1[0], 1]:+.2f}), q₂ A₁ ({Xa[q2[0], 0]:.2f}, {Xa[q2[0], 1]:+.2f}): верность {fid:.3f} '
              f'(«по паре от каждого» {w12:.3f}); T два настоящих источника = {Tr_:.3f}; T «копия руками» = {Ti:.3f}; настроек {len(As)}/{len(Cs)}')
        sys.stdout.flush()
