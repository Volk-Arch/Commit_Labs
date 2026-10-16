#!/usr/bin/env python3
"""СЕТЬ RENOU БЕЗ ДВУХ ПОДПОРОК: два настоящих источника, развязанных причинностью, и Боб через слияние (обратный распад).

Геометрия (лаб. система; всё зеркально по P):
  источник 1 — накачка (заряд 2q) слева x ≈ −6, среда M₁ — полоса t ∈ [−11, −9], x ∈ [−7, −5]; источник 2 = P(источник 1);
  Алиса — события (A₁, A₂) слева x ∈ [−11, −8.5]; кубит 1 Боба — (Q₁, Q₂) у центра слева; Чарли = P(A), кубит 2 Боба = P(Q).
  РАЗВЯЗКА: Q и A лежат вне будущего среды M₂ (и зеркально) — пары «не туда» и двойные пары от одного источника дают
  ровно 0 (проверяется). Накачка поглощается за своей средой (ловушка пучка).
Боб: оба его кубита (захват в симм. моду в Q и P(Q)) снова выпускаются в мир; в среде слияния M_f на оси два кванта на
  входных ногах a ≠ b события s сливаются в один (заряд 2q) — обращение вершины распада; слитый квант ловит детектор
  нечётной моды на оси (обращение нечётной накачки). Проектор слияния Φ[i, j] = 2 Σ_s d(s) (u_i u'_j − Σ_a r_i(a) r'_j(a)).
  Без поля и при точном зеркале Φ антисимметрична ⇒ ровно синглет. Ещё три исхода Белла — проводка кубита 2 перед
  слиянием (перекрест световодов = X, фаза π = Z) — локальные операции, руками. Отбор по успеху слияния несмещённый,
  если четыре проектора ортонормальны (тогда вероятность успеха не зависит от состояния).
Алиса и Чарли — приборы из событий кристалла (devices2) с полем; пластинок нет."""
import sys, math, itertools, numpy as np
_argv = sys.argv[1:]
src = open('/home/claude/code/rensym.py').read().split('\nfor B in ')[0]
sys.argv = ['rensym.py']; exec(src)
from renou import local_search
P_ = S.Pimg
import os
XL = float(os.environ.get('XL', '0'))      # поле только в лабораториях Алисы и Чарли: |x| > XL на обоих концах ноги (0 — однородное)
_fph = fph
def fph(a, b, Bq): return _fph(a, b, Bq) if (XL == 0 or min(abs(X[a, 1]), abs(X[b, 1])) > XL) else 1.0
_set = S.set_field
def set_field_local(Bq):
    _set(Bq)
    if XL and Bq:
        ea = np.array([e[0] for e in S.edges]); eb = np.array([e[1] for e in S.edges])
        m = np.minimum(np.abs(X[ea, 1]), np.abs(X[eb, 1])) > XL; S.f = np.where(m, S.f, 1.0)
S.set_field = set_field_local
box = lambda t0, t1, x0, x1: [j for j in range(n) if t0 <= X[j, 0] <= t1 and x0 <= X[j, 1] <= x1]
M1 = box(-11, -9, -7, -5); M2 = [int(P_[j]) for j in M1]
pump1 = {j: np.sign(X[j, 1] + 6) for j in box(-13.45, -12.55, -7.5, -4.5) if X[j, 1] != -6}
reach2 = R[np.array(M2)].any(0)                       # будущее среды M₂
candA = [j for j in box(-8, -5.5, -11, -8.5) if not reach2[j] and R[np.array(M1)][:, j].any()]
candQ = [j for j in box(-8, -5.5, -2.0, -0.5) if not reach2[j] and R[np.array(M1)][:, j].any()]
Mf = box(-3.5, -1, -1.5, 1.5)
det = {j: np.sign(X[j, 1]) for j in box(1.55, 2.45, -2.5, 2.5) if X[j, 1] != 0}
print(f'событий: M₁ {len(M1)}, накачка {len(pump1)}, кандидаты Алисы {len(candA)}, кубита Боба {len(candQ)}, среда слияния {len(Mf)}, детектор {len(det)}')

def pump_p(Bq, inj, medium):
    S.set_field(2 * Bq); _, _, _, r = S.run({k: np.array([v]) for k, v in inj.items()}, record=set(medium))
    return {s: r[s][0] for s in medium}

for B in [float(b) for b in (_argv or ['0', '0.3'])]:
    p1 = pump_p(B, {k: v / math.sqrt(len(pump1)) for k, v in pump1.items()}, M1)
    p2 = pump_p(B, {int(P_[k]): v / math.sqrt(len(pump1)) for k, v in pump1.items()}, M2)   # зеркальная накачка (тот же знак-узор, отражённый)
    S.set_field(B)
    capt = sorted(set(candA + candQ + [int(P_[j]) for j in candA + candQ]))
    beta, emit, col = S.backward(capt)
    def G(p):
        g = np.zeros((len(capt),) * 2, complex)
        for s, ps in p.items():
            lo = S.legs[s][1]
            if len(lo) < 2: continue
            b = beta[lo]; u = b.sum(0); g += ps * (np.outer(u, u) - b.T @ b)
        return g
    G1, G2 = G(p1), G(p2)
    ia = [col[j] for j in candA]; iq = [col[j] for j in candQ]
    ipa = [col[int(P_[j])] for j in candA]; ipq = [col[int(P_[j])] for j in candQ]
    leak = max(np.abs(G1[np.ix_(ia + iq, ipa + ipq)]).max(), np.abs(G1[np.ix_(ipa + ipq, ipa + ipq)]).max(), np.abs(G2[np.ix_(ia + iq, ia + iq)]).max())
    blk = G1[np.ix_(ia, iq)]
    # лучшая пара (A₁, A₂) × (Q₁, Q₂) по запутанности
    best = []
    for a1, a2 in itertools.combinations(range(len(ia)), 2):
        for q1, q2 in itertools.combinations(range(len(iq)), 2):
            M = blk[np.ix_([a1, a2], [q1, q2])]; nm = np.linalg.norm(M)
            if nm < 1e-300: continue
            best.append((2 * abs(np.linalg.det(M)) / nm ** 2, a1, a2, q1, q2, nm))
    best.sort(key=lambda t: -t[0])
    print(f'\nB = {B}: утечка «не туда» (пары и двойные пары) max |G| = {leak:.1e} при типичном |G| {np.abs(blk).max():.1e}; '
          f'кандидатов пар {len(best)}, C макс {best[0][0]:.4f}')
    done = 0
    for C, a1, a2, q1, q2, nm in best:
        if done >= int(os.environ.get("NDONE", 2)): break
        A = (candA[a1], candA[a2]); Q = (candQ[q1], candQ[q2]); PA = tuple(int(P_[k]) for k in A); PQ = tuple(int(P_[k]) for k in Q)
        psi = G1[np.ix_([col[k] for k in A], [col[k] for k in Q])]; phi = G2[np.ix_([col[k] for k in PQ], [col[k] for k in PA])]
        psi, phi = psi / np.linalg.norm(psi), phi / np.linalg.norm(phi)
        # ---- слияние у Боба ----
        rel = list(Q) + list(PQ)
        S.set_field(B); _, _, amp, _ = S.run({k: np.eye(4)[i].astype(complex) for i, k in enumerate(rel)})
        S.set_field(2 * B); _, emitD, colD = S.backward(sorted(det))
        d = sum(det[k] * emitD[:, colD[k]] for k in det)            # амплитуда слитого кванта из s до нечётного детектора
        Phi = np.zeros((2, 2), complex)
        for s in Mf:
            li = S.legs[s][0]
            if len(li) < 2: continue
            r = amp[li]; u = r.sum(0)
            Phi += 2 * d[s] * (np.outer(u[:2], u[2:]) - r[:, :2].T @ r[:, 2:])
        CPhi = 2 * abs(np.linalg.det(Phi)) / np.linalg.norm(Phi) ** 2; eff = np.linalg.norm(Phi) ** 2
        Xm = np.array([[0, 1], [1, 0]]); Zm = np.diag([1.0, -1.0])
        bobs = [np.conj(Phi @ Op.T) / np.linalg.norm(Phi) for Op in (np.eye(2), Xm, Zm, Xm @ Zm)]   # проводка кубита 2: Φ·Opᵀ
        V = np.array([b.reshape(4) for b in bobs]); orth = np.abs(V.conj() @ V.T - np.eye(4)).max()
        S.set_field(B)
        As, Cs = devices2(A[0], A[1], B, nmax=NMAX)
        if len(As) < 3: continue
        T = max(local_search(psi, phi, [bobs[i] for i in pm], As, Cs, restarts=8)[0] for pm in itertools.permutations(range(4)))
        Tid = max(local_search(psi, phi, [BELL[k] for k in pm], As, Cs, restarts=8)[0] for pm in itertools.permutations(BELL))
        caus = sum(bool(R[a, c] or R[c, a]) for a in A for c in PA)
        done += 1
        print(f'  Алиса ({X[A[0], 0]:.2f}, {X[A[0], 1]:.2f}), кубит Боба ({X[Q[0], 0]:.2f}, {X[Q[0], 1]:.2f}): C источника {C:.4f}; '
              f'слияние: C(Φ) = {CPhi:.6f}, неортогональность 4 исходов {orth:.1e}, |Φ|² {eff:.1e}; настроек {len(As)}; '
              f'T = {T:.3f} (Боб — слияние) / {Tid:.3f} (Боб — идеальный Белл); связей Алиса–Чарли {caus}  [вещ. 7.66]')
        sys.stdout.flush()
