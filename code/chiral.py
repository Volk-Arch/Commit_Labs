#!/usr/bin/env python3
"""ШАХМАТНАЯ ДОСКА ФЕЙНМАНА НА КРИСТАЛЛЕ: «поворот = i».

Откуда в физике берутся 90° у делителя без потерь: у Фейнмана (уравнение Дирака в 1+1) частица
идёт вправо или влево, и каждый поворот умножает амплитуду на i. Тогда у любого пути от «влево» до
«вправо» нечётное число поворотов — амплитуда чисто мнимая, а от «вправо» до «вправо» —
вещественная. Если веса звеньев вещественные, a ⊥ b получается автоматически, а зеркало делает
|столбцы| равными ⇒ делитель без потерь ТОЧНО.

Киральность звена на кристалле. «Лево/право» по знаку Δx не годится: буст его меняет. Но у кристалла
буст дискретный (шаг η₀), поэтому есть метка, которую не меняет ни один буст кристалла, а отражение
меняет на противоположную: половина, в которую попадает быстрота звена по модулю η₀:
    ℓ = 0, если frac(η/η₀) ∈ (0, ½);  ℓ = 1, если ∈ (½, 1).
Метка должна сохраняться под g^K и меняться под P и h = P·g^K, поэтому период — Kη₀:
    ℓ = 0, если frac(η/Kη₀) ∈ (0, ½);  ℓ = 1, если ∈ (½, 1).
Звенья ровно на границах (быстрота — кратная Kη₀/2, направления осей зеркал) зеркало оставляет
на месте, поэтому они «и влево, и вправо» (метка 2): несут монету в обе компоненты.
Вещественность (теорема чётности) от этого не страдает: в компоненту m любое звено приносит
s·(ψ_m + iεψ_{1−m}), а это всегда «тот же класс», что ψ_m.

Правило: у события две компоненты ψ₀, ψ₁. Звено J→K с меткой m несёт s·(ψ_m(J) + iε·ψ_{1−m}(J));
ψ_m(K) — сумма по «освещённым» входящим звеньям с меткой m, делённая на число всех освещённых
входящих звеньев (событие = среднее того прошлого, что до него дошло). s — вещественный знак:
1 или (−1)^N, N = s² − σ² — целое (след формы), вещественный остаток теневой фазы.
"""
import sys, math, numpy as np
from boostsplit import img, axis_frame, grid, G, Gs, P, TAU, smax, build_order, links_of
_a = sys.argv; sys.argv = sys.argv[:1]
from mirrorsplit import sym_world
sys.argv = _a
ETA0 = math.acosh(TAU)

def prep(w, Lk, K=2):
    I, J = np.nonzero(Lk); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    d = w.X[J] - w.X[I]; f = np.mod(np.arctanh(d[:, 1] / d[:, 0]) / (K * ETA0), 1.0)
    edge = (np.abs(f) < 1e-9) | (np.abs(f - 0.5) < 1e-9) | (np.abs(f - 1) < 1e-9)
    lab = np.where(edge, 2, (f > 0.5).astype(int))
    dw = w.W[J] - w.W[I]; N = d[:, 0] ** 2 - d[:, 1] ** 2 - (dw ** 2).sum(1)
    Ni = np.rint(N); assert np.abs(N - Ni).max() < 1e-6
    st = np.searchsorted(J, np.arange(w.n)); en = np.searchsorted(J, np.arange(w.n), side='right')
    return dict(I=I, J=J, lab=lab, edge=edge, N=Ni.astype(int), st=st, en=en, f=f)

def chiral(w, LK, R, src, m0, eps=1.0, signs=False, wb=1.0, field=None):
    I, lab, st, en = LK['I'], LK['lab'], LK['st'], LK['en']
    if signs == 'shadow': s = np.exp(1j * math.pi * ((w.W[LK['J']] - w.W[I]) ** 2).sum(1))   # контроль: комплексная фаза тени
    elif signs: s = np.where(LK['N'] % 2 == 1, -1.0, 1.0)
    else: s = np.ones(len(I))
    if field is not None: s = s * field          # калибровочное поле: фаза e^{i∫A} на звене
    coin = 1j * eps if COIN == 'i' else eps
    lit = (I == src) | R[src, I]
    psi = np.zeros((w.n, 2), complex); psi[src, m0] = 1
    for j in range(src + 1, w.n):
        a, b = st[j], en[j]
        if a >= b: continue
        li = lit[a:b]; nl = li.sum()
        if nl == 0: continue
        ii, mm, ss = I[a:b][li], lab[a:b][li], s[a:b][li]
        c0 = ss * (psi[ii, 0] + coin * psi[ii, 1]); c1 = ss * (psi[ii, 1] + coin * psi[ii, 0])
        if KAPPA:   # сумма историй с одним κ на мир
            psi[j, 0] = (c0[mm == 0].sum() + wb * c0[mm == 2].sum()) * KAPPA
            psi[j, 1] = (c1[mm == 1].sum() + wb * c1[mm == 2].sum()) * KAPPA
        else:
            psi[j, 0] = (c0[mm == 0].sum() + wb * c0[mm == 2].sum()) / nl
            psi[j, 1] = (c1[mm == 1].sum() + wb * c1[mm == 2].sum()) / nl
    return psi

COIN, LABELS, VERB = 'i', 'chiral', True
import os
KAPPA = float(os.environ.get('KAPPA', '0'))
def unitary(T):
    U, _, Vh = np.linalg.svd(T); return U @ Vh

if __name__ == '__main__':
    EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    SIGNS = {'signs': True, 'shadow': 'shadow'}.get(sys.argv[2], False) if len(sys.argv) > 2 else False
    COIN = sys.argv[5] if len(sys.argv) > 5 else 'i'; LABELS = sys.argv[6] if len(sys.argv) > 6 else 'chiral'
    VERB = '-q' not in sys.argv
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    RHO = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    Lr, PLO = 24.0, 4.0
    H, Hs = P @ np.linalg.matrix_power(G, K), P @ np.linalg.matrix_power(Gs, K)
    w = sym_world(RHO, Lr); R = build_order(w, 'proj'); Lk = links_of(R); X = w.X; n = w.n
    LABK = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7].isdigit() else K
    LK = prep(w, Lk, LABK)
    if LABELS == 'random': LK['lab'] = np.random.default_rng(5).integers(0, 3, len(LK['lab']))   # контроль: метки без симметрии
    # проверка: отражение P и зеркало h меняют метку, бусты — сохраняют
    pos = {}
    for k in range(n): pos[(round(X[k, 0], 6), round(X[k, 1], 6))] = k
    def lab_of(i, j):
        a, b = LK['st'][j], LK['en'][j]; k = np.nonzero(LK['I'][a:b] == i)[0]
        return None if len(k) == 0 else (a + k[0])
    chk = {'P': [0, 0], 'h': [0, 0], 'g': [0, 0]}
    rs = np.random.default_rng(0)
    for e in rs.choice(len(LK['I']), 3000, replace=False):
        i, j = LK['I'][e], LK['J'][e]
        for nm, (Mw, Ms) in {'P': (P, P), 'h': (H, Hs), 'g': (np.linalg.matrix_power(G, LABK), np.linalg.matrix_power(Gs, LABK))}.items():
            i2, j2 = img(w, i, Mw, Ms), img(w, j, Mw, Ms)
            if i2 < 0 or j2 < 0: continue
            e2 = lab_of(i2, j2)
            if e2 is None: continue
            l1 = LK['lab'][e]; exp = l1 if (l1 == 2 or nm == 'g') else 1 - l1
            chk[nm][0] += 1; chk[nm][1] += (LK['lab'][e2] == exp)
    print(f'монета {COIN}, метки {LABELS}, ε = {EPS}, фаза {SIGNS if SIGNS == "shadow" else ("(−1)^N" if SIGNS else "нет")}, k = {K}, период меток {LABK}η₀, ρ = {RHO}: событий {n}, звеньев {len(LK["I"])}, «и влево, и вправо» {LK["edge"].sum()}')
    print('метка ведёт себя как надо (P, h: 0↔1, 2 на месте; g^K: на месте): P {1}/{0}, h {3}/{2}, g^K {5}/{4}'.format(*chk['P'], *chk['h'], *chk['g']))
    sys.stdout.flush()
    beta = K * ETA0 / 2; Xa = axis_frame(X, -beta)
    src = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
    PS = {s: [chiral(w, LK, R, s, m, EPS, SIGNS) for m in (0, 1)] for s in src}
    candA = [k for k in range(n) if 2.0 <= Xa[k, 0] <= 3.2 and 0.3 <= Xa[k, 1] <= 2.0]
    allu, summary = [], []
    for A1 in candA:
        A2 = img(w, A1, H, Hs)
        if A2 < 0: continue
        B1, B2 = img(w, A1, P, P), img(w, A2, P, P)
        if any(R[a, b] or R[b, a] for a in (A1, A2) for b in (B1, B2)): continue
        PA = {(x, m): chiral(w, LK, R, x, m, EPS, SIGNS) for x in (A1, A2, B1, B2) for m in (0, 1)}
        for m0 in (0, 1):
            # входы Алисы (A1, m0), (A2, 1−m0); Боба — их отражения (P меняет метку)
            za = np.array([[PS[s][m][A1, m0], PS[s][m][A2, 1 - m0]] for s in src for m in (0, 1)])
            zb = np.array([[PS[s][m][B1, 1 - m0], PS[s][m][B2, m0]] for s in src for m in (0, 1)])
            M = za.T @ zb
            if np.linalg.norm(M) == 0: continue
            psi = M.reshape(4) / np.linalg.norm(M); S_id = smax(np.outer(psi, psi.conj()))
            C = 2 * abs(np.linalg.det(M)) / (np.abs(M) ** 2).sum()
            for m1 in (0, 1):
                cands = []
                for p in range(n):
                    if not (PLO <= Xa[p, 0] <= PLO + 2.5 and Xa[p, 1] > 0.05 and R[A1, p] and R[A2, p]): continue
                    q = img(w, p, H, Hs)
                    if q < 0: continue
                    Pp, Pq = img(w, p, P, P), img(w, q, P, P)
                    if any(R[a, b] or R[b, a] for a in (A1, A2, p, q) for b in (B1, B2, Pp, Pq)): continue
                    T = np.array([[PA[A1, m0][p, m1], PA[A2, 1 - m0][p, m1]], [PA[A1, m0][q, 1 - m1], PA[A2, 1 - m0][q, 1 - m1]]])
                    TB = np.array([[PA[B1, 1 - m0][Pp, 1 - m1], PA[B2, m0][Pp, 1 - m1]], [PA[B1, 1 - m0][Pq, m1], PA[B2, m0][Pq, m1]]])
                    nrm = (np.abs(T) ** 2).sum()
                    if nrm < 1e-24: continue
                    u = 2 * abs(np.linalg.det(T)) / nrm; allu.append(u)
                    bal = 2 * abs(T[0, 0]) * abs(T[0, 1]) / (abs(T[0, 0]) ** 2 + abs(T[0, 1]) ** 2 + 1e-300)
                    cands.append((bal, u, T, TB))
                if not cands: continue
                cands.sort(key=lambda c: -c[0])
                best = None
                for bal, u, T, TB in cands[:4]:
                    g = grid(T, TB, M, tols=(1e-9, 0.001))
                    if best is None or g[1e-9][0] > best[0]: best = (g[1e-9][0], g[0.001][0], u, bal, np.abs(TB - T).max() / np.abs(T).max())
                S0, S1, u, bal, dTB = best
                summary.append((S0, S_id, C))
                if VERB: print(f'A1 ({Xa[A1,0]:.2f},{Xa[A1,1]:+.2f}) вход {m0} выход {m1}: C {C:.3f}, S ид. {S_id:.3f}; портов {len(cands)}; '
                      f'делитель: 1−качество {1-u:.1e}, баланс {bal:.2f}, Боб≠Алиса {dTB:.0e} → S {S0:.3f} при сигнале 0 (≤1e−9), {S1:.3f} при ≤0.1 %')
                sys.stdout.flush()
    allu = np.array(allu); s = np.array(summary)
    print(f'== делителей всего {len(allu)}: 1−качество макс {(1-allu).max():.1e}, медиана {np.median(1-allu):.1e}')
    if len(s): print(f'== S при нулевом сигнале: макс {s[:,0].max():.3f}, S > 2 в {(s[:,0] > 2.0005).sum()} из {len(s)}; S ид. макс {s[:,1].max():.3f}; C макс {s[:,2].max():.3f}')
