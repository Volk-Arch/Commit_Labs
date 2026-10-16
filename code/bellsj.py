#!/usr/bin/env python3
"""БЕЛЛ НА ЯЗЫКЕ СОРКИНА–ДЖОНСТОНА.

Раньше вероятности шли из амплитуд правила (|T|²), теперь — из вакуума W (положительная часть
i(R − R†)), т.е. из той нормы, что сохраняется. Перевод «один к одному», без новых постулатов:
  * пара рождается в неизвестном месте полосы источника: |Ψ⟩ = Σ_s φ†(s)φ†(s)|0⟩;
    двухчастичная волновая функция Ψ(x, y) = ⟨0|φ(x)φ(y)|Ψ⟩ = 2·Σ_s W(x, s)·W(y, s) (теорема Вика);
  * детектор в событии (порт = событие и компонента) — по Глауберу: P(a, b) ∝ |Ψ(a, b)|²;
  * настройка — выбор пары портов в своей области (Алиса справа, Боб слева, всё пространственно
    разнесено), отбор «сработал один из двух портов у каждого».
Сравнение: (1) W полный, (2) W со срезкой мод, (3) контроль — запаздывающий пропагатор R вместо W
(старое правило «сумма историй»).
Отбор пар портов: в двухчастичной функции Ψ(Алиса × Боб) берём два главных шмидтовых направления;
каждый порт — 2-вектор в них; пара портов — «делитель» с качеством u = 2|det|/‖·‖² (1 — без потерь).
"""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from sj import sj
from beamsplit import smax
sys.argv = _a

def chsh_search(Psi, setsA, setsB, tols=(0.005, 0.001)):
    """Psi — матрица амплитуд [порты Алисы × порты Боба]; sets — списки пар индексов портов"""
    PA = len(setsA); PB = len(setsB)
    # таблицы вероятностей для всех (настройка А, настройка Б)
    P = np.zeros((PA, PB, 2, 2))
    for i, (p, q) in enumerate(setsA):
        sub = Psi[[p, q], :]
        for j, (r, s) in enumerate(setsB):
            t = np.abs(sub[:, [r, s]]) ** 2; P[i, j] = t / t.sum()
    E = P[..., 0, 0] - P[..., 0, 1] - P[..., 1, 0] + P[..., 1, 1]
    mA = P[..., 0, :].sum(-1); mB = P[..., :, 0].sum(-1)
    best = {t: (0, None) for t in tols}
    for i0, i1 in itertools.combinations(range(PA), 2):
        # все пары настроек Боба сразу
        S = np.abs(E[i0][:, None] + E[i0][None, :] + E[i1][:, None] - E[i1][None, :])
        S = np.maximum(S, np.abs(E[i0][:, None] - E[i0][None, :] + E[i1][:, None] + E[i1][None, :]))
        sig = np.maximum.reduce([np.abs(mA[i0][:, None] - mA[i0][None, :]), np.abs(mA[i1][:, None] - mA[i1][None, :]),
                                 np.abs(mB[i0][:, None] - mB[i1][:, None]) + 0 * S, np.abs(mB[i0][None, :] - mB[i1][None, :]) + 0 * S])
        np.fill_diagonal(S, 0)
        for t in tols:
            Sm = np.where(sig <= t, S, 0); k = Sm.argmax()
            if Sm.flat[k] > best[t][0]: best[t] = (Sm.flat[k], sig.flat[k])
    return best

def schmidt_settings(Psi, nbest=25):
    U, s, Vh = np.linalg.svd(Psi)
    C = 2 * s[0] * s[1] / (s[0] ** 2 + s[1] ** 2); leak = 1 - (s[0] ** 2 + s[1] ** 2) / (s ** 2).sum()
    va = U[:, :2] * s[:2]; vb = Vh[:2].T * s[:2]
    def best_pairs(v):
        out = []
        for p, q in itertools.combinations(range(len(v)), 2):
            T = np.array([v[p], v[q]]); nrm = (np.abs(T) ** 2).sum()
            if nrm < 1e-30: continue
            out.append((2 * abs(np.linalg.det(T)) / nrm, (p, q)))
        out.sort(key=lambda t: -t[0]); return out[:nbest]
    return C, leak, s, best_pairs(va), best_pairs(vb)

if __name__ == '__main__':
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    Lw = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
    w = CH.sym_world(4.0, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
    Rp, iD, lam, V, Wfull = sj(w, LK, eps); order = np.argsort(-lam)
    kA = int((lam > math.sqrt(n) / (4 * math.pi)).sum())
    src_ev = [k for k in range(n) if abs(X[k, 0] + 7) <= 0.45 and abs(X[k, 1]) <= 3.0]
    S = np.array([2 * k + m for k in src_ev for m in (0, 1)])
    Aev = [k for k in range(n) if 2.0 <= X[k, 0] <= 5.0 and 2.5 <= X[k, 1] <= 7.0 and np.abs(X[k]).sum() <= Lw - 1]
    Bev = [CH.img(w, k, CH.P, CH.P) for k in Aev]
    caus = sum(R[a, b] or R[b, a] for a in Aev for b in Bev)
    Ap = np.array([2 * k + m for k in Aev for m in (0, 1)]); Bp = np.array([2 * k + m for k in Bev for m in (0, 1)])
    print(f'ε = {eps}, ромб {Lw}: событий {n}; источник {len(src_ev)} событий; портов Алисы/Боба {len(Ap)}/{len(Bp)}; '
          f'причинных связей Алиса–Боб: {caus}; мод ААС {kA}')
    variants = [('запаздывающий R (старое правило)', Rp), ('SJ, все моды', Wfull)]
    for kk in (kA, 4 * kA, 16 * kA):
        sel = order[:kk]; variants.append((f'SJ, {kk} мод', (V[:, sel] * lam[sel]) @ V[:, sel].conj().T))
    for name, G in variants:
        Psi = G[np.ix_(Ap, S)] @ G[np.ix_(Bp, S)].T
        C, leak, sv, sa, sb = schmidt_settings(Psi)
        best = chsh_search(Psi, [p for _, p in sa], [p for _, p in sb])
        print(f'  {name}: C двух главных = {C:.3f}, вне двух главных {leak:.1%}; качество лучших «делителей» {sa[0][0]:.3f}/{sb[0][0]:.3f}; '
              f'S = {best[0.005][0]:.3f} при сигнале ≤0.5 %, {best[0.001][0]:.3f} при ≤0.1 %')
        sys.stdout.flush()
