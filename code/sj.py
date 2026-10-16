#!/usr/bin/env python3
"""СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ СОРКИНА–ДЖОНСТОНА (SJ) ДЛЯ ПРАВИЛА «ПОВОРОТ = i».

Правило сжимает норму на срезах. Причинные множества знают обход (Johnston 2009, Sorkin 2011,
Afshordi–Aslanbeigi–Sorkin 2012): не искать норму на срезах, а построить её из всего мира сразу.
  1) правило линейно: ψ = Aψ + f, A — «среднее по звеньям каждой компоненты, монета унитарна»;
     запаздывающий пропагатор R = (I − A)⁻¹ (нижнетреугольный: прошлое → будущее);
  2) iΔ = i(R − R†) — эрмитова матрица (аналог коммутатора поля);
  3) её положительная часть W = Σ_{λ>0} λ v v† — «вакуум» (двухточечная функция), скалярное
     произведение одночастичных состояний ⟨f|g⟩ = f†Wg. Срезов в определении нет.
Проверка сохранения (то, чего не было): частица |f⟩ (пакет, рождённый в прошлом). Вероятность найти
её на срезе S = доля |f⟩ в пространстве состояний, сидящих на событиях S:
      P_S = (Wf)_S† (W_SS)⁺ (Wf)_S / (f†Wf).
Если норма сохраняется, P_S ≈ 1 на КАЖДОМ достаточно толстом срезе — до и после рождения
(в теории поля состояние одно на все времена). Плотность по событиям среза (ортонормировка
Лёвдина) показывает, где частица; её сумма по срезу и есть P_S.
"""
import sys, math, numpy as np
from scipy.linalg import solve_triangular, eigh
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

def build_A(w, LK, eps, field=None):
    n = w.n; A = np.zeros((2 * n, 2 * n), complex); s0 = 1 / math.sqrt(1 + eps * eps); c = 1j * eps
    I, J, lab = LK['I'], LK['J'], LK['lab']
    for K in range(n):
        a, b = LK['st'][K], LK['en'][K]
        if a >= b: continue
        ii, mm = I[a:b], lab[a:b]
        n0, n1 = (mm != 1).sum(), (mm != 0).sum()
        for e, (j_, m) in enumerate(zip(ii, mm)):
            s = s0 if field is None else s0 * field[a + e]
            if m != 1 and n0:
                A[2 * K, 2 * j_] += s / n0; A[2 * K, 2 * j_ + 1] += s * c / n0
            if m != 0 and n1:
                A[2 * K + 1, 2 * j_ + 1] += s / n1; A[2 * K + 1, 2 * j_] += s * c / n1
    return A

def sj(w, LK, eps, field=None):
    A = build_A(w, LK, eps, field); N = A.shape[0]
    Rp = solve_triangular(np.eye(N) - A, np.eye(N, dtype=complex), lower=True)
    iD = 1j * (Rp - Rp.conj().T)
    lam, V = eigh(iD)
    pos = lam > 0
    Wm = (V[:, pos] * lam[pos]) @ V[:, pos].conj().T
    return Rp, iD, lam, V, Wm

def slice_prob(Wm, f, S, rcond=1e-10):
    """вероятность найти состояние f на событиях S и плотность по ним (Лёвдин)"""
    g = Wm @ f; norm = (f.conj() @ g).real
    WS = Wm[np.ix_(S, S)]; lam, U = eigh(WS); keep = lam > rcond * lam.max()
    # координаты в ортонормированном базисе пространства состояний среза
    coef = (U[:, keep].conj().T @ g[S]) / np.sqrt(lam[keep])
    P = (np.abs(coef) ** 2).sum() / norm
    # плотность по событиям: Лёвдин W_SS^{-1/2}
    half = (U[:, keep] / np.sqrt(lam[keep])) @ U[:, keep].conj().T
    dens = np.abs(half @ g[S]) ** 2 / norm
    return P, dens, keep.sum()

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    Lw = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    w = CH.sym_world(rho, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
    print(f'мир ρ = {rho}, ромб |t|+|x| ≤ {Lw}: событий {n}, размерность {2 * n}')
    idx = lambda evs, comps=(0, 1): np.array([2 * k + m for k in evs for m in comps])
    for eps in (0.0, 0.5, 1.0):
        Rp, iD, lam, V, Wm = sj(w, LK, eps)
        herm = np.abs(iD - iD.conj().T).max()
        lp, ln = np.sort(lam[lam > 0])[::-1], np.sort(-lam[lam < 0])[::-1]
        m = min(len(lp), len(ln))
        pair = np.abs(lp[:m] - ln[:m]).max() / lp[0]
        big = (lam > 1e-3 * lam.max()).sum()
        print(f'\nε = {eps}: iΔ эрмитова до {herm:.0e}; собственных + / −: {len(lp)} / {len(ln)}, пары ±λ совпадают до {pair:.1e}; '
              f'заметных (> 1e−3·макс) положительных {big}; макс λ {lam.max():.2f}')
        # пакет: гаусс по x на срезе t ∈ [−8, −7], компонента 0
        src = [k for k in range(n) if -8 <= X[k, 0] <= -7 and abs(X[k, 1]) < 4]
        f = np.zeros(2 * n, complex)
        for k in src: f[2 * k] = math.exp(-X[k, 1] ** 2 / 4)
        print('  срез t (толщина 1.0): вероятность найти частицу P_S | центр и ширина по x | наивная Σ|ψ|² (из правила)')
        psi = Rp @ f
        for t0 in (-9.0, -6.0, -3.0, 0.0, 3.0, 6.0):
            S_ev = [k for k in range(n) if t0 - 0.5 <= X[k, 0] <= t0 + 0.5]
            S = idx(S_ev)
            P, dens, rk = slice_prob(Wm, f, S)
            dx = dens.reshape(-1, 2).sum(1); xs = X[S_ev, 1]
            cx = (dx * xs).sum() / max(dx.sum(), 1e-30); sx = math.sqrt(max((dx * (xs - cx) ** 2).sum() / max(dx.sum(), 1e-30), 0))
            naive = (np.abs(psi[S]) ** 2).sum()
            print(f'   t = {t0:+.0f}: P_S = {P:.4f} (событий {len(S_ev)}, ранг {rk}) | x = {cx:+.2f} ± {sx:.2f} | Σ|ψ|² = {naive:.3f}')
        sys.stdout.flush()
