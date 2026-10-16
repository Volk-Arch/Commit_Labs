#!/usr/bin/env python3
"""ПУТЬ 1: БЛИЖАЙШАЯ ИЗОМЕТРИЯ «СРЕЗ → СРЕЗ» (по Котлеру–Строминджеру: эволюция сохраняет скалярные
произведения, но пространство растёт; новички «подхватывают ритм»).

Носители состояния на срезе t — события E(t), из которых звено пересекает срез (хвосты звеньев через t).
Любая история после t начинается с них, поэтому правило задаёт линейный переход T₂₁: ψ|E(t₁) → ψ|E(t₂).
Правило: «поворот = i», каждая компонента — среднее по всем своим входящим звеньям, монета унитарна, ε = 0.5.
Ближайшая изометрия V = U W† (из T = U Σ W†). Проверяем:
  * насколько T далёк от изометрии (разброс сингулярных чисел);
  * причинность V: какая доля ‖V‖² приходится на пары (x₁, x₂), НЕ связанные причинно (у T она ровно 0);
  * согласованность: V₃₁ против V₃₂·V₂₁ (если зависит от нарезки — скрытая выделенная система отсчёта);
  * расширяющаяся половина ромба против сужающейся."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a
eps = 0.5
Lw = float(sys.argv[1]) if len(sys.argv) > 1 else 16.0
w = CH.sym_world(4.0, Lw); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X; n = w.n
I, J, lab, st, en = LK['I'], LK['J'], LK['lab'], LK['st'], LK['en']
s0, coin = 1 / math.sqrt(1 + eps * eps), 1j * eps

def carriers(t):
    m = (X[I, 0] < t) & (X[J, 0] >= t); return np.unique(I[m])

def transfer(t1, t2):
    E1, E2 = carriers(t1), carriers(t2)
    nc = 2 * len(E1); psi = np.zeros((n, 2, nc), complex)
    for c, k in enumerate(E1):
        psi[k, 0, 2 * c] = 1; psi[k, 1, 2 * c + 1] = 1
    inE1 = np.zeros(n, bool); inE1[E1] = True
    for j in np.nonzero(X[:, 0] >= t1)[0]:
        a, b = st[j], en[j]
        if a >= b: continue
        ii, mm = I[a:b], lab[a:b]
        n0, n1 = (mm != 1).sum(), (mm != 0).sum()
        c0 = s0 * (psi[ii, 0] + coin * psi[ii, 1]); c1 = s0 * (psi[ii, 1] + coin * psi[ii, 0])
        psi[j, 0] = c0[mm != 1].sum(0) / max(n0, 1); psi[j, 1] = c1[mm != 0].sum(0) / max(n1, 1)
        if j in set(E2) and X[j, 0] < t2: pass
    T = psi[E2].reshape(2 * len(E2), nc)
    # причинная маска: x₂ из будущего x₁ или совпадает
    rel = (R[np.ix_(E1, E2)].T | (E2[:, None] == E1[None, :]))
    mask = np.repeat(np.repeat(rel, 2, 0), 2, 1)
    return E1, E2, T, mask

def polar(T):
    U, s, Wh = np.linalg.svd(T, full_matrices=False); return U @ Wh, s

if __name__ == '__main__':
    ts = [-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0]
    TR = {}
    print(f'мир ρ = 4, ромб {Lw}: событий {n}; носителей на срезах t = {ts}: {[len(carriers(t)) for t in ts]}')
    for t1, t2 in zip(ts[:-1], ts[1:]):
        E1, E2, T, mask = transfer(t1, t2); TR[t1, t2] = (E1, E2, T, mask)
        V, s = polar(T)
        acaus = (np.abs(V[~mask]) ** 2).sum() / (np.abs(V) ** 2).sum()
        iso = np.abs(V.conj().T @ V - np.eye(V.shape[1])).max()
        print(f'  {t1:+.0f} → {t2:+.0f}: размерности {T.shape[1]} → {T.shape[0]} ({"расширение" if T.shape[0] >= T.shape[1] else "СУЖЕНИЕ"}); '
              f'сингулярные числа T: {s.max():.3f}…{s.min():.2e}; изометрия V†V = 1 до {iso:.1e}; '
              f'доля ‖V‖² вне светового конуса {acaus:.1%} (у T — {(np.abs(T[~mask]) ** 2).sum() / (np.abs(T) ** 2).sum():.0%})')
        sys.stdout.flush()
    # согласованность нарезки: прямой переход через два шага против композиции
    for (ta, tb, tc) in ((-9.0, -6.0, -3.0), (-6.0, -3.0, 0.0)):
        E1, E3, T31, _ = transfer(ta, tc)
        V31, _ = polar(T31); V21, _ = polar(TR[ta, tb][2]); V32, _ = polar(TR[tb, tc][2])
        comp = V32 @ V21
        d = np.linalg.norm(V31 - comp) / np.linalg.norm(V31)
        dT = np.linalg.norm(T31 - TR[tb, tc][2] @ TR[ta, tb][2]) / np.linalg.norm(T31)
        print(f'  нарезка {ta:+.0f} → {tb:+.0f} → {tc:+.0f}: правило T согласовано до {dT:.1e}; изометрии: ‖V₃₁ − V₃₂V₂₁‖/‖V₃₁‖ = {d:.3f}')
