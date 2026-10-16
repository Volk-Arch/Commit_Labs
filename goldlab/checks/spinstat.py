"""СПИН–СТАТИСТИКА ДЛЯ ЗИГЗАГА (разведка). Запаздывающий пропагатор по состояниям «ребро» (событие + направление):
K = T(I − T)⁻¹, T[e_out, e_in] = 1 (прямо) или −i·m·Δt (поворот). Для скаляра Джонстона спектр iΔ = i(K − K†) строго симметричен
(±) ⇒ только коммутатор ⇒ бозоны (`maxwell-doc` §23). Здесь смотрим спектры i(K − K†) и K + K†: симметричны ли они, есть ли
знакоопределённость (положительность ⇒ возможен антикоммутатор)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
from zigzag import zig_graph
for L, m in ((8, 1.0), (10, 1.0), (10, 0.5)):
    w = Wd.gold11(4.0, L); X, t = w.X, w.t; EA, EB, Dir = zig_graph(X); E = len(EA)
    T = np.zeros((E, E), complex)
    for eo in range(E):
        prev = np.nonzero(EB == EA[eo])[0]
        for ei in prev: T[eo, ei] = 1.0 if Dir[ei] == Dir[eo] else -1j * m * (t[EB[eo]] - t[EA[eo]])
    Kz = T @ np.linalg.inv(np.eye(E) - T)                        # K[e_out, e_in]
    for name, Mx in (('i(K − K†)', 1j * (Kz - Kz.conj().T)), ('K + K†', Kz + Kz.conj().T)):
        lam = np.linalg.eigvalsh(Mx); big = np.abs(lam).max(); pos = lam[lam > 1e-9 * big]; neg = -lam[lam < -1e-9 * big]
        sym = np.abs(np.sort(pos)[::-1][:min(len(pos), len(neg))] - np.sort(neg)[::-1][:min(len(pos), len(neg))]).max() / big if len(pos) and len(neg) else float('nan')
        print(f'L = {L}, m = {m}, состояний {E}: {name}: положит. {len(pos)}, отрицат. {len(neg)}, наибольшее {lam.max():.3g}, наименьшее {lam.min():.3g}; '
              f'несимметрия ± (отн.) {sym:.2e}', flush=True)
