"""Гармоническая инверсия (многоканальный матричный пучок): сигнал многих каналов в коротком окне раскладывается в сумму
e^{(iω − γ)t}. Отделяет связанный уровень от порога непрерывного спектра, даже если в окне один период (dirac-doc §29)."""
import numpy as np

def pencil(Y, dt, K, L=None):
    """Y — (каналы, отсчёты) комплексный. Возвращает частоты ω, затухания γ и амплитуды (норма по каналам), по убыванию амплитуды."""
    Y = np.atleast_2d(Y); C, N = Y.shape; L = L or N // 2
    H0 = np.vstack([np.array([y[i:i + L] for i in range(N - L)]) for y in Y])       # строки — сдвиги, по всем каналам
    H1 = np.vstack([np.array([y[i + 1:i + 1 + L] for i in range(N - L)]) for y in Y])
    U, s, Vh = np.linalg.svd(H0, full_matrices=False); K = min(K, len(s))
    Uk, sk, Vk = U[:, :K], s[:K], Vh[:K].conj().T
    A = np.diag(1 / sk) @ Uk.conj().T @ H1 @ Vk; z = np.linalg.eigvals(A)
    om = np.angle(z) / dt; ga = -np.log(np.abs(z)) / dt
    V = np.vander(z, N, increasing=True).T                                            # N × K
    amps = np.array([np.linalg.lstsq(V, y, rcond=None)[0] for y in Y])                # C × K
    w = np.linalg.norm(amps, axis=0); o = np.argsort(-w)
    return om[o], ga[o], w[o]

def level(Y, dt, lo, hi, K=8):
    """уровень — мода в окне частот [lo, hi] с наибольшей амплитудой и малым затуханием."""
    om, ga, w = pencil(Y, dt, K); m = (om > lo) & (om < hi) & (np.abs(ga) < 0.1)
    if not m.any(): return np.nan, (om, ga, w)
    i = np.nonzero(m)[0][0]; return om[i], (om, ga, w)
