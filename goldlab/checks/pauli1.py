"""ПРАВИЛО ПЕРЕСЕЧЕНИЙ, ПЕРВЫЙ ОПЫТ: две одинаковые частицы (Джонстон, сумма по цепочкам: ½ на шаг, b = −m²/ρ на остановку).
H₀ — сумма по парам цепочек БЕЗ общих событий («встретиться нельзя», λ = 0), без знаков (бозоны с запретом встречи).
D — определитель K(a₁→j₁)K(a₂→j₂) − K(a₁→j₂)K(a₂→j₁) (фермионы). В непрерывной 1+1 они совпадают (Жирардо, ЛГВ) — если пересечь
путь, не встретившись, нельзя. На кристалле цепочки прыгают и могут поменяться местами без общего события: насколько это важно?
Счёт H₀ — через «первую встречу»: M(z) = P(z) − Σ_{w<z} M(w)·b²·K(w→z)², H₀ = P₁₂ − Σ_w M(w)·b²·K(w→j₁)K(w→j₂)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs
for L, m in ((12, 1.0), (12, 0.5), (16, 1.0), (16, 0.5)):
    rho = 4.0; w = Wd.gold11(rho, L); t, x, n = w.t, w.x, w.n
    Phi = 0.5 * Cs.causal(w.X).astype(float); b = -m * m / rho
    K = np.linalg.solve(np.eye(n) - b * Phi, np.eye(n)); K = Phi @ K            # K = Φ(I − bΦ)⁻¹: сумма по цепочкам
    near = lambda tt, xx: int(np.argmin(np.abs(t - tt) * 3 + np.abs(x - xx)))
    a1, a2 = near(-L + 4, -1.0), near(-L + 4, 1.0)
    P = K[a1] * K[a2]; M = np.zeros(n)
    for z in range(n):                                                          # события отсортированы по времени
        M[z] = P[z] - b * b * (M[:z] @ (K[:z, z] ** 2))
    H0 = np.outer(K[a1], K[a2]) - b * b * (K.T * M) @ K                        # H₀[j₁, j₂]
    D = np.outer(K[a1], K[a2]) - np.outer(K[a2], K[a1])
    B = np.outer(K[a1], K[a2]) + np.outer(K[a2], K[a1])
    for tl in (0.0, L / 3):
        Z = np.nonzero(np.abs(t - tl) < 0.5)[0]; Z = Z[np.abs(x[Z]) < 6]; J1, J2 = np.meshgrid(Z, Z, indexing='ij'); left = x[J1] < x[J2] - 1e-9
        h, d, bb = H0[J1, J2][left], D[J1, J2][left], B[J1, J2][left]
        r = (h @ d) / (d @ d); print(f'L = {L}, m = {m}, слой t ≈ {tl:.1f}: H₀ ≈ {r:.3f}·D, остаток {np.linalg.norm(h - r * d) / np.linalg.norm(h):.3f};  '
              f'для сравнения: |H₀ − D|/|D| = {np.linalg.norm(h - d) / np.linalg.norm(d):.3f}, |B − D|/|D| = {np.linalg.norm(bb - d) / np.linalg.norm(d):.3f}', flush=True)
