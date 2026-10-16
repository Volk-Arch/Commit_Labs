"""ОДНО ПРАВИЛО НА РАЗНЫХ УРОВНЯХ (вопрос пользователя). Подкристалл: события с тенью в шаре r/τ — это ТОТ ЖЕ золотой кристалл,
растянутый в τ раз (Z[τ]·τ, тень сжимается в 1/τ), и он лежит ВНУТРИ мелкого. Вопрос: если частица живёт по правилу Джонстона
на мелком уровне (сумма по цепочкам: ½ на шаг, b = −m²/ρ на остановку), то её амплитуды МЕЖДУ событиями крупного уровня —
снова ли это правило Джонстона на крупном уровне (с какими ½·z и b')? Если да и точно — правило само себя воспроизводит.
Подгонка: K_sub = z·Φ_c(I − b′Φ_c)⁻¹, Φ_c = ½C_c ⇔ K_sub = zΦ_c + b′Φ_cK_sub. Контроль — случайное прореживание Пуассона в τ² раз."""
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import core, causal as Cs
from scipy.optimize import least_squares
TAU = core.TAU
def gold_with_shadow(rho, L):
    r = core.r_for_density(rho, 2)
    c, cp = core.numbers(-L / core.ST, L / core.ST, -r / core.SI, r / core.SI); T, Tq = c * core.ST, cp * core.SI
    d, dp = core.numbers(-L, L, -r, r)
    i, j = np.nonzero(Tq[:, None] ** 2 + dp[None, :] ** 2 <= r * r)
    X = np.stack([T[i], d[j]], 1); sh = Tq[i] ** 2 + dp[j] ** 2; keep = np.abs(X[:, 0]) + np.abs(X[:, 1]) <= L
    X, sh = X[keep], sh[keep]; o = np.argsort(X[:, 0], kind='stable'); return X[o], sh[o], r
def johnston(X, rho, m):
    Phi = 0.5 * Cs.causal(X).astype(float); b = -m * m / rho; n = len(X)
    return np.linalg.solve((np.eye(n) - b * Phi).T, Phi.T).T, Phi
def fit_rule(Ksub, Phic):
    def res(p):
        z, bb = p; r = Ksub - z * Phic - bb * (Phic @ Ksub); return r[np.triu_indices(len(Ksub), 1)]
    sol = least_squares(res, [1.0, -0.1]); r = res(sol.x); scale = np.linalg.norm(Ksub[np.triu_indices(len(Ksub), 1)])
    return sol.x, np.linalg.norm(r) / scale
L, rho = 12.0, 4.0
X, sh, r = gold_with_shadow(rho, L); coarse = sh <= (r / TAU) ** 2 + 1e-12; Xc = X[coarse]; rho_c = rho / TAU ** 2
print(f'мелкий уровень: {len(X)} событий (ρ = {rho}); крупный (тень в r/τ): {coarse.sum()} — отношение {len(X) / coarse.sum():.3f} (τ² = {TAU ** 2:.3f})')
# проверка: крупный уровень = мелкий кристалл, растянутый в τ
Xs, _, _ = gold_with_shadow(rho, L / TAU); Xs = Xs * TAU
A = {tuple(np.round(p, 6)) for p in Xc}; B = {tuple(np.round(p, 6)) for p in Xs}
inn = lambda P: [p for p in P if abs(p[0]) + abs(p[1]) <= L - 1]
print(f'крупный уровень = τ·(кристалл): совпадает {len(set(inn(A)) & set(inn(B)))} из {len(inn(A))} внутренних событий')
rng = np.random.default_rng(5)
for m in (0.0, 0.5, 1.0):
    K, Phi = johnston(X, rho, m); idx = np.nonzero(coarse)[0]; Ksub = K[np.ix_(idx, idx)]; Phic = 0.5 * Cs.causal(Xc).astype(float)
    (z, bb), res = fit_rule(Ksub, Phic)
    # контроль: случайное прореживание того же мелкого кристалла (те же события, но выбор не по правилу)
    ridx = np.sort(rng.choice(len(X), size=len(idx), replace=False)); Kr = K[np.ix_(ridx, ridx)]; Phir = 0.5 * Cs.causal(X[ridx]).astype(float)
    (zr, br), resr = fit_rule(Kr, Phir)
    print(f'm = {m}: золотое укрупнение — z = {z:.4f}, b′ = {bb:+.4f} (−m²/ρ_c = {-m * m / rho_c:+.4f}, т.е. m′ = {math.sqrt(max(-bb * rho_c, 0)):.4f}), остаток {res:.2e};  '
          f'случайное — z = {zr:.4f}, b′ = {br:+.4f}, остаток {resr:.2e}', flush=True)
