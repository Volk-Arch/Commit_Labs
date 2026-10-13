"""Антропный вес (Вайнберг): миров много, Λ своя в каждом; наблюдателей тем больше,
чем больше вещества успело собраться в галактики к a = 1 (фотоны остыли до 2.7 K).
Доля в галактиках — Пресс–Шехтер: f = erfc(δc / (√2·S·D(1))), D — линейный рост
(D = a в ранней эпохе, одинаково во всех мирах), S — амплитуда флуктуаций масштаба
галактики, если бы Λ = 0 (S ≈ 1.5 — 1e13 M☉, 2.5 — 1e12, 4 — 1e10).
Вопрос: при α из молекул горизонта (~1) какая доля НАБЛЮДАТЕЛЕЙ видит мир,
согласный с SN+BAO?"""
import sys, math, numpy as np
sys.argv = sys.argv[:1]
from scipy.integrate import solve_ivp
from frozen import solve, past_cone, OMN, OMR
from frozenbao import chi_bao, chi_sn, rows
import everpresent as ep
np.seterr(all='ignore')
DC = 1.686
it = (rows[:, 1] + rows[:, 2]).argmin(); ctot0 = rows[it, 1] + rows[it, 2]

def growth(L):
    """D(a=1) при Λ (единицы H_ном), D = a рано."""
    def rhs(x, y):
        a = math.exp(x); H2 = OMN * a ** -3 + OMR * a ** -4 + L / 3
        if H2 <= 0: return [0, 0]
        dlnH = 0.5 * (-3 * OMN * a ** -3 - 4 * OMR * a ** -4) / H2
        Om = OMN * a ** -3 / H2
        return [y[1], -(2 + dlnH) * y[1] + 1.5 * Om * y[0]]
    a0 = 2e-3
    s = solve_ivp(rhs, (math.log(a0), 0), [a0, a0], rtol=1e-8, atol=1e-12)
    return s.y[0, -1]

if __name__ == '__main__':
    D_lcdm = growth(3 * 0.69 * 1.0)
    print(f'проверка: у ΛCDM (Ω_Λ ≈ 0.69) рост D(1)/a = {D_lcdm:.3f} (известно ≈ 0.78)')
    for alpha in (0.03, 0.3, 1.0):
        fine = np.linspace(1e-5, 0.3 / alpha * 1.6 if alpha > 0.05 else 3, 1500)
        coarse = np.linspace(-4, 4, 801)
        xs = np.unique(np.concatenate([fine, coarse])); xs = xs[xs != 0]
        W = {S: [] for S in (1.5, 2.5, 4.0)}; good = []; phi = []; neg = []
        for xi in xs:
            try:
                L = solve(alpha, xi)
            except Exception:
                L = np.nan
            if not np.isfinite(L):
                for S in W: W[S].append(0.0)
                good.append(False); phi.append(math.exp(-xi ** 2 / 2)); neg.append(xi < 0); continue
            V, Hn = past_cone(L); D1 = growth(L)
            for S in W: W[S].append(math.erfc(DC / (math.sqrt(2) * S * D1)))
            h0 = 67.7 * Hn; Om = OMN / Hn ** 2
            ok = (60 < h0 < 80 and 0.2 < Om < 0.45) and (chi_bao(L, Hn) + chi_sn(L, Hn) - ctot0 < 4)
            good.append(ok); phi.append(math.exp(-xi ** 2 / 2)); neg.append(xi < 0)
        good = np.array(good); phi = np.array(phi); neg = np.array(neg)
        dx = np.gradient(xs)
        base = (phi * good * dx).sum() / (phi * dx).sum()
        line = f'α = {alpha:4.2f}: без отбора {base*100:.3f} %'
        for S, w in W.items():
            w = np.array(w); p = (phi * w * good * dx).sum() / (phi * w * dx).sum()
            ppos = (phi * w * good * dx).sum() / (phi * w * dx * ~neg).sum()
            line += f' | S={S}: {p*100:.2f} % (только Λ>0: {ppos*100:.2f} %)'
        print(line); sys.stdout.flush()
