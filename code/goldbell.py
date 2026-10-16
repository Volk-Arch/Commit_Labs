#!/usr/bin/env python3
"""CHSH для правила пар с золотым сопряжением.

Амплитуда исхода (a, b) при настройках (x, y): ψ = Σ_λ s(λ)·α(a|x,λ)·β(b|y,λ)  — сумма по общему
прошлому λ (здесь два состояния, «кубит»). Всё живёт в Q(τ): у каждого числа есть мир и тень.
Правило пар: P(a,b|x,y) ∝ ψ·ψ′ (ψ′ — тень ψ: τ → −1/τ, золотая «норма»).
Если тень совпадает с миром (ψ′ = ψ), это обычная квантовая механика с вещественными
амплитудами (P ∝ ψ²) и потолок 2√2.

Настройки — повороты базиса на углы, косинусы которых лежат в Q(τ): кратные 36°
(cos 36° = τ/2). Тень поворота на 36° — поворот на 108° (cos → −1/(2τ)), то есть
θ_тень = 3θ_мир. Состояние s(λ) — числа a + bτ с малыми целыми a, b.
Проверяем: P ≥ 0, сумма 1, отсутствие сигнала (маргиналы Алисы не зависят от Боба), и CHSH.
"""
import itertools, math, numpy as np
TAU = (1 + 5 ** .5) / 2
def rot(th): c, s = math.cos(th), math.sin(th); return np.array([[c, s], [-s, c]])  # строки — базис исходов ±1

def probs(Psi_w, Psi_s, Aw, As, Bw, Bs):
    """P[a,b] для одной пары настроек. Psi — 2×2 матрица s(λ) по диагонали (состояние Σ s_λ |λ⟩|λ⟩)."""
    Pw = Aw @ Psi_w @ Bw.T; Ps = As @ Psi_s @ Bs.T
    raw = Pw * Ps; Z = raw.sum()
    return raw / Z, raw, Z

def chsh(P):  # P[x][y] — 2×2 матрицы по исходам a,b ∈ {+1,−1}
    E = lambda p: p[0, 0] - p[0, 1] - p[1, 0] + p[1, 1]
    return E(P[0][0]) + E(P[0][1]) + E(P[1][0]) - E(P[1][1])

def evaluate(sw, ss, ax, bx, shadow_map):
    """sw, ss — (s0, s1) в мире и тени; ax, bx — углы мира для настроек Алисы и Боба."""
    Psi_w = np.diag(sw); Psi_s = np.diag(ss); P = [[None] * 2 for _ in range(2)]; neg = 0; Zs = []
    for i in range(2):
        for j in range(2):
            p, raw, Z = probs(Psi_w, Psi_s, rot(ax[i]), rot(shadow_map(ax[i])), rot(bx[j]), rot(shadow_map(bx[j])))
            if Z <= 1e-12 or (raw / Z < -1e-12).any(): return None
            P[i][j] = p; Zs.append(Z)
    # отсутствие сигнала
    ns = 0.0
    for i in range(2): ns = max(ns, np.abs(P[i][0].sum(1) - P[i][1].sum(1)).max())
    for j in range(2): ns = max(ns, np.abs(P[0][j].sum(0) - P[1][j].sum(0)).max())
    return chsh(P), ns, P

if __name__ == '__main__':
    angles = [k * math.pi / 5 for k in range(10)]                # кратные 36°
    golden = lambda th: 3 * th                                    # тень поворота: ×3
    same = lambda th: th                                          # контроль: тень = мир (обычная КМ)
    # состояния a + bτ, |a|,|b| ≤ 2, их тени a − b/τ
    nums = [(a, b) for a in range(-2, 3) for b in range(-2, 3) if (a, b) != (0, 0)]
    def W(n): return n[0] + n[1] * TAU
    def S(n): return n[0] - n[1] / TAU
    for name, smap, use_shadow in (('контроль: обычная КМ (тень = мир)', same, False), ('ЗОЛОТОЕ правило пар', golden, True)):
        best = (0, None); best_ns = (0, None); count = 0; signal = 0
        for n0, n1 in itertools.product(nums, repeat=2):
            sw = (W(n0), W(n1)); ss = (S(n0), S(n1)) if use_shadow else sw
            for ax in itertools.combinations(angles, 2):
                for bx in itertools.combinations(angles, 2):
                    r = evaluate(sw, ss, ax, bx, smap)
                    if r is None: continue
                    count += 1; Sv, ns, _ = r
                    if ns > 1e-9: signal += 1
                    if abs(Sv) > best[0]: best = (abs(Sv), (n0, n1, ax, bx, ns))
                    if ns <= 1e-9 and abs(Sv) > best_ns[0]: best_ns = (abs(Sv), (n0, n1, ax, bx))
        print(f'{name}: допустимых (P ≥ 0) конфигураций {count}, из них с сигналом {signal}')
        print(f'   максимум |S| среди всех допустимых: {best[0]:.4f}  (сигнал {best[1][4]:.2e})' if best[1] else '   нет')
        print(f'   максимум |S| без сигнала: {best_ns[0]:.4f}   конфигурация {best_ns[1]}')
