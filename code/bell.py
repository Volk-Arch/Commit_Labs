"""Белл в нашей модели. Пара рождается в общем событии со скрытой фазой θ (тор); у каждой
частицы фаза дальше идёт своей родословной через золотую кошку (детерминированный хаос).
Результат измерения — детерминированная функция своей фазы и своей настройки:
A(a) = sign cos(2πθ₁ − a). Настройки берутся из своих, независимых скрытых фаз.
CHSH: S = E(a,b) + E(a,b') + E(a',b) − E(a',b'). Локальные скрытые переменные: |S| ≤ 2.
Квантовая механика: до 2√2 = 2.83."""
import numpy as np, math
TAU = (1 + 5 ** .5) / 2; OM = np.array([1 / TAU, 1 / TAU ** 2])
def cat(t, k, eps=0.6):
    for _ in range(k):
        t = (t + OM) % 1.0; p = eps / (2 * math.pi) * np.sin(2 * math.pi * t[..., 0])
        t = np.stack([2 * t[..., 0] + t[..., 1] + p, t[..., 0] + t[..., 1] + p], -1) % 1.0
    return t
rng = np.random.default_rng(1); N = 400000
th0 = rng.random((N, 2))                          # общая фаза пары (одно зерно на пару)
thA = cat(th0.copy(), 7); thB = cat((th0 + 0.5) % 1.0, 7)  # каждая частица — своя ветвь
sA = rng.random(N) < 0.5; sB = rng.random(N) < 0.5          # настройки — независимые фазы
best = 0
for a0, a1, b0, b1 in ((0, math.pi / 2, math.pi / 4, -math.pi / 4), (0, math.pi / 2, math.pi / 4, 3 * math.pi / 4)):
    for comp in (0, 1):
        A = np.where(sA, np.sign(np.cos(2 * math.pi * thA[:, comp] - a1)), np.sign(np.cos(2 * math.pi * thA[:, comp] - a0)))
        Bv = np.where(sB, np.sign(np.cos(2 * math.pi * thB[:, comp] - b1)), np.sign(np.cos(2 * math.pi * thB[:, comp] - b0)))
        E = lambda x, y: np.mean((A * Bv)[(sA == x) & (sB == y)])
        S = E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1); best = max(best, abs(S))
        print(f'настройки ({a0:.2f},{a1:.2f};{b0:.2f},{b1:.2f}), компонента {comp}: S = {S:+.3f}')
# лучшая детерминированная стратегия без хаоса: общая фаза напрямую (самая сильная корреляция)
A0 = np.sign(np.cos(2 * math.pi * th0[:, 0])); thetas = 2 * math.pi * th0[:, 0]
S_lhv = max(abs(sum(sg * np.mean(np.sign(np.cos(thetas - a)) * np.sign(np.cos(thetas - b))) for (a, b, sg) in
            ((0, math.pi / 4, 1), (0, -math.pi / 4, 1), (math.pi / 2, math.pi / 4, 1), (math.pi / 2, -math.pi / 4, -1)))), 0)
print(f'без хаоса, общая фаза напрямую (классика Белла): S = {S_lhv:.3f};  предел скрытых переменных 2, квантовый 2.828')
