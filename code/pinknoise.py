"""Проверка предложения «розового (самоподобного) шума». Дрожь числа событий —
дробное броуновское движение по объёму V с показателем Хёрста h:
ΔN ~ V^h. Тогда Λ = ΔN/V ~ V^{h−1} ~ t^{4h−4}, а слежение за H² ~ t⁻²
требует h = 1/2 — то есть ОБЫЧНОГО (белого) шума. Проверяем численно
и заодно смотрим, меняется ли форма (смены знака)."""
import numpy as np, math
T = np.exp(np.linspace(math.log(10), math.log(1e6), 300)); V = T ** 4; v = V / V[-1]
rs = np.random.default_rng(2)
for h in (0.35, 0.45, 0.5, 0.55, 0.65):
    C = 0.5 * (v[:, None] ** (2 * h) + v[None, :] ** (2 * h) - np.abs(v[:, None] - v[None, :]) ** (2 * h))
    Lc = np.linalg.cholesky(C + 1e-14 * np.eye(len(v)))
    X = (Lc @ rs.normal(size=(len(v), 300))).T * V[-1] ** h
    Lam = X / V
    a0 = np.median(np.abs(Lam[:, 60]) * T[60] ** 2); a1 = np.median(np.abs(Lam[:, -1]) * T[-1] ** 2)
    sg = np.mean(np.sum(np.diff(np.sign(Lam), axis=1) != 0, axis=1)) / math.log(T[-1] / T[0])
    trend = math.log(a1 / a0) / math.log(T[-1] / T[60])
    print(f'h = {h:.2f}:  |Λ|·t² меняется как t^{trend:+.2f} (нужно 0);  смен знака на e-кратное время {sg:.2f}')
