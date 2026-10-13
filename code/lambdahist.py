"""История Λ(t) = ΔN(t)/V(t) для «монеток» трёх видов. Прошлый конус растёт,
V ∝ t⁴. Считаем, как часто Λ меняет знак и насколько она скачет за e-кратный
рост времени. Виды: (а) независимые монетки (Соркин); (б) монетки, связанные
блоками по 10⁴ событий (местная корреляция); (в) анти-связанные: внутри блока
сумма отклонений почти нулевая (гипероднородная примесь)."""
import numpy as np, math
rs = np.random.default_rng(1)
T = np.exp(np.linspace(math.log(10), math.log(1e6), 400))     # время в условных единицах
V = T ** 4
def history(kind, reps=200):
    out = []
    for r in range(reps):
        dV = np.diff(np.concatenate([[0], V]))
        if kind == 'независимые':
            dN = rs.normal(0, np.sqrt(dV))
        elif kind == 'блоки 10⁴':
            # число событий в блоке 10⁴, все монетки блока одинаковы: дисперсия ×10⁴ на событие
            dN = rs.normal(0, np.sqrt(dV * 1e4))
        else:
            # внутри блока почти сохраняется: дисперсия от «краёв» блоков ~ (dV)^(3/4)
            dN = rs.normal(0, dV ** 0.375)
        L = np.cumsum(dN) / V
        out.append(L)
    return np.array(out)
for kind in ('независимые', 'блоки 10⁴', 'анти-связанные'):
    H = history(kind)
    amp = np.median(np.abs(H[:, -1]) * T[-1] ** 2)          # |Λ|·t² в конце: «следит ли» за H²
    amp0 = np.median(np.abs(H[:, 100]) * T[100] ** 2)
    sgn = np.mean(np.sum(np.diff(np.sign(H), axis=1) != 0, axis=1)) / math.log(T[-1] / T[0])
    print(f'{kind:15s}  |Λ|·t² в начале {amp0:9.3g}, в конце {amp:9.3g}  (постоянно ⇒ Λ ~ H²);  смен знака на e-кратный рост времени: {sgn:.2f}')
