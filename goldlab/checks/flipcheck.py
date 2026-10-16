"""СПИНОР ИЗ ПОРЯДКА — что выводится. Обратный оператор Дирака в 1+1 (киральный базис, u = t + x, v = t − x):
S = [[2∂_v G, −i m G], [−i m G, 2∂_u G]], G — запаздывающая функция Грина Клейна–Гордона. На кристалле G = K Джонстона (сумма по
всем цепочкам — из порядка). Значит, СМЕНА компоненты (вправо → влево) — это −i·m·K: «монета» не выбирается, её задаёт порядок.
Проверка: амплитуда смены компоненты у блуждания по нитям (точечный источник «вправо», читаем «влево») против −i·m·K(y → x)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, fermions as Fe
for L, m in ((12, 1.0), (16, 1.0), (16, 0.5)):
    w = Wd.gold11(4.0, L); t, x = w.t, w.x; W = Fe.Walk(w); K = Pr.johnston11(w.X, 4.0, m)
    y = int(np.argmin(np.abs(t + L + 3) + np.abs(x)))                   # источник низко в ромбе
    y = int(np.argmin(np.abs(t - (-L + 3)) * 4 + np.abs(x)))
    f = np.zeros(w.n); f[y] = 1.0; aR, aL, _ = W.run(m, f, 0 * f)        # вход «вправо» в событие y
    # выход из событий — после монеты; читаем входы (это и есть амплитуда, пришедшая в x)
    fut = (t > t[y] + 0.5) & (np.abs(x - x[y]) < (t - t[y])) & (t < L - 2)
    for name, cvel in (('кристалл (Джонстон)', 1.0),):
        kk = -1j * m * K[y, fut]; aw = aL[fut]
        r = np.vdot(kk, aw) / np.vdot(kk, kk); corr = abs(np.vdot(kk, aw)) / (np.linalg.norm(kk) * np.linalg.norm(aw))
        # где лежит амплитуда: внутри конуса скорости нитей c или у светового конуса
        inside_c = np.abs(x[fut] - x[y]) < W.c * (t[fut] - t[y])
        share = (np.abs(aw[inside_c]) ** 2).sum() / (np.abs(aw) ** 2).sum(); shareK = (np.abs(kk[inside_c]) ** 2).sum() / (np.abs(kk) ** 2).sum()
        print(f'L = {L}, m = {m}: корреляция |⟨−imK, ψ_L⟩| = {corr:.3f}, множитель {abs(r):.3f}; доля амплитуды внутри конуса нитей (c = {W.c:.2f}): блуждание {share:.2f}, −imK {shareK:.2f}', flush=True)
