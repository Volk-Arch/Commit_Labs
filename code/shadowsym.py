"""Скрытая симметрия тени. Буст кристалла g (элементы из Z[τ]) действует на физическое
пространство-время как преобразование Лоренца, а на тень — сопряжённой матрицей g' (τ → −1/τ),
и там это ВРАЩЕНИЕ (форма тени определённая). Смотрим: какой поворот тени отвечает
элементарному бусту, плотен ли образ группы в SO(4), есть ли свобода тени без движения мира."""
import numpy as np, math
from scipy.stats import special_ortho_group, ks_2samp
TAU = (1 + 5 ** .5) / 2; TP = -1 / TAU
def boost_lab(i, conj=False):
    """буст в плоскости (t, x_i) в метках: t' = τt + x, x' = τt + τx (сохраняет τt² − x²)"""
    tv = TP if conj else TAU
    g = np.eye(4); g[0, 0] = tv; g[0, i] = 1; g[i, 0] = tv; g[i, i] = tv; return g
def to_phys(g, conj=False):
    s = math.sqrt(-TP) if conj else math.sqrt(TAU)
    S = np.diag([s, 1, 1, 1]); return S @ g @ np.linalg.inv(S)
gens_lab = []
for i in (1, 2, 3):
    gens_lab.append(i)
# элементарный буст по x
P = to_phys(boost_lab(1)); Sh = to_phys(boost_lab(1, True), True)
eta0 = math.acosh(P[0, 0]); th0 = math.degrees(math.acos(Sh[0, 0]))
print(f'элементарный буст: в мире быстрота η₀ = {eta0:.4f} (cosh η₀ = τ);  в тени поворот на θ₀ = {th0:.2f}° (cos θ₀ = −1/τ)')
print(f'   проверка: тень ортогональна? {np.allclose(Sh @ Sh.T, np.eye(4))};  мир сохраняет интервал? '
      f'{np.allclose(P.T @ np.diag([1,-1,-1,-1]) @ P, np.diag([1,-1,-1,-1]))}')
print(f'   θ₀/360° = {th0/360:.6f} — иррационально: k бустов поворачивают тень на kθ₀, углы всюду плотны')
for k in (1, 2, 3, 5):
    Pk = np.linalg.matrix_power(P, k); Sk = np.linalg.matrix_power(Sh, k)
    print(f'   {k} буста: η = {math.acosh(Pk[0,0]):.4f} = {k}η₀;  поворот тени {math.degrees(math.acos(np.clip(Sk[0,0],-1,1))):.2f}° (kθ₀ mod 360 = {(k*th0)%360:.2f}°)')
# образующие: бусты ±x, ±y, ±z и перестановки/отражения осей
def perm(pi, signs):
    g = np.zeros((4, 4)); g[0, 0] = 1
    for a, (b, s) in enumerate(zip(pi, signs)): g[1 + a, 1 + b] = s
    return g
gensP, gensS = [], []
for i in (1, 2, 3):
    for inv in (False, True):
        gp, gs = to_phys(boost_lab(i)), to_phys(boost_lab(i, True), True)
        if inv: gp, gs = np.linalg.inv(gp), np.linalg.inv(gs)
        gensP.append(gp); gensS.append(gs)
for pi, sg in (((1, 0, 2), (1, 1, 1)), ((0, 2, 1), (1, 1, 1)), ((0, 1, 2), (-1, 1, 1))):
    g = perm(pi, sg); gensP.append(g); gensS.append(g)
rng = np.random.default_rng(3)
def word(L):
    gp, gs = np.eye(4), np.eye(4)
    for _ in range(L):
        k = rng.integers(len(gensP)); gp = gp @ gensP[k]; gs = gs @ gensS[k]
    return gp, gs
traces, maxeta = [], []
for _ in range(4000):
    gp, gs = word(25)
    if np.linalg.det(gs) < 0: gs = gs @ np.diag([1, 1, 1, -1])
    traces.append(np.trace(gs)); maxeta.append(math.acosh(max(1.0, gp[0, 0])))
haar = [np.trace(special_ortho_group.rvs(4, random_state=int(s))) for s in rng.integers(0, 2**31, 4000)]
st = ks_2samp(traces, haar)
print(f'\nслучайные слова длины 25: след поворота тени против равномерного (Хаар) SO(4): KS p = {st.pvalue:.3f} '
      f'(p > 0.05 — неотличимо, образ плотен); медианный буст в мире η = {np.median(maxeta):.1f}')
# элементы, не двигающие время (g00 = 1): какие повороты тени у них?
print('\nГруппа покоя (g₀₀ = 1): в тени тоже g\'₀₀ = 1, пространственная часть — целые ортогональные матрицы над Z[τ].')
print('Столбец v с v·v = 1 и v\'·v\' = 1 ⇒ каждый элемент |a| ≤ 1 и |a\'| ≤ 1 ⇒ a ∈ {0, ±1}: только знаковые перестановки,')
print('группа куба (48 элементов). Икосаэдра (и E8 через Маккея) у этого кристалла нет.')
