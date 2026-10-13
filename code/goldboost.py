"""Какие бусты есть у золотого мира 3+1 в плоскости (t, x).
Форма Q = τt² − x² (над Z[τ]). Буст переводит (1, 0) в (t, x) с Q = τ:
x² = τ(t² − 1), t, x ∈ Z[τ]. В физических единицах T = √τ·t, значит
cosh η = t. Ищем наименьшую быстроту η > 0 (это и есть «шаг» дискретной
симметрии Лоренца в этой плоскости; у решётки 1+1 он равен 2·ln τ = 0.962)."""
import math, itertools
TAU = (1 + 5 ** .5) / 2; TP = -1 / TAU
def val(a, b): return a + b * TAU
def is_square_in_Zt(p, q):
    """p + qτ = (c + dτ)²?  (c+dτ)² = c² + d² + (2cd + d²)τ"""
    v = p + q * TAU
    if v < 0: return None
    s = math.sqrt(v)
    for d in range(-200, 201):
        c = round(s - d * TAU)
        for cc in (c - 1, c, c + 1):
            if cc * cc + d * d == p and 2 * cc * d + d * d == q: return (cc, d)
    return None
best = []
for a in range(-300, 301):
    for b in range(-300, 301):
        t = val(a, b)
        if t <= 1.0000001 or t > 2000: continue
        # τ(t²−1) в виде p + qτ:  t² = a²+b² + (2ab+b²)τ ;  t² − 1 = (a²+b²−1) + (2ab+b²)τ
        P, Q = a * a + b * b - 1, 2 * a * b + b * b
        # умножение на τ: τ·(P + Qτ) = Qτ² + Pτ = Q + (P+Q)τ
        p2, q2 = Q, P + Q
        r = is_square_in_Zt(p2, q2)
        if r: best.append((math.acosh(t), (a, b), r))
best.sort()
for eta, (a, b), (c, d) in best[:6]:
    print(f'η = {eta:.4f}   t = {a}+{b}τ = {val(a,b):.4f}   x = {c}+{d}τ   (η/ln τ = {eta/math.log(TAU):.3f})')
print(f'для сравнения: 1+1 решётка, 2·ln τ = {2*math.log(TAU):.4f}')
