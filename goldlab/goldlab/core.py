"""ЯДРО. Весь мир — одна формула и одно отношение; всё остальное в goldlab либо выведено отсюда, либо помечено «взято руками».

    τ = (1 + √5)/2,  τ′ = −1/τ  (сопряжение Галуа: a + bτ ↦ a + bτ′)
    координаты события — числа a + bτ (время ещё × √τ, чтобы форма была τt² − x²);
    «тень» события — те же числа с τ → τ′ (время × √|τ′|).

    СОБЫТИЕ ЕСТЬ  ⟺  ТЕНЬ В ШАРЕ радиуса r          (r задаёт только плотность событий)
    i РАНЬШЕ j    ⟺  j − i во внутреннем световом конусе

Итератора у мира нет: правило задаёт сразу всё пространство-время. Но время у кристалла дискретно (значения t — конечный
набор чисел Z[τ] с тенью в окне), поэтому мир можно честно перечислять слоями: layers()."""
import math
import numpy as np

TAU = (1 + 5 ** .5) / 2          # золотое сечение
TAU_ = -1 / TAU                  # его «тень» τ′
S5 = 5 ** .5
ST, SI = math.sqrt(TAU), math.sqrt(-TAU_)   # масштаб времени в мире и в тени

def numbers(lo, hi, rlo, rhi):
    """все c = a + bτ с lo ≤ c ≤ hi и тенью c′ = a + bτ′ в [rlo, rhi]; возвращает (c, c′)."""
    C, Cp = [], []
    for b in range(math.floor((lo - rhi) / S5) - 1, math.ceil((hi - rlo) / S5) + 2):
        alo = max(math.ceil(lo - b * TAU), math.ceil(rlo - b * TAU_)); ahi = min(math.floor(hi - b * TAU), math.floor(rhi - b * TAU_))
        if ahi < alo: continue
        a = np.arange(alo, ahi + 1, dtype=float); C.append(a + b * TAU); Cp.append(a + b * TAU_)
    if not C: return np.empty(0), np.empty(0)
    return np.concatenate(C), np.concatenate(Cp)

def crystal(box, r):
    """ФОРМУЛА МИРА. box = [(t_lo, t_hi), (x_lo, x_hi), ...] — где смотрим (1+1, 2+1, 3+1 — одно и то же правило).
    Возвращает события (t, x, …), у которых |тень|² = |τ′|·t′² + x′² + … ≤ r²."""
    (tl, th), *space = box
    c, cp = numbers(tl / ST, th / ST, -r / SI, r / SI); axes = [(c * ST, (cp * SI) ** 2)]
    for lo, hi in space:
        c, cp = numbers(lo, hi, -r, r); axes.append((c, cp ** 2))
    P = np.zeros((1, len(box))); W = np.zeros(1)
    for k, (c, w2) in enumerate(axes):                       # перебор по осям, сразу выкидывая то, чья тень уже вне шара
        P = np.repeat(P, len(c), 0); W = np.repeat(W, len(c))
        P[:, k] = np.tile(c, len(P) // len(c)); W += np.tile(w2, len(W) // len(c))
        keep = W <= r * r; P, W = P[keep], W[keep]
    return P[np.argsort(P[:, 0], kind='stable')]

def precedes(X):
    """ПОРЯДОК. C[i, j] = True, если i раньше j (j − i во внутреннем конусе)."""
    d = X[None, :, :] - X[:, None, :]
    return (d[..., 0] > 0) & (d[..., 0] ** 2 - (d[..., 1:] ** 2).sum(-1) > 1e-12)

def layers(X, tol=1e-9):
    """ПЕРЕЧИСЛЕНИЕ СЛОЯМИ ВРЕМЕНИ: (t, индексы событий слоя) снизу вверх. Слои — точные значения t кристалла."""
    t = X[:, 0]; o = np.argsort(t, kind='stable'); cut = np.nonzero(np.diff(t[o]) > tol)[0] + 1
    for g in np.split(o, cut): yield float(t[g[0]]), g

# ---------- выведено: окно ↔ плотность (сколько событий на единицу объёма даёт шар радиуса r) ----------
def r_for_density(rho, dim):
    """1+1: ρ = πr²/5; 3+1: ρ = (π²/2)r⁴/25 (объём шара тени / √5^dim)."""
    if dim == 2: return math.sqrt(5 * rho / math.pi)
    if dim == 4: return (25 * rho / (math.pi ** 2 / 2)) ** 0.25
    raise ValueError(dim)
