import sys; sys.path.insert(0, '/mnt/user-data/outputs')
import numpy as np
exec(open('/home/claude/order8.py').read().split("if __name__")[0])
exec(open('/home/claude/weaken.py').read().split("def rel_majority")[0])

def base_rel(U):
    N = U.shape[1]; R = np.ones((N, N), dtype=bool)
    for u in U: R &= (u[None, :] < u[:, None])
    return R

def rel_lead(U, delta, scale=0):
    """Обычное правило ПЛЮС: шкала `scale` должна опережать минимум на delta.
    Убирает связи, выделяет одну шкалу, транзитивность сохраняет."""
    R = base_rel(U)
    u = U[scale]
    return R & ((u[None, :] - u[:, None]) >= delta)

def rel_chiral(U):
    """Приращения по трём шкалам ранжируются; связь есть, только если
    ранжировка — ЧЁТНАЯ перестановка. Инвариантно к циклическому сдвигу шкал
    (поворот), не инвариантно к обмену двух (зеркало)."""
    R = base_rel(U)
    N = U.shape[1]
    d = [U[m][None, :] - U[m][:, None] for m in range(3)]
    # знак перестановки, ранжирующей (d0,d1,d2): чётная <=> знак определителя порядка +
    a, b, c = d
    sign = np.sign((a - b))*np.sign((b - c))*np.sign((a - c))
    return R & (sign > 0)

def acyclic(R):
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    indeg = Rb.sum(axis=1).astype(np.int64)
    ready = list(np.flatnonzero(indeg == 0)); seen = 0
    while ready:
        j = ready.pop(); seen += 1
        kids = np.flatnonzero(Rb[:, j])
        if kids.size:
            indeg[kids] -= 1
            ready.extend(kids[indeg[kids] == 0].tolist())
    return seen == n

def transitive(R, samples=150000, rs=None):
    r = rs if rs is not None else np.random.default_rng(0)
    Rb = np.ascontiguousarray(R, dtype=bool); n = len(Rb)
    ok = tot = 0
    for _ in range(samples):
        a, b, c = (int(v) for v in r.integers(0, n, 3))
        if Rb[b, a] and Rb[c, b]:
            tot += 1; ok += bool(Rb[c, a])
    return ok/max(tot, 1), tot

def rel_mixed(U, frac, mask=None, rs=None):
    """Доля frac пар подчиняется КИРАЛЬНОМУ условию (чётность приращений),
    остальные — обычному правилу. Маска фиксирована и от шкал не зависит,
    иначе сравнение с зеркалом было бы нечестным."""
    r = rs if rs is not None else np.random.default_rng(0)
    N = U.shape[1]
    R = base_rel(U)
    if frac <= 0: return R, mask
    if mask is None:
        mask = r.random((N, N)) < frac
        mask = mask | mask.T
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    sign = np.sign(a - b)*np.sign(b - c)*np.sign(a - c)
    return R & (~mask | (sign > 0)), mask

def rel_margin(U, t):
    """Киральное условие применяется, ТОЛЬКО если приращения по шкалам явно
    различаются (минимальная попарная разница >= t). Когда близки — обычное правило."""
    N = U.shape[1]
    R = base_rel(U)
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    ab = np.abs(a - b); bc = np.abs(b - c); ac = np.abs(a - c)
    clear = (np.minimum(np.minimum(ab, bc), ac) >= t)
    sign = np.sign(a - b)*np.sign(b - c)*np.sign(a - c)
    return R & (~clear | (sign > 0)), clear

def rel_maximal(U, mask, mode='parity'):
    """Внутри отмеченных пар нарушение РАЗНОЙ глубины.
    parity  — чётность приращений (отсекает половину, как раньше)
    strict  — требуем строгий циклический порядок приращений (отсекает 5/6)
    total   — требуем ОДНУ конкретную ранжировку (отсекает всё, кроме 1/6)"""
    N = U.shape[1]
    R = base_rel(U)
    a, b, c = [U[m][None, :] - U[m][:, None] for m in range(3)]
    if mode == 'parity':
        keep = (np.sign(a-b)*np.sign(b-c)*np.sign(a-c)) > 0
    elif mode == 'strict':
        keep = ((a > b) & (b > c)) | ((b > c) & (c > a)) | ((c > a) & (a > b))
    else:
        keep = (a > b) & (b > c)
    return R & (~mask | keep)
