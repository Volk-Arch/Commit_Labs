#!/usr/bin/env python3
"""ПЕТЛЯ «ПОРЯДОК -> ЧИСЛА -> ПОРЯДОК» С КРУГЛЫМ КОНУСОМ.

Прежде (PICTURE §5д): числа = длины цепей до опор, правило = «раньше по
всем» (ортант). Неподвижная точка — цепь, из любого начала.

Теперь: правило = круглый конус. Отображение F(порядок):
  * в мире есть 5 «линий вещества» — цепей-часов (их события помечены, и
    метка переживает итерации; сами их отношения берутся из порядка);
  * радар по порядку: для события x и линии i  p = последний тик в прошлом,
    f = первый в будущем;  время t_i = (p+f)/2, радиус r_i = (f−p)/2;
  * положения линий: радар между линиями -> взаимные расстояния -> MDS;
    сдвиги часов — синхронизацией по Эйнштейну (тоже радаром);
  * положение события: пересечение 5 сфер (наименьшие квадраты), время —
    среднее синхронизированных t_i;
  * новый порядок: ΔT > |ΔX| на восстановленных точках.
Ничего, кроме порядка и меток линий, на входе F нет.
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '/home/claude')
from core import frac_pairs, mm_dim, height, links_of, spectral

NL, M = 5, 200                          # линий, тиков на линию

def make_world(N, rs):
    X = []
    while len(X) < N:
        c = rs.uniform(-1, 1, (4 * N, 4)); c = c[np.abs(c[:, 0]) + np.linalg.norm(c[:, 1:], axis=1) <= 1]; X.extend(c)
    X = np.array(X[:N])
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / math.sqrt(3)
    A = np.vstack([0.3 * v, [[0, 0, 0]]])
    lines = []
    for a in A:
        T = np.linspace(-2.5, 2.5, M)          # линии длиннее мира событий
        lines.append(np.column_stack([T, np.repeat(a[None], M, 0)]))
    P = np.vstack([X] + lines)
    lab = np.full(len(P), -1); tick = np.full(len(P), -1)
    for i in range(NL):
        idx = np.arange(N + i * M, N + (i + 1) * M); lab[idx] = i; tick[idx] = np.arange(M)
    return P, lab, tick

def mink(P):
    dt = P[:, None, 0] - P[None, :, 0]; dx = np.linalg.norm(P[:, None, 1:] - P[None, :, 1:], axis=-1)
    R = dt > dx; np.fill_diagonal(R, False); return R

def box(P):
    R = np.ones((len(P), len(P)), bool)
    # ортант по четырём «световым» шкалам t ± x, t ± y (покомпонентный порядок, как в проекте)
    S = np.column_stack([P[:, 0] + P[:, 1], P[:, 0] - P[:, 1], P[:, 0] + P[:, 2], P[:, 0] - P[:, 2]])
    for k in range(4): R &= S[None, :, k] < S[:, None, k]
    np.fill_diagonal(R, False); return R

def radar(R, lab, tick):
    n = len(R); T = np.full((n, NL), np.nan); Rr = np.full((n, NL), np.nan)
    for i in range(NL):
        idx = np.flatnonzero(lab == i); idx = idx[np.argsort(tick[idx])]
        before = R[:, idx]                        # [x, k]: тик k раньше x
        after = R[idx, :].T                       # [x, k]: x раньше тика k
        has_p = before.any(1); has_f = after.any(1)
        p = np.where(has_p, M - 1 - np.argmax(before[:, ::-1], 1), -1)
        f = np.where(has_f, np.argmax(after, 1), -1)
        ok = has_p & has_f
        T[ok, i] = (p[ok] + f[ok]) / 2; Rr[ok, i] = (f[ok] - p[ok] - 1) / 2   # несмещённо: тики строго до/после
        own = lab == i; T[own, i] = tick[own]; Rr[own, i] = 0
    return T, Rr

def F(R, lab, tick):
    T, Rr = radar(R, lab, tick)
    # расстояния и сдвиги между линиями — по их собственным тикам
    D = np.zeros((NL, NL)); off = np.zeros(NL)
    for j in range(NL):
        own = lab == j
        for i in range(NL):
            if i != j:
                D[i, j] = np.nanmedian(Rr[own, i])
        if j: off[j] = np.nanmedian(T[lab == j, 0] - T[lab == j, j])   # перевод часов j в часы 0
    D = (D + D.T) / 2
    if not np.isfinite(D).all(): return np.zeros_like(R), np.zeros(len(R), bool)
    J = np.eye(NL) - 1 / NL; B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); w, V = w[::-1][:3], V[:, ::-1][:, :3]
    A = V * np.sqrt(np.maximum(w, 1e-12))
    # события: время — среднее синхронизированных, положение — 5 сфер
    Tsync = T + off[None, :]
    ok = np.isfinite(Rr).sum(1) >= 4
    Tm = np.nanmean(Tsync, 1)
    X = np.zeros((len(R), 3))
    rr = Rr.copy()
    # линейное начальное приближение
    for it in range(12):
        d = X[:, None, :] - A[None]; dist = np.linalg.norm(d, axis=-1) + 1e-9
        res = np.where(np.isfinite(rr), dist - rr, 0.0)
        Jm = np.where(np.isfinite(rr)[..., None], d / dist[..., None], 0.0)
        H = np.einsum('nij,nik->njk', Jm, Jm) + 1e-6 * np.eye(3)
        g = np.einsum('nij,ni->nj', Jm, res)
        X -= np.linalg.solve(H, g[..., None])[..., 0]
    P = np.column_stack([Tm, X])
    for i in range(NL):                          # линия знает себя: свои часы и место
        own = lab == i; P[own, 0] = tick[own] + off[i]; P[own, 1:] = A[i]; ok[own] = True
    P[~ok] = np.nan
    F.last_P = P
    Rn = mink(np.nan_to_num(P, nan=1e9))
    Rn[~ok, :] = False; Rn[:, ~ok] = False
    return Rn, ok

def feats(R, keep):
    S = R[np.ix_(keep, keep)]; f = frac_pairs(S)
    return f, mm_dim(f) if 0 < f < 1 else float('nan'), height(S)[0], links_of(S).sum() / len(S)

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rs = np.random.default_rng(3)
    P, lab, tick = make_world(N, rs)
    Rtrue = mink(P)
    ev = lab < 0
    def report(name, R, Rprev, keep):
        f, d, h, l = feats(R, keep & ev)
        ch = (R ^ Rprev)[np.ix_(keep & ev, keep & ev)].sum() / max(Rprev[np.ix_(keep & ev, keep & ev)].sum(), 1)
        tr = (R ^ Rtrue)[np.ix_(keep & ev, keep & ev)].sum() / Rtrue[np.ix_(keep & ev, keep & ev)].sum()
        print(f'   {name:8s} событий {int((keep & ev).sum()):5d}  доля {f:.4f} разм {d:5.2f} выс {h:3d} связ/соб {l:6.2f}'
              f' | изменилось {ch:6.3f} | от Минковского {tr:6.3f}')
    starts = {'МИНКОВСКИЙ': Rtrue, 'ОРТАНТ (ящик)': box(P)}
    # шумный Минковский: 20% точек сдвинуты на 0.3
    Pn = P.copy(); sel = (lab < 0) & (rs.random(len(P)) < 0.2); Pn[sel] += rs.normal(0, 0.3, (sel.sum(), 4))
    starts['ШУМНЫЙ МИНК.'] = mink(Pn)
    # случайный 2D-порядок поверх тех же событий (линии — цепи)
    U = rs.random((len(P), 2)); U[lab >= 0] = np.column_stack([tick[lab >= 0] / M, tick[lab >= 0] / M]) * 0.999 + 1e-4 * lab[lab >= 0][:, None]
    R2 = (U[None, :, 0] < U[:, None, 0]) & (U[None, :, 1] < U[:, None, 1]); np.fill_diagonal(R2, False)
    starts['СЛУЧ. 2D'] = R2
    t0 = time.time()
    for name, R in starts.items():
        print(f'СТАРТ: {name}')
        keep = np.ones(len(P), bool); report('шаг 0', R, R, keep)
        for step in range(1, 9):
            Rn, ok = F(R, lab, tick); keep &= ok
            if keep.sum() < 50: print('   мир распался'); break
            report(f'шаг {step}', Rn, R, keep); R = Rn
        Pq = F.last_P[keep & ev]; Pq = Pq - np.nanmean(Pq, 0)
        rad = np.linalg.norm(Pq[:, 1:], axis=1); tt = Pq[:, 0]
        print(f'   точки неподвижной точки: разброс времени {np.nanstd(tt):6.2f}, радиус медиана {np.nanmedian(rad):6.2f}, '
              f'отношение {np.nanmedian(rad)/np.nanstd(tt):5.2f}   (у засева ромба: {np.std(P[ev,0]):.2f}, {np.median(np.linalg.norm(P[ev,1:],axis=1))*M/5:.2f} в тиках, отношение {np.median(np.linalg.norm(P[ev,1:],axis=1))/np.std(P[ev,0]):.2f})')
        print(f'   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
