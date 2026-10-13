#!/usr/bin/env python3
"""СБОЙ НАКЛОНЯЕТ КОНУСЫ: от Нордстрёма к Эйнштейну на языке порядка.

1. Мир: засев в слое; в центре неподвижный «сбой» — трубка событий r < 0.15.
2. ПОЛЕ ИЗ ПОРЯДКА: для каждого события — число связей от сбоя (defect-doc:
   идёт как 1/R), сглаженное по соседям. Φ = −κ·поле.
3. ПРАВИЛО «РАНЬШЕ» ТОЛЬКО МЕЖДУ БЛИЗКИМИ: 0 < Δt < ε и |Δx| < c·Δt, где
   c = 1 + 2Φ (так в слабом поле Шварцшильда: свет медленнее у массы;
   у Нордстрёма c = 1 и конусы не меняются).
4. Дальнее «раньше» — цепочкой близких (транзитивное замыкание).
5. ПРИБОР — чисто порядковый: испускатель слева, приёмник справа (часы-цепи);
   время прихода = первый тик приёмника в будущем тика испускателя.
   Задержка Шапиро = (приход с полем) − (приход без поля), те же точки.
Предсказание ОТО: Δ = 2M·ln((r₁+r₂+D)/(r₁+r₂−D)), растёт к центру как логарифм.
Нордстрём: Δ = 0 тождественно.
"""
import sys, math, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

T0, T1 = 0.0, 4.2
XB, YB, ZB = (-1.45, 1.45), (-0.55, 1.25), (-0.5, 0.5)
XE, XR = -1.2, 1.2; D = XR - XE
EPS = float(__import__('os').environ.get('EPS', '0.8'))
BS = (0.25, 0.4, 0.6, 0.9)
NT = 300                                  # тиков на линию часов

def build(rho, rs):
    V = (T1 - T0) * (XB[1] - XB[0]) * (YB[1] - YB[0]) * (ZB[1] - ZB[0])
    n = rs.poisson(rho * V)
    P = np.column_stack([rs.uniform(T0, T1, n), rs.uniform(*XB, n), rs.uniform(*YB, n), rs.uniform(*ZB, n)])
    # часы: испускатели и приёмники на высоте b (по y), z = 0
    lines = []
    for b in BS:
        tt = np.linspace(T0, T1, NT)
        lines.append(np.column_stack([tt, np.full(NT, XE), np.full(NT, b), np.zeros(NT)]))
        lines.append(np.column_stack([tt, np.full(NT, XR), np.full(NT, b), np.zeros(NT)]))
    Q = np.vstack([P] + lines)
    lab = np.concatenate([np.full(len(P), -1)] + [np.full(NT, k) for k in range(len(lines))])
    return Q, lab

def field_from_links(Q, lab, rs, nsrc=60):
    """число связей от событий сбоя (трубка r<0.15) к каждому событию"""
    r = np.linalg.norm(Q[:, 1:], axis=1)
    src = np.flatnonzero((lab < 0) & (r < 0.15))
    src = rs.choice(src, min(nsrc, len(src)), replace=False)
    cnt = np.zeros(len(Q))
    for y in src:
        d = Q - Q[y]; F = np.flatnonzero((d[:, 0] > 0) & (d[:, 0] > np.linalg.norm(d[:, 1:], axis=1)))
        Fp = Q[F]; minimal = np.ones(len(F), bool)
        for s in range(0, len(F), 1500):
            blk = Fp[s:s + 1500]
            dd = blk[:, None, :] - Fp[None, :, :]
            below = (dd[..., 0] > 0) & (dd[..., 0] > np.linalg.norm(dd[..., 1:], axis=-1))
            minimal[s:s + 1500] = ~below.any(1)
        cnt[F[minimal]] += 1
    return cnt * (len(np.flatnonzero((lab < 0) & (r < 0.15))) / len(src))

def smooth(Q, lab, cnt, rad=0.3):
    from scipy.spatial import cKDTree
    ev = lab < 0; tree = cKDTree(Q[ev])
    out = np.zeros(len(Q))
    for i, nb in enumerate(tree.query_ball_point(Q, rad)):
        out[i] = cnt[ev][nb].mean() if nb else 0.0
    return out

def local_dag(Q, phi):
    o = np.argsort(Q[:, 0]); Qs = Q[o]; ph = phi[o]
    rows, cols = [], []
    t = Qs[:, 0]
    for i in range(len(Qs)):
        j1 = np.searchsorted(t, t[i] + 1e-12); j2 = np.searchsorted(t, t[i] + EPS)
        if j2 <= j1: continue
        d = Qs[j1:j2] - Qs[i]
        c = 1 + 2 * 0.5 * (ph[i] + ph[j1:j2])
        ok = np.linalg.norm(d[:, 1:], axis=1) < c * d[:, 0]
        js = np.arange(j1, j2)[ok]; rows.extend([i] * len(js)); cols.extend(js)
    A = csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(len(Qs), len(Qs)))
    inv = np.empty(len(o), int); inv[o] = np.arange(len(o))
    return A, o, inv

def arrivals(Q, lab, A, o, inv):
    res = {}
    for k, b in enumerate(BS):
        em = np.flatnonzero(lab == 2 * k); rc = np.flatnonzero(lab == 2 * k + 1)
        tr = Q[rc, 0]; rc_sorted = inv[rc]
        out = []
        for e in em[(Q[em, 0] > 0.2) & (Q[em, 0] < 1.3)]:
            reach = breadth_first_order(A, inv[e], directed=True, return_predecessors=False)
            hit = np.isin(rc_sorted, reach)
            out.append(tr[hit].min() - Q[e, 0] if hit.any() else np.nan)
        res[b] = np.array(out)
    return res

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 350.0
    Mtarget = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    t0 = time.time(); D_all = {b: [] for b in BS}; Cfit = []; PRED = {}
    for s in range(seeds):
        rs = np.random.default_rng(700 + s)
        Q, lab = build(rho, rs)
        cnt = field_from_links(Q, lab, rs)
        f = smooth(Q, lab, cnt)
        ev = (lab < 0); R = np.linalg.norm(Q[ev, 1:], axis=1); sel = (R > 0.3) & (R < 1.2) & (Q[ev, 0] > 1.2)
        C = (f[ev][sel] @ (1 / R[sel])) / ((1 / R[sel]) @ (1 / R[sel])); Cfit.append(C)
        print(f'засев {s}: событий {ev.sum()}, поле·R ≈ {C:.3f} (подгонка f ≈ C/R)   [{time.time()-t0:.0f} с]'); sys.stdout.flush()
        kappa = Mtarget / C
        A0, o, inv = local_dag(Q, np.zeros(len(Q)))
        A1, _, _ = local_dag(Q, -kappa * f)
        print(f'   рёбер в местном графе: {A0.nnz}   κ = {kappa:.3f}')
        a0 = arrivals(Q, lab, A0, o, inv); a1 = arrivals(Q, lab, A1, o, inv)
        # предсказание по ФАКТИЧЕСКОМУ полю вдоль прямого светового пути: ∫(1/c − 1)dx
        from scipy.spatial import cKDTree
        tr = cKDTree(Q[ev]); fe = f[ev]
        for k, b in enumerate(BS):
            em = np.flatnonzero(lab == 2 * k); te = Q[em, 0][(Q[em, 0] > 0.2) & (Q[em, 0] < 1.3)]
            xs = np.linspace(XE, XR, 49); dx = xs[1] - xs[0]; preds = []
            for t_e in te[::4]:
                pts = np.column_stack([t_e + (xs - XE), xs, np.full_like(xs, b), np.zeros_like(xs)])
                nb = tr.query_ball_point(pts, 0.3)
                fv = np.array([fe[n].mean() if n else 0.0 for n in nb])
                c = 1 - 2 * kappa * fv; preds.append(((1 / c - 1) * dx).sum())
            PRED.setdefault(b, []).append(np.mean(preds))
        for b in BS:
            d = a1[b] - a0[b]; D_all[b].extend(d[np.isfinite(d)])
            print(f'   b={b}: свет без поля {np.nanmean(a0[b]):.3f} (D={D}),  задержка {np.nanmean(d):+.4f} ± {np.nanstd(d)/math.sqrt(np.isfinite(d).sum()):.4f}   [{time.time()-t0:.0f} с]')
            sys.stdout.flush()
    M = Mtarget
    print(f'\nМасса из поля: M = κ·C = {M:.4f}')
    print(f"{'b':>6}{'замер':>10}{'±':>8}{'∫(1/c−1) по полю':>18}{'ОТО 2M·ln':>12}{'Нордстрём':>11}")
    for b in BS:
        r1 = math.hypot(1.2, b); th = 2 * M * math.log((2 * r1 + D) / (2 * r1 - D))
        d = np.array(D_all[b]); print(f'{b:>6}{d.mean():>10.4f}{d.std()/math.sqrt(len(d)):>8.4f}{np.mean(PRED[b]):>18.4f}{th:>12.4f}{0:>11}')
