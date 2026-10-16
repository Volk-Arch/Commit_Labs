#!/usr/bin/env python3
"""ПРОВЕРКА «ПОЛЯРНОГО СОБЫТИЯ» (closedsym.py): интерференция, пара от накачки, Белл на световодах.
Динамика вещественна и точно зеркальна (P до 1e−14), норма мир + тень точная."""
import sys, math, itertools, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closedsym import ClosedSym, CH
from beamsplit import smax
sys.argv = _a
S = ClosedSym(); X = S.X; n = S.n; w = S.w
ea = np.array([e[0] for e in S.edges]); eb = np.array([e[1] for e in S.edges])

# ---------- 1) две щели ----------
def crossing(c):
    t0, t1 = X[ea, 0], X[eb, 0]; cr = (t0 < c) & (t1 >= c)
    xc = X[ea, 1] + (X[eb, 1] - X[ea, 1]) * (c - t0) / np.where(t1 - t0 == 0, 1, t1 - t0)
    return np.nonzero(cr)[0], xc[cr]
src = int(np.argmin(((X - np.array([-10.0, 0.0])) ** 2).sum(1)))
wallE, _ = crossing(-3.0); scrE, scrx = crossing(5.0); ws = set(wallE.tolist())
for dsl in (2.0, 4.0):
    cand = np.nonzero(np.abs(X[:, 0] + 3) < 0.45)[0]
    S1 = int(cand[np.argmin((X[cand, 1] - dsl / 2) ** 2)]); S2 = int(S.Pimg[S1])
    out = {}
    for name, op in (('обе', {S1, S2}), ('1', {S1}), ('2', {S2})):
        block = {int(e): 'w' for e in wallE if ea[e] not in op and eb[e] not in op}
        for e in scrE:
            if e not in ws: block[int(e)] = 's'
        ab, resk, amp, _ = S.run({src: np.array([1.0 + 0j])}, block=block)
        wall = sum(abs(ab[e][0]) ** 2 for e in ab if block[e] == 'w'); xs = dict(zip(scrE.tolist(), scrx))
        scr = {e: abs(ab[e][0]) ** 2 for e in ab if block[e] == 's'}
        bins = np.linspace(-9, 9, 37); Ib = np.histogram([xs[e] for e in scr], bins, weights=list(scr.values()))[0]
        out[name] = (Ib, wall + Ib.sum() + resk[X[:, 0] < 5.0, 0].sum())
    Ic, inc = out['обе'][0], out['1'][0] + out['2'][0]; m = inc > 0.05 * inc.max(); r = (Ic - inc) / np.maximum(inc, 1e-30)
    print(f'1) две щели x = ±{X[S1, 1]:.2f}: на экран {Ic.sum():.4f}; интерференционный член от {r[m].min():+.2f} до {r[m].max():+.2f}; '
          f'полная вероятность (экран + стенка + тень) {out["обе"][1]:.6f}'); sys.stdout.flush()

# ---------- 2) пара от накачки и Белл ----------
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
WIDE = '--wide' in sys.argv
lo_t, hi_t, lo_x, hi_x = (1.0, 5.0, 0.2, 3.5) if WIDE else (1.8, 3.4, 0.3, 2.0)
quads = []
for A1 in [k for k in range(n) if lo_t <= Xa[k, 0] <= hi_t and lo_x <= Xa[k, 1] <= hi_x]:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    B1, B2 = int(S.Pimg[A1]), int(S.Pimg[A2]); quads.append((A1, A2, B1, B2))
rec = sorted({k for q in quads for k in q})
beta, emit, col = S.backward(rec)

def pump_field(inj):
    _, _, _, r = S.run(inj, record=set(range(n))); return np.array([r[k][0] for k in range(n)]).real
beam = {j: 1.0 for j in range(n) if abs(X[j, 0] + 11) <= 0.45 and abs(X[j, 1]) <= 3.0}
p_beam = pump_field({k: np.array([v / math.sqrt(len(beam))]) for k, v in beam.items()})
strip = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]

def vertex(k, kind):
    lo = S.legs[k][1]; m = len(lo)
    if kind == 'бозон': return np.ones((m, m)) - np.eye(m)
    if kind == 'фермион':                 # антисимметрия по порядку быстрот (меняет знак при зеркале)
        r = np.argsort(np.argsort(S.d[lo, 1] / S.d[lo, 0])); return np.sign(r[None, :] - r[:, None]).astype(float)

def Mq(p, srcs, kind, q, mixed=False):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2)); rho = np.zeros((4, 4))
    for s in srcs:
        if kind == 'руками': Ms = np.outer(emit[s, iA], emit[s, iB])
        else:
            lo = S.legs[s][1]
            if len(lo) == 0: continue
            Ms = beta[lo][:, iA].T @ vertex(s, kind) @ beta[lo][:, iB]
        Ms = p[s] * Ms; M = M + Ms; v = Ms.reshape(4); rho += np.outer(v, v)
    return rho if mixed else M

def stats(M):
    nm = np.linalg.norm(M)
    if nm < 1e-300: return 0.0, 0.0
    Mn = M / nm; return 2 * abs(np.linalg.det(Mn)), smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj()))

ones = np.ones(n)
print(f'\n2) четвёрок {len(quads)} (зеркальные: B = P(A) точно; A₂ = h(A₁) — h уже не точная симметрия динамики)')
for name, p, srcs, kind in (('руками (впрыск в событие), p = 1, полоса', ones, strip, 'руками'),
                            ('бозон, накачка-пучок, полоса', p_beam, strip, 'бозон'),
                            ('фермион, накачка-пучок, полоса', p_beam, strip, 'фермион'),
                            ('бозон, накачка-пучок, весь мир', p_beam, list(range(n)), 'бозон')):
    r = np.array([stats(Mq(p, srcs, kind, q)) for q in quads])
    Smix = max(smax(Mq(p, srcs, kind, q, True) / np.trace(Mq(p, srcs, kind, q, True))) for q in quads[:12])
    sym = np.median([np.abs(Mq(p, srcs, kind, q) - Mq(p, srcs, kind, q).T).max() / max(np.abs(Mq(p, srcs, kind, q)).max(), 1e-300) for q in quads[:12]])
    print(f'   {name:42s}: C медиана {np.median(r[:, 0]):.3f}, макс {r[:, 0].max():.3f}; S ид. макс {r[:, 1].max():.3f}, > 2 у {np.sum(r[:, 1] > 2.0005)}/{len(r)}; '
          f'место записано: S ≤ {Smix:.3f}; |M − Mᵀ| {sym:.1e}'); sys.stdout.flush()

# ---------- 3) накачка нечётная по зеркалу (как пучок в первой поперечной моде) ----------
print('\n3) нечётная накачка: амплитуда пучка ∝ знак x (события на оси — 0); точное зеркало ⇒ M = −Mᵀ ⇒ синглет?')
beam_odd = {j: np.sign(X[j, 1]) for j in range(n) if abs(X[j, 0] + 11) <= 0.45 and abs(X[j, 1]) <= 3.0 and X[j, 1] != 0}
p_odd = pump_field({k: np.array([v / math.sqrt(len(beam_odd))]) for k, v in beam_odd.items()})
for srcname, srcs in (('полоса', strip), ('весь мир', list(range(n)))):
    for kind in ('бозон', 'фермион', 'руками'):
        r = []; asym = []; eta = []
        for q in quads:
            M = Mq(p_odd, srcs, kind, q); r.append(stats(M)); eta.append(np.linalg.norm(M) ** 2)
            asym.append(np.abs(M + M.T).max() / max(np.abs(M).max(), 1e-300))
        r = np.array(r)
        Smix = max(smax(Mq(p_odd, srcs, kind, q, True) / np.trace(Mq(p_odd, srcs, kind, q, True))) for q in quads)
        print(f'   {kind:8s}, {srcname:8s}: C мин {r[:, 0].min():.6f}, медиана {np.median(r[:, 0]):.6f}; S ид. мин {r[:, 1].min():.4f}; '
              f'|M + Mᵀ| ≤ {max(asym):.1e}; |M|² от {min(eta):.1e} до {max(eta):.1e}; место записано: S ≤ {Smix:.3f}')
        sys.stdout.flush()
