#!/usr/bin/env python3
"""ЧТО ВЫБИРАЕТ ВЕРШИНУ РАСПАДА И СТЕПЕНЬ ЗАПУТАННОСТИ (продолжение pairsrc.py).

1) Сохранение «импульса» в решётке: d_c = d_a + d_b для векторов Z[τ]² (мир + тень) — есть ли такие вершины?
2) Симметрия события. Квант накачки поглощается в симметричной моде; выход события U·s различает ноги только
   фазой «поворот = i» (классы L/R). Вершина, не различающая неразличимое, инвариантна к перестановкам ног
   внутри класса ⇒ симметричные части: диагональ и вне диагонали в каждом классе, перекрёстная (5 чисел);
   антисимметричная — ТОЛЬКО перекрёстная L×R (одна). Значит: бозонная вершина — «бесплатно», фермионная
   требует двух классов (хиральности) и меняет знак при зеркале.
   Проверяем: хиральный фермион ε(L,R); случайные бозоны из 5-параметрического семейства — зависит ли C?
3) Фазовый синхронизм: растёт ли запутанность (и «диагональность» — пары в сопряжённых направлениях),
   если среда толще/шире и накачка шире (больше когерентная сумма)?"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closed2 import Closed, CH
from beamsplit import smax
sys.argv = _a
rng = np.random.default_rng(7)

S = Closed(turn=True); X = S.X; n = S.n; w = S.w; Wsh = w.W
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)
WIDE = '--wide' in sys.argv

def vec(e):
    a, b = S.edges[e][:2]; return np.concatenate([X[b] - X[a], Wsh[b] - Wsh[a]])

# ---- 1) сохранение вектора решётки ----
hits = 0; near = []
for s in range(n):
    li, lo, _ = S.legs[s]
    li = [e for e in li if S.edges[e][2] == 0]; lo = [e for e in lo if S.edges[e][2] == 0]
    if len(lo) < 2 or not li: continue
    vo = np.array([vec(e) for e in lo]); vi = np.array([vec(e) for e in li])
    sm = vo[:, None, :] + vo[None, :, :]
    d = np.abs(sm[None] - vi[:, None, None, :])
    hits += int((d.sum(-1) < 1e-7).sum()); near.append(d[..., :2].sum(-1).min())
print(f'1) вершин с d_накачки = d_a + d_b точно (мир + тень): {hits}; ближайшее совпадение только в мире: {min(near):.3f}')
print('   (сумма двух звеньев никогда не звено — точного «импульса» у звена нет; синхронизм надо искать в интерференции)')

# ---- захват и обратный проход ----
lo_t, hi_t, lo_x, hi_x = (1.0, 5.0, 0.2, 3.5) if WIDE else (1.8, 3.4, 0.3, 2.0)
cands = [k for k in range(n) if lo_t <= Xa[k, 0] <= hi_t and lo_x <= Xa[k, 1] <= hi_x]
quads = []
for A1 in cands:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
    if min(B1, B2) < 0: continue
    quads.append((A1, A2, B1, B2))
rec = sorted({k for q in quads for k in q}); col = {k: i for i, k in enumerate(rec)}
beta = np.zeros((S.E, len(rec)), complex)
for k in reversed(S.order):
    li, lo, Um = S.legs[k]; D = len(Um)
    bout = np.zeros((D, len(rec)), complex); bout[:len(lo)] = beta[lo]
    bin_ = Um.T @ bout
    if k in col: bin_[:, col[k]] += 1 / math.sqrt(D)
    beta[li] = bin_[:len(li)]

def pump_field(inj):
    amp = np.zeros(S.E, complex); p = np.zeros(n, complex)
    for k in S.order:
        li, lo, Um = S.legs[k]; D = len(Um)
        vin = np.zeros(D, complex); vin[:len(li)] = amp[li]
        if k in inj: vin[len(li) if len(li) < D else 0] += inj[k]
        p[k] = vin.sum() / math.sqrt(D); amp[lo] = (Um @ vin)[:len(lo)]
    return p

def cls(k):
    lo = S.legs[k][1]; return np.array([S.edges[e][3] == 1 for e in lo])       # True = класс R (фаза i)
def ph(k): return np.where(cls(k), 1j, 1.0)

def vertex(k, kind, par=None):
    m = len(S.legs[k][1]); c = cls(k); P = ph(k); off = 1 - np.eye(m)
    if kind == 'бозон через U': return np.outer(P, P) * off
    if kind == 'фермион хиральный':                      # +1 для (R, L), −1 для (L, R): единственный антисимм. инвариант
        return (np.outer(c, ~c).astype(float) - np.outer(~c, c).astype(float)) * np.outer(P, P)
    if kind == 'бозон-семейство':                        # 5 чисел: диаг L, диаг R, вне диаг L, вне диаг R, перекрёстная
        dL, dR, oL, oR, x = par
        f = np.where(np.outer(c, c), oR, 0) + np.where(np.outer(~c, ~c), oL, 0) + np.where(np.outer(c, ~c) | np.outer(~c, c), x, 0)
        f = f * off + np.diag(np.where(c, dR, dL)); return f * np.outer(P, P)
    raise ValueError(kind)

def Mquad(p, srcs, kind, q, par=None):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2), complex)
    for s in srcs:
        lo = S.legs[s][1]
        if len(lo) == 0 or p[s] == 0: continue
        M += p[s] * (beta[lo][:, iA].T @ vertex(s, kind, par) @ beta[lo][:, iB])
    return M

def stats(M):
    nm = np.linalg.norm(M)
    if nm < 1e-300: return 0, 0, 0
    Mn = M / nm; C = 2 * abs(np.linalg.det(Mn)); Sid = smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj()))
    dg = abs(Mn[0, 0]) ** 2 + abs(Mn[1, 1]) ** 2
    return C, Sid, dg

def summary(p, srcs, kind, par=None):
    r = np.array([stats(Mquad(p, srcs, kind, q, par)) for q in quads]); r = r[r[:, 1] > 0]
    return r

beam = lambda t0, wd: {j: 1.0 for j in range(n) if abs(X[j, 0] - t0) <= 0.45 and abs(X[j, 1]) <= wd}
def norm_inj(d): z = math.sqrt(len(d)); return {k: v / z for k, v in d.items()}
p_beam = pump_field(norm_inj(beam(-11, 3.0)))
strip = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]

# случайное состояние: C и диагональность
v = rng.normal(size=(20000, 4)) + 1j * rng.normal(size=(20000, 4)); v /= np.linalg.norm(v, axis=1)[:, None]
Cr = 2 * np.abs(v[:, 0] * v[:, 3] - v[:, 1] * v[:, 2]); Dr = np.abs(v[:, 0]) ** 2 + np.abs(v[:, 3]) ** 2
print(f'\nчетвёрок захвата {len(quads)}{" (широкая область)" if WIDE else ""}; случайное состояние: C медиана {np.median(Cr):.3f}, '
      f'диагональность медиана {np.median(Dr):.3f}')

# ---- 2) вершины ----
print('\n2) вершины (накачка-пучок, полоса):')
for kind in ('бозон через U', 'фермион хиральный'):
    r = summary(p_beam, strip, kind)
    print(f'   {kind:20s}: C медиана {np.median(r[:, 0]):.3f}, макс {r[:, 0].max():.3f}; S ид. макс {r[:, 1].max():.3f}; '
          f'диагональность медиана {np.median(r[:, 2]):.3f}')
nfer = sum(1 for k in range(n) if len(S.legs[k][1]) and cls(k).any() and (~cls(k)).any())
print(f'   событий, где фермионная вершина вообще есть (выходы обоих классов): {nfer} из {n}')
meds = []
for _ in range(30):
    par = rng.normal(size=5) + 1j * rng.normal(size=5); meds.append(np.median(summary(p_beam, strip, 'бозон-семейство', par)[:, 0]))
print(f'   случайные бозоны из 5-параметрического семейства (30 штук): медиана C от {min(meds):.3f} до {max(meds):.3f}')
sys.stdout.flush()

# ---- 3) синхронизм: толщина и ширина среды, ширина накачки ----
print('\n3) когерентная сумма побольше — выбирается ли запутанность? (бозон через U)')
print(f'   {"среда: t-центр ± h, |x| ≤":28s} {"накачка |x| ≤":>13s} {"C медиана":>9s} {"C > 0.9":>8s} {"диаг. медиана":>13s} {"S ид. макс":>10s}')
for (h, xw, pw) in ((0.45, 3, 3), (1.5, 3, 3), (3.0, 3, 3), (3.0, 6, 6), (3.0, 10, 10), (99, 99, 3), (99, 99, 10)):
    srcs = [j for j in range(n) if abs(X[j, 0] + 7) <= h and abs(X[j, 1]) <= xw] if h < 50 else list(range(n))
    p = pump_field(norm_inj(beam(-11 if h < 3.5 else -11.5, pw)))
    r = summary(p, srcs, 'бозон через U')
    lab = f'±{h}, {xw}' if h < 50 else 'весь мир'
    print(f'   {lab:28s} {pw:13} {np.median(r[:, 0]):9.3f} {np.mean(r[:, 0] > 0.9):8.2f} {np.median(r[:, 2]):13.3f} {r[:, 1].max():10.3f}')
    sys.stdout.flush()
