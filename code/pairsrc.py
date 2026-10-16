#!/usr/bin/env python3
"""ВЫВОД ПРАВИЛА ПАРЫ: распад кванта накачки в событии на две ноги.

Было [руками]: амплитуда пары = когерентная сумма по полосе источника Σ_s z_A(s) z_B(s), z — частица,
впрыснутая в событие s (herald.py: лучшая четвёрка C 0.874, S ид. 2.657).

Здесь правило пары не задаётся, а собирается из двух шагов замкнутой системы (closed2.py):
1) квант накачки летит по системе свободно (та же унитарная динамика), в каждом событии s его амплитуда
   в симметричной моде p(s) = ⟨s|v_in⟩;
2) «нелинейная среда»: событие поглощает квант накачки в симметричной моде и выпускает ДВА кванта на свои
   выходные ноги, состояние Σ_{a,b} f(a,b)|a⟩|b⟩ (f — вершина); дальше каждая дочь летит свободно.
Амплитуда «одна дочь захвачена в A_i, другая в B_j»: M_ij = Σ_s p(s) Σ_{a,b} f(a,b) β_Ai(a) β_Bj(b),
где β_k(e) — амплитуда дойти с ноги e до захвата в k. β считаются ОБРАТНЫМ проходом (транспонированная
динамика, по столбцу на событие захвата) — так все ноги всех событий сразу.

Вершины: бозон «a ≠ b», бозон «все a, b», бозон «через матрицу события» (f = ph_a ph_b, a ≠ b),
фермион (f = знак порядка ног по быстроте — антисимметрично).
Накачка: точка, пучок (линия событий), без накачки (p = 1). Среда: полоса источника или весь мир."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
from closed2 import Closed, CH
from beamsplit import smax
sys.argv = _a

S = Closed(turn=True); X = S.X; n = S.n; w = S.w
K = 2
H, Hs = CH.P @ np.linalg.matrix_power(CH.G, K), CH.P @ np.linalg.matrix_power(CH.Gs, K)
Xa = CH.axis_frame(X, -K * CH.ETA0 / 2)

def turnph(k):
    li, lo, Um = S.legs[k]
    return np.array([1j if S.edges[e][3] == 1 else 1.0 for e in lo])

# ---- четвёрки захвата (как в herald.py) ----
cands = [k for k in range(n) if 1.8 <= Xa[k, 0] <= 3.4 and 0.3 <= Xa[k, 1] <= 2.0]
quads = []
for A1 in cands:
    A2 = CH.img(w, A1, H, Hs)
    if A2 < 0: continue
    B1, B2 = CH.img(w, A1, CH.P, CH.P), CH.img(w, A2, CH.P, CH.P)
    if min(B1, B2) < 0: continue
    quads.append((A1, A2, B1, B2))
rec = sorted({k for q in quads for k in q}); col = {k: i for i, k in enumerate(rec)}

# ---- обратный проход: β[e, c] = амплитуда с ноги e до захвата в rec[c] ----
beta = np.zeros((S.E, len(rec)), complex)
for k in reversed(S.order):
    li, lo, Um = S.legs[k]; D = len(Um)
    bout = np.zeros((D, len(rec)), complex); bout[:len(lo)] = beta[lo]
    bin_ = Um.T @ bout                                  # входной слот j → Σ_idx Um[idx, j] β(выход idx)
    if k in col: bin_[:, col[k]] += 1 / math.sqrt(D)    # захват: ⟨s|v_in⟩
    beta[li] = bin_[:len(li)]
# z «руками» для каждого события: тот же слот, что в herald.run_record
zhand = np.zeros((n, len(rec)), complex)
for k in range(n):
    li, lo, Um = S.legs[k]; D = len(Um); slot = len(li) if len(li) < D else 0
    bout = np.zeros((D, len(rec)), complex); bout[:len(lo)] = beta[lo]
    zhand[k] = (Um.T @ bout)[slot]
    if k in col: zhand[k, col[k]] += 1 / math.sqrt(D)

# ---- прямой проход накачки: p(s) = ⟨s|v_in⟩ ----
def pump_field(inj):
    amp = np.zeros(S.E, complex); p = np.zeros(n, complex)
    for k in S.order:
        li, lo, Um = S.legs[k]; D = len(Um)
        vin = np.zeros(D, complex); vin[:len(li)] = amp[li]
        if k in inj: vin[len(li) if len(li) < D else 0] += inj[k]
        p[k] = vin.sum() / math.sqrt(D)
        vout = Um @ vin
        amp[lo] = vout[:len(lo)]
    return p

def vertex(k, kind):
    li, lo, Um = S.legs[k]; m = len(lo)
    if kind == 'бозон a≠b': return np.ones((m, m)) - np.eye(m)
    if kind == 'бозон все': return np.ones((m, m))
    if kind == 'бозон через U': ph = turnph(k); return np.outer(ph, ph) * (1 - np.eye(m))
    if kind == 'фермион': idx = np.arange(m); return np.sign(idx[None, :] - idx[:, None]).astype(float)
    raise ValueError(kind)

def Mquad(p, sources, kind, q):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2), complex)
    for s in sources:
        if p[s] == 0: continue
        if kind == 'руками': M += p[s] * np.outer(zhand[s, iA], zhand[s, iB]); continue
        lo = S.legs[s][1]
        if len(lo) == 0: continue
        bA, bB = beta[lo][:, iA], beta[lo][:, iB]
        M += p[s] * (bA.T @ vertex(s, kind) @ bB)
    return M

def stats(M):
    nm = np.linalg.norm(M)
    if nm < 1e-300: return 0, 0, 0
    Mn = M / nm; C = 2 * abs(np.linalg.det(Mn)); Sid = smax(np.outer(Mn.reshape(4), Mn.reshape(4).conj()))
    return C, Sid, nm ** 2

strip = [j for j in range(n) if abs(X[j, 0] + 7) <= 0.45 and abs(X[j, 1]) <= 3.0]
allev = list(range(n))
ones = np.ones(n, complex)
pt = int(np.argmin(((X - np.array([-11.0, 0.0])) ** 2).sum(1)))
beam = [j for j in range(n) if abs(X[j, 0] + 11) <= 0.45 and abs(X[j, 1]) <= 3.0]
p_pt = pump_field({pt: 1.0 + 0j})
p_beam = pump_field({j: 1 / math.sqrt(len(beam)) for j in beam})
print(f'событий {n}; четвёрок {len(quads)}; полоса источника {len(strip)} событий; накачка: точка {np.round(X[pt], 2)}, пучок {len(beam)} событий при t ≈ −11')
print(f'накачка в полосе: точка Σ|p|² = {np.sum(np.abs(p_pt[strip]) ** 2):.3f}, пучок {np.sum(np.abs(p_beam[strip]) ** 2):.3f}')

# проверка: «руками» с p = 1 на полосе должно повторить herald.py
cases = [('руками, p = 1, полоса (= herald)', ones, strip, 'руками')]
for pname, p in (('p = 1', ones), ('накачка-точка', p_pt), ('накачка-пучок', p_beam)):
    for med, srcs in (('полоса', strip), ('весь мир', allev)):
        for kind in ('бозон a≠b', 'бозон все', 'бозон через U', 'фермион'):
            cases.append((f'{kind}, {pname}, {med}', p, srcs, kind))
    cases.append((f'руками, {pname}, полоса', p, strip, 'руками'))
print(f'\n{"вариант":44s} {"C макс":>7s} {"S ид. макс":>10s} {"S>2":>6s} {"лучшая A₁":>16s} {"η отн.":>9s}')
for name, p, srcs, kind in cases:
    r = []
    for q in quads:
        C, Sid, eta = stats(Mquad(p, srcs, kind, q)); r.append((Sid, C, eta, q))
    r.sort(key=lambda t: -t[0]); Sid, C, eta, q = r[0]
    print(f'{name:44s} {max(t[1] for t in r):7.3f} {Sid:10.3f} {sum(t[0] > 2.0005 for t in r):3d}/{len(r)} '
          f'({Xa[q[0], 0]:5.2f},{Xa[q[0], 1]:+5.2f}) {eta:9.2e}'); sys.stdout.flush()

# ================= КОНТРОЛИ =================
# (0) случайное чистое состояние двух кубитов (Хаар): каким бывает C «само по себе»
rng = np.random.default_rng(1)
Cr = []
for _ in range(20000):
    v = rng.normal(size=4) + 1j * rng.normal(size=4); v /= np.linalg.norm(v); Cr.append(2 * abs(v[0] * v[3] - v[1] * v[2]))
Cr = np.array(Cr); mx12 = np.array([Cr[rng.integers(0, len(Cr), 12)].max() for _ in range(5000)])
print(f'\nКОНТРОЛЬ «случайное состояние»: C медиана {np.median(Cr):.3f}; максимум из 12 — медиана {np.median(mx12):.3f}, '
      f'95 % {np.quantile(mx12, 0.95):.3f}')

def table(name, p, srcs, kind, fn=None):
    Cs = []
    for q in quads:
        M = fn(q) if fn else Mquad(p, srcs, kind, q); Cs.append(stats(M)[0])
    Cs = np.array(Cs)
    print(f'{name:52s} C: медиана {np.median(Cs):.3f}, макс {Cs.max():.3f}, доля C > 0.9: {np.mean(Cs > 0.9):.2f}')
    return Cs

print('\nмедианы по 12 четвёркам (сравнивать с медианой случайного 0.6):')
table('руками, p = 1, полоса', ones, strip, 'руками')
table('бозон через U, накачка-пучок, полоса', p_beam, strip, 'бозон через U')
table('бозон a≠b, накачка-пучок, полоса', p_beam, strip, 'бозон a≠b')
table('фермион, накачка-пучок, полоса', p_beam, strip, 'фермион')
# (1) случайная вершина: f(a,b) — случайная комплексная симметричная
fr = {}
def Mrand(q):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; M = np.zeros((2, 2), complex)
    for s in strip:
        lo = S.legs[s][1]; m = len(lo)
        if m == 0: continue
        if s not in fr:
            F = rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)); fr[s] = F + F.T
        M += p_beam[s] * (beta[lo][:, iA].T @ fr[s] @ beta[lo][:, iB])
    return M
table('КОНТРОЛЬ: случайная вершина, накачка-пучок', None, None, None, Mrand)
# (2) записанное место рождения: смесь по s — S для смешанного состояния
def S_mixed(p, srcs, kind, q):
    iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]; rho = np.zeros((4, 4), complex)
    for s in srcs:
        lo = S.legs[s][1]
        if len(lo) == 0 or p[s] == 0: continue
        if kind == 'руками': Ms = p[s] * np.outer(zhand[s, iA], zhand[s, iB])
        else: Ms = p[s] * (beta[lo][:, iA].T @ vertex(s, kind) @ beta[lo][:, iB])
        v = Ms.reshape(4); rho += np.outer(v, v.conj())
    return smax(rho / np.trace(rho).real)
for name, p, kind in (('руками, p = 1', ones, 'руками'), ('бозон через U, накачка-пучок', p_beam, 'бозон через U'),
                      ('бозон a≠b, накачка-пучок', p_beam, 'бозон a≠b'), ('фермион, накачка-пучок', p_beam, 'фермион')):
    Ss = [S_mixed(p, strip, kind, q) for q in quads]
    print(f'место рождения ЗАПИСАНО, {name:32s}: S ид. макс {max(Ss):.3f}, > 2 у {sum(x > 2.0005 for x in Ss)}/12')
# (3) одно событие-источник: запутана ли пара уже в одной вершине?
for kind in ('бозон через U', 'бозон a≠b', 'фермион'):
    best = []
    for q in quads:
        iA = [col[q[0]], col[q[1]]]; iB = [col[q[2]], col[q[3]]]
        Cs = [stats(beta[S.legs[s][1]][:, iA].T @ vertex(s, kind) @ beta[S.legs[s][1]][:, iB])[0] for s in strip if len(S.legs[s][1])]
        best.append(np.median(Cs))
    print(f'одна вершина ({kind}): медиана C по событиям полосы {np.median(best):.3f}')
