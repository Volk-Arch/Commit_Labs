#!/usr/bin/env python3
"""СПИН НА КВАНТЕ: полярное событие со спинором в золотом мире 3+1.

Нога несёт 2-спинор. Ядро события — то же произведение Минковского, но в форме Вейля:
    K_ji = exp(−σ̄(d_out_j) σ(d_in_i) / m₀),  σ(d) = d⁰ − d·σ⃗,  σ̄(d) = d⁰ + d·σ⃗,
след ½·σ̄(a)σ(b) = a·b, так что скалярная часть ядра — прежняя. Событие — ближайшая изометрия (полярное
разложение) к блочной матрице (2·выходы × 2·входы); лишнее — в резервуар тени.
Вращения мира (24 знаковые перестановки с det = 1) действуют на спиноры через SU(2): σ(Rd) = U σ(d) U†, и полярное
разложение с этим согласовано ⇒ ковариантность должна быть точной (проверка).
Меряем: (1) нарушение ковариантности при 24 вращениях; (2) норма мир + тень; (3) СПИРАЛЬНОСТЬ — проекция спина на
направление ноги h = a†(n̂·σ⃗)a/|a|² на поздних ногах (контроль: скалярное ядро с тем же начальным спином);
(4) смена σ̄σ ↔ σσ̄ (левый ↔ правый Вейль) меняет знак спиральности?"""
import sys, math, itertools, numpy as np
sys.path.insert(0, '/home/claude/code')
from gold4 import diamond
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
P = diamond(N); n = len(P)
d = P[None, :, :] - P[:, None, :]; R = d[..., 0] > np.linalg.norm(d[..., 1:], axis=2) + 1e-12
Rf = R.astype(np.float32); L = R & ~((Rf @ Rf) > 0)
I, J = np.nonzero(L); E = len(I); dv = P[J] - P[I]
sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]]); sz = np.diag([1.0 + 0j, -1])
SIG = np.array([sx, sy, sz])
def sig(v, bar=False):
    s = np.einsum('...k,kab->...ab', v[..., 1:], SIG); I2 = np.eye(2)
    return v[..., 0, None, None] * I2 + (s if bar else -s)
from scipy.linalg import expm
m0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
order = np.argsort(P[:, 0], kind='stable')
ein = [[] for _ in range(n)]; eout = [[] for _ in range(n)]
for e in range(E): eout[I[e]].append(e); ein[J[e]].append(e)

def build(kind):
    legs = []
    for k in range(n):
        li, lo = np.array(ein[k], int), np.array(eout[k], int)
        if len(li) and len(lo):
            if kind == 'скаляр':
                M = dv[lo][:, None, 0] * dv[li][None, :, 0] - (dv[lo][:, None, 1:] * dv[li][None, :, 1:]).sum(-1)
                Kb = np.kron(np.exp(-M / m0), np.eye(2))
            else:
                A = sig(dv[lo], bar=(kind == 'левый')); B = sig(dv[li], bar=(kind != 'левый'))
                prod = np.einsum('jab,ibc->jiac', A, B)
                Kb = np.zeros((2 * len(lo), 2 * len(li)), complex)
                for j in range(len(lo)):
                    for i in range(len(li)): Kb[2 * j:2 * j + 2, 2 * i:2 * i + 2] = expm(-prod[j, i] / m0)
            U_, s_, Vt = np.linalg.svd(Kb, full_matrices=False); keep = s_ > 1e-9 * s_.max()
            V = U_[:, keep] @ Vt[keep]
        else: V = np.zeros((2 * len(lo), 2 * len(li)))
        legs.append((li, lo, V))
    return legs

def run(legs, src, spin):
    amp = np.zeros((E, 2), complex); res = 0.0
    for k in order:
        li, lo, V = legs[k]
        vin = amp[li].reshape(-1)
        vout = V @ vin if len(li) and len(lo) else np.zeros(2 * len(lo), complex)
        res += (np.abs(vin) ** 2).sum() - (np.abs(vout) ** 2).sum()
        if k == src and len(lo): vout = vout + np.kron(np.ones(len(lo)) / math.sqrt(len(lo)), spin)
        amp[lo] = vout.reshape(-1, 2)
    return amp, res

def su2(M):
    w = math.sqrt(max(1 + np.trace(M), 0)) / 2
    if w > 1e-9: x, y, z = (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w), (M[1, 0] - M[0, 1]) / (4 * w)
    else:
        vals, vecs = np.linalg.eig(M); ax = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); ax /= np.linalg.norm(ax); x, y, z = ax
    return w * np.eye(2) - 1j * (x * sx + y * sy + z * sz)

key = {tuple(np.round(p, 9)): i for i, p in enumerate(P)}
axis = [i for i in range(n) if np.abs(P[i, 1:]).max() < 1e-12]; src = min(axis, key=lambda i: P[i, 0])
eid = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(I, J))}
rots = []
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1, -1), repeat=3):
        M = np.zeros((3, 3)); M[range(3), perm] = sg
        if np.linalg.det(M) > 0: rots.append(M)
def img(i, M): q = P[i].copy(); q[1:] = M @ q[1:]; return key.get(tuple(np.round(q, 9)), -1)
print(f'мир 3+1: событий {n}, звеньев {E} ({E / n:.1f} на событие); источник на оси T = {P[src, 0]:.3f}')
up = np.array([1.0, 0.0], complex)
late = P[J, 0] > P[src, 0] + 0.6
nrm = np.linalg.norm(dv[:, 1:], axis=1); nh = dv[:, 1:] / np.maximum(nrm, 1e-300)[:, None]
for kind in ('скаляр', 'левый', 'правый'):
    legs = build(kind); amp, res = run(legs, src, up)
    tot = res + 0  # норма: всё, что ушло в тень, + что осталось на последних ногах (ноги в конце мира)
    endlegs = np.array([len(legs[J[e]][1]) == 0 for e in range(E)])
    norm = res                      # всё в итоге оседает в событиях-концах или в тени
    # ковариантность при вращениях
    worst = 0
    for Mr in rots:
        U = su2(Mr)
        amp2, _ = run(legs, img(src, Mr), U @ up)   # img(src) = src (на оси)
        for sgn in (1, -1):
            pass
        mp = np.array([eid.get((img(a, Mr), img(b, Mr)), -1) for a, b in zip(I, J)])
        ok = mp >= 0
        d1 = np.abs(amp2[mp[ok]] - (amp[ok] @ U.T)).max(); d2 = np.abs(amp2[mp[ok]] + (amp[ok] @ U.T)).max()
        worst = max(worst, min(d1, d2) / np.abs(amp).max())
    w2 = (np.abs(amp) ** 2).sum(1)
    h = np.einsum('ea,kab,eb,ek->e', amp.conj(), SIG, amp, nh).real / np.maximum(w2, 1e-300)
    sz_ = np.einsum('ea,ab,eb->e', amp.conj(), sz, amp).real / np.maximum(w2, 1e-300)
    m = late & (w2 > 0) & (nrm > 0.3 * dv[:, 0])          # ноги с заметной пространственной частью
    print(f'{kind:7s}: норма мир + тень {norm:.9f}; нарушение ковариантности (24 вращения, спинор × U) {worst:.1e}; '
          f'поздние ноги: спиральность ⟨h⟩ = {(h[m] * w2[m]).sum() / w2[m].sum():+.3f}, доля |h| > 0.9: {(w2[m] * (np.abs(h[m]) > 0.9)).sum() / w2[m].sum():.2f}; '
          f'⟨σ_z⟩ = {(sz_[m] * w2[m]).sum() / w2[m].sum():+.3f}')
    sys.stdout.flush()
    # спиральность по скорости ноги
    spd = nrm / dv[:, 0]; bins = [(0.3, 0.6), (0.6, 0.85), (0.85, 0.95), (0.95, 1.01)]
    msg = []
    for lo_, hi_ in bins:
        mm = late & (w2 > 0) & (spd >= lo_) & (spd < hi_)
        if mm.any(): msg.append(f'v∈[{lo_},{hi_}): ⟨h⟩ {(h[mm] * w2[mm]).sum() / w2[mm].sum():+.3f} (вес {w2[mm].sum() / w2[late].sum():.2f})')
    print('         ' + '; '.join(msg)); sys.stdout.flush()
