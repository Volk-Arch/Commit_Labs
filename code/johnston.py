#!/usr/bin/env python3
"""СУММА ИСТОРИЙ С ФАЗОЙ ∝ СОБСТВЕННОЙ ДЛИНЕ ЗВЕНА (в духе пропагатора Джонстона).
ψ_K = Σ_{J→K} κ·e^{i m s_JK}·ψ_J — сумма, а не среднее; κ — одно число на весь мир (не по событию).
Число цепочек растёт экспоненциально, поэтому есть критическое κ_c, при котором амплитуда плоской волны
не растёт и не гаснет. Ожидание по стационарной фазе: сумма по путям выделяет пути с почти максимальным
собственным временем, и пропагатор ~ e^{i m_eff τ}.
Проверки: (1) κ_c(m); (2) при κ_c — сохраняется ли |ψ| плоской волны и Σ|ψ|² пакета на срезах;
(3) точечный источник: зависит ли фаза только от τ, какая частота против m."""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
sys.argv = _a

def setup(rho=4.0, L=24.0):
    w = CH.sym_world(rho, L); R = CH.build_order(w, 'proj'); Lk = CH.links_of(R)
    I, J = np.nonzero(Lk); o = np.argsort(J, kind='stable'); I, J = I[o], J[o]
    d = w.X[J] - w.X[I]; s = np.sqrt(d[:, 0] ** 2 - d[:, 1] ** 2)
    st = np.searchsorted(J, np.arange(w.n)); en = np.searchsorted(J, np.arange(w.n), side='right')
    return w, R, dict(I=I, J=J, s=s, st=st, en=en)

def psum(w, LK, init, kappa, m, extra=None):
    I, st, en = LK['I'], LK['st'], LK['en']
    wt = kappa * np.exp(1j * m * LK['s'])
    if extra is not None: wt = wt * extra
    psi = np.zeros(w.n, complex); first = min(init)
    for k, v in init.items(): psi[k] = v
    for j in range(first + 1, w.n):
        if j in init: continue
        a, b = st[j], en[j]
        if a < b: psi[j] = (psi[I[a:b]] * wt[a:b]).sum()
    return psi

def slices_mean(w, psi, times, f=lambda p: np.abs(p)):
    X = w.X; out = []
    for t in times:
        sl = (np.abs(X[:, 0] - t) < 0.35) & (np.abs(X[:, 1]) < 24 - abs(t) - 4)
        out.append(f(psi[sl]))
    return np.array(out)

if __name__ == '__main__':
    w, R, LK = setup(); X = w.X; n = w.n
    band0 = {k: 1.0 for k in np.nonzero(np.abs(X[:, 0] + 10) < 0.35)[0]}
    times = np.arange(-8, 9, 2.0)
    print(f'мир ρ = 4: событий {n}, звеньев {len(LK["I"])}, средняя собственная длина звена {LK["s"].mean():.3f}')
    kc = {}
    for m in (0.0, 1.0, 2.0, 4.0):
        lo, hi = 0.01, 1.0
        for _ in range(30):
            k = math.sqrt(lo * hi); v = slices_mean(w, psum(w, LK, band0, k, m), times, lambda p: np.abs(p).mean())
            slope = np.polyfit(times[2:], np.log(v[2:] + 1e-300), 1)[0]
            if slope > 0: hi = k
            else: lo = k
        kc[m] = k
        v = slices_mean(w, psum(w, LK, band0, k, m), times, lambda p: np.abs(p).mean())
        coh = slices_mean(w, psum(w, LK, band0, k, m), times, lambda p: abs(p.sum()) / np.abs(p).sum())
        print(f'm = {m}: κ_c = {k:.4f}; плоская волна |ψ| на t = −8…8: ' + ' '.join(f'{x:.3f}' for x in v / v[0]) +
              f'; когерентность по срезу {coh[-1]:.3f}')
        # пакет: Σ|ψ|² по срезам
        pk = {kk: math.exp(-X[kk, 1] ** 2 / 8) for kk in band0 if abs(X[kk, 1]) < 7}
        p2 = slices_mean(w, psum(w, LK, pk, k, m), times, lambda p: (np.abs(p) ** 2).sum())
        print(f'        пакет Σ|ψ|² (к первому срезу): ' + ' '.join(f'{x:.2f}' for x in p2 / p2[0]))
        sys.stdout.flush()
    np.save('/home/claude/data/johnston_kc.npy', kc, allow_pickle=True)
