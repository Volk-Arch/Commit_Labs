#!/usr/bin/env python3
"""Сумма историй: точечный источник. Зависит ли амплитуда только от τ (лоренц-инвариантность)? Какая частота?
Ожидание стационарной фазы: фаза ≈ m·τ (вклад путей с максимальным собственным временем)."""
import sys, math, numpy as np
from johnston import setup, psum
w, R, LK = setup(); X = w.X; n = w.n
kc = np.load('/home/claude/data/johnston_kc.npy', allow_pickle=True).item()
src = int(np.argmin(((X - np.array([-12.0, 0.0])) ** 2).sum(1)))
fut = np.nonzero(R[src])[0]; d = X[fut] - X[src]
tau = np.sqrt(np.maximum(d[:, 0] ** 2 - d[:, 1] ** 2, 0)); th = np.arctanh(d[:, 1] / d[:, 0])
ok = (np.abs(X[fut]).sum(1) < 22) & (np.abs(th) < 1.0)
# самая длинная цепь (в собственном времени) до каждого события: max Σ s
I, st, en, s = LK['I'], LK['st'], LK['en'], LK['s']
best = np.full(n, -np.inf); best[src] = 0
for j in range(src + 1, n):
    a, b = st[j], en[j]
    if a < b:
        v = best[I[a:b]] + s[a:b]; best[j] = v.max()
geo = best[fut]
print(f'максимальная сумма собственных длин звеньев / τ: {np.median(geo[ok & (tau > 4)] / tau[ok & (tau > 4)]):.3f} (1 = цепь идёт по геодезической)')
for m in (0.0, 1.0, 2.0, 4.0):
    psi = psum(w, LK, {src: 1.0}, kc[m], m)[fut]
    bins = np.arange(1, 11, 0.5); coh, ph, mag = [], [], []
    for b0 in bins:
        mm = ok & (tau >= b0) & (tau < b0 + 0.5)
        z = psi[mm]; coh.append(abs(z.sum()) / np.abs(z).sum()); ph.append(np.angle(z.sum())); mag.append(np.abs(z).mean())
    ph = np.unwrap(np.array(ph)); om = np.polyfit(bins + 0.25, ph, 1)[0]
    p = np.polyfit(np.log(bins[2:] + 0.25), np.log(np.array(mag)[2:]), 1)[0]
    print(f'm = {m} (κ_c {kc[m]:.3f}): частота dφ/dτ = {om:+.3f} (ожидание стац. фазы ≈ m = {m}); когерентность по быстроте '
          f'τ≈3: {coh[4]:.3f}, τ≈6: {coh[10]:.3f}, τ≈9: {coh[16]:.3f}; |ψ| ~ τ^{p:.2f}')
