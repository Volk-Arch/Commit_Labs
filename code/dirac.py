#!/usr/bin/env python3
"""ПРЕДЕЛ ДИРАКА? Точечный источник, правило «поворот = i» (монета нормирована, унитарна).

Что должно быть у частицы Дирака в 1+1:
  * безмассовая (ε = 0): правая компонента бежит только вправо по световому лучу, левая — влево;
  * массивная: амплитуда зависит от собственного времени τ = √(Δt² − Δx²) (лоренц-инвариантность),
    осциллирует с частотой m (каналы «частица/античастица» — с частотами ±m), спадает ~ τ^{−1/2}.
Меряем: асимметрию «право/лево» при ε = 0; для каналов u± = (ψ₀ ± ψ₁)/√2 — когерентность по
быстроте при фиксированном τ (1 — зависит только от τ), частоту ω(ε) = dφ/dτ, спад |u|(τ); и
сравниваем ω со «счётчиком шагов»: фаза монеты на шаг atan ε × среднее число звеньев ℓ на пути.
"""
import sys, math, numpy as np
_a = sys.argv; sys.argv = sys.argv[:1]
import chiral as CH
from prop import propagate
sys.argv = _a

def mean_steps(w, LK, R, src):
    I, st, en = LK['I'], LK['st'], LK['en']
    lit_ev = R[src].copy(); lit_ev[src] = True; lit = lit_ev[I]
    ell = np.full(w.n, np.nan); ell[src] = 0
    for j in range(src + 1, w.n):
        a, b = st[j], en[j]
        if a >= b: continue
        li = lit[a:b]
        if li.sum() == 0: continue
        ell[j] = (ell[I[a:b][li]] + 1).mean()
    return ell

if __name__ == '__main__':
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    w = CH.sym_world(rho, 24.0); R = CH.build_order(w, 'proj'); LK = CH.prep(w, CH.links_of(R), 1); X = w.X
    src = int(np.argmin(((X - np.array([-12.0, 0.0])) ** 2).sum(1)))
    fut = np.nonzero(R[src])[0]; d = X[fut] - X[src]; tau = np.sqrt(np.maximum(d[:, 0] ** 2 - d[:, 1] ** 2, 0)); th = np.arctanh(d[:, 1] / d[:, 0])
    ok = (np.abs(X[fut]).sum(1) < 22) & (np.abs(th) < 1.2)
    ell = mean_steps(w, LK, R, src)[fut]
    # шаги на единицу собственного времени
    sl = np.polyfit(tau[ok & (tau > 2)], ell[ok & (tau > 2)], 1)
    cv = np.std(ell[ok & (np.abs(tau - 8) < 0.5)]) / np.mean(ell[ok & (np.abs(tau - 8) < 0.5)])
    print(f'ρ = {rho}: источник {np.round(X[src], 2)}, событий в будущем {len(fut)} (берём |быстрота| < 1.2: {ok.sum()})')
    print(f'среднее число звеньев на пути ℓ ≈ {sl[0]:.3f}·τ + {sl[1]:.2f}; разброс ℓ при τ ≈ 8 по быстротам: {cv:.1%}')
    for eps in (0.0, 0.25, 0.5, 1.0):
        for m0 in (0, 1):
            psi = propagate(w, LK, R, {src: np.eye(2)[m0]}, eps=eps, unit=True)[fut]
            if eps == 0:
                right = (np.abs(psi[:, m0]) ** 2 * (d[:, 1] > 0))[ok].sum() / (np.abs(psi[:, m0]) ** 2)[ok].sum()
                print(f'  ε = 0, вход в компоненту {m0}: |ψ|² справа от источника {right:.1%} (Дирак: 100 % или 0 %); '
                      f'|ψ_{m0}| среднее {np.abs(psi[ok, m0]).mean():.3f}, разброс {np.abs(psi[ok, m0]).std():.1e}; вторая компонента {np.abs(psi[ok, 1 - m0]).max():.1e}')
                continue
            if m0 == 1: continue
            for name, u in (('u+', (psi[:, 0] + psi[:, 1]) / math.sqrt(2)), ('u−', (psi[:, 0] - psi[:, 1]) / math.sqrt(2))):
                bins = np.arange(1, 11, 0.5); coh, ph, mag = [], [], []
                for b0 in bins:
                    m = ok & (tau >= b0) & (tau < b0 + 0.5)
                    if m.sum() < 5: coh.append(np.nan); ph.append(np.nan); mag.append(np.nan); continue
                    z = u[m]; coh.append(abs(z.sum()) / np.abs(z).sum()); ph.append(np.angle(z.sum())); mag.append(np.abs(z).mean())
                ph = np.unwrap(np.array(ph)); good = ~np.isnan(ph)
                om = np.polyfit(bins[good] + 0.25, ph[good], 1)[0]
                # по шагам: фаза одного шага ±atan ε, шагов sl[0] на единицу τ
                pred = (1 if name == 'u+' else -1) * math.atan(eps) * sl[0]
                # спад: показатель степени |u| ~ τ^p
                mg = np.array(mag); gm = good & (bins > 2)
                p = np.polyfit(np.log(bins[gm] + 0.25), np.log(mg[gm]), 1)[0]
                print(f'  ε = {eps}, {name}: частота ω = dφ/dτ = {om:+.3f} (по шагам {pred:+.3f}); когерентность по быстроте '
                      f'(1 = только от τ) τ≈3: {coh[4]:.3f}, τ≈6: {coh[10]:.3f}, τ≈9: {coh[16]:.3f}; спад |u| ~ τ^{p:.2f}')
            sys.stdout.flush()
