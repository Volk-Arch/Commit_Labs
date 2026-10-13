"""Поле или поверхность: растёт ли дальнобойность с размером дырки.

radargrav.py намерил дефицит хода часов -0.0459 (19.3σ) у поверхности
дырки и в 500 раз меньше на одну длину дискретности дальше. Вопрос:
это поле (дальнобойность растёт с массой) или поверхностный эффект
(range стоит)?
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from radargrav import run_one

N, DENS, SEEDS = 2200, 120, 10
ell = (math.pi/24.0/N)**0.25
print(f'N = {N}, ell = {ell:.4f}, цепь {DENS} событий, засевов {SEEDS}')
print('дефицит хода часов = высота/радарное время, с дыркой минус без')
print()
print(f"{'радиус':>8}{'в ell':>7}{'удал':>6}{'у края':>11}{'сигм':>7}"
      f"{'на +1 ell':>11}{'сигм':>7}{'на +2 ell':>11}{'сигм':>7}"
      f"{'спад':>8}")
for rad in (0.08, 0.12, 0.18, 0.26):
    # корзины по радарному расстоянию: у края, +1 ell, +2 ell
    edges = [(rad, rad + ell), (rad + ell, rad + 2*ell),
             (rad + 2*ell, rad + 3*ell)]
    acc = [[] for _ in edges]; nrem = []
    for s in range(SEEDS):
        A = run_one(N, 1.0, np.random.default_rng(82000+137*s), 0.0, DENS)
        B = run_one(N, 1.0, np.random.default_rng(82000+137*s), rad, DENS)
        nrem.append(B['nrem'])
        # перевод радарных делений в долю мира: одно деление = 1/DENS
        for q, (lo, hi) in enumerate(edges):
            lo_d, hi_d = lo*DENS, hi*DENS
            mA = A['ok'] & (A['rho'] >= lo_d) & (A['rho'] < hi_d)
            mB = B['ok'] & (B['rho'] >= lo_d) & (B['rho'] < hi_d)
            if mA.sum() < 10 or mB.sum() < 10: continue
            rA = A['h'][mA]/np.maximum(A['tau'][mA],1e-9)
            rB = B['h'][mB]/np.maximum(B['tau'][mB],1e-9)
            acc[q].append(float(rB.mean()-rA.mean()))
        del A, B
    out = []
    for q in range(len(edges)):
        v = np.array(acc[q])
        if len(v) < 3: out.append((np.nan, np.nan)); continue
        se = v.std(ddof=1)/math.sqrt(len(v))
        out.append((v.mean(), abs(v.mean())/max(se,1e-12)))
    sp = (abs(out[1][0])/abs(out[0][0])) if (np.isfinite(out[0][0])
          and abs(out[0][0])>1e-12) else np.nan
    print(f'{rad:>8.2f}{rad/ell:>7.1f}{np.mean(nrem):>6.0f}'
          f'{out[0][0]:>+11.5f}{out[0][1]:>7.1f}'
          f'{out[1][0]:>+11.5f}{out[1][1]:>7.1f}'
          f'{out[2][0]:>+11.5f}{out[2][1]:>7.1f}{sp:>8.3f}')
print()
print('  «спад» = |на +1 ell| / |у края|. Если дальнобойность растёт с')
print('  радиусом, спад растёт; если стоит — поверхностный эффект')
