"""ПОЛНЫЙ ЗАКОН: дефицит ~ M^a / r^b, с ИСПРАВЛЕННОЙ высотой.

Закон по массе M^+0.918 был намерен со СЛОМАННОЙ высотой (считалась по
всем событиям, включая цепь наблюдателя длиной 120-300, тогда как
естественная цепь ~15; высота вырождалась в номер вдоль цепи). После
починки (высота ТОЛЬКО по фону) показатель по расстоянию стал r^-1.041
при R^2 = 0.955. Значит и показатель по массе надо перемерить.

Здесь оба показателя сразу, с погрешностями по складному ножу (jackknife
по засевам).
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from bigfall import build, heights_blocked, radar_chain

N, T, RAD, DENS, SEEDS = 8000, 1.0, 0.077, 300, 24
ell = (math.pi/24.0/N)**0.25
BANDS = [(RAD + k*ell, RAD + (k+1)*ell) for k in range(6)]

def one(rs, frac):
    bg = build(N, T, rs, 0.0, DENS)[0]
    # фон и цепь уже вместе; выделим фон и проредим шар
    isobs = np.all(np.abs(bg[:,1:]) < 1e-9, axis=1)
    B = bg[~isobs]; C = bg[isobs]
    rr = np.linalg.norm(B[:,1:], axis=1)
    ins = np.flatnonzero(rr < RAD)
    drop = ins[rs.random(len(ins)) > frac] if len(ins) else ins
    keep = np.ones(len(B), bool); keep[drop] = False
    B = B[keep]
    h = heights_blocked(B)                 # ТОЛЬКО по фону
    tau, rho, ok = radar_chain(B, C)
    return dict(h=h.astype(float), tau=tau, rho=rho, ok=ok,
                nrem=len(drop), hmax=int(h.max()))

fracs = [1.0, 0.6, 0.3, 0.0]
print(f'N={N}, ell={ell:.5f}, дырка {RAD}={RAD/ell:.1f} ell, '
      f'цепь {DENS}, засевов {SEEDS}')
print('высота ТОЛЬКО по фону (исправлено)')
print()
# per-seed данные: [seed][frac][band]
D = np.full((SEEDS, len(fracs), len(BANDS)), np.nan)
M = np.zeros((SEEDS, len(fracs)))
for s in range(SEEDS):
    rs0 = np.random.default_rng(91000+191*s)
    A = one(np.random.default_rng(91000+191*s), 1.0)
    for fi, f in enumerate(fracs):
        Bq = one(np.random.default_rng(91000+191*s), f)
        M[s, fi] = Bq['nrem']
        for q,(lo,hi) in enumerate(BANDS):
            mA = A['ok'] & (A['rho']>=lo*DENS) & (A['rho']<hi*DENS)
            mB = Bq['ok'] & (Bq['rho']>=lo*DENS) & (Bq['rho']<hi*DENS)
            if mA.sum()<20 or mB.sum()<20: continue
            rA = A['h'][mA]/np.maximum(A['tau'][mA],1e-9)
            rB = Bq['h'][mB]/np.maximum(Bq['tau'][mB],1e-9)
            D[s, fi, q] = rB.mean()-rA.mean()
        del Bq
    del A

rc = np.array([0.5*(lo+hi) for lo,hi in BANDS])
print(f"{'масса M':>9}" + ''.join(f'{f"{c/ell:.1f}ell":>11}' for c in rc)
      + f"{'показ. r':>11}")
mas, bees = [], []
for fi, f in enumerate(fracs):
    m = np.nanmean(M[:, fi])
    row = np.nanmean(D[:, fi, :], axis=0)
    ok = row < 0
    b = (np.polyfit(np.log(rc[ok]), np.log(-row[ok]), 1)[0]
         if ok.sum() >= 4 else np.nan)
    mas.append(m); bees.append(b)
    print(f'{m:>9.0f}' + ''.join(f'{v:>+11.6f}' for v in row)
          + f'{b:>11.3f}')

def jack(fn):
    """Складной нож по засевам: оценка и погрешность."""
    full = fn(np.arange(SEEDS))
    vals = [fn(np.delete(np.arange(SEEDS), i)) for i in range(SEEDS)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    n = len(vals)
    return full, math.sqrt((n-1)/n * ((vals-vals.mean())**2).sum())

def expo_r(idx):
    row = np.nanmean(D[idx, -1, :], axis=0)      # максимальная масса
    ok = row < 0
    if ok.sum() < 4: return np.nan
    return np.polyfit(np.log(rc[ok]), np.log(-row[ok]), 1)[0]

def expo_M(idx):
    y, x = [], []
    for fi in range(len(fracs)):
        v = np.nanmean(D[idx, fi, 0])            # ближайшая полоса
        mm = np.nanmean(M[idx, fi])
        if v < 0 and mm > 0: y.append(-v); x.append(mm)
    if len(x) < 3: return np.nan
    return np.polyfit(np.log(x), np.log(y), 1)[0]

br, sbr = jack(expo_r)
bm, sbm = jack(expo_M)
print()
print(f'  показатель по РАССТОЯНИЮ: r^({br:+.3f} ± {sbr:.3f})   '
      f'ОТО: -1')
print(f'  показатель по МАССЕ:      M^({bm:+.3f} ± {sbm:.3f})   '
      f'ОТО: +1')
print()
print(f'  ЗАКОН: дефицит ~ M^{bm:.2f} / r^{abs(br):.2f}')
print('  слабое поле ОТО: M / r')
