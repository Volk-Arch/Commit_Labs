"""ЗАКОН СПАДАНИЯ: профиль дефицита по радарному расстоянию, с силой.

Предыдущий замер дал линейность по массе у поверхности (M^+0.918, до
12.7 сигм), но вдали всего 1.1-1.5 сигмы — показатели были бессмысленны.
1.4 сигмы при 14 засевах значит, что 60 дадут около 3. Здесь набирается
сила на максимальной массе (дырка пустая), и мерится профиль по r.

Слабое поле ОТО: дефицит ~ 1/r.
КОНТРОЛЬ: рассыпанное удаление того же числа — профиля быть не должно.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import rel_of, radar
from radargrav import clock_rate

N, DENS, RAD, SEEDS = 2200, 120, 0.20, 60
ell = (math.pi/24.0/N)**0.25

def one(rs, rad, scatter=False):
    out, got = [], 0
    while got < N:
        m = (N-got)*10 + 256
        Y = np.empty((m,4)); Y[:,0] = rs.uniform(0,1,m)
        Y[:,1:] = rs.uniform(-0.5,0.5,(m,3))
        r = np.linalg.norm(Y[:,1:],axis=1)
        Y = Y[(Y[:,0]>=r)&(1-Y[:,0]>=r)]
        if len(Y): out.append(Y); got += len(Y)
    bg = np.vstack(out)[:N]
    nrem = 0
    if rad > 0:
        rr = np.linalg.norm(bg[:,1:],axis=1)
        if scatter:
            k = int((rr<rad).sum())
            d = rs.choice(len(bg), k, replace=False)
            keep = np.ones(len(bg),bool); keep[d] = False; nrem = k
        else:
            keep = rr >= rad; nrem = int((~keep).sum())
        bg = bg[keep]
    Z = np.zeros((DENS,4)); Z[:,0] = np.linspace(0.05,0.95,DENS)
    X = np.vstack([bg,Z]); X = X[np.argsort(X[:,0])]
    isobs = np.all(np.abs(X[:,1:]) < 1e-12, axis=1)
    R = rel_of(X); cr,h,pst = clock_rate(X,R); del R
    C = X[isobs]; bi = np.flatnonzero(~isobs)
    tau, rho, ok = radar(np.vstack([X[bi],C]), len(bi), C)
    return dict(rho=rho,tau=tau,ok=ok,h=h[bi],nrem=nrem)

bands = [(RAD + k*ell, RAD + (k+1)*ell) for k in range(5)]
print(f'N={N}, дырка радиуса {RAD} = {RAD/ell:.1f} ell ПУСТАЯ, '
      f'цепь {DENS}, засевов {SEEDS}')
print(f'ell = {ell:.4f}; полосы по одной длине дискретности от края')
print()
for lbl, sc in (('ДЫРКА', False), ('КОНТРОЛЬ рассыпано', True)):
    acc = [[] for _ in bands]; nr = []
    for s in range(SEEDS):
        A = one(np.random.default_rng(84000+151*s), 0.0)
        B = one(np.random.default_rng(84000+151*s), RAD, scatter=sc)
        nr.append(B['nrem'])
        for q,(lo,hi) in enumerate(bands):
            mA = A['ok'] & (A['rho']>=lo*DENS) & (A['rho']<hi*DENS)
            mB = B['ok'] & (B['rho']>=lo*DENS) & (B['rho']<hi*DENS)
            if mA.sum()<8 or mB.sum()<8: continue
            rA = A['h'][mA]/np.maximum(A['tau'][mA],1e-9)
            rB = B['h'][mB]/np.maximum(B['tau'][mB],1e-9)
            acc[q].append(float(rB.mean()-rA.mean()))
        del A,B
    print(f'--- {lbl}: удалено {np.mean(nr):.0f}')
    print(f"{'r от центра':>12}{'в ell':>7}{'дефицит':>12}{'ст.ош':>10}"
          f"{'сигм':>7}{'миров':>7}")
    xs, ys = [], []
    for q,(lo,hi) in enumerate(bands):
        v = np.array(acc[q])
        if len(v) < 5: continue
        se = v.std(ddof=1)/math.sqrt(len(v))
        c = 0.5*(lo+hi)
        print(f'{c:>12.3f}{c/ell:>7.1f}{v.mean():>+12.5f}{se:>10.5f}'
              f'{abs(v.mean())/max(se,1e-12):>7.1f}{len(v):>7}')
        if v.mean() < 0 and abs(v.mean())/max(se,1e-12) > 2:
            xs.append(c); ys.append(-v.mean())
    if len(xs) >= 3:
        b = np.polyfit(np.log(xs), np.log(ys), 1)[0]
        print(f'  СПАДАНИЕ ~ r^{b:+.3f}   (слабое поле ОТО: -1)'
              f'   точек {len(xs)}')
    else:
        print(f'  точек выше 2 сигм: {len(xs)} — для показателя мало')
    print(flush=True)
