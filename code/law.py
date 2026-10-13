"""ЗАКОН: показатель по массе при ФИКСИРОВАННОЙ геометрии.

Предыдущий замер менял радиус, то есть массу И геометрию вместе.
Здесь радиус фиксирован, меняется ДОЛЯ оставленных событий — значит
меняется только масса. Показатель по массе тогда чист.

Слабое поле ОТО: замедление ~ GM/r, значит дефицит ~ M^1.
"""
import numpy as np, math, sys
sys.path.insert(0, '/home/claude/code')
from frames import rel_of, radar
from radargrav import clock_rate

N, DENS, RAD, SEEDS = 2200, 120, 0.20, 14
ell = (math.pi/24.0/N)**0.25

def one(rs, frac):
    out, got = [], 0
    while got < N:
        m = (N-got)*10 + 256
        Y = np.empty((m,4)); Y[:,0] = rs.uniform(0,1,m)
        Y[:,1:] = rs.uniform(-0.5,0.5,(m,3))
        r = np.linalg.norm(Y[:,1:],axis=1)
        Y = Y[(Y[:,0]>=r)&(1-Y[:,0]>=r)]
        if len(Y): out.append(Y); got += len(Y)
    bg = np.vstack(out)[:N]
    rr = np.linalg.norm(bg[:,1:],axis=1)
    ins = np.flatnonzero(rr < RAD)
    drop = ins[rs.random(len(ins)) > frac] if len(ins) else ins
    keep = np.ones(len(bg),bool); keep[drop] = False
    bg = bg[keep]
    t = np.linspace(0.05,0.95,DENS)
    Z = np.zeros((DENS,4)); Z[:,0] = t
    X = np.vstack([bg,Z]); X = X[np.argsort(X[:,0])]
    isobs = np.all(np.abs(X[:,1:]) < 1e-12, axis=1)
    R = rel_of(X); cr,h,pst = clock_rate(X,R); del R
    C = X[isobs]; bi = np.flatnonzero(~isobs); Xb = X[bi]
    tau, rho, ok = radar(np.vstack([Xb,C]), len(bi), C)
    return dict(rho=rho,tau=tau,ok=ok,h=h[bi],nrem=len(drop))

print(f'N={N}, радиус ФИКСИРОВАН {RAD} = {RAD/ell:.1f} ell, '
      f'цепь {DENS}, засевов {SEEDS}')
print('меняется только ДОЛЯ оставленных => только масса')
print()
print(f"{'доля f':>8}{'масса M':>9}{'у края':>11}{'сигм':>7}"
      f"{'+1 ell':>11}{'сигм':>7}{'+2 ell':>11}{'сигм':>7}")
rows = []
for frac in (0.75, 0.50, 0.25, 0.0):
    acc = [[],[],[]]; nr = []
    for s in range(SEEDS):
        A = one(np.random.default_rng(83000+149*s), 1.0)
        B = one(np.random.default_rng(83000+149*s), frac)
        nr.append(B['nrem'])
        for q,(lo,hi) in enumerate([(RAD,RAD+ell),(RAD+ell,RAD+2*ell),
                                    (RAD+2*ell,RAD+3*ell)]):
            mA = A['ok'] & (A['rho']>=lo*DENS) & (A['rho']<hi*DENS)
            mB = B['ok'] & (B['rho']>=lo*DENS) & (B['rho']<hi*DENS)
            if mA.sum()<10 or mB.sum()<10: continue
            rA = A['h'][mA]/np.maximum(A['tau'][mA],1e-9)
            rB = B['h'][mB]/np.maximum(B['tau'][mB],1e-9)
            acc[q].append(float(rB.mean()-rA.mean()))
        del A,B
    o = []
    for q in range(3):
        v = np.array(acc[q]); se = v.std(ddof=1)/math.sqrt(len(v))
        o.append((v.mean(), abs(v.mean())/max(se,1e-12)))
    M = np.mean(nr)
    rows.append([M]+[abs(x[0]) for x in o])
    print(f'{frac:>8.2f}{M:>9.0f}{o[0][0]:>+11.5f}{o[0][1]:>7.1f}'
          f'{o[1][0]:>+11.5f}{o[1][1]:>7.1f}{o[2][0]:>+11.5f}'
          f'{o[2][1]:>7.1f}', flush=True)
A = np.array(rows)
print()
for q,nm in ((1,'у края'),(2,'на +1 ell'),(3,'на +2 ell')):
    ok = A[:,q] > 1e-9
    if ok.sum() >= 3:
        b = np.polyfit(np.log(A[ok,0]), np.log(A[ok,q]), 1)[0]
        print(f'  дефицит {nm:<12} ~ M^{b:+.3f}')
print('  ожидание слабого поля ОТО: M^+1 на всех расстояниях;')
print('  у края насыщение (показатель меньше), вдали ближе к единице')
