"""Симметрично размытый световой конус 3+1 (dirac-doc §24–25). У кристалла нет световых пар, поэтому острый свет 3+1 (δ(τ²)) можно
только размыть на зерно. Связи размывают его с одной стороны — отсюда δ′ и разгон; здесь — в обе стороны: все пары с t_i < t_j и
|τ²| < τ_c², вес (c/2)·e^{−ατ⁴}·(1 ∓ β) внутри/снаружи конуса, α = ρπ/24, c = √ρ/(2π√6), плюс нить Вильсона. β ≈ 0.115 гасит
остаток раскачки (средняя теория, не зависит от ρ). Голая масса m_b² = m²/(1 − m²·Re G(m)) — выведена из того же ядра."""
import sys, math, time, numpy as np
try:
    import numba
except ImportError as e:
    raise ImportError('для размытого конуса нужен numba: pip install numba') from e
from . import worlds as Wd, observe as Ob
GX,GW=np.polynomial.legendre.leggauss(4)
@numba.njit(parallel=False)
def respond_fly(X,tc2,alpha,coef,al,a,gx,gw,f,B,use_phase,beta=0.0):
    n=X.shape[0]; phi=np.zeros((n,2),np.complex128); src=f.copy()
    for j in range(n):
        a0=0j; a1=0j
        for i in range(j):
            dt=X[j,0]-X[i,0]
            if dt<=0: continue
            dx=X[j,1]-X[i,1]; dy=X[j,2]-X[i,2]; dz=X[j,3]-X[i,3]; u=dt*dt-dx*dx-dy*dy-dz*dz
            if u>=tc2 or u<=-tc2: continue
            w=0.5*math.exp(-alpha*u*u)*((1.0-beta) if u>0 else (1.0+beta))
            if use_phase:
                s=0.0
                for q in range(4):
                    fq=0.5*(gx[q]+1); x=X[i,1]+fq*dx; y=X[i,2]+fq*dy; z=X[i,3]+fq*dz
                    s+=0.5*gw[q]*(-al/math.sqrt(x*x+y*y+z*z+a*a))
                ph=complex(math.cos(s*dt),math.sin(s*dt)); c=w*ph
            else: c=complex(w,0.0)
            a0+=c*src[i,0]; a1+=c*src[i,1]
        phi[j,0]=coef*a0; phi[j,1]=coef*a1
        src[j,0]+=B[j,0,0]*phi[j,0]+B[j,0,1]*phi[j,1]; src[j,1]+=B[j,1,0]*phi[j,0]+B[j,1,1]*phi[j,1]
    return phi
def _spec(tt, sig, om, dt=0.01):
    """спектр Ханна по слоям времени (бинирование dt — экономия памяти на больших мирах)"""
    if len(tt)*len(om) < 3e7: return Ob.spectrum(tt, sig, om, 'hann')
    w = np.sin(np.pi * (tt - tt.min()) / np.ptp(tt)) ** 2; e = np.arange(tt.min(), tt.max() + dt, dt); k = np.digitize(tt, e) - 1
    ys = np.bincount(k, (w * sig).real, minlength=len(e)) + 1j * np.bincount(k, (w * sig).imag, minlength=len(e)); tc = e + dt / 2
    return np.abs(np.exp(-1j * np.outer(om, tc)) @ ys) ** 2

def measure(rho,H,Rc,Rs,REF,tcf=2.8,shift=None,label='',lam=1.0,mb2=(1.0,0.81),Z=1.0,beta=0.0):
    al,a,m=0.45,0.5,1.0; alpha=lam*rho*math.pi/24; coef=math.sqrt(lam*rho)/(2*math.pi*math.sqrt(6)); tc2=(tcf*(lam*rho)**-0.25)**2
    if shift=='poisson':
        rng=np.random.default_rng(1); nn=rng.poisson(rho*(H+0.7)*(2*Rc)**3); P=np.c_[rng.uniform(-0.6,H+0.1,nn),rng.uniform(-Rc,Rc,(nn,3))]
        rr=np.linalg.norm(P[:,1:],axis=1); P=P[rr<=np.minimum(Rc,Rs+np.maximum(P[:,0],0)+0.6)]; X=np.ascontiguousarray(P[np.argsort(P[:,0])])
    else:
        w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=Rs,shift=shift); X=np.ascontiguousarray(w.X)
    n=len(X); T0=time.time()
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); src0=t<0.45
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; eye=np.broadcast_to(np.eye(2),(n,2,2)); g=X[:,3]*np.exp(-r/2.0); cst=0.12/np.abs(g).max(); e=np.array([1,0],complex)
    f=np.ascontiguousarray(np.where(src0[:,None],prof[:,None]*e[None,:],0).astype(np.complex128)); om=np.linspace(0.3,1.3,2001)
    C=lambda A: np.ascontiguousarray(A,dtype=np.complex128)
    def go(B):
        s=respond_fly(X,tc2,alpha,coef,al,a,GX,GW,f,C(B),True,beta)@e; S=_spec(t[sel],prof[sel]*s[sel],om); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0], s
    M1,M2=mb2; E1,s1=go((-M1/rho)*eye); E2,_=go((-M2/rho)*eye); Es,_=go(((-M1+Z*cst*g)/rho)[:,None,None]*eye); Ed,_=go((-M1*eye+Z*1j*dU[:,None,None]*srr)/rho)
    ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
    amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc_=0.5*(ed[1:]+ed[:-1]); ss=(tc_>2)&(tc_<H-1)&(cnt>0)
    gr=np.polyfit(tc_[ss],np.log(amp[ss]+1e-300),1)[0]; res=dict(E=E1,growth=gr,slope=(E1-E2)/0.19,stark=Es-E1,spin=Ed-E1,n=n)
    if label is not None: print('[%4.0f c] %sρ=%g H=%g R=%g Rs=%g S1: n=%d, 1s %.4f (%.4f), рост %+.3f, наклон %.3f (%.3f), штарк %+.5f (%+.5f), спин %+.5f (%+.5f)'%(time.time()-T0,label,rho,H,Rc,Rs,n,E1,REF['E'],gr,(E1-E2)/0.19,REF['slope'],Es-E1,REF['stark'],Ed-E1,REF['spin']),flush=True)
    return res

def mean_kernel_delta(rho, beta, T=10.0, N=1500, shape=1):
    """средняя теория (k = 0): δ(t) = c′A(t) − t для ядра (c/2)e^{−αu²}(1 ∓ β)."""
    c=math.sqrt(rho)/(2*math.pi*math.sqrt(6)); al=rho*math.pi/24; t=np.linspace(1e-6,T,N); A=np.empty(N)
    for i,tt in enumerate(t):
        r=np.linspace(0,math.sqrt(tt*tt+7/math.sqrt(al)),5001); u=tt*tt-r*r; x=al*u*u; h=np.exp(-x) if shape==1 else (1.5-x)*np.exp(-x)
        A[i]=np.trapezoid(4*np.pi*r*r*h*np.where(u>0,1-beta,1+beta),r)
    return t,0.5*c*A-t
def bare_mass2(rho, beta, m):
    """голая масса, при которой полюс средней теории стоит на m: m_b² = m²/(1 − m²·Re G(m)), G(ω) = ∫e^{iωt}δ(t)dt."""
    t,d=mean_kernel_delta(rho,beta); G=np.trapezoid(np.exp(1j*m*t)*d,t); return m*m/(1-m*m*G.real)


@numba.njit
def respond_tab(X, tc2, alpha, coef, rg, Ug, gx, gw, f, B, beta, shape=1):
    """то же, что respond_fly, но потенциал — таблица U(r) на равномерной сетке rg (линейная интерполяция) — для любого поля,
    в т. ч. поля облака, посчитанного из порядка."""
    n = X.shape[0]; phi = np.zeros((n, 2), np.complex128); src = f.copy(); dr = rg[1] - rg[0]; nr = len(rg)
    for j in range(n):
        a0 = 0j; a1 = 0j
        for i in range(j):
            dt = X[j, 0] - X[i, 0]
            if dt <= 0: continue
            dx = X[j, 1] - X[i, 1]; dy = X[j, 2] - X[i, 2]; dz = X[j, 3] - X[i, 3]; u = dt * dt - dx * dx - dy * dy - dz * dz
            if u >= tc2 or u <= -tc2: continue
            xx = alpha * u * u; w = 0.5 * math.exp(-xx) * ((1.0 - beta) if u > 0 else (1.0 + beta)); s = 0.0
            if shape == 3: w *= (1.5 - xx)
            for q in range(4):
                fq = 0.5 * (gx[q] + 1); x = X[i, 1] + fq * dx; y = X[i, 2] + fq * dy; z = X[i, 3] + fq * dz
                rr = math.sqrt(x * x + y * y + z * z) / dr; k = int(rr)
                if k >= nr - 1: uu = Ug[nr - 1]
                else: uu = Ug[k] + (rr - k) * (Ug[k + 1] - Ug[k])
                s += 0.5 * gw[q] * uu
            c = w * complex(math.cos(s * dt), math.sin(s * dt)); a0 += c * src[i, 0]; a1 += c * src[i, 1]
        phi[j, 0] = coef * a0; phi[j, 1] = coef * a1
        src[j, 0] += B[j, 0, 0] * phi[j, 0] + B[j, 0, 1] * phi[j, 1]; src[j, 1] += B[j, 1, 0] * phi[j, 0] + B[j, 1, 1] * phi[j, 1]
    return phi

@numba.njit
def field_static(X, tc2, alpha, coef, beta, sigma, targets, shape=1):
    """поле статического облака ИЗ ПОРЯДКА тем же размытым конусом (фотон = тот же конус): A(x) = Σ_y K(x, y)·σ(y),
    сумма по всем y раньше x (нормировка ρ — снаружи)."""
    out = np.zeros(len(targets))
    for a in range(len(targets)):
        j = targets[a]; acc = 0.0
        for i in range(X.shape[0]):
            dt = X[j, 0] - X[i, 0]
            if dt <= 0: continue
            u = dt * dt - (X[j, 1] - X[i, 1]) ** 2 - (X[j, 2] - X[i, 2]) ** 2 - (X[j, 3] - X[i, 3]) ** 2
            if u >= tc2 or u <= -tc2: continue
            xx = alpha * u * u; w = 0.5 * math.exp(-xx) * ((1.0 - beta) if u > 0 else (1.0 + beta))
            if shape == 3: w *= (1.5 - xx)
            acc += w * sigma[i]
        out[a] = coef * acc
    return out


# ---------------- быстрый счёт: пространственный индекс + параллельные слои времени ----------------
def build_index(X, cell=1.0):
    """клетки по пространству (размер cell); внутри клетки события по времени. Возвращает (lo, nc, cstart, cev, ct, cbox)."""
    lo = X[:, 1:].min(0) - 1e-9; nc = (np.floor((X[:, 1:].max(0) - lo) / cell).astype(np.int64) + 1)
    ci = np.floor((X[:, 1:] - lo) / cell).astype(np.int64); cid = (ci[:, 0] * nc[1] + ci[:, 1]) * nc[2] + ci[:, 2]
    order = np.argsort(cid, kind='stable'); ncell = int(nc[0] * nc[1] * nc[2]); counts = np.bincount(cid, minlength=ncell)
    cstart = np.zeros(ncell + 1, np.int64); cstart[1:] = np.cumsum(counts); cev = order.astype(np.int64); ct = X[cev, 0].copy()
    nz = np.nonzero(counts)[0]; cx = nz // (nc[1] * nc[2]); cy = (nz // nc[2]) % nc[1]; cz = nz % nc[2]
    cbox = np.c_[lo[0] + cx * cell, lo[1] + cy * cell, lo[2] + cz * cell]; cbox = np.c_[cbox, cbox + cell]
    return nz.astype(np.int64), cstart, cev, ct, np.ascontiguousarray(cbox)

@numba.njit(inline='always')
def _pair_w(X, i, j, tc2, alpha, beta, shape, rg, Ug, gx, gw, use_phase):
    dt = X[j, 0] - X[i, 0]
    if dt <= 0: return 0j
    dx = X[j, 1] - X[i, 1]; dy = X[j, 2] - X[i, 2]; dz = X[j, 3] - X[i, 3]; u = dt * dt - dx * dx - dy * dy - dz * dz
    if u >= tc2 or u <= -tc2: return 0j
    xx = alpha * u * u; w = 0.5 * math.exp(-xx) * ((1.0 - beta) if u > 0 else (1.0 + beta))
    if shape == 3: w *= (1.5 - xx)
    if use_phase == 0: return complex(w, 0.0)
    if use_phase == 2:                      # ядро −α/√(r² + a²) точно: rg = [α, a], ∫ по прямой — через логарифм
        al = rg[0]; aa = rg[1]; px = X[i, 1]; py = X[i, 2]; pz = X[i, 3]
        Aq = dx * dx + dy * dy + dz * dz; Bq = 2 * (px * dx + py * dy + pz * dz); Cq = px * px + py * py + pz * pz + aa * aa
        if Aq < 1e-12: I = 1.0 / math.sqrt(Cq)
        else:
            sA = math.sqrt(Aq); I = math.log((2 * sA * math.sqrt(Aq + Bq + Cq) + 2 * Aq + Bq) / (2 * sA * math.sqrt(Cq) + Bq)) / sA
        s = -al * I * dt; return w * complex(math.cos(s), math.sin(s))
    dr = rg[1] - rg[0]; nr = len(rg); s = 0.0
    for q in range(4):
        fq = 0.5 * (gx[q] + 1); x = X[i, 1] + fq * dx; y = X[i, 2] + fq * dy; z = X[i, 3] + fq * dz
        rr = math.sqrt(x * x + y * y + z * z) / dr; k = int(rr)
        uu = Ug[nr - 1] if k >= nr - 1 else Ug[k] + (rr - k) * (Ug[k + 1] - Ug[k])
        s += 0.5 * gw[q] * uu
    return w * complex(math.cos(s * dt), math.sin(s * dt))

@numba.njit(parallel=True)
def _slab_far(X, a, b, tlim, tc2, alpha, beta, shape, rg, Ug, gx, gw, use_phase, nz, cstart, cev, ct, cbox, src, acc):
    """вклады в события j ∈ [a, b) от всех i с t_i < tlim (параллельно по j)."""
    for jj in numba.prange(b - a):
        j = a + jj; xj = X[j, 1]; yj = X[j, 2]; zj = X[j, 3]; tj = X[j, 0]; a0 = 0j; a1 = 0j
        for q in range(len(nz)):
            c = nz[q]
            ddx = max(cbox[q, 0] - xj, 0.0, xj - cbox[q, 3]); ddy = max(cbox[q, 1] - yj, 0.0, yj - cbox[q, 4]); ddz = max(cbox[q, 2] - zj, 0.0, zj - cbox[q, 5])
            dmin2 = ddx * ddx + ddy * ddy + ddz * ddz
            fx = max(abs(cbox[q, 0] - xj), abs(cbox[q, 3] - xj)); fy = max(abs(cbox[q, 1] - yj), abs(cbox[q, 4] - yj)); fz = max(abs(cbox[q, 2] - zj), abs(cbox[q, 5] - zj))
            dmax2 = fx * fx + fy * fy + fz * fz
            tlo = tj - math.sqrt(dmax2 + tc2); thi = min(tj - math.sqrt(max(dmin2 - tc2, 0.0)), tlim)
            if thi <= tlo: continue
            s0 = cstart[c]; s1 = cstart[c + 1]
            if ct[s0] >= thi or ct[s1 - 1] < tlo: continue
            lo_ = s0; hi_ = s1
            while lo_ < hi_:
                mid = (lo_ + hi_) // 2
                if ct[mid] < tlo: lo_ = mid + 1
                else: hi_ = mid
            p = lo_
            while p < s1 and ct[p] < thi:
                i = cev[p]; wv = _pair_w(X, i, j, tc2, alpha, beta, shape, rg, Ug, gx, gw, use_phase)
                a0 += wv * src[i, 0]; a1 += wv * src[i, 1]; p += 1
        acc[jj, 0] = a0; acc[jj, 1] = a1

@numba.njit
def _slab_near(X, a, b, tlim_idx, tc2, alpha, beta, shape, rg, Ug, gx, gw, use_phase, coef, src, acc, phi, B):
    """внутри слоя: вклады от i ∈ [tlim_idx, j) последовательно, затем вставка B."""
    for j in range(a, b):
        a0 = acc[j - a, 0]; a1 = acc[j - a, 1]
        for i in range(tlim_idx, j):
            wv = _pair_w(X, i, j, tc2, alpha, beta, shape, rg, Ug, gx, gw, use_phase); a0 += wv * src[i, 0]; a1 += wv * src[i, 1]
        phi[j, 0] = coef * a0; phi[j, 1] = coef * a1
        src[j, 0] += B[j, 0, 0] * phi[j, 0] + B[j, 0, 1] * phi[j, 1]; src[j, 1] += B[j, 1, 0] * phi[j, 0] + B[j, 1, 1] * phi[j, 1]

def respond_fast(X, idx, tc2, alpha, coef, rg, Ug, f, B, beta, shape=1, slab=0.5, use_phase=True):
    """то же, что respond_tab, но с индексом клеток и параллельными слоями времени (X отсортирован по t)."""
    nz, cstart, cev, ct, cbox = idx; n = len(X); phi = np.zeros((n, 2), np.complex128); src = f.copy(); t = X[:, 0]
    edges = np.searchsorted(t, np.arange(t[0], t[-1] + slab, slab)); edges = np.unique(np.r_[edges, n])
    a = 0
    for b in edges[1:]:
        if b <= a: continue
        tlim = t[a]; acc = np.zeros((b - a, 2), np.complex128)
        _slab_far(X, a, b, tlim, tc2, alpha, beta, shape, rg, Ug, GX, GW, use_phase, nz, cstart, cev, ct, cbox, src, acc)
        # события с t_i == t[a] (одновременные) — тоже во внутренний проход
        k0 = a
        while k0 > 0 and t[k0 - 1] >= tlim: k0 -= 1
        _slab_near(X, a, b, k0, tc2, alpha, beta, shape, rg, Ug, GX, GW, use_phase, coef, src, acc, phi, np.ascontiguousarray(B))
        a = b
    return phi
