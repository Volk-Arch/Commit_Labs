"""Связи по всему кристаллу, а не по обрезку: пустота интервала проверяется по событиям с запасом pad вокруг области;
потом пары ограничиваются областью. Сравнение: рост, уровень, наклон, второй порядок, спин — обрезок против запаса."""
import sys, math, time, json, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, causal as Cs, propagate as Pr, observe as Ob, core
rho=float(sys.argv[1]); pad=float(sys.argv[2]); nw=int(sys.argv[3]); H,Rc,Rs=12.0,5.0,3.0; al,a,m=0.45,0.5,1.0
rng=np.random.default_rng(int(rho*100+H))
def region(P, extra):
    rr=np.linalg.norm(P[:,1:],axis=1); return rr<=np.minimum(Rc+extra,Rs+np.maximum(P[:,0],0)+0.6+extra)
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
for k in range(nw):
    v=None if k==0 and nw==1 else tuple(np.array(rng.normal(size=4))*0)  # заполним ниже
    if nw>1:
        vv=rng.normal(size=4); vv*=core.r_for_density(rho,4)*rng.random()**0.25/np.linalg.norm(vv); v=tuple(vv)
    else: v=None
    P=core.crystal([(-0.6,H+0.1)]+[(-Rc-pad,Rc+pad)]*3,Wd.r_for_density31(rho),v); P=P[region(P,pad)]
    T0=time.time(); Ib,Jb=Cs.links_sparse(P,rho,1.6); inside=region(P,0.0); idx=-np.ones(len(P),int); idx[inside]=np.arange(inside.sum())
    keep=inside[Ib]&inside[Jb]; I=idx[Ib[keep]]; J=idx[Jb[keep]]; X=P[inside]; n=len(X)
    I0,J0=Cs.links_sparse(X,rho,1.6)
    res={}
    for lab,(II,JJ) in (('обрезок',(I0,J0)),('запас %.1f'%pad,(I,J))):
        r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); sel=(t>1.5)&(t<H-0.3); src0=t<0.45
        ph=Pr.wilson31(X,II,JJ,lambda q: -al/np.sqrt(q*q+a*a)); e=np.array([1,0],complex)
        f=np.where(src0[:,None],prof[:,None]*e[None,:],0)
        def go(Bx):
            s=Pr.johnston31_respond_spinor(n,II,JJ,rho,m,f,ph,Bx)@e
            S=Ob.spectrum(t[sel],prof[sel]*s[sel],np.linspace(0.5,1.3,1601),'hann'); pk=Ob.peaks(np.linspace(0.5,1.3,1601),S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0], s
        eye=np.broadcast_to(np.eye(2),(n,2,2)); E1,s1=go(-m*m/rho*eye); E2,_=go(-0.81/rho*eye)
        ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1); cnt=np.bincount(kk[ok],minlength=len(ed)-1)
        amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(cnt,1)); tc=0.5*(ed[1:]+ed[:-1]); ss=(tc>2)&(tc<H-1)&(cnt>0)
        gr=np.polyfit(tc[ss],np.log(amp[ss]+1e-300),1)[0]
        g=X[:,3]*np.exp(-r/2.0); c=0.12/np.abs(g).max(); Es,_=go(((-m*m+c*g)/rho)[:,None,None]*eye)
        rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
        dU=al*r/(r*r+a*a)**1.5; Ed,_=go((-m*m*eye+1j*dU[:,None,None]*srr)/rho)
        res[lab]=dict(E=E1,growth=gr,slope=(E1-E2)/0.19,stark=Es-E1,spin=Ed-E1,links=len(II))
    print('[%4.0f c] ρ=%g мир %d: '%(time.time()-T0,rho,k)+' | '.join('%s: связей %d, 1s %.4f, рост %+.3f, наклон %.3f, штарк %+.5f, спин %+.5f'%(l,d['links'],d['E'],d['growth'],d['slope'],d['stark'],d['spin']) for l,d in res.items()),flush=True)
    with open('padded31_rho%g.jsonl'%rho,'a') as fo: fo.write(json.dumps({'shift':None if v is None else list(v),'res':res})+'\n')
