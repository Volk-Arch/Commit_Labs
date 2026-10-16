"""Ансамбль сдвигов фазона в 3+1: тот же закон, шар тени сдвинут — другие события. На каждом мире: уровень 1s (КГ), рост,
наклон по массе, второй порядок (вставка z), спиновый сдвиг Дирак − КГ. Пишет по строке JSON на мир."""
import sys, os, math, time, json, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob, causal as Cs
from goldlab.core import r_for_density
rho=float(sys.argv[1]); nw=int(sys.argv[2]); H=float(sys.argv[3]) if len(sys.argv)>3 else 12.0; Rc=float(sys.argv[4]) if len(sys.argv)>4 else 5.0
out=open('ens31loc_rho%g_H%g.jsonl'%(rho,H),'a'); SH=[json.loads(l)['shift'] for l in open('ens31_rho%g_H%g.jsonl'%(rho,H))]; T0=time.time(); m=1.0; al,a=0.45,0.5
rw=r_for_density(rho,4); rng=np.random.default_rng(int(rho*100+H))
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex); e=np.array([1,0],complex)
for k in range(nw):
    v=np.array(SH[k]); w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0,shift=tuple(v)); X=w.X
    rr_=np.linalg.norm(X[:,1:],axis=1); reff=((rr_<4)&(X[:,0]>2)&(X[:,0]<10)).sum()/(4/3*np.pi*64*8)   # фактическая плотность у ядра
    I,J=Cs.links_sparse(X,rho,1.6); I=I.astype(np.int32); J=J.astype(np.int32)
    r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; src0=t<0.45; sel=(t>1.5)&(t<H-0.3); tt=t[sel]; prof=np.exp(-al*r)
    order=np.argsort(J,kind='stable'); I2=I[order]; starts=np.searchsorted(J[order],np.arange(w.n+1)); Jf=np.repeat(np.arange(w.n),np.diff(starts)).astype(np.int32)
    wv=math.sqrt(reff)/(2*math.pi*math.sqrt(6))*Pr.wilson31(X,I2,Jf,lambda q: -al/np.sqrt(q*q+a*a)); del I,J,Jf
    def run(Bx):
        phi=np.zeros((w.n,2),complex); src=np.where(src0[:,None],prof[:,None]*e[None,:],0).astype(complex)
        for j in range(w.n):
            s0,s1=starts[j],starts[j+1]
            if s1>s0: phi[j]=wv[s0:s1]@src[I2[s0:s1]]
            src[j]+=Bx[j]@phi[j]
        return phi@e
    def peak(sig,om=np.linspace(0.5,1.3,1601)):
        S=Ob.spectrum(tt,prof[sel]*sig[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0]
    eye=np.broadcast_to(np.eye(2),(w.n,2,2))
    s1=run(-m*m/reff*eye); E1=peak(s1); E2=peak(run(-0.81/reff*eye))
    ed=np.arange(0,H+1e-9,1.0); kk=np.digitize(t,ed)-1; ok=(kk>=0)&(kk<len(ed)-1)
    amp=np.sqrt(np.bincount(kk[ok],np.abs(prof*s1)[ok]**2,minlength=len(ed)-1)/np.maximum(np.bincount(kk[ok],minlength=len(ed)-1),1))
    tc=0.5*(ed[1:]+ed[:-1]); ss=(tc>2)&(tc<H-1); gr=float(np.polyfit(tc[ss],np.log(amp[ss]+1e-300),1)[0])
    g=X[:,3]*np.exp(-r/2.0); c=0.12/np.abs(g).max(); Es=peak(run(((-m*m+c*g)/reff)[:,None,None]*eye))
    rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
    dU=al*r/(r*r+a*a)**1.5; Ed=peak(run((-m*m*eye+1j*dU[:,None,None]*srr)/reff))
    rec=dict(reff=reff,rho=rho,H=H,Rc=Rc,shift=v.tolist(),n=w.n,E=E1,growth=gr,slope=(E1-E2)/0.19,stark=Es-E1,spin=Ed-E1)
    out.write(json.dumps(rec)+'\n'); out.flush()
    print('[%5.0f c] ρ=%g мир %d: n=%d, 1s %.4f, рост %+.3f, наклон %.3f, штарк %+.5f, спин %+.5f'%(time.time()-T0,rho,k,w.n,E1,gr,(E1-E2)/0.19,Es-E1,Ed-E1),flush=True)
