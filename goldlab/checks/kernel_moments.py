"""Средняя связь по времени: c·A(t) = c·Σ_y w_{n(x,y)} по y с t_x − t_y ∈ [t, t+dt] / dt. Непрерывно c·A(t) = t.
Хвост t·(cA − t) → −0.78/√ρ у связей (посыпка, вывод) → 0 у слоёв. Что на кристалле?"""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
rho=float(sys.argv[1]); kind=sys.argv[2]; S=9.0; T=8.0
if kind=='gold': G=Wd.points_in_box31([(-T-0.5,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho))
else:
    rng=np.random.default_rng(5); V=(T+1.1)*(2*S)**3; n=rng.poisson(rho*V); G=np.c_[rng.uniform(-T-0.5,0.6,n),rng.uniform(-S,S,(n,3))]
tc2=(3.0*rho**-0.25)**2; c=math.sqrt(rho)/(2*math.pi*math.sqrt(6))
tg=np.nonzero((np.abs(G[:,0])<0.3)&(np.linalg.norm(G[:,1:],axis=1)<0.5))[0][:40]
bins=np.arange(0,T+1e-9,0.5); acc={nm:np.zeros(len(bins)-1) for nm in ('связи','2 слоя','3 слоя')}
W={'связи':(1,),'2 слоя':(2,-2),'3 слоя':(3,-7,4)}
for j in tg:
    d=G[j]-G; tau2=d[:,0]**2-(d[:,1:]**2).sum(1); c_=np.nonzero((d[:,0]>0)&(tau2>1e-12)&(tau2<tc2))[0]
    Q=G[c_]; e=Q[None,:,:]-Q[:,None,:]; prec=(e[...,0]>0)&(e[...,0]**2-(e[...,1:]**2).sum(-1)>1e-12); nn=prec.sum(1)
    dt=d[c_,0]; b=np.digitize(dt,bins)-1
    for nm,w in W.items():
        ww=np.zeros(len(c_))
        for k,wk in enumerate(w): ww[nn==k]=wk
        ok=(b>=0)&(b<len(bins)-1); acc[nm]+=np.bincount(b[ok],ww[ok],minlength=len(bins)-1)
tm=0.5*(bins[1:]+bins[:-1])
for nm in acc:
    cA=c*acc[nm]/len(tg)/0.5; dl=cA-tm
    print('%s %s ρ=%g: c·A(t)/t на t = %s'%(kind,nm,rho,' '.join('%.0f:%.3f'%(a,b) for a,b in zip(tm[1::2],(cA/tm)[1::2]))))
    print('      хвост t·δ(t) при t = 4…7.5: %.3f (посыпка, связи: теория %.3f)'%(np.mean((tm*dl)[8:]),-0.783/math.sqrt(rho)))
