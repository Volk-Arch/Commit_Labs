# Поле облака электрона из порядка в 3+1: U(x) = 4π·e²·(√ρ/(2π√6))/ρ · Σ_{связи i→x} σ(r_i)  (то же правило, что Кулон из связей)
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd
rho=float(sys.argv[1]) if len(sys.argv)>1 else 2.0; S=10.0; T=S+4.0
c=np.load('helium31_cloud.npz'); rr,sg,Uc=c['r'],c['sig_r'],c['Uc']; e2=0.225
G=Wd.points_in_box31([(-T,0.6),(-S,S),(-S,S),(-S,S)],Wd.r_for_density31(rho)); print('событий',len(G),flush=True)
sigG=np.interp(np.linalg.norm(G[:,1:],axis=1),rr,sg,right=0.0); tc2=(1.6*rho**-0.25)**2
coef=4*math.pi*e2*math.sqrt(rho)/(2*math.pi*math.sqrt(6))/rho
rad=np.linalg.norm(G[:,1:],axis=1); rng=np.random.default_rng(3); rows=[]
for R in (0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0):
    idx=np.nonzero((np.abs(G[:,0])<0.3)&(np.abs(rad-R)<0.25))[0]; vals=[]
    for j in rng.choice(idx,min(25,len(idx)),replace=False):
        d=G[j]-G; tau2=d[:,0]**2-(d[:,1:]**2).sum(1); cnd=np.nonzero((d[:,0]>0)&(tau2>1e-12)&(tau2<tc2))[0]
        Q=G[cnd]; e=Q[None,:,:]-Q[:,None,:]; prec=(e[...,0]>0)&(e[...,0]**2-(e[...,1:]**2).sum(-1)>1e-12)
        lk=cnd[~prec.any(1)]; vals.append(coef*sigG[lk].sum())
    ue=np.interp(R,rr,Uc); rows.append((R,np.mean(vals),np.std(vals)/math.sqrt(len(vals)),ue))
    print('r=%.1f: поле из порядка %.4f ± %.4f, точное %.4f, отношение %.3f'%(R,rows[-1][1],rows[-1][2],ue,rows[-1][1]/ue),flush=True)
np.savez('field31_cloud_rho%g.npz'%rho,rows=np.array(rows))
