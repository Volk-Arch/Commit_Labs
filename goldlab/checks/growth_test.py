# Растёт ли отклик со временем (неустойчивость)? Темп роста γ и частота при разных m² и плотностях.
import sys, os, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, causal as Cs
rho,H,Rc=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); al,a=0.45,0.5
w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X
cache='links31_rho%g_H%g_R%g.npz'%(rho,H,Rc)
if rho==1 and os.path.exists('links31_H%.0f_R%.0f.npz'%(H,Rc)): cache='links31_H%.0f_R%.0f.npz'%(H,Rc)
z=np.load(cache); I,J=z['I'],z['J']
r=np.linalg.norm(X[:,1:],axis=1); prof=np.exp(-r*al); src0=X[:,0]<0.45; t=X[:,0]
order=np.argsort(J,kind='stable'); I2=I[order]; J2=J[order]; starts=np.searchsorted(J2,np.arange(w.n+1))
indeg=np.diff(starts); print('ρ=%g H=%g R=%g: событий %d, связей в прошлое на событие %.0f (медиана), коэффициент·число = %.2f'%(rho,H,Rc,w.n,np.median(indeg),np.median(indeg)*math.sqrt(rho)/(2*math.pi*math.sqrt(6))*(1/rho)),flush=True)
def run(m2,U):
    ph=Pr.wilson31(X,I,J,U)[order] if U else np.ones(len(I)); wv=ph*math.sqrt(rho)/(2*math.pi*math.sqrt(6))
    phi=np.zeros(w.n,complex); src=np.where(src0,prof,0).astype(complex); b=-m2/rho
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=(wv[s0:s1]*src[I2[s0:s1]]).sum()
        src[j]+=b*phi[j]
    e=np.arange(0,H+1e-9,1.0); k=np.digitize(t,e)-1; ok=(k>=0)&(k<len(e)-1)
    amp=np.sqrt(np.bincount(k[ok],np.abs(prof*phi)[ok]**2,minlength=len(e)-1)/np.maximum(np.bincount(k[ok],minlength=len(e)-1),1))
    tc=0.5*(e[1:]+e[:-1]); s=(tc>2)&(tc<H-1); g=np.polyfit(tc[s],np.log(amp[s]+1e-300),1)[0]
    return g, ' '.join('%.2g'%x for x in amp[::2])
Uc=lambda q: -al/np.sqrt(q*q+a*a)
for lab,U in (('свободно',None),('ядро',Uc)):
    for m2 in (0.25,1.0,4.0):
        g,s=run(m2,U); print('  %s m²=%.2f: темп роста γ = %+.3f   |φ|(t=0.5,2.5,…): %s'%(lab,m2,g,s),flush=True)
