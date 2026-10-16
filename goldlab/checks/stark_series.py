# Сохраняет временные ряды отклика (проекция на форму 1s) для разных вставок — для разбора измерителем
import sys, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr
H,Rc,al,a,m=20.0,7.0,0.45,0.5,1.0
w=Wd.gold31(1.0,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X; z=np.load('links31_H20_R7.npz'); I,J=z['I'],z['J']
U=lambda q: -al/np.sqrt(q*q+a*a); ph=Pr.wilson31(X,I,J,U)
r=np.linalg.norm(X[:,1:],axis=1); prof=np.exp(-r*al); src0=X[:,0]<0.45
order=np.argsort(J,kind='stable'); I2,J2=I[order],J[order]; p2=ph[order]; starts=np.searchsorted(J2,np.arange(w.n+1)); wv=p2/(2*math.pi*math.sqrt(6))
def run(B):
    phi=np.zeros(w.n,complex); src=np.where(src0,prof,0).astype(complex)
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=(wv[s0:s1]*src[I2[s0:s1]]).sum()
        src[j]+=B[j]*phi[j]
    return phi
cfg={'base':np.full(w.n,-m*m),'uni':np.full(w.n,-m*m+0.02)}
for L in (2.0,4.0): cfg['s%d'%L]=-m*m+0.02*np.exp(-r/L)
g=X[:,3]*np.exp(-r/2.0); cfg['z2']=-m*m+0.12/np.abs(g).max()*g
out={'t':X[:,0],'r':r,'prof':prof}
for k,B in cfg.items(): out[k]=run(B); print(k,flush=True); np.savez('stark_series.npz',**out)
print('готово')
