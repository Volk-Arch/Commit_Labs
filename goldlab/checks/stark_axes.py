import sys, math, numpy as np
sys.path.insert(0, '/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob
H,Rc,al,a,m=20.0,7.0,0.45,0.5,1.0
w=Wd.gold31(1.0,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X; z=np.load('links31_H20_R7.npz'); I,J=z['I'],z['J']
U=lambda q: -al/np.sqrt(q*q+a*a); ph=Pr.wilson31(X,I,J,U)
r=np.linalg.norm(X[:,1:],axis=1); prof=np.exp(-r*al); src0=X[:,0]<0.45; sel=(X[:,0]>1.5)&(X[:,0]<H-0.3); tt=X[sel,0]; om=np.linspace(0.8,1.0,4001)
order=np.argsort(J,kind='stable'); I2,J2=I[order],J[order]; p2=ph[order]; starts=np.searchsorted(J2,np.arange(w.n+1)); wv=p2/(2*math.pi*math.sqrt(6))
def run(B):
    phi=np.zeros(w.n,complex); src=np.where(src0,prof,0).astype(complex)
    for j in range(w.n):
        s0,s1=starts[j],starts[j+1]
        if s1>s0: phi[j]=(wv[s0:s1]*src[I2[s0:s1]]).sum()
        src[j]+=B[j]*phi[j]
    S=Ob.spectrum(tt,prof[sel]*phi[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0]
E0=run(np.full(w.n,-m*m)); print('E0 %.5f'%E0, flush=True)
for ax,name in ((1,'x'),(2,'y'),(3,'z')):
    g=X[:,ax]*np.exp(-r/2.0); c=0.12/np.abs(g).max()
    print('ось %s: +c %+.5f  −c %+.5f'%(name, run(-m*m+c*g)-E0, run(-m*m-c*g)-E0), flush=True)
