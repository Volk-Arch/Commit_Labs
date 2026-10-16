# Непрерывный эталон «гелия» 3+1 в первом порядке: КГ 1s в U = −α/√(r²+a²); облако второго электрона (заряд КГ (E−U)|φ|²),
# его потенциал e²·[Q(r)/r + ∫_r σ 4π r′ dr′]; сдвиг уровня 1s от добавки облака = J (с поправкой второго порядка — тоже).
import numpy as np, scipy.linalg as sl, scipy.optimize as so
al,a,m,e2=0.45,0.5,1.0,0.225; N=1500; R=50.0; h=R/N; r=h*np.arange(1,N+1); U=-al/np.sqrt(r*r+a*a)
D=(np.diag(np.full(N,2.0))-np.diag(np.ones(N-1),1)-np.diag(np.ones(N-1),-1))/h**2
def lam(om,Ut): return sl.eigh(D+np.diag(m*m-(om-Ut)**2),eigvals_only=True,subset_by_index=[0,0])[0]
def level(Ut): return so.brentq(lambda o:lam(o,Ut),0.5,0.99999)
E=level(U); w,v=sl.eigh(D+np.diag(m*m-(E-U)**2),subset_by_index=[0,0]); u=v[:,0]
sig=(E-U)*u*u; sig/=np.sum(sig)*h; sig_r=sig/(4*np.pi*r*r)        # плотность заряда на единицу объёма, ∫σ d³x = 1
Q=np.cumsum(sig)*h; out=np.cumsum((sig/r)[::-1])[::-1]*h
Uc=e2*(Q/r+out)
E2=level(U+Uc); print('КГ 1s %.5f; с облаком %.5f; J = %+.5f; ⟨r⟩ облака %.3f'%(E,E2,E2-E,np.sum(sig*r)*h))
np.savez('helium31_cloud.npz',r=r,sig_r=sig_r,Uc=Uc,E=E,J=E2-E)
