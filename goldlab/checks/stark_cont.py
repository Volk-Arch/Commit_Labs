# Непрерывный эталон штарковской вставки: КГ, U=-α/sqrt(r²+a²), m² -> m² - c·z·e^{-r/L}; два канала s и p.
import numpy as np, scipy.linalg as sl, scipy.optimize as so
al,a,m=0.45,0.5,1.0; N=1600; R=60.0; h=R/N; r=h*np.arange(1,N+1)
U=-al/np.sqrt(r*r+a*a)
D=(np.diag(np.full(N,2.0))-np.diag(np.ones(N-1),1)-np.diag(np.ones(N-1),-1))/h**2
def lam(om,c,L):
    V=m*m-(om-U)**2; f=c*r*np.exp(-r/L)/np.sqrt(3)
    if c==0: return sl.eigh(D+np.diag(V),eigvals_only=True,subset_by_index=[0,0])[0]
    M=np.block([[D+np.diag(V),-np.diag(f)],[-np.diag(f),D+np.diag(V+2/r**2)]])
    return sl.eigh(M,eigvals_only=True,subset_by_index=[0,0])[0]
for L in (1.0,2.0,4.0):
    c=0.12/(L/np.e)  # c·max|z e^{-r/L}| = 0.12, максимум при r=L
    E0=so.brentq(lambda o:lam(o,0,L),0.8,0.99); E=so.brentq(lambda o:lam(o,c,L),0.8,0.99)
    print('L=%.0f: 1s %.5f, со вставкой %.5f, сдвиг %+.5f'%(L,E0,E,E-E0))
print('s-вставка m² -> m² - 0.02·e^{-r/L}:')
def lam_s(om,L,d):
    V=m*m-d*np.exp(-r/L)-(om-U)**2
    return sl.eigh(D+np.diag(V),eigvals_only=True,subset_by_index=[0,0])[0]
for L in (1.0,2.0,4.0):
    E0=so.brentq(lambda o:lam_s(o,L,0),0.8,0.99); E=so.brentq(lambda o:lam_s(o,L,0.02),0.8,0.99)
    print('L=%.0f: сдвиг %+.5f'%(L,E-E0))
