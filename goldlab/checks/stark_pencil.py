import numpy as np, sys
d=np.load('stark_series.npz'); t=d['t']; prof=d['prof']
def series(phi,dt=0.2,t0=1.5,t1=19.6):
    e=np.arange(t0,t1+1e-9,dt); k=np.digitize(t,e)-1; ok=(k>=0)&(k<len(e)-1)
    s=np.bincount(k[ok],prof[ok]*phi[ok].real,minlength=len(e)-1)+1j*np.bincount(k[ok],prof[ok]*phi[ok].imag,minlength=len(e)-1)
    n=np.bincount(k[ok],minlength=len(e)-1); return 0.5*(e[1:]+e[:-1]), s/np.maximum(n,1)
def pencil(y,dt,K):
    N=len(y); Lp=N//2; Y=np.array([y[i:i+Lp+1] for i in range(N-Lp)])
    U,S,Vh=np.linalg.svd(Y,full_matrices=False); V=Vh[:K].conj().T
    z=np.linalg.eigvals(np.linalg.pinv(V[:-1])@V[1:])
    om=np.angle(z)/dt; ga=np.log(np.abs(z))/dt
    A=np.linalg.lstsq(np.vander(z,N,increasing=True).T,y,rcond=None)[0]
    o=np.argsort(-np.abs(A)); return [(om[i],ga[i],abs(A[i])) for i in o]
keys=[k for k in ('base','uni','s2','s4','z2') if k in d.files]
for K in (2,3,4,6):
  for dt in (0.2,0.3):
    row=[]
    for k in keys:
        tc,y=series(d[k],dt); md=pencil(y,dt,K); md=[x for x in md if abs(abs(x[0])-0.95)<0.15]
        row.append('%s %.4f'%(k,abs(md[0][0])) if md else k+' —')
    print('K=%d dt=%.1f: '%(K,dt)+' | '.join(row))
tc,y=series(d['base'],0.2); print('моды base K=4:',[('%.4f'%o,'%.3f'%g,'%.2g'%A) for o,g,A in pencil(y,0.2,4)])
