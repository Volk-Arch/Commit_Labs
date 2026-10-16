"""Паули в 3+1 без взаимодействия: два электрона со спином в поле ядра. Амплитуда двух одинаковых фермионов —
антисимметризованное произведение откликов кристалла (спинор Вейля с членом Паули): Ψ = φ_A ⊗ φ_B − φ_B ⊗ φ_A.
Источники: A — облако 1s со сдвигом +1.5 по x, B — со сдвигом −1.5 по x. Чтение — проекция на 1s ⊗ 1s со спинами (c₁, c₂)
по слоям времени. Ожидание: ↑↑ — тождественный ноль; ↑↓ — частота 2·E₁ₛ (на орбитали два электрона с разным спином);
бозоны (симметрично) ↑↑ — тоже 2·E₁ₛ."""
import sys, os, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob
rho,H,Rc=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]); al,a,m=0.45,0.5,1.0
w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X; z=np.load('links31_rho%g_H%g_R%g.npz'%(rho,H,Rc)); I,J=z['I'],z['J']
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); src0=t<0.45
ph=Pr.wilson31(X,I,J,lambda q: -al/np.sqrt(q*q+a*a))
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.array([[1,0],[0,-1]],complex)
rh=np.where(r[:,None]>1e-9,X[:,1:]/np.maximum(r,1e-9)[:,None],0); srr=rh[:,0,None,None]*sx+rh[:,1,None,None]*sy+rh[:,2,None,None]*sz
dU=al*r/(r*r+a*a)**1.5; Bx=(-m*m*np.eye(2)[None]+1j*dU[:,None,None]*srr)/rho
def resp(x0,spin):
    rr=np.linalg.norm(X[:,1:]-np.array([x0,0,0]),axis=1); f=np.where(src0[:,None],np.exp(-al*rr)[:,None]*np.eye(2)[spin][None,:],0)
    return Pr.johnston31_respond_spinor(w.n,I,J,rho,m,f,ph,Bx)
dt=0.25; ed=np.arange(1.5,H-0.3+1e-9,dt); k=np.digitize(t,ed)-1; ok=(k>=0)&(k<len(ed)-1); tc=0.5*(ed[1:]+ed[:-1])
def slices(ps,c):                                       # проекция на 1s с компонентой c по слоям времени
    v=prof*ps[:,c]; return np.bincount(k[ok],v[ok].real,minlength=len(tc))+1j*np.bincount(k[ok],v[ok].imag,minlength=len(tc))
A=resp(+1.5,0); Bu=resp(-1.5,0); Bd=resp(-1.5,1)
p={(n,c):slices(ps,c) for n,ps in (('A',A),('Bu',Bu),('Bd',Bd)) for c in (0,1)}
om1=np.linspace(0.6,1.3,1401); om2=np.linspace(1.4,2.2,1601)
def pk(sig,om):
    w_=np.sin(np.pi*(tc-tc.min())/np.ptp(tc))**2; S=np.abs(np.exp(-1j*np.outer(om,tc))@(w_*sig))**2; i=int(np.argmax(S)); return om[i],S.max()
E1=pk(p[('A',0)],om1)[0]; print('одна частица: 1s %.4f (2·E = %.4f)'%(E1,2*E1))
F_uu=p[('A',0)]*p[('Bu',0)]-p[('Bu',0)]*p[('A',0)]; B_uu=p[('A',0)]*p[('Bu',0)]+p[('Bu',0)]*p[('A',0)]
F_ud=p[('A',0)]*p[('Bd',1)]-p[('Bd',0)]*p[('A',1)]
nrm=np.abs(B_uu).max()
print('фермионы ↑↑ в 1s⊗1s: max|Ψ| / бозоны = %.1e'%(np.abs(F_uu).max()/nrm))
print('фермионы ↑↓ в 1s⊗1s: пик %.4f, амплитуда / бозоны ↑↑ = %.2f'%(pk(F_ud,om2)[0],np.abs(F_ud).max()/nrm))
print('бозоны ↑↑ в 1s⊗1s: пик %.4f'%pk(B_uu,om2)[0])
# с членом Паули спин смешивается в p-волне; в s-чтении переворот мал:
print('доля «чужой» компоненты в s-чтении (спин-орбита): %.1e'%(np.abs(p[('A',1)]).max()/np.abs(p[('A',0)]).max()))
