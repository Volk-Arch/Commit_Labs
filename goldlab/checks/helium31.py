"""«Гелий» 3+1 в первом порядке: второй электрон в поле ядра + облака первого (1s, синглет — Паули разрешает два на 1s
только с разным спином). J = сдвиг уровня 1s от облака. Три варианта: непрерывная теория (точно и тем же прибором),
кристалл с точным полем облака, кристалл с полем облака ИЗ ПОРЯДКА (связи, field31_cloud.py)."""
import sys, math, numpy as np
sys.path.insert(0,'/home/claude/goldlab')
from goldlab import worlds as Wd, propagate as Pr, observe as Ob
rho,H,Rc,frho=float(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),sys.argv[4]; al,a,m=0.45,0.5,1.0
c=np.load('helium31_cloud.npz'); rr,Uc,Jc=c['r'],c['Uc'],float(c['J'])
F=np.load('field31_cloud_rho%s.npz'%frho)['rows']; Rf,Uf=F[:,0],F[:,1]
def Uord(q):                                       # профиль поля из порядка; дальше последней точки — точное × последнее отношение
    q=np.asarray(q); last=Uf[-1]/np.interp(Rf[-1],rr,Uc); return np.where(q<=Rf[-1],np.interp(q,Rf,Uf),np.interp(q,rr,Uc)*last)
w=Wd.gold31(rho,'cylinder',H=H,Rc=Rc,Rs=3.0); X=w.X; z=np.load('links31_rho%g_H%g_R%g.npz'%(rho,H,Rc)); I,J=z['I'],z['J']
r=np.linalg.norm(X[:,1:],axis=1); t=X[:,0]; prof=np.exp(-al*r); f=np.where(t<0.45,prof,0); sel=(t>1.5)&(t<H-0.3); tt=t[sel]; om=np.linspace(0.6,1.3,1401)
def level(Ufun):
    phi=Pr.johnston31_respond(w.n,I,J,rho,m,f,Pr.wilson31(X,I,J,Ufun)); S=Ob.spectrum(tt,prof[sel]*phi[sel],om,'hann'); pk=Ob.peaks(om,S,0.0); return pk[int(np.argmax([h for _,h in pk]))][0]
Un=lambda q: -al/np.sqrt(q*q+a*a)
E0=level(Un); Ee=level(lambda q: Un(q)+np.interp(q,rr,Uc)); Eo=level(lambda q: Un(q)+Uord(q))
print('ρ=%g: 1s %.4f; J кристалл (точное поле облака) %+.4f; J кристалл (поле из порядка, ρ_поля=%s) %+.4f; J непрерывная точно %+.4f'%(rho,E0,Ee-E0,frho,Eo-E0,Jc))
