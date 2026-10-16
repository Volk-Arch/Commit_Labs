"""Один проход на мир + гармоническая инверсия по каналам-оболочкам: уровень лёгкого атома (фаза Вильсона и вставка при энергии E)."""
import sys, json, numpy as np; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af, fuzzy as Fz, hinv as Hv
Af.AL=0.6; M=0.381966; R=11.0; H=float(sys.argv[2]) if len(sys.argv)>2 else 20.0; kind=sys.argv[1]
X=Af.make_world(1.0,H,R,poisson_seed=1 if kind=='poisson' else None); at=Af.Atom(X,1.0,M,H,R); at.idx=Fz.build_index(X,2.0)
shells=np.linspace(0,R,7); sh=np.digitize(at.r,shells)-1; dtb=0.5; ed=np.arange(1.5,H-0.3+1e-9,dtb); kb=np.digitize(at.t,ed)-1
def channels(phi):
    Y=[]
    for c in range(6):
        ok=(sh==c)&(kb>=0)&(kb<len(ed)-1); cnt=np.bincount(kb[ok],minlength=len(ed)-1)
        v=(np.bincount(kb[ok],phi[ok].real,minlength=len(ed)-1)+1j*np.bincount(kb[ok],phi[ok].imag,minlength=len(ed)-1))/np.maximum(cnt,1)
        Y.append(v*(4/3*np.pi*(shells[c+1]**3-shells[c]**3)))
    return np.array(Y)
e=np.array([1,0]); U=-Af.AL/np.sqrt(at.r**2+Af.A**2); f=at.src()
def run_phase(pauli=0.0): return at.respond(at.B(M,pauli=pauli))@e
def run_energy(E,pauli=0.0):
    b=-(at.mb2(M)+2*E*U-U*U); B=(b[:,None,None]*at.eye+pauli*1j*at.dU[:,None,None]*at.sr)
    return Fz.respond_fast(X,at.idx,at.tc2,at.alpha,at.coef,at.rg,at.Unuc,f,np.ascontiguousarray(B,dtype=np.complex128),Af.BETA,1,use_phase=0)@e
lv=lambda phi: Hv.level(channels(phi),dtb,0.05,M)[0]
Eph=lv(run_phase()); Ephs=lv(run_phase(1.0))
# вставка при энергии: неподвижная точка по инверсии
def om(E): return lv(run_energy(E))
E0=0.30; w0=om(E0); E1=w0; w1=om(E1)
for _ in range(4):
    if abs(w1-E1)<3e-4: break
    E2=E1-(w1-E1)*(E1-E0)/((w1-E1)-(w0-E0)); E0,w0,E1=E1,w1,E2; w1=om(E1)
Es=lv(run_energy(E1,1.0))
print(json.dumps(dict(kind=kind,H=H,phase_level=Eph,phase_spin=Ephs-Eph,energy_level=E1,energy_check=w1,energy_spin_at_E=Es-w1)),flush=True)
# спин с самосогласованием (неподвижная точка с членом Паули)
def om_s(E): return lv(run_energy(E,1.0))
E0=E1; w0=om_s(E0); Ea=w0; wa=om_s(Ea)
for _ in range(4):
    if abs(wa-Ea)<3e-4: break
    E2=Ea-(wa-Ea)*(Ea-E0)/((wa-Ea)-(w0-E0)); E0,w0,Ea=Ea,wa,E2; wa=om_s(Ea)
print(json.dumps(dict(kind=kind,spin_selfconsistent=Ea-E1,slope_dw_dE=(w1-w0) if False else None)),flush=True)
