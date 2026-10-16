# Разностный измеритель: скорость набега фазы отношения «возмущённый / базовый» ряд (зерно одно и то же — сокращается)
import numpy as np
d=np.load('stark_series.npz'); t=d['t']; prof=d['prof']
def proj(phi,om0=0.904,dt=0.1,T=3.05):
    e=np.arange(0.0,20.0+1e-9,dt); k=np.digitize(t,e)-1; ok=(k>=0)&(k<len(e)-1)
    y=np.bincount(k[ok],(prof*phi*np.exp(-1j*om0*t))[ok].real,minlength=len(e)-1)+1j*np.bincount(k[ok],(prof*phi*np.exp(-1j*om0*t))[ok].imag,minlength=len(e)-1)
    tc=0.5*(e[1:]+e[:-1]); n=int(round(T/dt)); ker=np.ones(n)/n   # бегущее среднее на период биения с античастицей
    yy=np.convolve(y,ker,"valid"); return np.convolve(tc,ker,"valid"), yy
tb,yb=proj(d['base'])
for k,ref in (('uni',-0.00932),('s2',-0.00232),('s4',-0.00430),('z2',-0.00399)):
    _,yk=proj(d[k]); ph=np.unwrap(np.angle(yk/yb))
    for a0,b0 in ((4,17),(6,17),(8,17)):
        s=(tb>a0)&(tb<b0); sl=np.polyfit(tb[s],ph[s],1)[0]
        print('%s окно %d–%d: Δω %+.5f'%(k,a0,b0,sl), end='; ')
    print(' непрерывная %+.5f'%(-ref if k=='z2' else ref) if False else ' непрерывная %+.5f'%ref)
# сам уровень: наклон фазы базового ряда
ph=np.unwrap(np.angle(yb)); s=(tb>6)&(tb<17); print('базовый: ω = 0.904 %+.5f'%np.polyfit(tb[s],ph[s],1)[0])
