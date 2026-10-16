import sys; H=float(sys.argv[1]); Rb=float(sys.argv[2])
src=open('instr_cont.py').read().split('E0=peak')[0].replace('ball=r<7','ball=r<%g'%Rb).replace('T=19.8','T=%g'%(H-0.2)).replace('ts<19.7','ts<%g'%(H-0.3))
exec(src)
E1=peak(*evolve(m2=1.0)); E2=peak(*evolve(m2=0.81)); print('непрерывная, тот же прибор H=%g: %.5f / %.5f, наклон %.3f (истина 0.487)'%(H,E1,E2,(E1-E2)/0.19))
