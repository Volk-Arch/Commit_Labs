import sys,json; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
Af.AL=float(sys.argv[1]); M=float(sys.argv[2]); R=float(sys.argv[3]); H=float(sys.argv[4])
for tr in ((False,) if len(sys.argv)<6 else (False,True)):
    ref=Af.cont_refs(M,R,H,trunc=tr); print('обрезка' if tr else 'без обрезки', json.dumps({k:round(v,5) for k,v in ref.items()}),flush=True)
    json.dump(ref,open('atom31f_ref_al%g_M%.3f_R%g_H%g_trunc%d.json'%(Af.AL,M,R,H,tr),'w'))
