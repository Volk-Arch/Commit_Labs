import sys, json; sys.path.insert(0,'/home/claude/goldlab')
from goldlab import atom31f as Af
M=float(sys.argv[1]); R=float(sys.argv[2]); H=float(sys.argv[3])
ref=Af.cont_refs(M,R,H); print(json.dumps(ref)); json.dump(ref,open('atom31f_ref_M%.3f_R%g_H%g.json'%(M,R,H),'w'))
