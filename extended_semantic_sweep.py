#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
USING_REAL_QISKIT = True
try:
    import qiskit  # type: ignore
except Exception:
    USING_REAL_QISKIT = False
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
    import qiskit  # type: ignore
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
import under1000_eea_shared_s835_fastdual_wrapped as shared

ALIASES={'cnot':'cx','tof':'ccx','toffoli':'ccx'}
IGN={'barrier','id','delay'}

def name_of(op):
    base=getattr(op,'base_name',None)
    s=str(base if base is not None else op.name).lower()
    while s.endswith('_dg'): s=s[:-3]
    return ALIASES.get(s,s)

def idx(c,b): return int(c.find_bit(b).index)

def items(c):
    for it in c.data:
        if hasattr(it,'operation'): yield it.operation,tuple(it.qubits),tuple(it.clbits)
        else:
            op,q,cargs=it; yield op,tuple(q),tuple(cargs)

def cond_mask(op,definition,cmap,cbits,allmask):
    cond=getattr(op,'condition',None)
    if cond is None:return allmask
    lhs,expected=cond
    # single Clbit
    try:
        isreg = hasattr(lhs,'__iter__') and not hasattr(lhs,'register')
    except Exception:isreg=False
    if not isreg:
        local=idx(definition,lhs)
        val=cbits[cmap[local]]
        return val if int(expected)==1 else (allmask ^ val)
    # register equality; build truth mask casewise with bitset algebra
    mask=allmask
    for i,b in enumerate(lhs):
        local=idx(definition,b); val=cbits[cmap[local]]
        want=(int(expected)>>i)&1
        mask &= val if want else (allmask ^ val)
    return mask

class VecSim:
    def __init__(self,nq,nc,cases,initial_by_q):
        self.ncases=len(cases);self.allmask=(1<<self.ncases)-1
        self.q=[0]*nq;self.c=[0]*nc
        for qid,mask in initial_by_q.items():self.q[qid]=mask & self.allmask
        self.counts={};self.composites=0;self.max_depth=0
    def run_circuit(self,circ,qmap,cmap,depth=0,active_mask=None):
        if active_mask is None:
            active_mask=self.allmask
        active_mask &= self.allmask
        if active_mask==0:
            return
        self.max_depth=max(self.max_depth,depth)
        for op,qargs,cargs in items(circ):
            qids=[qmap[idx(circ,q)] for q in qargs]
            cids=[cmap[idx(circ,c)] for c in cargs]
            nm=name_of(op); definition=getattr(op,'definition',None)

            # Real Qiskit 2.x represents feed-forward as IfElseOp.  Different
            # packed test cases can take different branches, so propagate an
            # explicit case mask into each selected block.
            if nm=='if_else' and hasattr(op,'blocks'):
                self.composites+=1
                truth=cond_mask(op,circ,cmap,self.c,self.allmask)
                blocks=list(op.blocks)
                if blocks:
                    self.run_circuit(blocks[0],qids,cids,depth+1,active_mask & truth)
                if len(blocks)>1:
                    self.run_circuit(blocks[1],qids,cids,depth+1,active_mask & (self.allmask ^ truth))
                continue

            if definition is not None and nm not in {'x','cx','ccx','mcx','swap','h','z','cz','measure','reset',*IGN}:
                self.composites+=1;self.run_circuit(definition,qids,cids,depth+1,active_mask);continue
            self.counts[nm]=self.counts.get(nm,0)+1
            cm=active_mask & cond_mask(op,circ,cmap,self.c,self.allmask)
            if cm==0:continue
            if nm=='x':self.q[qids[0]] ^= cm
            elif nm=='cx':self.q[qids[1]] ^= self.q[qids[0]] & cm
            elif nm in {'ccx','mcx'}:
                m=cm
                for k in qids[:-1]:m &= self.q[k]
                self.q[qids[-1]] ^= m
            elif nm=='swap':
                # conditional swap: xor-difference under mask
                d=(self.q[qids[0]]^self.q[qids[1]])&cm
                self.q[qids[0]]^=d;self.q[qids[1]]^=d
            elif nm=='h':
                # Deferred to following measurement. Computational bit after H
                # is not classical; all emitted EEA Hs are measurement-uncompute Hs.
                pass
            elif nm=='measure':
                # Choose the valid all-zero branch. H-basis MBU outcomes are uniform;
                # phase corrections do not affect computational-basis semantics.
                self.c[cids[0]] &= self.allmask ^ cm
                self.q[qids[0]] &= self.allmask ^ cm
            elif nm=='reset':
                self.q[qids[0]] &= self.allmask ^ cm
            elif nm in {'z','cz',*IGN}:pass
            else:raise NotImplementedError(f'leaf {nm} op={op!r}')

def exact_steps(p,x):
    r0,r1=p,min(x,p-x);w=0
    while r1:q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
    return 4*w

def choose_cases(p,count,seed):
    vals={1,2,3,p//2,p//2+1,p-3,p-2,p-1}
    rng=random.Random(seed)
    while len(vals)<min(count,p-1): vals.add(rng.randrange(1,p))
    return sorted(x for x in vals if 1<=x<p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('p',type=lambda s:int(s,0));ap.add_argument('--all',action='store_true');ap.add_argument('--count',type=int,default=128);ap.add_argument('--seed',type=int,default=12345);ap.add_argument('--out',required=True);ap.add_argument('--require-real-qiskit',action='store_true')
    a=ap.parse_args();
    if a.require_real_qiskit and not USING_REAL_QISKIT: raise RuntimeError('real Qiskit is required but unavailable')
    p=a.p;n=p.bit_length();cases=list(range(1,p)) if a.all else choose_cases(p,a.count,a.seed)
    layout=shared.shared_eea_layout(n)
    X=QuantumRegister(n,'X');A=QuantumRegister(n,'A');S=QuantumRegister(layout.s_qubits,'S');c=ClassicalRegister(n,'c');qc=QuantumCircuit(X,A,S,c)
    qc.append(shared.eea_forward_shared_instruction(n,p,T_max=layout.T_max,lazy_definition=True),[*X,*A,*S],[*c])
    # bitset input masks
    init={}
    for i,qbit in enumerate(X):
        m=0
        for k,x in enumerate(cases):
            if (x>>i)&1:m|=1<<k
        init[idx(qc,qbit)]=m
    sim=VecSim(qc.num_qubits,qc.num_clbits,cases,init)
    t=time.time();sim.run_circuit(qc,list(range(qc.num_qubits)),list(range(qc.num_clbits)));elapsed=time.time()-t
    rows=[];passed=0
    for k,x in enumerate(cases):
        got=0
        for i,qbit in enumerate(X):got|=((sim.q[idx(qc,qbit)]>>k)&1)<<i
        az=all(((sim.q[idx(qc,q)]>>k)&1)==0 for q in A)
        exp=pow(x,-1,p);ok=(got==exp and az and (x*got)%p==1)
        passed+=ok
        rows.append({'x':x,'got':got,'expected':exp,'A_zero':az,'ok':bool(ok),'exact_steps':exact_steps(p,x)})
    report={'qiskit_version':getattr(qiskit,'__version__','unknown'),'real_qiskit':USING_REAL_QISKIT,'p':p,'n':n,'T_max':layout.T_max,'cases':len(cases),'passed':passed,'failed':len(cases)-passed,'elapsed_s':elapsed,'primitive_counts':sim.counts,'composite_calls':sim.composites,'max_depth':sim.max_depth,'failures':[r for r in rows if not r['ok']][:100],'results':rows}
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['p','n','T_max','cases','passed','failed','elapsed_s','composite_calls','max_depth']},indent=2))
    if report['failures']:print('first_failure',json.dumps(report['failures'][0],indent=2))
    raise SystemExit(0 if report['failed']==0 else 1)
if __name__=='__main__':main()
