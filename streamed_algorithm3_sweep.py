"""Stream Algorithm-3 one emitted step at a time over several basis inputs."""
import argparse,json,random,time,gc,sys
from pathlib import Path

USING_REAL_QISKIT=True
try:
    import qiskit  # type: ignore
except Exception:
    USING_REAL_QISKIT=False
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
    import qiskit  # type: ignore
from qiskit import QuantumCircuit

import eea_circuit_s835_fastdual as eea
import under1000_eea_shared_s835_fastdual_wrapped as shared
from extended_semantic_sweep import VecSim,idx


def exact_steps(p,x):
    r0,r1=p,min(x,p-x);w=0
    while r1:
        q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
    return 4*w


def choose(p,count,seed):
    vals={1,2,3,p//2,p//2+1,p-3,p-2,p-1};rng=random.Random(seed)
    while len(vals)<min(count,p-1):vals.add(rng.randrange(1,p))
    return sorted(x for x in vals if 1<=x<p)


def qregs_by_name(qc):return {r.name:r for r in qc.qregs}

def put(init,qid,k):init[qid]=init.get(qid,0)|(1<<k)


def flatten_classical_ops(circuit):
    """Recursively flatten a coherent emitted step to ordered bit operations."""
    out=[]
    def rec(circ,qmap):
        for item in circ.data:
            if hasattr(item,'operation'):
                op=item.operation;qargs=tuple(item.qubits);cargs=tuple(item.clbits)
            else:
                op,qargs,cargs=item
            if cargs:
                raise AssertionError(f'coherent streamed step unexpectedly uses clbits: {op.name}')
            if getattr(op,'condition',None) is not None:
                raise AssertionError(f'coherent streamed step unexpectedly has a classical condition: {op.name}')
            qids=[qmap[idx(circ,q)] for q in qargs]
            nm=str(getattr(op,'base_name',None) or op.name).lower()
            while nm.endswith('_dg'): nm=nm[:-3]
            nm={'cnot':'cx','tof':'ccx','toffoli':'ccx'}.get(nm,nm)
            definition=getattr(op,'definition',None)
            if definition is not None and nm not in {'x','cx','ccx','mcx','swap','z','cz','barrier','id'}:
                rec(definition,qids)
            elif nm in {'x','cx','ccx','mcx','swap'}:
                out.append((nm,tuple(qids)))
            elif nm in {'z','cz','barrier','id'}:
                # Computational-basis endpoint semantics are phase-insensitive.
                continue
            else:
                raise NotImplementedError(f'unsupported coherent primitive {nm!r}')
    rec(circuit,list(range(circuit.num_qubits)))
    return out


def apply_flat_ops(sim,ops):
    q=sim.q
    for nm,ids in ops:
        if nm=='x':
            q[ids[0]] ^= sim.allmask
        elif nm=='cx':
            q[ids[1]] ^= q[ids[0]]
        elif nm in {'ccx','mcx'}:
            mask=sim.allmask
            for c in ids[:-1]: mask &= q[c]
            q[ids[-1]] ^= mask
        elif nm=='swap':
            q[ids[0]],q[ids[1]]=q[ids[1]],q[ids[0]]
        else:
            raise AssertionError(nm)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('p',type=lambda s:int(s,0));ap.add_argument('--count',type=int,default=8);ap.add_argument('--seed',type=int,default=835);ap.add_argument('--out',required=True);ap.add_argument('--progress-every',type=int,default=100);ap.add_argument('--require-real-qiskit',action='store_true');a=ap.parse_args()
    if a.require_real_qiskit and not USING_REAL_QISKIT: raise SystemExit('real Qiskit is required but is not installed')
    p=a.p;n=p.bit_length();cases=choose(p,a.count,a.seed);layout=shared.shared_eea_layout(n,p=p)
    regs=eea.make_global_registers_noctrl(n=n,len_width=layout.len_width,shift_width=layout.shift_width,T_max=layout.T_max,aux_size=layout.step_aux)
    base=QuantumCircuit(*regs);by=qregs_by_name(base);W1=by['Work1'];W2=by['Work2'];lt=by['l_t'];lq=by['l_q'];ls=by['l_s'];lrp=by['l_rp'];Iter=by['Iter'];Aux=by['Aux']
    init={}
    for k,x in enumerate(cases):
        xu=min(x,p-x)
        if x>p//2:put(init,idx(base,Iter[0]),k)
        put(init,idx(base,W1[0]),k)
        for i in range(n):
            if (p>>(n-1-i))&1:put(init,idx(base,W1[3+i]),k)
            if (xu>>(n-1-i))&1:put(init,idx(base,W2[3+i]),k)
        for q in lq:put(init,idx(base,q),k)
        for q in ls:put(init,idx(base,q),k)
        enc=xu.bit_length()-1
        for i,q in enumerate(lrp):
            if (enc>>i)&1:put(init,idx(base,q),k)
    sim=VecSim(base.num_qubits,base.num_clbits,cases,init);t0=time.time()

    # Cache only genuinely identical emitted step circuits.  With the latest
    # paper windows, a step is determined by its five window pairs and by T mod 4.
    op_cache={}
    for T in range(1,layout.T_max+1):
        wins=eea._step_windows(n,T)
        key=(tuple(sorted(wins.items())),T%4)
        ops=op_cache.get(key)
        if ops is None:
            step=eea.build_step_circuit(n,T,T_max=layout.T_max,aux_size=layout.step_aux,measurement_uncompute=False)
            ops=flatten_classical_ops(step);op_cache[key]=ops
        apply_flat_ops(sim,ops)
        if a.progress_every and T%a.progress_every==0:
            print(f'{T}/{layout.T_max} elapsed={time.time()-t0:.1f}s',flush=True)
    # Public wrapper endpoint canonicalization and epoch packing.
    final=QuantumCircuit(*regs)
    eea.canonical_rotate_work2(final,by['Work2'],by['l_s'],by['Aux'][2:],shift_epoch=by['Aux'][1])
    final.x(by['Aux'][0]);eea.compress_terminal_shift_epoch(final,shift_epoch=by['Aux'][1],l_s=by['l_s'],outer_ctrl=by['Aux'][0],scratch=by['Aux'][2:]);final.x(by['Aux'][0])
    apply_flat_ops(sim,flatten_classical_ops(final))
    rows=[];passed=0
    for k,x in enumerate(cases):
        tp=sum(((sim.q[idx(base,q)]>>k)&1)<<i for i,q in enumerate(W2[:n]));it=(sim.q[idx(base,Iter[0])]>>k)&1;inv=tp%p if it else (-tp)%p
        lrpzero=all(((sim.q[idx(base,q)]>>k)&1)==1 for q in lrp);auxzero=all(((sim.q[idx(base,q)]>>k)&1)==0 for q in Aux);exp=pow(x,-1,p);ok=inv==exp and lrpzero and auxzero
        passed+=int(ok);rows.append({'x':x,'got_inverse':inv,'expected':exp,'tprime':tp,'Iter':it,'lrp_zero':lrpzero,'Aux_zero':auxzero,'exact_steps':exact_steps(p,x),'ok':bool(ok)})
    report={'qiskit_version':getattr(qiskit,'__version__','unknown'),'real_qiskit':USING_REAL_QISKIT,'p':p,'n':n,'T_max':layout.T_max,'len_width':layout.len_width,'shift_width':layout.shift_width,'aux_size':layout.step_aux,'cases':len(cases),'passed':passed,'failed':len(cases)-passed,'elapsed_s':time.time()-t0,'failures':[r for r in rows if not r['ok']],'results':rows}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('p','n','T_max','shift_width','aux_size','cases','passed','failed','elapsed_s')},indent=2))
    if report['failed']:raise SystemExit(1)
if __name__=='__main__':main()
