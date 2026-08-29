"""Exhaustive per-microstep oracle comparison for an arbitrary small prime.

All 418 nonzero inputs are propagated through the emitted Algorithm-3 step
circuits in parallel.  After every microstep up to each input's exact EEA
horizon, the complete packed work registers, length metadata, phase bits,
iteration parity, Sign bit, and scratch cleanliness are compared against the
independent classical ``one_iter_opt`` model.
"""

import argparse, json, math, sys, time
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
import eea_circuit as eea
from extended_semantic_sweep import VecSim, idx

HERE=Path(__file__).resolve().parent
REF=HERE/'validation_reference'
sys.path.insert(0,str(REF))
from register import Registers  # type: ignore
from one_iter_opt import one_iter_opt  # type: ignore


def exact_steps(p:int,x:int)->int:
    r0,r1=p,min(x,p-x);w=0
    while r1:
        q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
    return 4*w

def dec_len(enc:int,width:int)->int:
    return 0 if enc==((1<<width)-1) else enc+1

def model_work2_bits(regs:Registers)->str:
    w=regs.work2
    if w.l_r_prime==0:
        return bin(int(w.t_prime))[2:].zfill(w.n+3)[::-1]
    return w.bin().replace('|','')

def bit_at(mask:int,k:int)->int:return (mask>>k)&1

def main():
    ap=argparse.ArgumentParser();ap.add_argument('p',type=lambda s:int(s,0));ap.add_argument('--out',required=True);ap.add_argument('--verbose',action='store_true');ap.add_argument('--require-real-qiskit',action='store_true');a=ap.parse_args()
    if a.require_real_qiskit and not USING_REAL_QISKIT: raise SystemExit('real Qiskit is required but is not installed')
    p=int(a.p);n=p.bit_length();cases=list(range(1,p));cfg=eea.get_n_config(n);Tmax=int(cfg['T_max']);lw=int(cfg['len_width']);sw=int(cfg['shift_width']);aux=int(eea.qiskit_paper_aux_size(n,lw,sw,Tmax))
    first=eea.build_step_circuit(n,1,T_max=Tmax,aux_size=aux,measurement_uncompute=False);by={r.name:r for r in first.qregs};sim=VecSim(first.num_qubits,first.num_clbits,cases,{});allmask=(1<<len(cases))-1
    def setc(q,v):sim.q[idx(first,q)]=allmask if v else 0
    setc(by['Work1'][0],1)
    for j in range(n):setc(by['Work1'][3+j],(p>>(n-1-j))&1)
    models=[];horizons=[]
    for x in cases:
        xu=min(x,p-x);models.append(Registers(p,xu,1 if x>p//2 else 0));horizons.append(exact_steps(p,x))
    for j in range(n):
        m=0
        for k,x in enumerate(cases):
            xu=min(x,p-x)
            if (xu>>(n-1-j))&1:m|=1<<k
        sim.q[idx(first,by['Work2'][3+j])]=m
    for q in by['l_q']:sim.q[idx(first,q)]=allmask
    for q in by['l_s']:sim.q[idx(first,q)]=allmask
    for b,q in enumerate(by['l_rp']):
        m=0
        for k,x in enumerate(cases):
            if ((min(x,p-x).bit_length()-1)>>b)&1:m|=1<<k
        sim.q[idx(first,q)]=m
    im=0
    for k,x in enumerate(cases):
        if x>p//2:im|=1<<k
    sim.q[idx(first,by['Iter'][0])]=im
    del first
    fields=['Phase1','Phase2','Iter','Sign','Work1','Work2','l_t','l_q','l_s','l_rp','Aux']
    mismatches=[];checked=0;t0=time.time()
    for T in range(1,Tmax+1):
        step=eea.build_step_circuit(n,T,T_max=Tmax,aux_size=aux,measurement_uncompute=False);by={r.name:r for r in step.qregs};sim.run_circuit(step,list(range(step.num_qubits)),list(range(step.num_clbits)))
        for k,(x,m,h) in enumerate(zip(cases,models,horizons)):
            if T>h:continue
            one_iter_opt(m);checked+=1
            got_w1=''.join(str(bit_at(sim.q[idx(step,q)],k)) for q in by['Work1'])
            got_w2=''.join(str(bit_at(sim.q[idx(step,q)],k)) for q in by['Work2'])
            got={
                'work1':got_w1,'work2':got_w2,
                'ell_t':dec_len(sum(bit_at(sim.q[idx(step,q)],k)<<i for i,q in enumerate(by['l_t'])),lw),
                'ell_q':dec_len(sum(bit_at(sim.q[idx(step,q)],k)<<i for i,q in enumerate(by['l_q'])),lw),
                'ell_rp':dec_len(sum(bit_at(sim.q[idx(step,q)],k)<<i for i,q in enumerate(by['l_rp'])),lw),
                'ell_s':dec_len(sum(bit_at(sim.q[idx(step,q)],k)<<i for i,q in enumerate(by['l_s'])),sw),
                'phase1':bit_at(sim.q[idx(step,by['Phase1'][0])],k),
                'phase2':bit_at(sim.q[idx(step,by['Phase2'][0])],k),
                'iter':bit_at(sim.q[idx(step,by['Iter'][0])],k),
                'sign':bit_at(sim.q[idx(step,by['Sign'][0])],k),
                'aux_clean':all(bit_at(sim.q[idx(step,q)],k)==0 for q in by['Aux']),
            }
            s=m.snapshot();exp={'work1':m.work1.bin().replace('|',''),'work2':model_work2_bits(m),'ell_t':m.work1.l_t,'ell_q':m.work1.l_q,'ell_rp':m.work2.l_r_prime,'ell_s':m.work2.l_s,'phase1':m.control.phase1,'phase2':m.control.phase2,'iter':m.control.iter,'sign':m.control.sign,'aux_clean':True}
            diff={name:{'got':got[name],'expected':exp[name]} for name in got if got[name]!=exp[name]}
            if diff:
                mismatches.append({'x':x,'T':T,'diff':diff});break
        if mismatches:break
        if a.verbose and T%4==0:print(f'[PASS] through T={T}: active cases={sum(h>=T for h in horizons)}')
    rep={'qiskit_version':getattr(qiskit,'__version__','unknown'),'real_qiskit':USING_REAL_QISKIT,'p':p,'n':n,'T_max':Tmax,'len_width':lw,'shift_width':sw,'aux_size':aux,'inputs':len(cases),'state_checks':checked,'passed':not mismatches,'mismatches':mismatches,'elapsed_s':time.time()-t0};Path(a.out).write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2));raise SystemExit(0 if not mismatches else 1)
if __name__=='__main__':main()
