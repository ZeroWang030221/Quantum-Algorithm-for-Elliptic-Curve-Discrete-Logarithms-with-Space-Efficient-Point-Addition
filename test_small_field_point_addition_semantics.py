"""End-to-end computational-basis Figure-14 semantics on small curves.

This recursively executes the actual emitted point-addition circuit.  The
computational-basis sweep selects the valid all-zero H-basis measurement branch;
``test_p419_x153_quantum_semantics.py`` separately checks relative phases and a
coherent forward/inverse EEA round trip.
"""
import json
from pathlib import Path

USING_REAL_QISKIT=True
try:
    import qiskit  # type: ignore
except Exception:
    USING_REAL_QISKIT=False
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()

from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from extended_semantic_sweep import VecSim, idx
from test_point_addition_correctness_s835_domainfix_v10 import enumerate_points, affine_add, fig14_schedule_reference


def reg_value(sim, qc, reg, case_i):
    return sum(((sim.q[idx(qc,q)] >> case_i) & 1) << i for i,q in enumerate(reg))


def reg_zero(sim, qc, reg, case_i):
    return all(((sim.q[idx(qc,q)] >> case_i) & 1)==0 for q in reg)


def run_curve(p,a,b,max_p1=16):
    n=p.bit_length();pts=enumerate_points(p,a,b)
    if len(pts)<4: raise AssertionError(f'not enough points p={p}')
    selected=None
    for P2 in pts:
        candidates=[];rejected=[]
        for P1 in pts:
            if P1[0]==P2[0]: continue
            R=affine_add(P1,P2,p=p,a=a,b=b)
            if R is None: continue
            if (P2[0]-R[0])%p==0:
                rejected.append({'P1':P1,'R':R,'reason':'internal multiplier zero'});continue
            candidates.extend(((0,P1),(1,P1)))
            if len(candidates)>=2*max_p1: break
        if candidates:
            selected=(P2,candidates,rejected);break
    if selected is None: raise AssertionError(f'no valid domain p={p}')
    P2,candidates,rejected=selected;x2,y2=P2
    qc=build_point_addition_fig14_quadratic(n=n,p=p,x2=x2,y2=y2)
    regs={r.name:r for r in qc.qregs};ctrl_reg=regs['ctrl'];X=regs['X_x1_to_x3'];Y=regs['Y_y1_to_y3'];A=regs['A_shared_work'];S=regs['S_shared_eea_arith']
    init={}
    def put(qid,k): init[qid]=init.get(qid,0)|(1<<k)
    for k,(ctrl,P1) in enumerate(candidates):
        x1,y1=P1
        if ctrl: put(idx(qc,ctrl_reg[0]),k)
        for i,q in enumerate(X):
            if (x1>>i)&1: put(idx(qc,q),k)
        for i,q in enumerate(Y):
            if (y1>>i)&1: put(idx(qc,q),k)
    sim=VecSim(qc.num_qubits,qc.num_clbits,candidates,init)
    sim.run_circuit(qc,list(range(qc.num_qubits)),list(range(qc.num_clbits)))
    rows=[];passed=0
    for k,(ctrl,P1) in enumerate(candidates):
        got=(reg_value(sim,qc,X,k),reg_value(sim,qc,Y,k))
        expected=fig14_schedule_reference(ctrl,*P1,*P2,p);expxy=(expected[0],expected[1])
        az=reg_zero(sim,qc,A,k);sz=reg_zero(sim,qc,S,k);ctrl_same=((sim.q[idx(qc,ctrl_reg[0])]>>k)&1)==ctrl
        ok=got==expxy and az and sz and ctrl_same;passed+=int(ok)
        rows.append({'ctrl':ctrl,'P1':P1,'P2':P2,'got':got,'expected':expxy,'A_zero':az,'S_zero':sz,'ctrl_restored':ctrl_same,'ok':ok})
    return {'p':p,'a':a,'b':b,'n':n,'P2':P2,'num_qubits':qc.num_qubits,'num_clbits':qc.num_clbits,'cases':len(rows),'passed':passed,'failed':len(rows)-passed,'failures':[r for r in rows if not r['ok']], 'primitive_counts':sim.counts,'composite_calls':sim.composites,'max_depth':sim.max_depth,'rejected':rejected}


def main():
    reports=[run_curve(*cfg) for cfg in ((13,0,7),(17,2,2),(31,0,7))]
    total=sum(r['cases'] for r in reports);passed=sum(r['passed'] for r in reports)
    report={'curves':reports,'cases':total,'passed':passed,'failed':total-passed}
    out=Path('validation_835/small_point_addition_semantics.json');out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2,default=list)+'\n')
    print(json.dumps({'cases':total,'passed':passed,'failed':total-passed,'curves':[{k:r[k] for k in ('p','n','P2','num_qubits','cases','passed','failed')} for r in reports]},indent=2,default=list))
    if report['failed']: raise SystemExit(1)

if __name__=='__main__': main()
