import random
try:
    import qiskit
except Exception:
    import mini_qiskit_runtime as _mini; _mini.install_as_qiskit()
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import eea_circuit_s835_fastdual as f
from extended_semantic_sweep import VecSim, idx


def run_gate_pair(gf, gi, reg_sizes, cases=128, seed=1):
    regs=[QuantumRegister(s, f'r{i}') for i,s in enumerate(reg_sizes)]
    c=ClassicalRegister(1,'m')
    qc=QuantumCircuit(*regs,c)
    qargs=[q for r in regs for q in r]
    # dynamic blocks are circuits; compose helper preserving clbits
    f._e._append_with_optional_clbits(qc,gf,qargs)
    f._e._append_with_optional_clbits(qc,gi,qargs)
    rng=random.Random(seed)
    vals=[]
    init={}
    # all scratch register (last) zero; randomize everything else
    for k in range(cases):
        row=[]
        for ri,s in enumerate(reg_sizes):
            v=0 if ri==len(reg_sizes)-1 else rng.randrange(1<<s)
            row.append(v)
        vals.append(row)
    off=0
    for ri,r in enumerate(regs):
        for bi,q in enumerate(r):
            m=0
            for k,row in enumerate(vals):
                if (row[ri]>>bi)&1:m|=1<<k
            init[idx(qc,q)]=m
    sim=VecSim(qc.num_qubits,qc.num_clbits,list(range(cases)),init)
    sim.run_circuit(qc,list(range(qc.num_qubits)),list(range(qc.num_clbits)))
    bad=[]
    for k,row in enumerate(vals):
        for ri,r in enumerate(regs):
            got=0
            for bi,q in enumerate(r):got|=((sim.q[idx(qc,q)]>>k)&1)<<bi
            if got!=row[ri]:
                bad.append((k,ri,row[ri],got));break
    print(gf.name if hasattr(gf,'name') else gf, 'bad',len(bad), 'first',bad[:3], 'clbits',qc.num_clbits)
    return not bad

def main():
    f.set_measurement_uncompute(False)
    # interval M=5, n=4, k=2,K=6, lw=3,sw=3
    args=dict(n=4,k=2,K=6,len_width=3,shift_width=3,mode='sub',sign_update=True,target='work1')
    gf=f.lc_interval_addsub_unary_gate(name='F',inverse=False,**args)
    gi=f.lc_interval_addsub_unary_gate(name='I',inverse=True,**args)
    # Ctrl,Sign,W1(5),W2(5),lt3,lq3,ls3,Scratch
    ok1=run_gate_pair(gf,gi,[1,1,5,5,3,3,3,gf.num_qubits-(2+10+9)],cases=256)
    args2=dict(k=1,K=6,len_width=3,mode='add',sign_update=True,target='work2')
    gf2=f.lc_prefix_addsub_prepared_boundary_gate(name='PF',inverse=False,**args2)
    gi2=f.lc_prefix_addsub_prepared_boundary_gate(name='PI',inverse=True,**args2)
    # Ctrl Sign W1 W2 Boundary Scratch
    ok2=run_gate_pair(gf2,gi2,[1,1,6,6,3,gf2.num_qubits-(2+12+3)],cases=256,seed=2)
    raise SystemExit(0 if ok1 and ok2 else 1)
if __name__=='__main__':main()
