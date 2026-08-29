"""Reference-entangled gate-level Figure-14 phase/cleanup regression.

The test executes the actual emitted p=13 point-addition circuit on

    (|0>|P0> + |1>|P1>)/sqrt(2)

using several independent dynamic-measurement trajectories.  It checks the two
affine outputs, their relative phase, preservation of the external reference,
and cleanup of every A/S work qubit.  Python affine arithmetic is used only as
an independent assertion oracle.
"""

import argparse
import json
import math
from pathlib import Path

USING_REAL_QISKIT=True
try:
    import qiskit  # type: ignore
except Exception:
    USING_REAL_QISKIT=False
    import mini_qiskit_runtime as _mini
    _mini.install_as_qiskit()
from qiskit import QuantumCircuit, QuantumRegister

from point_addition_fig14_s835_fastdual_wrapped_quadratic import (
    build_point_addition_fig14_quadratic,
)
from test_p419_x153_quantum_semantics import SparseTrajectorySimulator, _find_index
from test_point_addition_correctness_s835_domainfix_v10 import affine_add

P=13; A_CURVE=0; B_CURVE=7
P2=(7,5)
P0=(8,5)
P1=(11,5)


def _set_reg_basis(basis:int,circuit,reg,value:int)->int:
    for i,q in enumerate(reg):
        if (int(value)>>i)&1:
            basis |= 1 << _find_index(circuit,q)
    return basis


def _reg_value(basis:int,circuit,reg)->int:
    return sum(((basis>>_find_index(circuit,q))&1)<<i for i,q in enumerate(reg))


def _reg_zero(basis:int,circuit,reg)->bool:
    return all(((basis>>_find_index(circuit,q))&1)==0 for q in reg)


def build_test_circuit():
    point=build_point_addition_fig14_quadratic(n=P.bit_length(),p=P,x2=P2[0],y2=P2[1])
    ref=QuantumRegister(1,'ref')
    qc=QuantumCircuit(ref,*point.qregs,*point.cregs,name='SMALL_PA_REFERENCE_ENTANGLED')
    qc.compose(point,qubits=list(qc.qubits[1:]),clbits=list(qc.clbits),inplace=True)
    regs={r.name:r for r in qc.qregs}
    return qc,ref,regs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shots',type=int,default=8)
    ap.add_argument('--seed',type=int,default=130835)
    ap.add_argument('--out',default='validation_835/small_point_addition_coherence.json')
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args()

    qc,ref,regs=build_test_circuit()
    ctrl=regs['ctrl'];X=regs['X_x1_to_x3'];Y=regs['Y_y1_to_y3']
    W=regs['A_shared_work'];S=regs['S_shared_eea_arith']
    R0=affine_add(P0,P2,p=P,a=A_CURVE,b=B_CURVE)
    R1=affine_add(P1,P2,p=P,a=A_CURVE,b=B_CURVE)
    assert R0 is not None and R1 is not None and R0 != R1

    b0=0
    b0 |= 1 << _find_index(qc,ctrl[0])
    b0=_set_reg_basis(b0,qc,X,P0[0]);b0=_set_reg_basis(b0,qc,Y,P0[1])
    b1=1 << _find_index(qc,ref[0])
    b1 |= 1 << _find_index(qc,ctrl[0])
    b1=_set_reg_basis(b1,qc,X,P1[0]);b1=_set_reg_basis(b1,qc,Y,P1[1])
    amp=1/math.sqrt(2)
    initial={b0:amp+0j,b1:amp+0j}

    rows=[]
    for shot in range(args.shots):
        sampled=SparseTrajectorySimulator(qc.num_qubits,qc.num_clbits,initial_state=initial,seed=args.seed+shot).run(qc)
        replay=SparseTrajectorySimulator(
            qc.num_qubits,qc.num_clbits,initial_state=initial,seed=0,
            forced_measurements=sampled.measurement_record,
        ).run(qc)
        if len(replay.state)!=2:
            raise AssertionError(f'shot {shot}: support={len(replay.state)}, expected 2')
        branches={}
        for basis,a in replay.state.items():
            rb=(basis>>_find_index(qc,ref[0]))&1
            branches[rb]=((_reg_value(basis,qc,X),_reg_value(basis,qc,Y)),a)
            if not _reg_zero(basis,qc,W): raise AssertionError(f'shot {shot}: A not zero')
            if not _reg_zero(basis,qc,S): raise AssertionError(f'shot {shot}: S not zero')
            if ((basis>>_find_index(qc,ctrl[0]))&1)!=1: raise AssertionError(f'shot {shot}: ctrl changed')
        got={k:v[0] for k,v in branches.items()}
        expected={0:R0,1:R1}
        if got!=expected: raise AssertionError(f'shot {shot}: got {got}, expected {expected}')
        a0,a1=branches[0][1],branches[1][1]
        if abs(abs(a0)-amp)>1e-10 or abs(abs(a1)-amp)>1e-10:
            raise AssertionError(f'shot {shot}: branch magnitudes {(abs(a0),abs(a1))}')
        ratio=a1/a0
        if abs(ratio-1)>1e-10:
            raise AssertionError(f'shot {shot}: relative phase={ratio}')
        row={'shot':shot,'measurements':len(replay.measurement_record),'outputs':{str(k):list(v) for k,v in got.items()},'relative_phase':[ratio.real,ratio.imag],'A_zero':True,'S_zero':True}
        rows.append(row)
        if args.verbose: print(row,flush=True)

    report={'p':P,'curve':[A_CURVE,B_CURVE],'P2':list(P2),'inputs':[list(P0),list(P1)],'expected_outputs':[list(R0),list(R1)],'num_qubits':qc.num_qubits,'num_clbits':qc.num_clbits,'shots':args.shots,'passed':True,'trajectories':rows}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('p','num_qubits','num_clbits','shots','passed')},indent=2))

if __name__=='__main__':main()
