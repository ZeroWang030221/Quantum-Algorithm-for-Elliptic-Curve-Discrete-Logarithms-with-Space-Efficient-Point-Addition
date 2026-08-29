"""Exhaustive n=256 terminal-padding regression at exactly 835 qubits.

The uniform schedule has T_max=1620 and every EEA run takes at least 4n=1024
microsteps, so the reachable terminal-padding counts are the multiples of four
from 0 through 596.  This test executes the emitted borrowed-epoch padding
permutation for every nonzero reachable count, canonicalizes Work2, folds the
epoch back into the paper-width l_s register, and verifies clean scratch.  It
also checks full forward/inverse round trips at the low-word wrap boundaries.
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
from qiskit import QuantumCircuit, QuantumRegister

import eea_circuit_s835_fastdual as eea
from extended_semantic_sweep import VecSim, idx
from streamed_algorithm3_sweep import flatten_classical_ops, apply_flat_ops

N=256;T_MAX=1620;SHIFT_WIDTH=9;WORK_SIZE=N+3;AUX_SIZE=20
MAX_PADDING=T_MAX-4*N


def bit(sim,qc,q): return sim.q[idx(qc,q)] & 1

def reg(sim,qc,qs): return sum(bit(sim,qc,q)<<i for i,q in enumerate(qs))

def zero(sim,qc,qs): return all(bit(sim,qc,q)==0 for q in qs)


def make_registers(name):
    terminal=QuantumRegister(1,'terminal');epoch=QuantumRegister(1,'epoch')
    W=QuantumRegister(WORK_SIZE,'Work2');ls=QuantumRegister(SHIFT_WIDTH,'l_s')
    scratch=QuantumRegister(AUX_SIZE-2,'scratch')
    return QuantumCircuit(terminal,epoch,W,ls,scratch,name=name),(terminal,epoch,W,ls,scratch)


def emitted_blocks():
    step,regs=make_registers('TERMINAL_EPOCH_STEP')
    terminal,epoch,W,ls,scratch=regs
    eea._append_terminal_padding_rotation(
        step,terminal=terminal[0],shift_epoch=epoch[0],Work2=W,l_s=ls,
        scratch=scratch,n=N,T_max=T_MAX,
    )

    final,_=make_registers('TERMINAL_EPOCH_CANONICALIZE')
    tr,er,Wr,lsr,scr=final.qregs
    eea.canonical_rotate_work2(final,Wr,lsr,scr,shift_epoch=er[0],outer_ctrl=tr[0])
    eea.compress_terminal_shift_epoch(
        final,shift_epoch=er[0],l_s=lsr,outer_ctrl=tr[0],scratch=scr,
    )

    unfinal,_=make_registers('TERMINAL_EPOCH_DECANONICALIZE')
    tr,er,Wr,lsr,scr=unfinal.qregs
    eea.compress_terminal_shift_epoch(
        unfinal,shift_epoch=er[0],l_s=lsr,outer_ctrl=tr[0],scratch=scr,
    )
    eea.canonical_rotate_work2(
        unfinal,Wr,lsr,scr,shift_epoch=er[0],outer_ctrl=tr[0],inverse=True,
    )
    return regs,flatten_classical_ops(step),flatten_classical_ops(step.inverse()),flatten_classical_ops(final),flatten_classical_ops(unfinal)


def initial_sim(qc,regs):
    terminal,epoch,W,ls,scratch=regs
    init={idx(qc,terminal[0]):1,idx(qc,W[0]):1}
    for q in ls:init[idx(qc,q)]=1
    return VecSim(qc.num_qubits,qc.num_clbits,[0],init)


def state_summary(sim,qc,regs,k):
    terminal,epoch,W,ls,scratch=regs
    marker=[i for i,q in enumerate(W) if bit(sim,qc,q)]
    low_before=(k-1)&((1<<SHIFT_WIDTH)-1)
    physical_epoch_before=1-(((k-1)>>SHIFT_WIDTH)&1)
    expected_low=(low_before^3) if physical_epoch_before==1 else low_before
    return {
        'k':k,'marker':marker,'epoch':bit(sim,qc,epoch[0]),
        'low':reg(sim,qc,ls),'expected_low':expected_low,
        'scratch_zero':zero(sim,qc,scratch),'terminal':bit(sim,qc,terminal[0]),
    }


def main():
    qc,regs=make_registers('TERMINAL_EPOCH_STATE')
    # Only the register identity/order matters to the flattened emitted blocks.
    regs0,step_ops,step_inv_ops,final_ops,unfinal_ops=emitted_blocks()
    assert [len(r) for r in regs]==[len(r) for r in regs0]

    sim=initial_sim(qc,regs)
    checked=[]
    for k in range(1,MAX_PADDING+1):
        apply_flat_ops(sim,step_ops)
        if k%4: continue
        probe=VecSim(qc.num_qubits,qc.num_clbits,[0],{})
        probe.q=list(sim.q);probe.c=list(sim.c)
        apply_flat_ops(probe,final_ops)
        row=state_summary(probe,qc,regs,k)
        row['ok']=(row['marker']==[0] and row['epoch']==0 and row['low']==row['expected_low'] and row['scratch_zero'] and row['terminal']==1)
        checked.append(row)
        if not row['ok']: raise AssertionError(row)

    roundtrips=[]
    for k in (4,508,512,516,596):
        sim=initial_sim(qc,regs)
        for _ in range(k):apply_flat_ops(sim,step_ops)
        apply_flat_ops(sim,final_ops)
        apply_flat_ops(sim,unfinal_ops)
        for _ in range(k):apply_flat_ops(sim,step_inv_ops)
        terminal,epoch,W,ls,scratch=regs
        row={
            'k':k,
            'marker':[i for i,q in enumerate(W) if bit(sim,qc,q)],
            'epoch':bit(sim,qc,epoch[0]),'low':reg(sim,qc,ls),
            'scratch_zero':zero(sim,qc,scratch),'terminal':bit(sim,qc,terminal[0]),
        }
        row['ok']=(row['marker']==[0] and row['epoch']==0 and row['low']==(1<<SHIFT_WIDTH)-1 and row['scratch_zero'] and row['terminal']==1)
        roundtrips.append(row)
        if not row['ok']: raise AssertionError(row)

    report={
        'n':N,'T_max':T_MAX,'max_padding':MAX_PADDING,
        'reachable_padding_multiple':4,'shift_width':SHIFT_WIDTH,'aux_size':AUX_SIZE,
        's_qubits':66,'point_addition_qubits':835,
        'canonicalization_cases':len(checked),'canonicalization_passed':len(checked),
        'roundtrip_cases':roundtrips,'sample_canonicalization_cases':checked[:3]+checked[-3:],
        'passed':True,
    }
    out=Path('validation_835/terminal_shift_epoch_835.json');out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
