"""Prove that the expanded harness catches classical, phase, and ancilla bugs."""
import argparse,json
from official_style_qiskit_harness import FlatOp,PhaseBatchSimulator,bit_index,canonical_flat_opstream_bytes,encode_values,flatten_circuit_fail_closed,write_json
from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from test_point_addition_correctness_s835_domainfix_v10 import fig14_schedule_reference

P=13;P2=(7,5);CASES=[(1,(8,5)),(1,(11,5)),(0,(8,5)),(0,(11,5))]*128

def evaluate(qc,ops):
 regs={r.name:r for r in qc.qregs};ctrl=regs['ctrl'];X=regs['X_x1_to_x3'];Y=regs['Y_y1_to_y3'];A=regs['A_shared_work'];S=regs['S_shared_eea_arith'];init={}
 init[bit_index(qc,ctrl[0])]=encode_values([c for c,_ in CASES],1)[0]
 for i,m in enumerate(encode_values([p[0] for _,p in CASES],len(X))):init[bit_index(qc,X[i])]=m
 for i,m in enumerate(encode_values([p[1] for _,p in CASES],len(Y))):init[bit_index(qc,Y[i])]=m
 sim=PhaseBatchSimulator(qc.num_qubits,qc.num_clbits,len(CASES),seed_material=canonical_flat_opstream_bytes(qc.num_qubits,qc.num_clbits,ops),initial_qubit_masks=init);sim.run_ops(ops);gx=sim.register_values(qc,X);gy=sim.register_values(qc,Y);az=sim.register_zero_mask(qc,A);sz=sim.register_zero_mask(qc,S);co=sim.register_values(qc,ctrl)
 classical=ancilla=0
 for i,(c,p1) in enumerate(CASES):
  exp=fig14_schedule_reference(c,*p1,*P2,P);classical+=((gx[i],gy[i])!=(exp[0],exp[1]) or co[i]!=c);ancilla+=(not((az>>i)&1) or not((sz>>i)&1))
 return {'classical_errors':classical,'phase_errors':sim.phase.bit_count(),'ancilla_errors':ancilla}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();qc=build_point_addition_fig14_quadratic(n=4,p=P,x2=P2[0],y2=P2[1]);ops=flatten_circuit_fail_closed(qc);regs={r.name:r for r in qc.qregs};X=regs['X_x1_to_x3'];A=regs['A_shared_work'];mutants={
  'clean':list(ops),
  'classical_output_flip':list(ops)+[FlatOp('x',(bit_index(qc,X[0]),))],
  'phase_leak':list(ops)+[FlatOp('z',(bit_index(qc,X[0]),))],
  'ancilla_leak':list(ops)+[FlatOp('x',(bit_index(qc,A[0]),))],
 }
 mutants['drop_all_phase_corrections']=[op for op in ops if op.kind not in {'z','cz'}]
 rows={name:evaluate(qc,stream) for name,stream in mutants.items()};ok=(rows['clean']=={'classical_errors':0,'phase_errors':0,'ancilla_errors':0} and rows['classical_output_flip']['classical_errors']>0 and rows['phase_leak']['phase_errors']>0 and rows['ancilla_leak']['ancilla_errors']>0 and sum(rows['drop_all_phase_corrections'].values())>0);rep={'shots':len(CASES),'passed':ok,'channels':rows};write_json(a.out,rep);print(json.dumps(rep,indent=2));raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
