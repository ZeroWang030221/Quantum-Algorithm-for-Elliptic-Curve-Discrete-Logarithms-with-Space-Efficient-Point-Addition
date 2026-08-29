import sys
from pathlib import Path
try:
 import qiskit
except Exception:
 import mini_qiskit_runtime as _mini; _mini.install_as_qiskit()
from qiskit import QuantumCircuit
import eea_circuit_s835_fastdual as f
from extended_semantic_sweep import VecSim, idx


def exact_steps(p,x):
 r0,r1=p,min(x,p-x);w=0
 while r1:q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
 return 4*w

def inv_step(n,T,Tmax,aux,measurement=False):
 cfg=f.get_n_config(n); lw=int(cfg['len_width']); sw=max(int(cfg['shift_width']),int(f.terminal_safe_shift_width(n,Tmax)))
 f.set_measurement_uncompute(measurement)
 regs=f.make_global_registers_noctrl(n=n,len_width=lw,shift_width=sw,T_max=Tmax,aux_size=aux)
 qc=QuantumCircuit(*regs,name=f'INV_T{T}')
 P1,P2,It,Sg,W1,W2,lt,lq,ls,lrp,A=regs
 f.append_one_step_T_inverse(qc,T=T,n=n,len_width=lw,shift_width=sw,Phase1=P1,Phase2=P2,Iter=It,Sign=Sg,Work1=W1,Work2=W2,l_t=lt,l_q=lq,l_s=ls,l_rp=lrp,Aux=A,T_max=Tmax)
 return qc

def main(p=37):
 n=p.bit_length();cfg=f.get_n_config(n);Tmax=int(cfg['T_max']);lw=int(cfg['len_width']);sw=max(int(cfg['shift_width']),int(f.terminal_safe_shift_width(n,Tmax)));aux=f.qiskit_paper_aux_size(n,lw,sw,Tmax)
 cases=list(range(1,p)); first=f.build_step_circuit(n,1,T_max=Tmax,aux_size=aux,measurement_uncompute=False);by={r.name:r for r in first.qregs}; sim=VecSim(first.num_qubits,first.num_clbits,cases,{})
 allmask=(1<<len(cases))-1
 def setc(q,v):sim.q[idx(first,q)]=allmask if v else 0
 setc(by['Work1'][0],1)
 for j in range(n):setc(by['Work1'][3+j],(p>>(n-1-j))&1)
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
 for T in range(1,Tmax+1):
  step=f.build_step_circuit(n,T,T_max=Tmax,aux_size=aux,measurement_uncompute=False)
  before=list(sim.q)
  sim.run_circuit(step,list(range(step.num_qubits)),list(range(step.num_clbits)))
  after=list(sim.q)
  inv=inv_step(n,T,Tmax,aux,measurement=False)
  sim.run_circuit(inv,list(range(inv.num_qubits)),list(range(inv.num_clbits)))
  if sim.q!=before:
   dif=[i for i,(a,b) in enumerate(zip(sim.q,before)) if a!=b]
   print('FAIL T',T,'qdiff',dif[:20]);return 1
  sim.q=after
  if T%4==0: print('pass',T)
 print('ALL PASS',p,Tmax);return 0
if __name__=='__main__':raise SystemExit(main(int(sys.argv[1]) if len(sys.argv)>1 else 37))
