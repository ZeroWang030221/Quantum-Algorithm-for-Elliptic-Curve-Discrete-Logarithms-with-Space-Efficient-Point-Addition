"""Exhaustive terminal-padding/canonicalization matrix across field widths."""
import argparse,json,time
from pathlib import Path
try: import qiskit  # type: ignore
except Exception:
 import mini_qiskit_runtime as m;m.install_as_qiskit()
from qiskit import QuantumCircuit,QuantumRegister
import eea_circuit_s835_fastdual as eea
import under1000_eea_shared_s835_fastdual_wrapped as shared
from extended_semantic_sweep import VecSim,idx
from streamed_algorithm3_sweep import flatten_classical_ops,apply_flat_ops


def make(n,w,aux,name):
 t=QuantumRegister(1,'terminal');e=QuantumRegister(1,'epoch');W=QuantumRegister(n+3,'Work2');ls=QuantumRegister(w,'l_s');scratch=QuantumRegister(aux-2,'scratch');return QuantumCircuit(t,e,W,ls,scratch,name=name),(t,e,W,ls,scratch)

def blocks(n,T,w,aux):
 q,regs=make(n,w,aux,'STEP');t,e,W,ls,s=regs;eea._append_terminal_padding_rotation(q,terminal=t[0],shift_epoch=e[0],Work2=W,l_s=ls,scratch=s,n=n,T_max=T)
 f,_=make(n,w,aux,'FINAL');t2,e2,W2,ls2,s2=f.qregs;eea.canonical_rotate_work2(f,W2,ls2,s2,shift_epoch=e2[0],outer_ctrl=t2[0]);eea.compress_terminal_shift_epoch(f,shift_epoch=e2[0],l_s=ls2,outer_ctrl=t2[0],scratch=s2)
 u,_=make(n,w,aux,'UNFINAL');t3,e3,W3,ls3,s3=u.qregs;eea.compress_terminal_shift_epoch(u,shift_epoch=e3[0],l_s=ls3,outer_ctrl=t3[0],scratch=s3);eea.canonical_rotate_work2(u,W3,ls3,s3,shift_epoch=e3[0],outer_ctrl=t3[0],inverse=True)
 return regs,flatten_classical_ops(q),flatten_classical_ops(q.inverse()),flatten_classical_ops(f),flatten_classical_ops(u)

def initial(qc,regs):
 t,e,W,ls,s=regs;init={idx(qc,t[0]):1,idx(qc,W[0]):1}
 for q in ls:init[idx(qc,q)]=1
 return VecSim(qc.num_qubits,qc.num_clbits,[0],init)
def bit(sim,qc,q):return sim.q[idx(qc,q)]&1
def reg(sim,qc,qs):return sum(bit(sim,qc,q)<<i for i,q in enumerate(qs))
def zero(sim,qc,qs):return all(bit(sim,qc,q)==0 for q in qs)

def run_n(n):
 lay=shared.shared_eea_layout(n);T=lay.T_max;w=lay.shift_width;aux=lay.step_aux;maxp=max(0,T-4*n);qc,regs=make(n,w,aux,'STATE');regs0,step,stepi,final,unfinal=blocks(n,T,w,aux);assert [len(x) for x in regs]==[len(x) for x in regs0]
 sim=initial(qc,regs);checked=0
 for k in range(1,maxp+1):
  apply_flat_ops(sim,step)
  if k%4:continue
  probe=VecSim(qc.num_qubits,qc.num_clbits,[0],{});probe.q=list(sim.q);probe.c=list(sim.c);apply_flat_ops(probe,final);t,e,W,ls,s=regs
  marker=[i for i,q in enumerate(W) if bit(probe,qc,q)];low_before=(k-1)&((1<<w)-1);physical_epoch_before=1-(((k-1)>>w)&1);expected_low=(low_before^3) if physical_epoch_before==1 else low_before
  ok=marker==[0] and bit(probe,qc,e[0])==0 and reg(probe,qc,ls)==expected_low and zero(probe,qc,s) and bit(probe,qc,t[0])==1
  if not ok:raise AssertionError({'n':n,'k':k,'marker':marker,'epoch':bit(probe,qc,e[0]),'low':reg(probe,qc,ls),'expected_low':expected_low,'scratch':zero(probe,qc,s)})
  checked+=1
 candidates={4,maxp};cap=1<<w
 for x in (cap-4,cap,cap+4):
  if 4<=x<=maxp and x%4==0:candidates.add(x)
 rounds=[]
 for k in sorted(x for x in candidates if 0<x<=maxp):
  sim=initial(qc,regs)
  for _ in range(k):apply_flat_ops(sim,step)
  apply_flat_ops(sim,final);apply_flat_ops(sim,unfinal)
  for _ in range(k):apply_flat_ops(sim,stepi)
  t,e,W,ls,s=regs;ok=[i for i,q in enumerate(W) if bit(sim,qc,q)]==[0] and bit(sim,qc,e[0])==0 and reg(sim,qc,ls)==(1<<w)-1 and zero(sim,qc,s) and bit(sim,qc,t[0])==1
  rounds.append({'k':k,'ok':ok})
  if not ok:raise AssertionError({'n':n,'roundtrip':k})
 return {'n':n,'T_max':T,'max_padding':maxp,'shift_width':w,'aux_size':aux,'capacity_with_epoch':1<<(w+1),'canonicalization_cases':checked,'roundtrips':rounds,'passed':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--widths',default='4,5,6,7,8,9,16,32,64,128,160,192,224,256,384,512');ap.add_argument('--out',required=True);a=ap.parse_args();rows=[];t=time.time()
 for n in [int(x) for x in a.widths.split(',') if x.strip()]:
  row=run_n(n);rows.append(row);print(f'n={n} T={row["T_max"]} padding={row["max_padding"]} w={row["shift_width"]} cases={row["canonicalization_cases"]} PASS',flush=True)
 rep={'widths':len(rows),'canonicalization_cases':sum(r['canonicalization_cases'] for r in rows),'roundtrip_cases':sum(len(r['roundtrips']) for r in rows),'passed':all(r['passed'] for r in rows),'elapsed_s':time.time()-t,'results':rows};Path(a.out).write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({k:rep[k] for k in ('widths','canonicalization_cases','roundtrip_cases','passed','elapsed_s')},indent=2))
if __name__=='__main__':main()
