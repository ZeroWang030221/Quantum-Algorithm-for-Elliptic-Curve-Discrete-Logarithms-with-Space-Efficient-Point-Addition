"""Exhaust the complete nonzero inversion domain for many small primes."""
import argparse,json,time,gc
from pathlib import Path
from official_style_qiskit_harness import USING_REAL_QISKIT,bit_index
from qiskit import ClassicalRegister,QuantumCircuit,QuantumRegister  # type: ignore
import under1000_eea_shared_s835_fastdual_wrapped as shared
import eea_circuit_s835_fastdual as fast
from extended_semantic_sweep import VecSim

def is_prime(n):
 if n<2:return False
 if n%2==0:return n==2
 d=3
 while d*d<=n:
  if n%d==0:return False
  d+=2
 return True


def clear_project_caches():
 fast.clear_gate_construction_caches()
 for obj in vars(shared).values():
  clear=getattr(obj,'cache_clear',None)
  if callable(clear):clear()
 gc.collect()

def run_p(p):
 n=p.bit_length();cases=list(range(1,p));layout=shared.shared_eea_layout(n,p=p);X=QuantumRegister(n,'X');A=QuantumRegister(n,'A');S=QuantumRegister(layout.s_qubits,'S');c=ClassicalRegister(n,'c');qc=QuantumCircuit(X,A,S,c);qc.append(shared.eea_forward_shared_instruction(n,p,T_max=layout.T_max,lazy_definition=True),[*X,*A,*S],[*c]);init={}
 for i,q in enumerate(X):
  mask=0
  for k,x in enumerate(cases):
   if (x>>i)&1:mask|=1<<k
  init[bit_index(qc,q)]=mask
 sim=VecSim(qc.num_qubits,qc.num_clbits,cases,init);t=time.time();sim.run_circuit(qc,list(range(qc.num_qubits)),list(range(qc.num_clbits)));passed=0;fails=[]
 for k,x in enumerate(cases):
  got=sum(((sim.q[bit_index(qc,q)]>>k)&1)<<i for i,q in enumerate(X));az=all(((sim.q[bit_index(qc,q)]>>k)&1)==0 for q in A);exp=pow(x,-1,p);ok=got==exp and az and (x*got)%p==1;passed+=ok
  if not ok and len(fails)<10:fails.append({'x':x,'got':got,'expected':exp,'A_zero':az})
 return {'p':p,'n':n,'T_max':layout.T_max,'cases':len(cases),'passed':passed,'failed':len(cases)-passed,'elapsed_s':time.time()-t,'composite_calls':sim.composites,'max_depth':sim.max_depth,'failures':fails}

def main():
 import subprocess,sys,tempfile
 ap=argparse.ArgumentParser()
 ap.add_argument('--max-prime',type=int,default=127)
 ap.add_argument('--extra',default='251,257,419,509,1021,2039,4093')
 ap.add_argument('--single-p',type=int,default=None,help=argparse.SUPPRESS)
 ap.add_argument('--jobs',type=int,default=4)
 ap.add_argument('--require-real-qiskit',action='store_true')
 ap.add_argument('--out',required=True)
 a=ap.parse_args()
 if a.require_real_qiskit and not USING_REAL_QISKIT:raise RuntimeError('real Qiskit required but unavailable')
 if a.single_p is not None:
  row=run_p(int(a.single_p))
  rep={'real_qiskit':USING_REAL_QISKIT,'primes_requested':[int(a.single_p)],'primes_completed':1,'total_cases':row['cases'],'total_passed':row['passed'],'all_passed':row['failed']==0,'elapsed_s':row['elapsed_s'],'results':[row]}
  Path(a.out).write_text(json.dumps(rep,indent=2)+'\n')
  print(f'p={a.single_p}: {row["passed"]}/{row["cases"]}',flush=True)
  raise SystemExit(0 if rep['all_passed'] else 1)
 primes=[p for p in range(3,a.max_prime+1) if is_prime(p)]
 for x in a.extra.split(','):
  if x.strip():
   prime=int(x.strip(),0)
   if not is_prime(prime):raise ValueError(f'{prime} is not prime')
   if prime not in primes:primes.append(prime)
 rows=[];t=time.time()
 # Isolate each modulus in a fresh process.  Qiskit circuit factories use large
 # unbounded LRU caches, and keeping all widths/moduli in one process can make
 # later cases much slower even after public cache-clear hooks are called.
 with tempfile.TemporaryDirectory() as td:
  from concurrent.futures import ThreadPoolExecutor,as_completed
  def launch(prime):
   child=Path(td)/f'p{prime}.json'
   cmd=[sys.executable,str(Path(__file__).resolve()),'--single-p',str(prime),'--out',str(child)]
   if a.require_real_qiskit:cmd.append('--require-real-qiskit')
   proc=subprocess.run(cmd,capture_output=True,text=True)
   if proc.returncode!=0:
    raise RuntimeError(f'child p={prime} failed:\nstdout={proc.stdout}\nstderr={proc.stderr}')
   return prime,json.loads(child.read_text())['results'][0]
  got={}
  with ThreadPoolExecutor(max_workers=max(1,int(a.jobs))) as ex:
   futures={ex.submit(launch,prime):prime for prime in primes}
   for fut in as_completed(futures):
    prime,row=fut.result();got[prime]=row
    print(f'[done] p={prime}: {row["passed"]}/{row["cases"]}',flush=True)
  for i,prime in enumerate(primes,1):
   row=got[prime];rows.append(row)
   print(f'[{i}/{len(primes)}] p={prime}: {row["passed"]}/{row["cases"]}',flush=True)
   if row['failed']:break
 rep={'real_qiskit':USING_REAL_QISKIT,'primes_requested':primes,'primes_completed':len(rows),'total_cases':sum(r['cases'] for r in rows),'total_passed':sum(r['passed'] for r in rows),'all_passed':len(rows)==len(primes) and all(r['failed']==0 for r in rows),'elapsed_s':time.time()-t,'results':rows}
 Path(a.out).write_text(json.dumps(rep,indent=2)+'\n')
 print(json.dumps({k:rep[k] for k in ('primes_completed','total_cases','total_passed','all_passed','elapsed_s')},indent=2))
 raise SystemExit(0 if rep['all_passed'] else 1)

if __name__=='__main__':main()
