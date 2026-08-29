"""Run many reference-entangled p=419 forward/inverse regressions.

Pairs differ in exactly one input bit so a single reference-controlled X gate
prepares the superposition.  Selection covers shortest/longest EEA horizons and
additional artifact-fingerprint-derived pairs.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from test_official_style_eea_harness import build_circuit
from official_style_qiskit_harness import circuit_fingerprint,flatten_circuit_fail_closed

P=419

def steps(x):
 r0,r1=P,min(x,P-x);w=0
 while r1:q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
 return 4*w

def all_pairs():
 out=[]
 for x in range(1,P):
  for b in range(P.bit_length()):
   y=x^(1<<b)
   if x<y<P:out.append((x,y))
 return sorted(set(out))

def select_pairs(count):
 pairs=all_pairs();qc,*_=build_circuit(P,roundtrip=True);fp=circuit_fingerprint(qc,flatten_circuit_fail_closed(qc));selected=[]
 def add(pair):
  if pair not in selected:selected.append(pair)
 add((153,155))
 add(min(pairs,key=lambda z:max(steps(z[0]),steps(z[1]))))
 add(max(pairs,key=lambda z:max(steps(z[0]),steps(z[1]))))
 add(max(pairs,key=lambda z:abs(steps(z[0])-steps(z[1]))))
 i=0
 while len(selected)<min(count,len(pairs)):
  h=hashlib.sha256(bytes.fromhex(fp)+i.to_bytes(8,'little')).digest();add(pairs[int.from_bytes(h,'little')%len(pairs)]);i+=1
 return fp,selected

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',type=int,default=16);ap.add_argument('--shots',type=int,default=3);ap.add_argument('--require-real-qiskit',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();fp,pairs=select_pairs(a.pairs);root=Path(__file__).resolve().parent;logdir=Path(a.out).with_suffix('');logdir.mkdir(parents=True,exist_ok=True);rows=[];t0=time.time()
 for i,(x,y) in enumerate(pairs):
  cmd=[sys.executable,str(root/'test_p419_x153_quantum_semantics.py'),'--x0',str(x),'--x1',str(y),'--shots',str(a.shots)]
  if a.require_real_qiskit:cmd.append('--require-real-qiskit')
  cp=subprocess.run(cmd,cwd=root,text=True,capture_output=True);(logdir/f'pair_{i:02d}_{x}_{y}.log').write_text(cp.stdout+'\n[stderr]\n'+cp.stderr)
  row={'x0':x,'x1':y,'steps0':steps(x),'steps1':steps(y),'returncode':cp.returncode,'passed':cp.returncode==0};rows.append(row);print(f'[{i+1}/{len(pairs)}] ({x},{y}) steps=({row["steps0"]},{row["steps1"]}) pass={row["passed"]}',flush=True)
  if cp.returncode!=0:break
 rep={'artifact_fingerprint_sha256':fp,'pairs_requested':a.pairs,'pairs_completed':len(rows),'shots_per_pair':a.shots,'trajectories_completed':len(rows)*a.shots,'passed':len(rows)==len(pairs) and all(r['passed'] for r in rows),'elapsed_s':time.time()-t0,'pairs':rows};Path(a.out).write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({k:rep[k] for k in ('pairs_completed','shots_per_pair','trajectories_completed','passed','elapsed_s')},indent=2));raise SystemExit(0 if rep['passed'] else 1)
if __name__=='__main__':main()
