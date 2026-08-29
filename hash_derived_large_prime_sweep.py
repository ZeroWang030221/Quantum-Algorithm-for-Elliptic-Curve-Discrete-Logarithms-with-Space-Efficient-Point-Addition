#!/usr/bin/env python3
"""Artifact-hash-derived large-prime regressions on emitted Algorithm-3 circuits."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import streamed_algorithm3_sweep as base

def source_hash(root:Path)->str:
 h=hashlib.sha256()
 for name in ['eea_circuit_updated.py','eea_circuit_s835_lowaux.py','eea_circuit_s835_fastdual.py','under1000_eea_shared_s835_fastdual_wrapped.py']:
  p=root/name;h.update(name.encode());h.update(p.read_bytes())
 return h.hexdigest()

def cases_for(p:int,count:int,artifact:str):
 vals={1,2,3,p//2-1,p//2,p//2+1,p-3,p-2,p-1}
 i=0
 while len(vals)<min(p-1,count+9):
  d=hashlib.sha256(f'{artifact}|{p}|{i}'.encode()).digest();vals.add(1+int.from_bytes(d,'big')%(p-1));i+=1
 return sorted(x for x in vals if 1<=x<p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('p',type=lambda s:int(s,0));ap.add_argument('--count',type=int,default=16);ap.add_argument('--out',required=True);ap.add_argument('--require-real-qiskit',action='store_true');a=ap.parse_args()
 root=Path(__file__).resolve().parent;artifact=source_hash(root);cases=cases_for(a.p,a.count,artifact)
 # Reuse the emitted-circuit runner but pass an exact deterministic case list through a temporary monkey patch.
 old=base.choose;base.choose=lambda p,count,seed:list(cases)
 try:
  sys.argv=['streamed_algorithm3_sweep.py',str(a.p),'--count',str(len(cases)),'--seed','0','--out',a.out,'--progress-every','0']+(['--require-real-qiskit'] if a.require_real_qiskit else [])
  base.main()
 finally:base.choose=old
 d=json.loads(Path(a.out).read_text());d['artifact_source_sha256']=artifact;d['case_derivation']='SHA256(source-hash || p || index), plus fixed boundary cases';Path(a.out).write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':main()
