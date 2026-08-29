"""Execute the published 1,536-microstep secp256k1 schedule witness."""
import argparse,json,sys
from pathlib import Path
import streamed_algorithm3_sweep as base

P=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
X=0x5db3d742c265539d92ba16b83c5c1dc492ec1a6629ed23cc63905323d950963b
EXPECTED_STEPS=1536

def exact_steps(p,x):
 r0,r1=p,min(x,p-x);w=0
 while r1:q,r2=divmod(r0,r1);w+=q.bit_length();r0,r1=r1,r2
 return 4*w

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--require-real-qiskit',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();got=exact_steps(P,X)
 if got!=EXPECTED_STEPS:raise AssertionError(f'witness step count {got}, expected {EXPECTED_STEPS}')
 old=base.choose;base.choose=lambda p,count,seed:[X]
 try:
  sys.argv=['streamed_algorithm3_sweep.py',hex(P),'--count','1','--seed','0','--out',a.out,'--progress-every','200']+(['--require-real-qiskit'] if a.require_real_qiskit else [])
  base.main()
 except SystemExit as e:
  if int(e.code or 0)!=0:raise
 finally:base.choose=old
 path=Path(a.out);report=json.loads(path.read_text());report['witness_hex']=hex(X);report['certified_exact_steps']=got;report['uniform_margin']=report['T_max']-got;path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'T_max':report['T_max'],'witness_steps':got,'margin':report['uniform_margin'],'passed':report['passed']},indent=2))
if __name__=='__main__':main()
