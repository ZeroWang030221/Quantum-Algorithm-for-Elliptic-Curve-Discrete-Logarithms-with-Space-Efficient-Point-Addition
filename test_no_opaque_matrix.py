"""Fail-closed recursive-decomposition audit over a width/step matrix."""
import argparse,json,time
from pathlib import Path
try: import qiskit  # type: ignore
except Exception:
 import mini_qiskit_runtime as m;m.install_as_qiskit()
import eea_circuit_s835_fastdual as eea
import under1000_eea_shared_s835_fastdual_wrapped as shared
from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from ccx_recursive_block_counter import CounterPolicy,count_circuit_recursive,summarize_counter


def audit(name,obj,policy,extra):
 ops=count_circuit_recursive(obj,policy=policy);summary=summarize_counter(ops,policy=policy)
 if summary['opaque_terms'] or summary['stopped_terms']:
  raise AssertionError({'name':name,'opaque':summary['opaque_terms'],'stopped':summary['stopped_terms']})
 return {'name':name,**extra,'x':summary['x'],'cx':summary['cx'],'ccx':summary['ccx'],'total_counted_terms':summary['total_counted_terms'],'opaque_terms':summary['opaque_terms'],'stopped_terms':summary['stopped_terms'],'passed':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();widths=[4,5,9,16,32] if not a.full else [4,5,9,16,32,64,128,160,192,224,256,384,512];rows=[];t0=time.time();policy=CounterPolicy()
 for n in widths:
  lay=shared.shared_eea_layout(n);Ts=sorted(set([1,2,3,4,max(1,lay.T_max//4),max(1,lay.T_max//2),max(1,3*lay.T_max//4),lay.T_max-1,lay.T_max]));
  if not a.full and n>=16:Ts=Ts[:4]+Ts[-2:]
  for T in Ts:
   c=eea.build_step_circuit(n,T,T_max=lay.T_max,aux_size=lay.step_aux,measurement_uncompute=True);row=audit(f'eea_n{n}_T{T}',c,policy,{'kind':'eea_step','n':n,'T':T,'T_max':lay.T_max,'qubits':c.num_qubits,'clbits':c.num_clbits});rows.append(row);print(f'EEA n={n} T={T}: ccx={row["ccx"]} PASS',flush=True)
 curves=[(13,7,5),(17,0,6),(31,0,10)]
 if a.full:curves += [(61,0,14)]
 for p,x2,y2 in curves:
  c=build_point_addition_fig14_quadratic(n=p.bit_length(),p=p,x2=x2,y2=y2);row=audit(f'pa_p{p}',c,policy,{'kind':'point_addition','p':p,'n':p.bit_length(),'qubits':c.num_qubits,'clbits':c.num_clbits});rows.append(row);print(f'PA p={p}: ccx={row["ccx"]} PASS',flush=True)
 root=Path(__file__).resolve().parent;legacy=[]
 for path in root.glob('*.py'):
  if path.name.startswith('test_') or path.name=='mini_qiskit_runtime.py':continue
  txt=path.read_text(errors='ignore')
  if '.condition =' in txt or '.condition=' in txt:legacy.append(path.name)
 rep={'full':a.full,'objects':len(rows),'all_lowered':all(r['passed'] for r in rows),'legacy_condition_assignments':legacy,'passed':all(r['passed'] for r in rows) and not legacy,'elapsed_s':time.time()-t0,'results':rows};Path(a.out).write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({k:rep[k] for k in ('objects','all_lowered','legacy_condition_assignments','passed','elapsed_s')},indent=2));raise SystemExit(0 if rep['passed'] else 1)
if __name__=='__main__':main()
