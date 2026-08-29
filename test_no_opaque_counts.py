import json
from pathlib import Path
try: import qiskit
except Exception:
 import mini_qiskit_runtime as m;m.install_as_qiskit()
import eea_circuit_s835_fastdual as eea
from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from ccx_recursive_block_counter import CounterPolicy,count_circuit_recursive,summarize_counter
policy=CounterPolicy()
step=eea.build_step_circuit(9,1,T_max=60,aux_size=20,measurement_uncompute=True)
pa=build_point_addition_fig14_quadratic(n=4,p=13,x2=3,y2=6)
rep={}
for name,obj in [('eea_step',step),('point_addition_n4',pa)]:
 ops=count_circuit_recursive(obj,policy=policy);s=summarize_counter(ops,policy=policy)
 assert not s['opaque_terms'] and not s['stopped_terms'];rep[name]=s
out = Path('validation/no_opaque_counts.json'); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=2)+'\n'); print(json.dumps(rep, indent=2))
