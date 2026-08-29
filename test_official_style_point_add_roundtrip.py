"""Phase-clean semantic round trip: add P2, then add -P2.

The two Figure-14 calls are independently built as actual Qiskit circuits,
then their fail-closed lowered operation streams are concatenated with disjoint
classical-bit namespaces.  This avoids relying on a dynamic-circuit inverse
placeholder while testing the natural semantic inverse.
"""


import argparse
import hashlib
import json

from official_style_qiskit_harness import (
    USING_REAL_QISKIT, FlatOp, PhaseBatchSimulator, bit_index,
    canonical_flat_opstream_bytes, encode_values, flat_opstream_fingerprint,
    flatten_circuit_fail_closed, write_json,
)
from point_addition_fig14_s835_fastdual_wrapped_quadratic import build_point_addition_fig14_quadratic
from test_point_addition_correctness_s835_domainfix_v10 import affine_add, enumerate_points


def valid_roundtrip_domain(p, a, b, p2):
    out=[];minus=(p2[0],(-p2[1])%p)
    for p1 in enumerate_points(p,a,b):
        if p1[0]==p2[0]: continue
        r=affine_add(p1,p2,p=p,a=a,b=b)
        if r is None or r[0]==minus[0]: continue
        if (p2[0]-r[0])%p==0: continue
        back=affine_add(r,minus,p=p,a=a,b=b)
        if back!=p1 or (minus[0]-back[0])%p==0: continue
        out.append(p1)
    if not out: raise RuntimeError('empty round-trip generic domain')
    return out


def offset_cargs(ops, delta):
    return [FlatOp(op.kind,op.qargs,tuple(c+delta for c in op.cargs),op.expected) for op in ops]


def derive_targets(fp,domain,count):
    out=[]
    for i in range(count):
        h=hashlib.shake_256();h.update(b'qiskit-ecdsafail-style-pa-roundtrip-v1');h.update(bytes.fromhex(fp));h.update(i.to_bytes(8,'little'))
        out.append(domain[int.from_bytes(h.digest(32),'little')%len(domain)])
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--p',type=lambda s:int(s,0),default=31);ap.add_argument('--a',type=lambda s:int(s,0),default=0);ap.add_argument('--b',type=lambda s:int(s,0),default=7);ap.add_argument('--x2',type=lambda s:int(s,0),default=0);ap.add_argument('--y2',type=lambda s:int(s,0),default=10);ap.add_argument('--shots',type=int,default=9024);ap.add_argument('--include-domain',action='store_true');ap.add_argument('--require-real-qiskit',action='store_true');ap.add_argument('--out',required=True);args=ap.parse_args()
    if args.require_real_qiskit and not USING_REAL_QISKIT: raise RuntimeError('real Qiskit is required but unavailable')
    p,a,b=args.p,args.a,args.b;p2=(args.x2%p,args.y2%p);minus=(p2[0],(-p2[1])%p);domain=valid_roundtrip_domain(p,a,b,p2)
    fwd=build_point_addition_fig14_quadratic(n=p.bit_length(),p=p,x2=p2[0],y2=p2[1]);rev=build_point_addition_fig14_quadratic(n=p.bit_length(),p=p,x2=minus[0],y2=minus[1])
    if fwd.num_qubits!=rev.num_qubits: raise AssertionError('forward/reverse PA widths differ')
    ops1=flatten_circuit_fail_closed(fwd);ops2=offset_cargs(flatten_circuit_fail_closed(rev),fwd.num_clbits);ops=ops1+ops2
    num_clbits=fwd.num_clbits+rev.num_clbits;fp=flat_opstream_fingerprint(fwd.num_qubits,num_clbits,ops);targets=(list(domain) if args.include_domain else [])+derive_targets(fp,domain,args.shots)
    regs={r.name:r for r in fwd.qregs};ctrl=regs['ctrl'];X=regs['X_x1_to_x3'];Y=regs['Y_y1_to_y3'];Areg=regs['A_shared_work'];Sreg=regs['S_shared_eea_arith']
    init={bit_index(fwd,ctrl[0]):(1<<len(targets))-1}
    for i,mask in enumerate(encode_values([x for x,_ in targets],len(X))):init[bit_index(fwd,X[i])]=mask
    for i,mask in enumerate(encode_values([y for _,y in targets],len(Y))):init[bit_index(fwd,Y[i])]=mask
    seed=canonical_flat_opstream_bytes(fwd.num_qubits,num_clbits,ops);sim=PhaseBatchSimulator(fwd.num_qubits,num_clbits,len(targets),seed_material=seed,initial_qubit_masks=init);sim.run_ops(ops)
    gx=sim.register_values(fwd,X);gy=sim.register_values(fwd,Y);co=sim.register_values(fwd,ctrl);az=sim.register_zero_mask(fwd,Areg);sz=sim.register_zero_mask(fwd,Sreg);fail=[]
    for i,t in enumerate(targets):
        ok=(gx[i],gy[i])==t and co[i]==1 and bool((az>>i)&1) and bool((sz>>i)&1)
        if not ok and len(fail)<100:fail.append({'index':i,'input':list(t),'output':[gx[i],gy[i]],'ctrl':co[i],'A_zero':bool((az>>i)&1),'S_zero':bool((sz>>i)&1)})
    phases=[i for i in range(len(targets)) if (sim.phase>>i)&1]
    if phases:fail.append({'error':'phase_garbage','count':len(phases),'first_indices':phases[:32]})
    rep={'qiskit_version':__import__('qiskit').__version__ if USING_REAL_QISKIT else 'mini-data-model','real_qiskit':USING_REAL_QISKIT,'p':p,'curve':[a,b],'P2':list(p2),'roundtrip_domain_size':len(domain),'shots':len(targets),'artifact_fingerprint_sha256':fp,'lowered_ops':len(ops),'measurement_ops':sim.stats.measurement_ops,'phase_errors':len(phases),'passed':not fail,'failures':fail};write_json(args.out,rep);print(json.dumps({k:rep[k] for k in ('p','P2','roundtrip_domain_size','shots','lowered_ops','measurement_ops','phase_errors','passed')},indent=2));raise SystemExit(0 if not fail else 1)
if __name__=='__main__':main()
