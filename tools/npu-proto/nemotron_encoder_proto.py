#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Full Nemotron FastConformer encoder (24 blocks + projectors) as ONNX, validated
vs pooler.f32 golden — the final math gate before the Rust port."""
import numpy as np, os
import onnx
from onnx import helper, TensorProto as TP
import openvino as ov
from safetensors.numpy import load_file

D = "testdata/asr/nemotron/hf"; GD = "testdata/asr/golden/nemotron"
W = load_file(os.path.join(D, "model.safetensors"))
def wt(n): return np.array(W[n])

C, HEADS, HD, FFN, NLAYERS = 1024, 8, 128, 4096, 24
NP_PROMPTS, PROMPT_INT, DEC_H = 128, 2048, 640
sub = np.fromfile(f"{GD}/subsampling.f32", np.float32).reshape(-1, C)
T = sub.shape[0]; VALID = 74; LEFT, RIGHT = 56, 3; L = 2*T-1
SCALE = 1.0/np.sqrt(HD); EPS = 1e-5; PROMPT_ID = 0

nodes, inits = [], []; _seen=set()
def I(name, arr, itype=TP.FLOAT):
    if name in _seen: return
    _seen.add(name)
    inits.append(helper.make_tensor(name, itype, list(arr.shape), arr.flatten().tolist()))
def N(op, ins, outs, **kw): nodes.append(helper.make_node(op, ins, outs, **kw))
cnt=[0]
def tmp(t="t"): cnt[0]+=1; return f"{t}{cnt[0]}"
def ln(x, prefix):
    I(prefix+".g", wt(prefix+".weight")); I(prefix+".b", wt(prefix+".bias"))
    m=tmp(); N("ReduceMean",[x],[m],axes=[-1],keepdims=1); xm=tmp(); N("Sub",[x,m],[xm])
    sq=tmp(); N("Mul",[xm,xm],[sq]); v=tmp(); N("ReduceMean",[sq],[v],axes=[-1],keepdims=1)
    I("eps",np.array([EPS],np.float32)); ve=tmp(); N("Add",[v,"eps"],[ve]); st=tmp(); N("Sqrt",[ve],[st])
    nr=tmp(); N("Div",[xm,st],[nr]); s=tmp(); N("Mul",[nr,prefix+".g"],[s]); o=tmp(); N("Add",[s,prefix+".b"],[o]); return o
def lin(x, wname, out=None):
    I(wname+".T", wt(wname).T.copy()); o=out or tmp(); N("MatMul",[x,wname+".T"],[o]); return o
def resh(x, shape):
    sn=tmp("sh"); I(sn,np.array(shape,np.int64),TP.INT64); o=tmp(); N("Reshape",[x,sn],[o]); return o
def ff(x, p):
    h=lin(x,p+".linear1.weight"); sg=tmp(); N("Sigmoid",[h],[sg]); si=tmp(); N("Mul",[h,sg],[si]); return lin(si,p+".linear2.weight")
def rel_pos_enc():
    half=C//2; inv=np.power(10000.0,-2.0*np.arange(half)/C).astype(np.float32); pe=np.zeros((L,C),np.float32)
    for idx in range(L):
        pos=float(T-1-idx); pe[idx,0::2]=np.sin(pos*inv); pe[idx,1::2]=np.cos(pos*inv)
    return pe
# shared consts
I("pe", rel_pos_enc()); I("scale", np.array([SCALE],np.float32)); I("half", np.array([0.5],np.float32))
mask=np.zeros((T,T),np.float32)
for i in range(T):
    for j in range(T):
        qc,kc=i//(RIGHT+1),j//(RIGHT+1)
        ok=(j<VALID) and (qc>=kc) and (qc-kc<=LEFT//(RIGHT+1)); mask[i,j]=0.0 if ok else -1e9
I("attmask", mask)
I("pad", np.array([0,0,1,0,0,0],np.int64),TP.INT64)
I("s_start",np.array([T],np.int64),TP.INT64); I("s_end",np.array([T*(L+1)],np.int64),TP.INT64); I("s_ax",np.array([1],np.int64),TP.INT64)
I("bd_s",np.array([0],np.int64),TP.INT64); I("bd_e",np.array([T],np.int64),TP.INT64); I("bd_a",np.array([2],np.int64),TP.INT64)
gmask=np.array([1.0 if i<VALID else 0.0 for i in range(T)],np.float32).reshape(T,1); I("gmask",gmask)
I("gl_s",np.array([0],np.int64),TP.INT64); I("gl_m",np.array([C],np.int64),TP.INT64); I("gl_e",np.array([2*C],np.int64),TP.INT64); I("gl_ax",np.array([1],np.int64),TP.INT64)

def heads(t, nT):
    r=resh(t,[-1,HEADS,HD]); o=tmp(); N("Transpose",[r],[o],perm=[1,0,2]); return o
def attention(x, p):
    q=lin(x,p+".q_proj.weight"); k=lin(x,p+".k_proj.weight"); v=lin(x,p+".v_proj.weight")
    rel_k=lin("pe",p+".relative_k_proj.weight")
    qh,kh,vh,rkh=heads(q,T),heads(k,T),heads(v,T),heads(rel_k,L)
    I(p+".bu", wt(p+".bias_u").reshape(HEADS,1,HD)); I(p+".bv", wt(p+".bias_v").reshape(HEADS,1,HD))
    qbv=tmp(); N("Add",[qh,p+".bv"],[qbv])
    rkT=tmp(); N("Transpose",[rkh],[rkT],perm=[0,2,1]); bdraw=tmp(); N("MatMul",[qbv,rkT],[bdraw])
    padded=tmp(); N("Pad",[bdraw,"pad"],[padded]); flat=resh(padded,[HEADS,T*(L+1)])
    sl=tmp(); N("Slice",[flat,"s_start","s_end","s_ax"],[sl]); bd=resh(sl,[HEADS,T,L])
    bdT=tmp(); N("Slice",[bd,"bd_s","bd_e","bd_a"],[bdT])
    qbu=tmp(); N("Add",[qh,p+".bu"],[qbu]); kT=tmp(); N("Transpose",[kh],[kT],perm=[0,2,1]); ac=tmp(); N("MatMul",[qbu,kT],[ac])
    sc=tmp(); N("Add",[ac,bdT],[sc]); ssc=tmp(); N("Mul",[sc,"scale"],[ssc]); mk=tmp(); N("Add",[ssc,"attmask"],[mk])
    pr=tmp(); N("Softmax",[mk],[pr],axis=-1); ctx=tmp(); N("MatMul",[pr,vh],[ctx])
    ct=tmp(); N("Transpose",[ctx],[ct],perm=[1,0,2]); cr=resh(ct,[T,C]); return lin(cr,p+".o_proj.weight")
def conv_mod(x, p):
    I(p+".pc1T", wt(p+".pointwise_conv1.weight").reshape(2*C,C).T.copy()); pc1=tmp(); N("MatMul",[x,p+".pc1T"],[pc1])
    a=tmp(); N("Slice",[pc1,"gl_s","gl_m","gl_ax"],[a]); b=tmp(); N("Slice",[pc1,"gl_m","gl_e","gl_ax"],[b])
    bs=tmp(); N("Sigmoid",[b],[bs]); glu=tmp(); N("Mul",[a,bs],[glu]); glm=tmp(); N("Mul",[glu,"gmask"],[glm])
    gct=tmp(); N("Transpose",[glm],[gct],perm=[1,0]); glc=resh(gct,[1,C,T])
    I(p+".dww", wt(p+".depthwise_conv.weight")); dwc=tmp(); N("Conv",[glc,p+".dww"],[dwc],kernel_shape=[9],strides=[1],pads=[8,0],group=C)
    dwr=resh(dwc,[C,T]); dwt=tmp(); N("Transpose",[dwr],[dwt],perm=[1,0]); cn=ln(dwt,p+".norm")
    cs=tmp(); N("Sigmoid",[cn],[cs]); ci=tmp(); N("Mul",[cn,cs],[ci])
    I(p+".pc2T", wt(p+".pointwise_conv2.weight").reshape(C,C).T.copy()); o=tmp(); N("MatMul",[ci,p+".pc2T"],[o]); return o
def block(h, layer):
    p=f"encoder.layers.{layer}"
    f1=ff(ln(h,p+".norm_feed_forward1"),p+".feed_forward1"); f1h=tmp(); N("Mul",[f1,"half"],[f1h]); h1=tmp(); N("Add",[h,f1h],[h1])
    att=attention(ln(h1,p+".norm_self_att"),p+".self_attn"); h2=tmp(); N("Add",[h1,att],[h2])
    cv=conv_mod(ln(h2,p+".norm_conv"),p+".conv"); h3=tmp(); N("Add",[h2,cv],[h3])
    f2=ff(ln(h3,p+".norm_feed_forward2"),p+".feed_forward2"); f2h=tmp(); N("Mul",[f2,"half"],[f2h]); h4=tmp(); N("Add",[h3,f2h],[h4])
    return ln(h4,p+".norm_out")

h="sub_in"
for l in range(NLAYERS): h=block(h,l)
# projectors: cat(h, onehot(prompt,128)) -> lin1 relu -> lin2 -> encoder_projector
onehot=np.zeros((T,NP_PROMPTS),np.float32); onehot[:,PROMPT_ID]=1.0; I("onehot",onehot)
cat=tmp(); N("Concat",[h,"onehot"],[cat],axis=1)   # [T, C+128]
I("pp1T", wt("prompt_projector.linear_1.weight").T.copy()); I("pp1b", wt("prompt_projector.linear_1.bias"))
mm1=tmp(); N("MatMul",[cat,"pp1T"],[mm1]); a1=tmp(); N("Add",[mm1,"pp1b"],[a1]); r1=tmp(); N("Relu",[a1],[r1])
I("pp2T", wt("prompt_projector.linear_2.weight").T.copy()); I("pp2b", wt("prompt_projector.linear_2.bias"))
mm2=tmp(); N("MatMul",[r1,"pp2T"],[mm2]); fused=tmp(); N("Add",[mm2,"pp2b"],[fused])
I("epT", wt("encoder_projector.weight").T.copy()); I("epb", wt("encoder_projector.bias"))
mm3=tmp(); N("MatMul",[fused,"epT"],[mm3]); pooler=tmp(); N("Add",[mm3,"epb"],[pooler])

g=helper.make_graph(nodes,"encoder",[helper.make_tensor_value_info("sub_in",TP.FLOAT,[T,C])],
                    [helper.make_tensor_value_info(pooler,TP.FLOAT,[T,DEC_H])],inits)
m=helper.make_model(g,opset_imports=[helper.make_opsetid("",13)]); m.ir_version=8
onnx.checker.check_model(m)
core=ov.Core(); cm=core.compile_model(core.read_model(m.SerializeToString(),b""),"CPU")
res=cm(sub.astype(np.float32))[cm.output(0)]
ref=np.fromfile(f"{GD}/pooler.f32",np.float32).reshape(-1,DEC_H)
print("pooler out",res.shape,"ref",ref.shape)
print("valid-frame maxdiff:", np.abs(res[:VALID]-ref[:VALID]).max())
print("res[0,:4]",res[0,:4]); print("ref[0,:4]",ref[0,:4])
