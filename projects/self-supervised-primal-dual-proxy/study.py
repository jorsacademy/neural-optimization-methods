"""Label-free simplex-QP primal/dual learning with independent KKT audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def data(seed,n=256,units=4,shift=False):
    g=np.random.default_rng(seed)
    a=g.uniform(.5,2,(n,units));b=g.uniform(0,1,(n,units));d=g.uniform(.5,3,n)
    if shift:b*=3;d*=1.5
    return np.column_stack([a,b,d])


def validate(Z):
    Z=np.asarray(Z,float)
    if (Z.ndim!=2 or len(Z)<1 or Z.shape[1]<3 or Z.shape[1]%2!=1
        or not np.isfinite(Z).all()):raise ValueError('Invalid parameter matrix')
    n=(Z.shape[1]-1)//2
    if np.any(Z[:,:n]<=0) or np.any(Z[:,-1]<=0):raise ValueError('Positive curvature and demand required')
    return Z,n


def oracle(Z):
    Z,n=validate(Z);a,b,d=Z[:,:n],Z[:,n:2*n],Z[:,-1]
    lo=-b.max(axis=1)-a.max(axis=1)*d;hi=-b.min(axis=1)
    for _ in range(80):
        nu=(lo+hi)/2;x=np.maximum(-(b+nu[:,None])/a,0)
        large=x.sum(axis=1)>d;lo=np.where(large,nu,lo);hi=np.where(large,hi,nu)
    nu=(lo+hi)/2;return np.maximum(-(b+nu[:,None])/a,0),nu


def objectives(Z,x,nu):
    n=(Z.shape[1]-1)//2;a,b,d=Z[:,:n],Z[:,n:2*n],Z[:,-1]
    primal=(.5*a*x*x+b*x).sum(dim=1)
    dual=-nu*d-.5*(torch.relu(-(b+nu[:,None]))**2/a).sum(dim=1)
    return primal,dual


class Proxy(nn.Module):
    def __init__(self,units):
        super().__init__();self.units=units
        self.primal=nn.Sequential(nn.Linear(2*units+1,32),nn.Tanh(),nn.Linear(32,units))
        self.dual=nn.Sequential(nn.Linear(2*units+1,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,Z):
        x=Z[:,-1,None]*torch.softmax(self.primal(Z),dim=1)
        return x,self.dual(Z).squeeze(1)


def train(Z,mode='self_supervised',seed=0,epochs=180):
    Z,n=validate(Z)
    if mode not in ('self_supervised','supervised') or epochs<1:raise ValueError('Invalid training mode')
    torch.manual_seed(seed);m=Proxy(n);o=torch.optim.Adam(m.parameters(),lr=.01)
    t=torch.tensor(Z,dtype=torch.float32);target=None
    if mode=='supervised':target=torch.tensor(oracle(Z)[0],dtype=torch.float32)
    history=[]
    for _ in range(epochs):
        x,nu=m(t);f,g=objectives(t,x,nu)
        # Separable primal and dual objectives; not a reproduction of conic DLL.
        loss=(f-g).mean() if target is None else ((x-target)**2).mean()-g.mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        o.zero_grad();loss.backward();o.step();history.append(float(loss.detach()))
    m.eval();return m,history


def evaluate(m,Z):
    Z,_=validate(Z);t=torch.tensor(Z,dtype=torch.float64)
    with torch.no_grad():x,nu=m(torch.tensor(Z,dtype=torch.float32))
    x=x.double();nu=nu.double();f,g=objectives(t,x,nu)
    optimal,_=oracle(Z);fo,_=objectives(t,torch.tensor(optimal),nu)
    actual=(f-fo).numpy();certificate=(f-g).numpy();violation=np.maximum(g.numpy()-fo.numpy(),0)
    residual=np.max(np.abs(x.numpy().sum(axis=1)-Z[:,-1]))
    if violation.max()>1e-6 or np.any(actual>certificate+1e-6) or residual>1e-5:
        raise RuntimeError('Primal-dual audit failure')
    # Certify only numerically feasible outputs; epsilon is a declared tolerance.
    return {'mean_actual_gap':float(actual.mean()),'mean_certificate':float(certificate.mean()),
            'max_balance_residual':float(residual),'max_dual_violation':float(violation.max()),
            'acceptance_at_0_05':float(np.mean(certificate<=.05)),
            'mean_cost':float(f.mean())}


def benchmark():
    torch.set_num_threads(1);rows=[]
    for seed in (31,32,33):
        Z=data(seed)
        for mode in ('self_supervised','supervised'):
            m,h=train(Z,mode,seed)
            rows.append({'seed':seed,'method':mode,'training_labels':0 if mode=='self_supervised' else len(Z),
                         'first_loss':h[0],'last_loss':h[-1],
                         'nominal':evaluate(m,data(901,150)),
                         'shifted':evaluate(m,data(902,150,shift=True))})
    return {'scope':'nonnegative simplex convex QP; no upper bounds; numerical certificates',
            'labels_used_only_for_supervised_baseline_and_evaluation':True,'results':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json')
    args=p.parse_args();Path(args.output).write_text(json.dumps(benchmark(),indent=2))
