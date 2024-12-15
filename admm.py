import numpy as np
import scipy.sparse as sp
import torch


class admm():
    def __init__(self,shape,As,projector,gamma,device,B=None,metric=None,verificators=None,solver='factorize'):
        """Implements a projection over a set of constraints and solve it with ADMM

        Arguments:
            shape (tuple): shape of the variable to project, it should be of
            format (L, N, D) with L the batch size, N the number of points and
            D the dimension
        """
        self.dtype = torch.float64
        self.device = device
        self.metric=sp.eye(shape[1]) if metric is None else metric
        self.metric = torch.tensor(self.metric.toarray(), device=self.device, dtype=self.dtype)
        self.verificators=len(As)*[None] if verificators is None else verificators
        if solver!='factorize': raise Exception("The only solver available for the moment is factorize")
        self.solver=solver
        self.check_init(shape,As,projector,gamma,B,self.metric,self.verificators,self.solver)
        self.Alist=As
        for i in range(len(self.Alist)):
            self.Alist[i] = torch.tensor(self.Alist[i].toarray(), device=self.device).type(self.dtype)
        self.gamma=gamma
        self.ATlist=[A.T for A in self.Alist]
        M=self.metric.clone()
        for (A,AT,g) in zip(self.Alist,self.ATlist,self.gamma):
            t = torch.mm(AT,A)
            M=M+g*g*t
        self.M= M if B==None else bmat([[M,B.T],[B,None]])
        self.Minv = torch.linalg.inv(self.M)
        self.projector=projector
        self.shape_x=(shape[0],shape[1],shape[2])
        self.indices_y=[]
        begin=0
        for  A in self.Alist:
            self.indices_y.append((begin,begin+A.shape[0]))
            begin+=A.shape[0]
        self.shape_y=(self.shape_x[0],begin,self.shape_x[2])

    def gA(self,x):
        y=torch.zeros(self.shape_y, dtype=self.dtype, device=self.device)
        for (ind,A,g) in zip(self.indices_y,self.Alist,self.gamma):
            y[:,ind[0]:ind[1]] = g*torch.matmul(A,x)
        return y

    def gAT(self,y):
        x=torch.zeros(self.shape_x, dtype=self.dtype, device=self.device)
        for (ind,AT,g) in zip(self.indices_y,self.ATlist,self.gamma):
            x += g*torch.matmul(AT,y[:,ind[0]:ind[1]])
        return x

    def solveM(self,z,b):
        x=torch.zeros(self.shape_x, dtype=self.dtype, device=self.device)
        mu=None if b is None else torch.zeros_like(b)
        t = self.Minv.matmul(z)
        x = t[:, :self.shape_x[1]]
        if b is not None:
            mu = t[:,self.shape_x[1]:]
        return (x,mu)

    def project(self,z,list_bnd):
        y=torch.zeros(self.shape_y, dtype=self.dtype, device=self.device)
        for (ind,projector,g,bnd) in zip(self.indices_y,self.projector,self.gamma,list_bnd):
            y[:,ind[0]:ind[1],:]=projector(z[:,ind[0]:ind[1],:],g,bnd)
        return y

    def verif(self,z):
        result=[]
        for (ind,verif,g) in zip(self.indices_y,self.verificators,self.gamma):
            if verif is not None:
                result.append(verif(z[:,ind[0]:ind[1]],g))
            else:
                result.append(None)
        return result


    def solve(self,z,list_bound,b=None,nitermax=100,xInit=None,lagr=None,tol=1.e-6,verbose=0,verificators=None, fixedNbStep=-1):
        list_bound = [torch.tensor(bnd, dtype=self.dtype, device=self.device) if isinstance(bnd, np.ndarray) else bnd for bnd in list_bound]
        niter=0
        lagrange=torch.zeros(self.shape_y, dtype=self.dtype, device=self.device) if lagr is None else lagr
        x=torch.zeros(z.shape, dtype=self.dtype, device=self.device) if xInit is None else xInit.clone()
        self.check_solve(z,list_bound,b,xInit,lagrange)
        tmp0=self.gA(z)
        tmp=self.project(tmp0.clone(),list_bound)

        if fixedNbStep<0 and torch.norm(tmp-tmp0) <tol*torch.norm(tmp0):
            x=z.clone()
            tab = self.verif(tmp0)
            tab=[t/l*100 for (t,l) in zip(tab,list_bound)]
            if verbose > 1:
                print('Immediate convergency |z-proj(z)| : %1.2e  |z| : %1.2e, bnds=[%2.1f%% (%1.2e),%2.1f%%  (%1.2e)] '%(np.linalg.norm((tmp-tmp0).detach().numpy()),np.linalg.norm(tmp0.detach().numpy()),tab[0],tab[1],list_bound[0],list_bound[1]))
            return z,(True,0,torch.zeros(self.shape_y, device=self.device, dtype=self.dtype),tab)
        if verbose > 2:
            tab = self.verif(self.gA(x))
            tab=[t/l*100 for (t,l) in zip(tab,list_bound)]
            print(' niter : %4i  distance : %1.2e --- evol(L) :%1.2e ---- evol(x) :%1.2e bnds=[%2.1f%% (%1.2e),%2.1f%%  (%1.2e)]'%(niter, np.linalg.norm((x-z).detach().numpy()),0.,0.,tab[0],list_bound[0],tab[1],list_bound[1]))
        while True:
                niter+=1
                #update y
                y=self.project(self.gA(x) +lagrange,list_bound)
                #update x
                xOld=x.clone()
                tmp = torch.zeros_like(z)
                for i in range(tmp.shape[0]):
                    tmp[i] = torch.mm(self.metric, z[i])
                x,mu=self.solveM(self.gAT(y - lagrange)+tmp,b)

                #update lagrange
                lOld=lagrange.clone()
                lagrange+=self.gA(x)-y
                # calcul des criteres de convergence

                erreurl=torch.norm(lagrange-lOld)/(torch.norm(lagrange)+1.e-12)
                erreurx=torch.norm(x-xOld)/(torch.norm(x)+1.e-12)
                tab = self.verif(self.gA(x))
                tab=[t/l*100 for (t,l) in zip(tab,list_bound)]
                if verbose > 3:
                    print(' niter : %4i  distance : %1.2e --- evol(L) :%1.2e ---- evol(x) :%1.2e bnds=[%2.1f%% (%1.2e),%2.1f%%  (%1.2e)]'%(niter, np.linalg.norm((x-z).detach().numpy()),erreurl,erreurx,tab[0],list_bound[0],tab[1],list_bound[1]))
                if verbose >2:
                    if (niter % (nitermax//10)) == 0:
                        if verbose ==2:
                            print(' niter : %4i  distance : %1.2e --- evol(L) :%1.2e ---- evol(x) :%1.2e bnds=[%2.1f%% (%1.2e),%2.1f%%  (%1.2e)]'%(niter, np.linalg.norm((x-z).detach().numpy()),erreurl,erreurx,tab[0],list_bound[0],tab[1],list_bound[1]))
                if niter>nitermax+1:
                    if verbose > 1:
                        print((' niter : %4i MAXITER REACHED distance : %1.2e --- Lagrange :%1.2e ---- x :%1.2e [%1.2e,%1.2e]'%(niter, np.linalg.norm((x-z).detach().numpy()),erreurl,erreurx,tab[0],tab[1])).center(128,'#'))
                    return x,(False,niter,erreurl,erreurx,lagrange,tab)
                if (fixedNbStep<0 and (erreurx<tol) and (erreurl<tol)) or (fixedNbStep>=0 and niter>fixedNbStep):
                    if verbose > 1:
                        print((' niter : %4i CONVERGENCE --- distance : %1.2e --- Lagrange :%1.2e ---- x :%1.2e [%1.2e,%1.2e]'%(niter, np.linalg.norm((x-z).detach().numpy()),erreurl,erreurx,tab[0],tab[1])).center(128,'#'))
                    return x,(True,niter,erreurl,erreurx,lagrange,tab)
    def check_init(self,shape,As,projector,gamma,B,metric,verificators,solver):
        for A in As:
            if not A.shape[1]==shape[1]:
                print(A.shape)
                print(shape)
                raise ValueError("All the matrices must have the shape[1] = %3i (found = %3i)"%(shape[1],A.shape[1]))
        if not len(As)==len(projector):
            raise ValueError("number of matrices (%3i) and of projectors must match (%3i)"%(len(As),len(projector)))
        if not len(As)==len(gamma):
            raise ValueError("number of matrices (%3i) and of multipliers must match (%3i)"%(len(As),len(gamma)))
        if not len(As)==len(verificators):
            raise ValueError("number of matrices (%3i) and of verificators must match (%3i)"%(len(As),len(verificators)))
        if not metric.shape==(shape[1],shape[1]):
            raise ValueError('The metric does not have the correct size, found : '+str(metric.shape)+' must have '+str((shape[1],shape[1])))
        if not solver in ['factorize','spsolve','cholesky']:
            raise ValueError('The solver parameter is not acceptable, found : '+str(solver)+' must have in [factorize,spsolve,cholesky]')
    def check_solve(self,z,list_bound,b,xInit,lagrange):
        if not z.shape==self.shape_x:
            raise ValueError('z does not have the correct size, found : '+str(z.shape)+' must have :'+str(self.shape_x))
        if not len(self.Alist)==len(list_bound):
            raise ValueError("number of matrices (%3i) and of bounds must match (%3i)"%(len(self.Alist),len(list_bound)))
        if not xInit.shape==self.shape_x:
            raise ValueError('xInit does not have the correct size, found : '+str(xInit.shape)+' must have :'+str(self.shape_x))
        if not lagrange.shape==self.shape_y:
            raise ValueError('lagrange does not have the correct size, found : '+str(lagrange.shape)+' must have :'+str(self.shape_y))

def proj_inf_inf(x,g,bnd):
    return torch.sign(x)*torch.clamp(torch.abs(x),max=g*bnd)

def proj_inf_2(x,g,bnd):
    xx = torch.zeros_like(x)
    norm=torch.norm(x,dim=2)+1e-16
    I= norm>g*bnd+1e-16
    for i in range(x.shape[2]):
        xx[...,i][I] = x[...,i][I]/norm[I]*g*bnd
        xx[...,i][torch.logical_not(I)] = x[...,i][torch.logical_not(I)]
    return xx
def proj_pt0(x, g, bnd):
    t = x.clone()
    t[:,0] *= 0
    return t
def proj_ring(x, g, bnd):
    t = x.clone()
    t[:,0] = bnd*g
    return t
def proj_ball_center(x, g, bnd):
    xx = torch.zeros_like(x)
    norm=torch.norm(x[:,0],dim=1)+1e-16
    I= norm>g*bnd+1e-16
    for i in range(x.shape[2]):
        xx[:,0,i][I] = x[:,0,i][I]/norm[I]*g*bnd
        xx[:,0,i][torch.logical_not(I)] = x[:,0,i][torch.logical_not(I)]
    xx[:,1:,:] = x[:,1:,:]
    return xx

def verif_inf_inf(x,g):
    return torch.max(torch.abs(x))/g
def verif_inf_2(x,g):
    return torch.max(torch.norm(x,dim=2))/g
def verif_pt0(x,g):
    return torch.norm(x[:,0])+1
def verif_ring(x,g):
    return g # TODO implement valid verification
def verif_ball_center(x,g):
    return torch.max(torch.norm(x[:,0], dim=1))/g


def mat_deriv(n):
    return sp.spdiags([-np.ones(n), np.ones(n)], [0,1], n-1, n)

def mat_deriv_periodic(n):
    D=sp.spdiags([-np.ones(n), np.ones(n)], [0,1], n, n).tolil()
    D[-1,0]=1
    return D

def mat_deriv2(n):
    return sp.spdiags([-np.ones(n), 2*np.ones(n),-np.ones(n)], [0,1,2], n-2, n)

def mat_deriv2_periodic(n):
    D=sp.spdiags([-np.ones(n), 2*np.ones(n),-np.ones(n)], [-1,0,1], n, n).tolil()
    D[0,-1]=-1
    D[-1,0]=-1
    return D

def mat_pt0(n):
    row = np.array([0])
    col = np.array([0])
    data = np.array([1])
    return sp.coo_matrix((data, (row, col)), shape=(n, n))
