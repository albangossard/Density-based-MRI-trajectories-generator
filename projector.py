import numpy as np
import scipy.sparse as sps
import torch

from admm import admm,proj_inf_2,mat_deriv,mat_deriv2,mat_pt0,verif_inf_2, proj_inf_inf, verif_inf_inf,proj_ring, verif_ring


class ProjectorNonPeriodic:
    def __init__(self, Nlines, K, alpha1, alpha2, k, device, VERBOSE=1, radius_proj_ring=None, bnd=np.pi, multipliersKinematic=(100/2., 100/4.), multiplierRing=100.):
        self.Nlines = Nlines
        self.VERBOSE = VERBOSE
        if radius_proj_ring is not None:
            x0 = np.zeros((Nlines, 2))
            x0[:,0] = radius_proj_ring*np.cos(np.linspace(0, 2*np.pi, self.Nlines, endpoint=False))
            x0[:,1] = radius_proj_ring*np.sin(np.linspace(0, 2*np.pi, self.Nlines, endpoint=False))

        if radius_proj_ring is None:
            projectors=[proj_inf_2,proj_inf_2,proj_inf_inf,proj_inf_inf]
            verificators=[verif_inf_2,verif_inf_2,verif_inf_inf,verif_inf_inf]
            multipliers=[100./2.,100./4.,1.,1.]
            self.bounds=[alpha1*2**k,alpha2*4**k,bnd,bnd]
            matrices=[mat_deriv(K),mat_deriv2(K),sps.eye(K),-sps.eye(K)]
        else:
            projectors=[proj_inf_2,proj_inf_2,proj_inf_inf,proj_inf_inf, proj_ring]
            verificators=[verif_inf_2,verif_inf_2,verif_inf_inf,verif_inf_inf, verif_ring]
            multipliers=[multipliersKinematic[0], multipliersKinematic[1], 1., 1., multiplierRing]
            self.bounds=[alpha1*2**k,alpha2*4**k,bnd,bnd,x0]
            matrices=[mat_deriv(K),mat_deriv2(K),sps.eye(K),-sps.eye(K), mat_pt0(K)]
        self.Admm=admm((Nlines,K,2),matrices,projectors,multipliers,device,metric=sps.eye(K),verificators=verificators,solver='factorize')
    def project(self, xi, fixedNbStep=-1, tol=1e-5, nitermax=200):
        inoutnp = isinstance(xi, np.ndarray)
        if inoutnp:
            xi = torch.tensor(xi, device=self.Admm.device, dtype=self.Admm.dtype)
        xi_proj, res = self.Admm.solve(xi,self.bounds,nitermax=nitermax,xInit=xi,tol=tol,verbose=self.VERBOSE, fixedNbStep=fixedNbStep)
        if self.VERBOSE>0:
            if res[1] == 0:
                print(("   ADMM finished niter = %i Convergence %i DD1 : %3.2f %% DD2 : %3.2f %% DD3 : %3.2f %% DD4 : %3.2f %%"%(res[1],int(res[0]),res[3][0],res[3][1],res[3][2],res[3][3])))
            else :
                print(("   ADMM finished niter = %i Convergence %i lagrange : %1.3e x : %1.3e DD1 : %3.2f %% DD2 : %3.2f %% DD3 : %3.2f %% DD4 : %3.2f %%"%(res[1],int(res[0]),res[2],res[3],res[5][0],res[5][1],res[5][2],res[5][3])))
        if inoutnp:
            xi_proj = xi_proj.cpu().numpy()
        return xi_proj
    def _compute_bounds(self, xi):
        actual_bound=self.Admm.verif(self.Admm.gA(xi))
        assert len(actual_bound)==len(self.bounds)
        for b_target, v in zip(self.bounds, actual_bound):
            print("target=",b_target, "actual=",v, "ratio=",(v/b_target)*100, " %")
