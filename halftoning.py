import numpy as np
import torch

import attraction_repulsion as ar

class Halftoning:
    def __init__(self, device, ovsf, eps=1e-8, dist='linear', delta=None, scaling=None, radiuscrop=None):
        self.ovsf = ovsf
        self.eps = eps
        self.device = device
        self.dist = dist
        self.scaling = scaling
        self.radiuscrop = radiuscrop
        if self.dist=='linear':
            self.R = ar.R_linear
            self.kwargs = {}
        elif self.dist=='log':
            self.R = ar.R_log
            self.kwargs = {'scaling': 1. if scaling is None else scaling}
        elif self.dist=='sqrt':
            self.R = ar.R_sqrt
            self.kwargs = {'scaling': 1. if scaling is None else scaling}
        elif self.dist=='pwlinear':
            self.R = ar.R_pwlinear
            self.kwargs = {'scaling': 1. if scaling is None else scaling, 'delta': delta}
        elif self.dist=='loglin':
            self.R = ar.R_loglin
            self.kwargs = {'scaling': 1. if scaling is None else scaling}
    def cost_function(self, p, compute_loss=True, compute_grad=True):
        lossF, gradF = self.A.compute(p, compute_loss, compute_grad)
        lossG, gradG = self.R(p, self.eps, compute_loss, compute_grad, **self.kwargs)
        loss=lossF-lossG
        grad=gradF-gradG
        return loss, grad

    def oracle(self, p, pi, compute_loss=True, compute_grad=True):
        ptorch = torch.tensor(p.reshape(-1, 2), device=self.device)
        pitorch = torch.tensor(pi, device=self.device)
        self.A=ar.Attraction(self.ovsf, pitorch, self.eps, self.dist, scaling=self.scaling, radiuscrop=self.radiuscrop)
        loss, grad = self.cost_function(ptorch, compute_loss=compute_loss, compute_grad=compute_grad)
        return loss.cpu().numpy(), grad.cpu().numpy().reshape(p.shape)

    def sample(self, p, pi, Niter, step, freq_aff=0, iterBB=None, compute_loss=True):
        ptorch = torch.tensor(p, device=self.device)
        pitorch = torch.tensor(pi, device=self.device)
        poptim, list_CF, list_norm_grad, list_step = self.sampletorch(ptorch, pitorch, Niter, step, freq_aff, iterBB, compute_loss)
        return poptim.detach().cpu().numpy(), list_CF, list_norm_grad, list_step

    def sampletorch(self, p, pi, Niter, step, freq_aff=0, iterBB=None, compute_loss=True):
        self.A=ar.Attraction(self.ovsf, pi, self.eps, self.dist, scaling=self.scaling, radiuscrop=self.radiuscrop)

        list_CF=[]
        list_norm_grad=[]
        list_step=[]
        if freq_aff: import matplotlib.pyplot as plt; plt.ion()
        for niter in range(Niter):

            cost,grad = self.cost_function(p, compute_loss=compute_loss, compute_grad=True)
            if iterBB is not None and niter>=iterBB:
                # Barzilai Borwein step
                diff_grad = grad-grad_last
                gamma=((p-plast)*diff_grad).sum()/(diff_grad.pow(2).sum() + 1e-16)
                list_step.append(gamma.item())
            else:
                gamma=step
                list_step.append(gamma)
            if niter>0:
                grad_last=grad.clone()
                plast=p.clone()
            p=self.proj(p-gamma*grad)

            list_CF.append(cost.item())
            normGrad = grad.pow(2).sum().sqrt().item()
            list_norm_grad.append(normGrad)

            if freq_aff and niter%freq_aff==freq_aff-1:
                plt.figure(1000)
                plt.clf()
                plt.scatter(p.detach().cpu().numpy()[:,0],p.detach().cpu().numpy()[:,1],s=1)
                plt.axis('equal')
                plt.grid(True)
                plt.title(str(niter))
                plt.pause(0.001)

                if compute_loss:
                    plt.figure(1001)
                    plt.clf()
                    plt.plot(list_CF)
                    plt.pause(0.001)
                plt.figure(1002)
                plt.clf()
                plt.semilogy(list_norm_grad)
                plt.pause(0.001)

            if normGrad<1e-8:
                break

        return p, list_CF, list_norm_grad, list_step


class HalftoningFree(Halftoning):
    def __init__(self, device, ovsf, eps=1e-8, dist='linear', delta=None, scaling=None, radiuscrop=None):
        super().__init__(device, ovsf, eps, dist, delta, scaling, radiuscrop)
    def proj(self, p, eps=5e-3):
        # Ensure points remain within the box
        return p.clamp(min=-np.pi+eps,max=np.pi-eps)


class HalftoningCons(Halftoning):
    def __init__(self, device, ovsf, projfunc, kwargsproj, eps=1e-8, dist='linear', delta=None, scaling=None, radiuscrop=None):
        super().__init__(device, ovsf, eps, dist, delta, scaling, radiuscrop)
        self.projfunc = projfunc
        self.kwargsproj = kwargsproj
    def proj(self, p, eps=None):
        return self.projfunc.project(
            p.reshape(self.projfunc.Nlines, -1, 2).to(self.projfunc.Admm.device).type(self.projfunc.Admm.dtype),
            **self.kwargsproj
        ).type(p.dtype).to(p.device).reshape(-1, 2)
