import numpy as np
import torch
from pykeops.torch import LazyTensor


def H_linear(grid,eps):
    return (grid.pow(2).sum(dim=2)+eps**2).sqrt()
def H_pwlinear(grid,eps,delta=None,scaling=1):
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    alpha = (scaling+1)/2.
    beta = -alpha
    return alpha*(d2+eps**2).sqrt() + beta*((d-delta)**2+eps**2).sqrt()
def H_loglin(grid,eps,scaling=1):
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    alpha = 1/(scaling+eps)
    beta = np.log(scaling+eps)-alpha*scaling
    return torch.log(d+eps)*(d<=scaling)+(alpha*d+beta)*(d>scaling)
def H_log(grid,eps,scaling=1):
    d = grid.pow(2).sum(dim=2).sqrt()
    return torch.log(scaling*d+eps)
def H_sqrt(grid,eps,scaling=1):
    d = grid.pow(2).sum(dim=2).sqrt()
    return torch.sqrt(scaling*d)
def Hp_linear(grid,eps):
    size_grid=grid.shape[:-1]
    return grid/(grid.pow(2).sum(dim=2)+eps**2).sqrt().reshape(size_grid+(1,))
def Hp_pwlinear(grid,eps,delta=None,scaling=1):
    size_grid=grid.shape[:-1]
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    alpha = (scaling+1)/2.
    beta = -alpha
    return grid/(d+1e-16).reshape(size_grid+(1,)) * ( alpha * d /(d2+eps**2).sqrt() + beta * ((d-delta)/((d-delta)**2+eps**2).sqrt()) ).reshape(size_grid+(1,))
def Hp_loglin(grid,eps,scaling=1):
    size_grid=grid.shape[:-1]
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    alpha = 1/(scaling+eps)
    return ( grid/((d+1e-16)*(d+eps)).reshape(size_grid+(1,)) )*(d.reshape(size_grid+(1,))<=scaling) + (alpha*grid/(d+1e-16).reshape(size_grid+(1,)))*(d.reshape(size_grid+(1,))>scaling)
def Hp_log(grid,eps,scaling=1):
    size_grid=grid.shape[:-1]
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    return (scaling*grid/(1e-16+d*(scaling*d+eps)).reshape(size_grid+(1,)))
def Hp_sqrt(grid,eps,scaling=1):
    size_grid=grid.shape[:-1]
    d2 = grid.pow(2).sum(dim=2)
    d = d2.sqrt()
    return (scaling*grid/(1e-16+2*d*(scaling*d+eps).sqrt()).reshape(size_grid+(1,)))


def convolution(pi_interp, grid_oversampled, eps, H, **kwargs):
    pad=int((grid_oversampled.shape[0]-pi_interp.shape[0])/2)

    tmp = torch.zeros(grid_oversampled.shape[:-1], dtype=pi_interp.dtype, device=pi_interp.device)
    tmp[pad:-pad,pad:-pad]=pi_interp
    pi_interp=tmp

    Hgrid = H(grid_oversampled, eps, **kwargs)

    conv=torch.fft.ifft2(torch.fft.fft2(pi_interp)*torch.fft.fft2(Hgrid)).real
    conv=torch.fft.fftshift(conv)
    conv=conv[pad:-pad,pad:-pad]

    return conv

def convolution_prime(pi_interp, grid_oversampled, eps, Hp, **kwargs):
    pad=int((grid_oversampled.shape[0]-pi_interp.shape[0])/2)

    tmp = torch.zeros((2,)+grid_oversampled.shape[:-1], dtype=pi_interp.dtype, device=pi_interp.device)
    tmp[0,pad:-pad,pad:-pad]=pi_interp
    tmp[1,pad:-pad,pad:-pad]=pi_interp
    pi_interp=tmp

    Hpgrid = Hp(grid_oversampled, eps, **kwargs)

    tmp = torch.zeros((2,)+Hpgrid.shape[:-1], dtype=pi_interp.dtype, device=pi_interp.device)
    tmp[:,:Hpgrid.shape[0],:Hpgrid.shape[1]]=Hpgrid.permute(2,0,1)
    Hpgrid=tmp

    conv=torch.fft.ifft2(torch.fft.fft2(pi_interp)*torch.fft.fft2(Hpgrid))
    conv[0]=torch.fft.fftshift(conv[0])
    conv[1]=torch.fft.fftshift(conv[1])
    conv=conv[:,pad:-pad,pad:-pad].real.permute(1,2,0)

    return conv


class Attraction:
    def __init__(self, ovsf, pi, eps, dist, delta=None, scaling=None, radiuscrop=None):
        self.dtype = pi.dtype
        self.device = pi.device
        self.ovsf = ovsf
        self.pi = pi
        self.eps = eps
        self.n = self.pi.shape[0]
        if dist=='linear':
            H = H_linear
            Hp = Hp_linear
            kwargs = {}
        elif dist=='log':
            H = H_log
            Hp = Hp_log
            if scaling is None:
                kwargs = {'scaling': 1.}
            else:
                kwargs = {'scaling': scaling}
        elif dist=='sqrt':
            H = H_sqrt
            Hp = Hp_sqrt
            if scaling is None:
                kwargs = {'scaling': 1.}
            else:
                kwargs = {'scaling': scaling}
        elif dist=='pwlinear':
            H = H_pwlinear
            Hp = Hp_pwlinear
            if scaling is None:
                kwargs = {'scaling': 1., 'delta': None}
            else:
                kwargs = {'scaling': scaling, 'delta': delta}
        elif dist=='loglin':
            H = H_loglin
            Hp = Hp_loglin
            if scaling is None:
                kwargs = {'scaling': 1.}
            else:
                kwargs = {'scaling': scaling}

        self.sf = 3
        self.factor = self.sf**self.ovsf

        grid = -np.pi+2*np.pi*(torch.arange(self.n, device=self.device).type(self.dtype)+0.5)/self.n
        self.n_padded = self.n+2*int(np.ceil((self.n+1)/2.))
        S = grid[-1] + (grid[1]-grid[0])*(self.n_padded-self.n)/2.
        grid_padded = torch.linspace(-S, S, self.n_padded, dtype=self.dtype, device=self.device)

        n_interp_padded = self.factor*(self.n_padded-1)+1
        theta=torch.zeros(n_interp_padded, self.n_padded, dtype=self.dtype, device=self.device)
        linsp=torch.linspace(0, 1, self.factor+1, dtype=self.dtype, device=self.device)
        for i in range(self.n_padded):
            if i>0:
                theta[(i-1)*self.factor:i*self.factor+1,i] = linsp
            if i<self.n_padded-1:
                theta[i*self.factor:(i+1)*self.factor+1,i] = torch.flip(linsp, dims=[0])
        grid_padded_oversampled = torch.mm(theta, grid_padded.view(-1,1)).view(-1)


        gg_padded=torch.zeros(n_interp_padded, n_interp_padded, 2, dtype=self.dtype, device=self.device)
        gx,gy=torch.meshgrid(grid_padded_oversampled, grid_padded_oversampled)
        gg_padded[...,0]=gx
        gg_padded[...,1]=gy

        pi_interp = pi.reshape(1,1,self.n,self.n)
        for i in range(self.ovsf):
            pi_interp = torch.nn.functional.interpolate(pi_interp, scale_factor=self.sf, mode='bicubic', align_corners=False)/(self.sf**2.)
        pi_interp=pi_interp[0,0].clamp(min=0)
        if radiuscrop is not None:
            xx = 2*np.pi*(torch.arange(pi_interp.shape[0])-pi_interp.shape[0]/2)/pi_interp.shape[0]
            xx, xy = torch.meshgrid(xx, xx)
            mask = torch.sqrt(xx**2+xy**2)<=radiuscrop
            pi_interp[mask] = 0.
        pi_interp = pi_interp/pi_interp.sum()
        if radiuscrop is not None: self.pi_interp = pi_interp.detach().cpu().numpy()

        self.oversampled_data = convolution(pi_interp, gg_padded, eps, H, **kwargs)
        self.oversampled_data = self.oversampled_data.reshape(1,1,self.factor*self.n,self.factor*self.n)
        self.oversampled_data_prime = convolution_prime(pi_interp, gg_padded, eps, Hp, **kwargs).permute(2,0,1)
        self.oversampled_data_prime = self.oversampled_data_prime.reshape(1,2,self.factor*self.n,self.factor*self.n)


    def interpolate(self, loc):
        # loc: shape (N,2)
        pts = loc.reshape(1,1,-1,2)/np.pi
        loc = torch.nn.functional.grid_sample(self.oversampled_data, pts, mode='bilinear', align_corners=True)
        return loc[0,0,0]
    def interpolate_prime(self, loc):
        # loc: shape (N,2)
        pts = loc.reshape(1,1,-1,2)/np.pi
        loc = torch.nn.functional.grid_sample(self.oversampled_data_prime, pts, mode='bilinear', align_corners=True)
        return torch.flip(loc[0,:,0,:].permute(1,0), dims=[1])

    def compute(self, p, compute_loss=True, compute_grad=True):
        loss=torch.tensor(0)
        if compute_loss:
            interp = self.interpolate(p)
            loss = interp.sum()/p.shape[0]
        grad=torch.zeros_like(p)
        if compute_grad:
            interp_prime = self.interpolate_prime(p)
            grad = interp_prime/p.shape[0]
        return loss, grad



def R_linear(p, eps, compute_loss=True, compute_grad=True):
    # p: shape (N,d) with d the dimension (ie d=2)
    N=p.shape[0]
    p_k = LazyTensor(p[:,None,:])
    p_j = LazyTensor(p[None,:,:])
    grid=p_k-p_j
    grad=torch.zeros_like(p)
    loss=torch.tensor(0)
    if compute_grad and compute_loss:
        norm=((grid**2).sum(dim=2)+eps**2).sqrt()
        Hp=grid/norm
        grad=Hp.sum(dim=1)/(N*N)
        loss=norm.sum(dim=0).sum(dim=0)/(2*N*N)
    elif compute_grad:
        Hp=grid/((grid**2).sum(dim=2)+eps**2).sqrt()
        grad=Hp.sum(dim=1)/(N*N)
    elif compute_loss:
        loss=((grid**2).sum(dim=2)+eps**2).sqrt().sum(dim=0).sum(dim=0)/(2*N*N)
    return loss, grad
def R_pwlinear(p, eps, compute_loss=True, compute_grad=True, delta=None, scaling=1):
    # p: shape (N,d) with d the dimension (ie d=2)
    N=p.shape[0]
    p_k = LazyTensor(p[:,None,:])
    p_j = LazyTensor(p[None,:,:])
    grid=p_k-p_j
    grad=torch.zeros_like(p)
    loss=torch.tensor(0)
    norm2 = ((grid**2).sum(dim=2))
    norm = norm2.sqrt()
    alpha = (scaling+1)/2.
    beta = -alpha
    if compute_grad and compute_loss:
        losstmp = alpha*(norm2+eps**2).sqrt() + beta*((norm-delta)**2+eps**2).sqrt()
        loss = losstmp.sum(dim=0).sum(dim=0)/(2*N*N)
        gradtmp = grid * ( alpha/((grid**2).sum(dim=2)+eps**2).sqrt() + beta*((norm-delta)/((1e-16+norm)*((norm-delta)**2+eps**2).sqrt())) )
        grad = gradtmp.sum(dim=1)/(N*N)
    elif compute_grad:
        gradtmp = grid * ( alpha/((grid**2).sum(dim=2)+eps**2).sqrt() + beta*((norm-delta)/((1e-16+norm)*((norm-delta)**2+eps**2).sqrt())) )
        grad = gradtmp.sum(dim=1)/(N*N)
    elif compute_loss:
        losstmp = alpha*(norm2+eps**2).sqrt() + beta*((norm-delta)**2+eps**2).sqrt()
        loss = losstmp.sum(dim=0).sum(dim=0)/(2*N*N)
    return loss, grad
def R_loglin(p, eps, compute_loss=True, compute_grad=True, scaling=1):
    # p: shape (N,d) with d the dimension (ie d=2)
    N=p.shape[0]
    p_k = LazyTensor(p[:,None,:])
    p_j = LazyTensor(p[None,:,:])
    grid=p_k-p_j
    grad=torch.zeros_like(p)
    loss=torch.tensor(0)
    norm2 = ((grid**2).sum(dim=2))
    norm = norm2.sqrt()
    alpha = 1/(scaling+eps)
    beta = np.log(scaling+eps)-alpha*scaling
    beta=torch.tensor(beta, dtype=p.dtype, device=p.device)
    indgeq1 = ((norm-scaling).sign()+1)/2
    indleq1 = 1-indgeq1
    if compute_grad:
        gradtmp = ( grid/((norm+1e-16)*(norm+eps)) )*indleq1 + (alpha*grid/(norm+1e-16))*indgeq1
        grad = gradtmp.sum(dim=1)/(N*N)
    if compute_loss:
        losstmp = (norm+eps).log()*indleq1+(alpha*norm+beta)*indgeq1
        loss = losstmp.sum(dim=0).sum(dim=0)/(2*N*N)
    return loss, grad
def R_log(p, eps, compute_loss=True, compute_grad=True, scaling=1):
    # p: shape (N,d) with d the dimension (ie d=2)
    N=p.shape[0]
    p_k = LazyTensor(p[:,None,:])
    p_j = LazyTensor(p[None,:,:])
    grid=p_k-p_j
    grad=torch.zeros_like(p)
    loss=torch.tensor(0)
    norm2 = ((grid**2).sum(dim=2))
    norm = norm2.sqrt()
    if compute_grad and compute_loss:
        grad = ( scaling*grid/(1e-16+norm*(scaling*norm+eps)) ).sum(dim=1)/(N*N)
        loss = ( (scaling*norm+eps).log() ).sum(dim=0).sum(dim=0)/(2*N*N)
    elif compute_grad:
        grad = ( scaling*grid/(1e-16+norm*(scaling*norm+eps)) ).sum(dim=1)/(N*N)
    elif compute_loss:
        loss = ( (scaling*norm+eps).log() ).sum(dim=0).sum(dim=0)/(2*N*N)
    return loss, grad
def R_sqrt(p, eps, compute_loss=True, compute_grad=True, scaling=1):
    # p: shape (N,d) with d the dimension (ie d=2)
    N=p.shape[0]
    p_k = LazyTensor(p[:,None,:])
    p_j = LazyTensor(p[None,:,:])
    grid=p_k-p_j
    grad=torch.zeros_like(p)
    loss=torch.tensor(0)
    norm2 = ((grid**2).sum(dim=2))
    norm = norm2.sqrt()
    if compute_grad and compute_loss:
        grad = ( scaling*grid/(1e-16+2*norm*(scaling*norm+eps).sqrt()) ).sum(dim=1)/(N*N)
        loss = ( (scaling*norm+eps).sqrt() ).sum(dim=0).sum(dim=0)/(2*N*N)
    elif compute_grad:
        grad = ( scaling*grid/(1e-16+2*norm*(scaling*norm+eps).sqrt()) ).sum(dim=1)/(N*N)
    elif compute_loss:
        loss = ( (scaling*norm+eps).sqrt() ).sum(dim=0).sum(dim=0)/(2*N*N)
    return loss, grad
