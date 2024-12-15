import numpy as np


class LinearMultiScaleNonPeriodic:
    def __init__(self, Nlevel, resMin=0, Kinit=None, Ktarget=None):
        if Ktarget is None and Kinit is None:
            raise Exception("Either Ktarget or Kinit should be given.")
        elif Ktarget is None:
            Ktarget = Kinit*2**(Nlevel-resMin) - 2**(Nlevel-resMin)+1
            assert int(Ktarget)==Ktarget
            Ktarget = int(Ktarget)
        elif Kinit is None:
            Kinit = (Ktarget+2**(Nlevel-resMin)-1)/(2**(Nlevel-resMin))
            assert int(Kinit)==Kinit
            Kinit = int(Kinit)
        else:
            assert Ktarget==Kinit*2**(Nlevel-resMin) - 2**(Nlevel-resMin)+1
        self.Kinit=Kinit
        self.Ktarget=Ktarget
        self.Nlevel=Nlevel
    def _upSampleByM(self, xi, M):
        xiRes = np.zeros((M*(xi.shape[0]-1)+1,xi.shape[1]))
        n = xi.shape[0]
        theta=np.linspace(0,1,M,endpoint=False)
        for i in range(n-1):
            xiRes[M*i:M*(i+1),:] = np.outer(1-theta,xi[i,:])+np.outer(theta,xi[(i+1)%n,:])
        xiRes[-1] = xi[-1]
        return xiRes
    def _downSampleByM(self, g, M):
        gRes = np.zeros((int((g.shape[0]-1)/M + 1), g.shape[1]))
        n = int((g.shape[0]-1)/M + 1)
        theta=np.linspace(0,1,M,endpoint=False)
        for i in range(n-1):
            gRes[i,:]       += g[M*i:M*(i+1),:].T.dot(1-theta)
            gRes[(i+1)%n,:] += g[M*i:M*(i+1),:].T.dot(theta)
        gRes[-1] += g[-1]
        return gRes
    def upSample(self, xi_optim):
        self.Kinit = 2*self.Kinit-1
        ndim = len(xi_optim.shape)
        if ndim==3:
            Nschemes = xi_optim.shape[0]
            upsampled = np.zeros((Nschemes, self.Kinit, xi_optim.shape[-1]))
            for i in range(Nschemes):
                upsampled[i] = self._upSampleByM(xi_optim[i], 2)
            return upsampled
        elif ndim==2:
            return self._upSampleByM(xi_optim, 2)
        else: raise NotImplementedError
    def downSample(self, g):
        ndim = len(g.shape)
        if ndim==3:
            Nschemes = g.shape[0]
            t = self._downSampleByM(g[0], 2)
            downsampled = np.zeros((Nschemes, t.shape[0], g.shape[-1]))
            downsampled[0] = t
            for i in range(1, Nschemes):
                downsampled[i] = self._downSampleByM(g[i], 2)
        elif ndim==2:
            return self._downSampleByM(g, 2)
        else: raise NotImplementedError
    def compute_sampling(self, xi_optim):
        ndim = len(xi_optim.shape)
        if ndim==3:
            Nschemes=xi_optim.shape[0]
            n=xi_optim.shape[1]
            level=int(np.log2((self.Ktarget-1)/(n-1)))
            upsampled = np.zeros((Nschemes, self.Ktarget, xi_optim.shape[-1]))
            for i in range(Nschemes):
                upsampled[i] = self._upSampleByM(xi_optim[i], 2**level)
            return upsampled
        elif ndim==2:
            n=xi_optim.shape[0]
            level=int(np.log2((self.Ktarget-1)/(n-1)))
            return self._upSampleByM(xi_optim, 2**level)
        else: raise NotImplementedError
