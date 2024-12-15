import numpy as np

def _ptsmaxspeedaccel(Nlines, subfac, nx, alpha1, alpha2):
    xn=0
    xnm1=0
    lx = [xn]
    lim = 1/(nx*subfac/Nlines)
    lim = np.pi/(subfac*nx*np.sin(np.pi/Nlines))
    while xn<lim:
        c1 = alpha1+xn
        c2 = alpha2+2*xn-xnm1
        if abs(c1+xnm1-2*xn)>alpha2:
            xnp1 = c2
        elif abs(c2-xn)>alpha1:
            xnp1 = c1
        else:
            raise Exception
        lx.append(xnp1)
        xnm1 = xn
        xn = xnp1
    return lx, lim
def computeRadiusRing(Nlines, subfac, nx, alpha1, alpha2):
    lx, lim = _ptsmaxspeedaccel(Nlines, subfac, nx, alpha1, alpha2)
    return lx[-1], lim
def centerSchemeGeneration(Nlines, subfac, nx, alpha1, alpha2):
    lx = _ptsmaxspeedaccel(Nlines, subfac, nx, alpha1, alpha2)[0]
    lx = lx[:-1]
    linspaceIn = np.array(lx)
    nbPtsToAdd = (len(linspaceIn)-1)*Nlines+1

    ptsCenter = np.zeros((nbPtsToAdd, 2))
    ptsCenter[0] = linspaceIn[0]
    for i in range(Nlines):
        for j in range(1,len(linspaceIn)):
            ptsCenter[1+i*(len(linspaceIn)-1)+(j-1), 0] = linspaceIn[j] * np.cos(2*np.pi*i/Nlines)
            ptsCenter[1+i*(len(linspaceIn)-1)+(j-1), 1] = linspaceIn[j] * np.sin(2*np.pi*i/Nlines)
    return ptsCenter

class DensityGen:
    def __init__(self, radius_proj_ring, n_half):
        x = np.linspace(-np.pi, np.pi, n_half); xx, xy=np.meshgrid(x, x)
        self.mask = np.sqrt(xx**2+xy**2)<=radius_proj_ring
        self.n_half = n_half
    def forward(self, den):
        pi = den.copy()
        pi[self.mask] = 0
        return pi/np.sum(pi)

class PtsMerge:
    def __init__(self, Nlines, subfaccenter, nx, alpha1, alpha2):
        self.xicenter = centerSchemeGeneration(Nlines, subfaccenter, nx, alpha1, alpha2)
        self.nbPtsToAdd = self.xicenter.shape[0]
    def sample(self, xihalf):
        return np.concatenate((xihalf, self.xicenter))
    def removeCenterPts(self, xi):
        return xi[:-self.nbPtsToAdd]


def threshold_density(G,u,r=1.01):
    # Limiting the density to a given threshold
    # G: density
    # u: undersampling factor
    # r: oversampling factor at the center
    n=G.shape[0]
    target=r/u # target density at the center since we sample u*n^2 points
    G/=np.sum(G)/(n*n)
    ind=np.zeros_like(G)
    ind[G>target]=1
    while (G>target).any():
        G[G>target]=target
        area_out = np.sum(G*(1-ind))
        area_in = np.sum(G*ind)
        factor=(n*n-area_in)/area_out
        G[ind==0]*=factor
    return G/np.sum(G)
