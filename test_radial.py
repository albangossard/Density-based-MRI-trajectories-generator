import numpy as np
import torch
import matplotlib.pyplot as plt
from halftoning import HalftoningCons
from projector import ProjectorNonPeriodic
from multiscale import LinearMultiScaleNonPeriodic
import center

torch.manual_seed(42)
np.random.seed(42)



device = torch.device('cpu') # torch.device('cuda:0')
nx = ny = 320 # Size of the image
n_half = 512
sfden = 0.1
kres = 0
resMax = 7
resMin = 0
Kinit = 6
Nlines = 16

Ktarget = (Kinit-1)*2**resMax+1 # Number of sampling points per line
subfac = Nlines*Ktarget/(nx*ny)



multipliersKinematic = (5, 2.5)
dist = 'linear'
scalingsampler = None



lms=LinearMultiScaleNonPeriodic(resMax, resMin=resMin, Ktarget=Ktarget)
Kcurrent = lms.Kinit
print("size:",nx,ny)
print("nb pts per line:",int(nx*ny/(Nlines/subfac)))
print('Kcurrent=',Kcurrent)
print('Ktarget=',Ktarget)
print('nb points=',Nlines*Ktarget)
print('ratio=',Nlines*Ktarget/(nx*ny))



# Parameters of the kinematic constraints projector
g=42.57*1e3
Gm=40*1e-3
Sm=180.
Km=np.pi*2
alpha=g*Gm/Km
beta=g*Sm/Km
dt=2*np.pi/(alpha*nx)
print("dt=",dt)
alpha1=alpha*dt
alpha2=beta*dt**2
print("alpha1=",alpha1)
print("alpha2=",alpha2)

sf = 1.8



merger = center.PtsMerge(Nlines, sf, nx, alpha1, alpha2)
radius_proj_ring, lim = center.computeRadiusRing(Nlines, sf, nx, alpha1, alpha2)
dengen = center.DensityGen(radius_proj_ring, n_half)


kwargsproj = {}



# Sampler
ovsf = 1
eps = 2*np.pi/(100*nx)




Niter = 10000
iterBB = 8000
step = 1e2
tolProj = 1e-4
nitermax = 15




x = np.linspace(-np.pi, np.pi, n_half); xx, xy=np.meshgrid(x, x)
den = 1/np.sqrt(xx**2+xy**2+1)
den = center.threshold_density(den, sfden, r=sf)
den = dengen.forward(den)


s = HalftoningCons(device, ovsf, None, kwargsproj, eps=eps, dist=dist, scaling=scalingsampler) # We can instantiate here even if the projection is not defined as we won't use the sample methods in this script



saveData = False
plot = True
freq_aff = 100
dirname='results_radial'



xi_optim = np.zeros((Nlines,Kcurrent,2))
for i in range(Nlines):
    xi_optim[i,:,0] = np.cos(2*np.pi*i/Nlines)*np.linspace(radius_proj_ring,np.pi*0.7,Kinit)
    xi_optim[i,:,1] = np.sin(2*np.pi*i/Nlines)*np.linspace(radius_proj_ring,np.pi*0.7,Kinit)


xistart = np.zeros((Nlines, 2))
xistart[:, 0] = radius_proj_ring*np.cos(np.linspace(0,2*np.pi, Nlines, endpoint=False))
xistart[:, 1] = radius_proj_ring*np.sin(np.linspace(0,2*np.pi, Nlines, endpoint=False))



grad = None

if plot: plt.ion()
for k in range(resMax,resMin-1,-1):
    if k<resMax:
        xi_optim = lms.upSample(xi_optim)
        xi_optim += np.random.randn(*(xi_optim.shape))*2*np.pi*np.sqrt(2**k)/(100*nx)
        Kcurrent = Kcurrent*2-1

    proj = ProjectorNonPeriodic(Nlines, Kcurrent, alpha1, alpha2, k, device, VERBOSE=1, radius_proj_ring=radius_proj_ring, bnd=np.pi-np.pi/nx, multipliersKinematic=multipliersKinematic)

    xi_optim = proj.project(xi_optim, tol=tolProj, nitermax=nitermax)
    xi_optim[:, 0] = xistart

    if np.isnan(xi_optim).sum()>0:
        raise Exception("Nan in xi")

    list_CF=[]
    for niter in range(1,Niter+1):

        gradold = grad
        cost, grad = s.oracle(xi_optim, den, compute_loss=True, compute_grad=True)
        grad[:,0]*=0
        if np.isnan(grad).sum()>0:
            raise Exception("Nan in grad xi")

        if niter >= iterBB:
            diff_grad = grad-gradold
            gamma=((xi_optim-xi_optim_old)*diff_grad).sum()/((diff_grad**2).sum() + 1e-16)
        else:
            gamma = step
        xi_optim_new = xi_optim - gamma * grad
        xi_optim_new = proj.project(xi_optim_new, tol=tolProj, nitermax=nitermax)

        xi_optim_old = xi_optim
        xi_optim = xi_optim_new

        list_CF.append(cost.item())

        if (plot and (niter%freq_aff==freq_aff-1)):

            fig = plt.figure(1)
            plt.clf()
            ax = fig.add_subplot(1,1,1)


            y=proj.Admm.gA(torch.tensor(xi_optim, dtype=proj.Admm.dtype, device=proj.Admm.device)).cpu()
            resultvx=[]
            resultvy=[]
            resultax=[]
            resultay=[]

            e1=e2=0.97

            indvit = proj.Admm.indices_y[0]
            indacc = proj.Admm.indices_y[1]

            saturates_speed = np.zeros(xi_optim.shape[:2])
            saturates_accel = np.zeros(xi_optim.shape[:2])

            # Compute the speed
            norm = torch.norm(y[:,indvit[0]:indvit[1]],dim=2).numpy()
            norm = np.concatenate([norm, np.zeros((xi_optim.shape[0],1))], axis=1)
            saturates_speed[(norm/proj.Admm.gamma[0])/proj.bounds[0]>e1] = 1

            # Compute the accel
            norm = torch.norm(y[:,indacc[0]:indacc[1]],dim=2).numpy()
            norm = np.concatenate([norm, np.zeros((xi_optim.shape[0],1))], axis=1)
            saturates_accel[:,1:][(norm/proj.Admm.gamma[1])/proj.bounds[1]>e2] = 1

            first = True
            for i in range(norm.shape[0]):
                plt.plot(xi_optim[i,:,0],xi_optim[i,:,1],linewidth=0.5,markersize=0.5,marker='o',color='black')
                for j in range(norm.shape[1]):
                    if saturates_speed[i,j]:
                        resultvx.append(xi_optim[i,j,0])
                        resultvy.append(xi_optim[i,j,1])
                    if saturates_accel[i,j]:
                        resultax.append(xi_optim[i,j,0])
                        resultay.append(xi_optim[i,j,1])

                    if saturates_speed[i,j]:
                        if first:
                            plt.plot(xi_optim[i,j:j+2,0], xi_optim[i,j:j+2,1], color='blue')
                            first = False
                        else:
                            plt.plot(xi_optim[i,j:j+2,0], xi_optim[i,j:j+2,1], color='blue')
            plt.scatter(resultax,resultay,marker="o",color='red',s=2)
            plt.axis('equal')
            plt.grid(True)
            plt.title(str(niter)+"/"+str(Niter)+"  k="+str(k))
            if plot: plt.pause(0.001)
            if saveData: plt.savefig(dirname+"/xi_k="+str(k)+".pdf")
            plt.figure(2)
            plt.clf()
            plt.plot(list_CF)
            if plot: plt.pause(0.001)
            if saveData: plt.savefig(dirname+"/cf_k="+str(k)+".pdf")

        if saveData and (niter%2000==0 or niter==Niter):
            np.save(dirname+"/xi_optim_k="+str(k)+"_i="+str(niter)+".npy", xi_optim)

            xi_ech = lms.compute_sampling(xi_optim)
            np.save(dirname+"/xi_ech_k="+str(k)+".npy", xi_ech)
if plot: plt.ioff(); plt.show()
