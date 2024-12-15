import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from halftoning import HalftoningFree

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cpu') # torch.device('cuda:0')
n=64*3*3
ovsf=0
N=(320*320)//4
eps=2*np.pi/(10*320)
print('eps=',eps)
iterBB=None
dtype=np.float32
# dtype=np.float64
x=np.linspace(-np.pi,np.pi,n,dtype=dtype)
xx,xy=np.meshgrid(x,x)
pi=np.exp(-(xx**2+xy**2)/(2*(np.pi/4)**2))
pi/=pi.sum()
# plt.imshow(pi)
# plt.show()

p=np.random.uniform(-np.pi,np.pi,size=(N,2)).astype(dtype) / 2


Niter=2000
freq_aff=100
step=5e4

s=HalftoningFree(device, ovsf, eps)
for i in range(1):
    print("\n i=",i)
    tic = time.time()
    p, _, _, _ = s.sample(p, pi, Niter, step, freq_aff=freq_aff, iterBB=iterBB, compute_loss=False)
    toc = time.time()
    print("Elapsed time =",toc-tic)
    plt.ioff(); plt.show()
plt.figure(-1)
pp=p+np.random.randn(N,2)*1e-3
plt.scatter(pp[:,0],pp[:,1],s=2)
plt.ioff()
plt.show()
