
import prettyplotlib as ppl
from prettyplotlib import plt
import scipy.optimize as opt
import os

import numpy as np

def d_eps_dt(eps,gamma,eta,alpha,lamb):
	return -2*gamma*eps+eta**2-lamb*eps**2/(alpha**2+eps)
	#return -np.dot(gamma,eps)-np.dot(eps,gamma.T)+np.dot(eta.T,eta)-lamb*np.linalg.solve(alpha+eps,np.dot(eps,eps))

def get_eq_eps(gamma,eta,alpha,lamb):
	f = lambda e : d_eps_dt(e,gamma,eta,alpha,lamb)
	return opt.fsolve(f,1.0)

def full_stoc_sigma(sigma0,dt,N,a,eta,alpha,la,NSamples,rands=None, discard=0):
    sigmas = np.zeros((NSamples,N))
    sigmas[:,0] = sigma0
    if rands==None:
        rng = np.random.RandomState(12345)
        rands = (rng.uniform(size=(NSamples,N))<la*dt).astype('int')
    else:
        assert rands.shape == (NSamples, N)
        rands = (rands<la*dt).astype('int')
    for i in xrange(0, discard):
        rand_sample = (rng.uniform(size=(NSamples,1))<la*dt).astype('int')
        splus1 = np.asarray([sigmas[:,0]+dt*(2*a*sigmas[:,0]+eta**2),
                             alpha**2*sigmas[:,0]/(alpha**2+sigmas[:,0])])
        sigmas[:,0] = splus1[rand_sample[:,0],range(NSamples)]
    
    for i in xrange(1,N):
        splus1 = np.asarray([sigmas[:,i-1]+dt*(2*a*sigmas[:,i-1]+eta**2),
                             alpha**2*sigmas[:,i-1]/(alpha**2+sigmas[:,i-1])])
        sigmas[:,i] = splus1[rands[:,i],range(NSamples)]
    return np.mean(sigmas, axis = 0)

def replica_eps(gamma, eta, alpha, lamb, tol=1e-9):
    eps = eta**2/(2.0*gamma)
    U = lamb/(alpha**2+eps)
    for i in range(1000):
        eps = eta**2/2 *(1.0/np.sqrt(gamma**2+U*eta**2))
        U = lamb/(alpha**2+eps)
        if np.abs(eps -  eta**2/2 *(1.0/np.sqrt(gamma**2+U*eta**2))) < tol:
            break
    phi = (np.sqrt(gamma**2+U*eta**2)-gamma) + lamb*np.log(1.0+eps/alpha**2)-U*eps
    phi = 0.5*phi
    return alpha**2*(np.exp(2.0*phi/lamb) - 1.0)

if __name__=='__main__':

    alphas = np.arange(0.001,2.0,0.01)
    phis = np.arange(0.001,2.0,0.01)
    eps = np.zeros((alphas.size,alphas.size))
    stoc_eps = np.zeros((alphas.size,alphas.size))

    gamma = 1.0
    eta = 1.0
    phi = 0.1
    N = 10000
    dt = 0.0001
    discard = 1000
    for n,alpha in enumerate(alphas):
        print n, alphas.size
        for m, phi in enumerate(phis):
            lamb = phi*np.sqrt(2*np.pi*alpha)
            eps[n,m] = get_eq_eps( gamma, eta, alpha, lamb )
            stoc_eps[n,m] =  np.mean(full_stoc_sigma(0.01, dt, N, -gamma,
                                           eta, alpha, lamb, 1000,
                                           discard=discard))
    fig, (ax1,ax2) = ppl.subplots(2)

    p = ax1.pcolormesh(eps) 
    fig.colorbar(p) 

    ax2.plot( alphas, eps[:,alphas.size/2], 'r', label='mean-field')
    ax2.plot( alphas, stoc_eps[:,alphas.size/2], 'b.', label='stochastic average')
    ax2.legend()
    plt.show()
    plt.savefig('../figures/figure_5_3.eps')
    os.system("echo \"all done\" | mutt -a \"../figures/figure_5_3.eps\" -s \"Plot\" -- alexsusemihl@gmail.com")
