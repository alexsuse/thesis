
import prettyplotlib as ppl
from prettyplotlib import plt
from prettyplotlib import brewer2mpl
import scipy.optimize as opt
import os
import cPickle as pic

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

    dalpha = 0.04
    dphi = 0.04
    alphas = np.arange(0.001,2.0,dalpha)
    phis = np.arange(0.001,2.0,dphi)
    eps = np.zeros((alphas.size,alphas.size))
    stoc_eps = np.zeros((alphas.size,alphas.size))

    try:
        dic = pic.load(open("figure_5_3.pik","r"))
        eps = dic['mean_field']
        stoc_eps = dic['stochastic']
        print "Found Pickle, skipping simulation"

    except:

        gamma = 1.0
        eta = 1.0
        phi = 0.1
        N = 10000
        dt = 0.001
        discard = 1000
        for n,alpha in enumerate(alphas):
            print n, alphas.size
            for m, phi in enumerate(phis):
                print n,m,alphas.size
                lamb = phi*np.sqrt(2*np.pi*alpha)
                eps[n,m] = get_eq_eps( gamma, eta, alpha, lamb )
                stoc_eps[n,m] =  np.mean(full_stoc_sigma(0.01, dt, N, -gamma,
                                               eta, alpha, lamb, 200,
                                               discard=discard))

        with open("figure_5_3.pik","w") as fi:
            print "Writing pickle to figure_5_3.pik"
            pic.dump({'alphas':alphas,'phis':phis,'mean_field':eps,'stochastic':stoc_eps},fi)

    print "Plotting..."
    
    font = {'size':12}
    plt.rc('font',**font)

    fig, (ax1,ax2) = ppl.subplots(1,2,figsize = (12,5))
    
    alphas2,phis2 = np.meshgrid(np.arange(alphas.min(),alphas.max()+dalpha,dalpha)-dalpha/2,
                                np.arange(phis.min(),phis.max()+dphi,dphi)-dphi/2)

    yellorred = brewer2mpl.get_map('YlOrRd','Sequential',9).mpl_colormap

    p = ax1.pcolormesh(alphas2,phis2,eps.T,cmap=yellorred, rasterized=True)
    ax1.axis([alphas2.min(),alphas2.max(),phis2.min(),phis2.max()])

    #xticks = np.arange(alphas.min(),alphas.max(),0.5)
    #xlabels = np.arange(alphas.min(),alphas.max(),0.5)-alphas.min()

    #yticks = np.arange(phis.min(),phis.max(),0.5)
    #ylabels = np.arange(phis.min(),phis.max(),0.5)-phis.min()

    #plt.xticks(xticks,xlabels,axes=ax1)
    #plt.yticks(yticks,ylabels,axes=ax1)

    cb = plt.colorbar(p, ax=ax1) 
    cb.set_ticks(np.array([0.3,0.4,0.5]))
    cb.set_ticklabels(np.array([0.3,0.4,0.5]))

    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title(r'MMSE ($\epsilon$)')

    l1,= ppl.plot( alphas, eps[:,10], label=r'$\phi = '+ str(phis[10]-0.001) + r'$',ax=ax2)
    ppl.plot( alphas[np.argmin(eps[:,10])], np.min(eps[:,10]), 'o',color=l1.get_color(),ax=ax2)
    l2, = ppl.plot( alphas, eps[:,20], label=r'$\phi = '+ str(phis[20]-0.001) + r'$',ax=ax2)
    ppl.plot( alphas[np.argmin(eps[:,20])] , np.min(eps[:,20]), 'o',color=l2.get_color(),ax=ax2)
    l3, = ppl.plot( alphas, eps[:,30], label=r'$\phi = '+ str(phis[30]-0.001) + r'$',ax=ax2)
    ppl.plot( alphas[np.argmin(eps[:,30])] , np.min(eps[:,30]), 'o',color=l3.get_color(),ax=ax2)
    l4, = ppl.plot( alphas, eps[:,40], label=r'$\phi = '+ str(phis[40]-0.001) + r'$',ax=ax2)
    ppl.plot( alphas[np.argmin(eps[:,40])] , np.min(eps[:,40]), 'o',color=l4.get_color(),ax=ax2)
    ppl.plot( alphas , stoc_eps[:,10], '-.',ax=ax2, color = l1.get_color())
    ppl.plot( alphas , stoc_eps[:,20], '-.',ax=ax2, color = l2.get_color())
    ppl.plot( alphas , stoc_eps[:,30], '-.',ax=ax2, color = l3.get_color())
    ppl.plot( alphas , stoc_eps[:,40], '-.',ax=ax2, color = l4.get_color())
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\epsilon$')
    ax2.set_title(r'MMSE as a function of $\alpha$')
    ppl.legend(ax2).get_frame().set_alpha(0.7)
    plt.savefig('../figures/figure_5_3.png',dpi=300)
    plt.savefig('../figures/figure_5_3.pdf')
    os.system("echo \"all done\" | mutt -a \"../figures/figure_5_3.eps\" -s \"Plot\" -- alexsusemihl@gmail.com")
