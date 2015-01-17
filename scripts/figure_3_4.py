"""
Solve the integral delayed equation

C(d) = K_0(d) - l/(a**2+C(0)) * \int_0^infinity C(s)C(s+d)ds

which gives the filtering kernel in the mean-field
approximation. The equation is solved by discretizing
C over a range from 0 to B, and assuming C = 0 for 
d>B. Then everything is done numerically. AWESOMES!
"""

import numpy
import prettyplotlib as ppl
from prettyplotlib import plt

def SquaredDistance(f,g):
    """
    Squared distance between two functions
    discretized as arrays.
    """
    assert f.shape == g.shape
    dist = 0.0
    for i,j in zip(f,g):
        dist += (i-j)**2
    return dist

def IterateOnce(C, K0, dx, la = 1.0, alpha=1.0):
    """
    Iterate the integral equation once
    """
    newC = numpy.zeros_like(C)
    for i in range(newC.size):
        Cdelta = numpy.roll(C,i)
        Cdelta[:i] = 0.0
        newC[i] -= la*numpy.sum(C*Cdelta)*dx/(alpha**2+C[0])
        newC[i] += K0(i*dx)
    return newC

if __name__=="__main__":
    k = 1.0
    K_rbf = lambda x : numpy.exp(-x**2/(2.0*k**2))
    K_matern = lambda x : (1.0+numpy.sqrt(3.0)*numpy.abs(x)/k)*numpy.exp(-numpy.sqrt(3.0)*numpy.abs(x)/k)
    K_ou = lambda x : numpy.exp(-numpy.abs(x)/k)
    dx = 0.001
    xs = numpy.arange(0.0,12000*dx,dx)
    rbf0 = K_rbf(xs)
    rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha=0.1)
    dist = SquaredDistance(rbf0,rbf1)
    while dist > 1e-10:
        rbf0 = rbf1
        rbf1 = IterateOnce(rbf0, K_rbf, dx,alpha =0.1)
        dist = SquaredDistance(rbf0,rbf1)
        print dist

    matern0 = K_matern(xs)
    matern1 = IterateOnce(matern0, K_matern, dx,alpha=0.1)
    dist = SquaredDistance(matern0,matern1)
    while dist > 1e-10:
        matern0 = matern1
        matern1 = IterateOnce(matern0, K_matern, dx,alpha =0.1)
        dist = SquaredDistance(matern0,matern1)
        print dist

    OU0 = K_ou(xs)
    OU1 = IterateOnce(OU0, K_ou, dx,alpha=0.1)
    dist = SquaredDistance(OU0,OU1)
    while dist > 1e-10:
        OU0 = OU1
        OU1 = IterateOnce(OU0, K_ou, dx,alpha =0.1)
        dist = SquaredDistance(OU0,OU1)
        print dist

    fig, (ax1,ax2,ax3) = ppl.subplots(3,1)
    #ax1 = plt.subplot(311)
    #ax2 = plt.subplot(312)
    #ax3 = plt.subplot(313)
    plt.gcf().suptitle("Posterior kernels")

    xs = xs[:6000]
    OU1 = OU1[:6000]
    matern1 = matern1[:6000]
    rbf1 = rbf1[:6000]

    ppl.plot(xs,K_ou(xs),label=r'OU prior kernel',ax=ax1)
    ppl.plot(xs,OU1,label = r'OU posterior kernel',ax=ax1)
    ppl.legend(ax1)
    ppl.plot(xs,K_matern(xs),label=r'Matern prior kernel',ax=ax2)
    ppl.plot(xs,matern1, label=r'Matern posterior kernel',ax=ax2)
    ppl.legend(ax2)
    ppl.plot(xs,K_rbf(xs),label=r'RBF prior kernel',ax=ax3)
    ppl.plot(xs,rbf1,label=r'RBF posterior kernel',ax=ax3)
    ppl.legend(ax3)

    plt.savefig("../figures/figure_3_4.eps")
