import numpy

import prettyplotlib as ppl
from prettyplotlib import plt

def drift(x,eq):
    return 4.0*x*(eq-x**2)

def bistablesample(x0,eq,eta,T,dt,NSamples):
    xs = numpy.zeros((NSamples,T))
    xs[:,0] = x0
    s_dt = numpy.sqrt(dt)
    for i in range(1,T):
        rands = numpy.random.normal(0,1,(NSamples,))
        xs[:,i] = xs[:,i-1]+dt*drift(xs[:,i-1],eq) + s_dt*eta*rands
    return xs


if __name__=='__main__':

    samples = bistablesample(0.0,1,0.9,300000,0.001,1)
    ts = numpy.arange(0.0,30000*0.001,0.001)
    print ts.shape
    print samples.shape

    fig, ax = ppl.subplots(1)
    ppl.plot(ts,samples,ax=ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [cm]')
    plt.savefig('../figures/figure_5_10.eps')
    plt.savefig('../figures/figure_5_10.png',dpi=300)
    plt.show()
