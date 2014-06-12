

import prettyplotlib as ppl
from prettyplotlib import plt

import numpy

T = 10.
dt = 0.001

plt.figure(figsize=(6,5))

def sample_and_plot(X0, A, H, dt, N, NSamples, ax):
    Xs = numpy.zeros((NSamples,N,2))
    Xs[:,0] = X0 
    for i in range(1,N):
        for j in range(NSamples):
            Xs[j,i] = Xs[j,i-1] +dt*numpy.dot(A,Xs[j,i-1])\
                    + numpy.sqrt(dt)*numpy.dot(H, numpy.random.normal(size=(2,)))
    ts = numpy.arange(0.0,N*dt,dt)
    [ax.plot(ts,Xs[i,:,0]) for i in range(NSamples)]

if __name__=='__main__':
    font = {'size':8}
    plt.rc('font',**font)
    plt.rcParams['text.usetex']=True

    X0 = numpy.zeros((2,))
    A_ou = numpy.array([[-1.0,0.0],[0.0,-1.0]])
    H_ou = numpy.array([[0.2,0.0],[0.0,0.6]])
    axou = plt.subplot(221)
    sample_and_plot(X0,A_ou,H_ou,0.001,10000,4,axou)
    axou.set_title("Ornstein-Uhlenbeck")
    
    X0 = numpy.ones((2,))
    A_under = numpy.array([[0.0,1.0],[-2.0,-2.0]])
    H_under = numpy.array([[0.0,0.0],[0.0,0.8]])
    axunder = plt.subplot(222)
    sample_and_plot(X0,A_under,H_under,0.001,10000,10,axunder)
    axunder.set_title("Underdamped")

    A_crit = numpy.array([[0.0,1.0],[-2.0,-4.0]])
    H_crit = numpy.array([[0.0,0.0],[0.0,0.8]])
    axcrit = plt.subplot(223)
    sample_and_plot(X0,A_crit,H_crit,0.001,10000,10,axcrit)
    axcrit.set_title("Critical")

    A_over = numpy.array([[0.0,1.0],[-2.0,-5.0]])
    H_over = numpy.array([[0.0,0.0],[0.0,0.8]])
    axover = plt.subplot(224)
    sample_and_plot(X0,A_over,H_over,0.001,10000,10,axover)
    axover.set_title("Overdamped")

    axover.set_xlabel('Time [s]')
    axcrit.set_ylabel('Position [cm]')
    axou.set_ylabel('Position [cm]')
    axcrit.set_xlabel('Time [s]')

    plt.gcf().suptitle('Linear Stochastic Processes')

    plt.savefig('../figures/figure_5_2.eps')

