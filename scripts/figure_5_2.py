

import prettyplotlib as ppl
from prettyplotlib import plt

import numpy

T = 20.
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
    [ppl.plot(ts,Xs[i,:,0],ax=ax) for i in range(NSamples)]

def sample(X0, A, H, dt, N, NSamples):
    Xs = numpy.zeros((NSamples,N,2))
    Xs[:,0] = X0 
    for i in range(1,N):
        for j in range(NSamples):
            Xs[j,i] = Xs[j,i-1] +dt*numpy.dot(A,Xs[j,i-1])\
                    + numpy.sqrt(dt)*numpy.dot(H, numpy.random.normal(size=(2,)))
    ts = numpy.arange(0.0,N*dt,dt)
    return ts, Xs

if __name__=='__main__':
    font = {'size':8}
    plt.rc('font',**font)
    plt.rcParams['text.usetex']=True

    fig, ax =ppl.subplots(2,2)
    axou = ax[0][0]
    axunder = ax[0][1]
    axcrit = ax[1][0]
    axover = ax[1][1]

    N =int(T/dt)

    X0 = numpy.zeros((2,))
    A_ou = numpy.array([[-1.0,0.0],[0.0,-1.0]])
    H_ou = numpy.array([[0.2,0.0],[0.0,0.6]])
    sample_and_plot(X0,A_ou,H_ou,dt,N,1,axou)
    axou.set_title("Ornstein-Uhlenbeck")
    
    X0 = numpy.ones((2,))
    A_under = numpy.array([[0.0,1.0],[-2.0,-2.0]])
    H_under = numpy.array([[0.0,0.0],[0.0,1.5]])
    ts,Xunder2 = sample(X0,A_under, H_under, dt, N, 1)
    ts,Xunder1 = sample(X0,numpy.array([[0.0,1.0],[-1.0,-1.0]]), 0.5*H_under, dt,N,1)
    ts,Xunder4 = sample(X0,numpy.array([[0.0,1.0],[-4.0,-4.0]]), 2.0*H_under, dt,N,1)
    ppl.plot(ts, Xunder1[0,:,0], label=r'$\omega$ = 1.0', ax=axunder)
    ppl.plot(ts, Xunder2[0,:,0], label=r'$\omega$ = 2.0', ax=axunder)
    ppl.plot(ts, Xunder4[0,:,0], label=r'$\omega$ = 4.0', ax=axunder)
    p = ppl.legend(axunder)
    fr = p.get_frame()
    fr.set_alpha(0.4)
    axunder.set_title("Underdamped")

    A_crit = numpy.array([[0.0,1.0],[-2.0,-4.0]])
    H_crit = numpy.array([[0.0,0.0],[0.0,1.5]])
    ts,Xcrit2 = sample(X0,A_crit, H_crit, dt, N, 1)
    ts,Xcrit1 = sample(X0,numpy.array([[0.0,1.0],[-1.0,-2.0]]), 0.5*H_crit, dt,N,1)
    ts,Xcrit4 = sample(X0,numpy.array([[0.0,1.0],[-4.0,-8.0]]), 2.0*H_crit, dt,N,1)
    ppl.plot(ts, Xcrit1[0,:,0], label=r'$\omega$ = 1.0', ax=axcrit)
    ppl.plot(ts, Xcrit2[0,:,0], label=r'$\omega$ = 2.0', ax=axcrit)
    ppl.plot(ts, Xcrit4[0,:,0], label=r'$\omega$ = 4.0', ax=axcrit)
    p = ppl.legend(axcrit)
    fr = p.get_frame()
    fr.set_alpha(0.4)
    axcrit.set_title("Critical")

    A_over = numpy.array([[0.0,1.0],[-2.0,-5.0]])
    H_over = numpy.array([[0.0,0.0],[0.0,1.5]])
    ts,Xover2 = sample(X0,A_over, H_over, dt, N, 1)
    ts,Xover1 = sample(X0,numpy.array([[0.0,1.0],[-1.0,-2.5]]), 0.5*H_over, dt,N,1)
    ts,Xover4 = sample(X0,numpy.array([[0.0,1.0],[-4.0,-10.0]]), 2.0*H_over, dt,N,1)
    ppl.plot(ts, Xover1[0,:,0], label=r'$\omega$ = 1.0', ax=axover)
    ppl.plot(ts, Xover2[0,:,0], label=r'$\omega$ = 2.0', ax=axover)
    ppl.plot(ts, Xover4[0,:,0], label=r'$\omega$ = 4.0', ax=axover)
    p = ppl.legend(axover)
    fr = p.get_frame()
    fr.set_alpha(0.4)
    axover.set_title("Overdamped")

    axover.set_xlabel('Time [s]')
    axcrit.set_ylabel('Position [cm]')
    axou.set_ylabel('Position [cm]')
    axcrit.set_xlabel('Time [s]')

    plt.gcf().suptitle('Linear Stochastic Processes')

    plt.savefig('../figures/figure_5_2.eps')
    plt.savefig('../figures/figure_5_2.pdf')

