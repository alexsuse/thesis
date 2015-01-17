import numpy
import prettyplotlib as ppl
from prettyplotlib import plt

def rate(x,theta,alpha):
    return numpy.exp(-(x-theta)**2/(2*alpha**2))

def generate_spike_train(X,dt,theta,alpha):
    spikes = numpy.zeros_like(X)
    for i,x in enumerate(X):
        if numpy.random.rand() < rate(x,theta,alpha)*dt:
            spikes[i] = 1
    return spikes

def make_OU(N,dt,gamma,sigma):
    ou = numpy.zeros(N)
    sqdt = numpy.sqrt(dt)*sigma
    for i in range(N):
        ou[i] = ou[i-1]*(1.0-gamma*dt) + sqdt*numpy.random.normal()
    return ou

if __name__=="__main__":
    ou = make_OU(50000,0.001,0.5,1.0)

    spike_1 = generate_spike_train(ou,0.001,0.5,1.0)
    spike_times_1 = numpy.where(spike_1==1)
    spike_2 = generate_spike_train(ou,0.001,0.5,0.2)
    spike_times_2 = numpy.where(spike_2==1)

    font = {'size':18}
    plt.rc('font',**font)

    ax1 = ppl.subplot2grid((2,4),[0,0],colspan=3)
    ax2 = ppl.subplot2grid((2,4),[1,0],colspan=3)
    ax3 = ppl.subplot2grid((2,4),[0,3],sharey=ax1)
    ax4 = ppl.subplot2grid((2,4),[1,3],sharey=ax2)

    times = 0.001*numpy.array(range(ou.size))

    ppl.plot(times,ou,ax=ax1)
    ppl.plot(times[spike_times_1],numpy.ones_like(spike_times_1).ravel(),'o',ax=ax1)
    
    ppl.plot(times,ou,ax=ax2)
    ppl.plot(times[spike_times_2],numpy.ones_like(spike_times_2).ravel(),'o',ax=ax2)

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [cm]')
    ax1.set_ylabel('Position [cm]')

    xs = numpy.arange(numpy.min(ou),numpy.max(ou),0.001)
    rates1 = rate(xs,0.5,1.0)
    rates2 = rate(xs,0.5,0.1)

    ppl.plot(rates1,xs,ax=ax3)
    ppl.plot(rates2,xs,ax=ax4)

    ppl.plt.gcf().suptitle('Precision and Frequency Trade-off in Poisson Processe')

    ax4.set_xlabel('Rate of neuron')

    ppl.plt.show()

    ppl.plt.show()
    ppl.plot(rates1,xs,ax=ax3)
