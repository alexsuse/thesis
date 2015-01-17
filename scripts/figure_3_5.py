import numpy
from prettyplotlib import plt

if __name__=='__main__':
    xs = numpy.arange(-5.0,5.0,0.001)
    def f(x):
        return 1.0*numpy.exp(-x**2/2)

    def fdash(x):
        return -x*numpy.exp(-x**2/2)

    
    tuning = f(xs)
    info = fdash(xs)**2/f(xs)

    plt.plot(xs,tuning,label='Tuning function')
    plt.plot(xs,info,label='Fisher Information')
    plt.gca().set_xlabel('x')
    plt.legend()

    plt.savefig('../figures/figure_3_5.eps')
