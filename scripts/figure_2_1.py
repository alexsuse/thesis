'''
Plot a counting Poisson process
'''

import prettyplotlib as ppl
from prettyplotlib import plt

import numpy

T = 10.
dt = 0.001

plt.figure(figsize=(6,5))

Ns = numpy.zeros((int(T/dt),3))
la = [0.5,1.0,2.]

for i in range(3):
    for n in range(Ns.shape[0]):
        if numpy.random.rand()<la[i]*dt:
            Ns[n,i] = Ns[n-1,i]+1
        else:
            Ns[n,i] = Ns[n-1,i]

plt.plot( numpy.arange(0.0,dt*Ns.shape[0],dt), Ns, linewidth=1.5 )
legends=[]
for i in la:
    legends.append(r'$\lambda = %lf$'%i)
plt.legend(legends,loc=2)
plt.savefig('../figures/figure_2_1.png',dpi=300)
