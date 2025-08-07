import numpy as np
from matplotlib import pyplot as plt

def h_asc(fr, to):
    return np.cos(2*np.pi*(to - fr))

def h_dsc(fr, to):
    return np.cos(2*np.pi*(to - fr + 0.5))

oscils = [0, 0, 0, 0]
freqs = [1, 1, 1, 1]

toplot = [[i for i in oscils]]

dt = 0.001
# seq(0, 10, dt)
ts = np.arange(0, 10, dt)
for t in ts[1:]:
    oscils[0] += (freqs[0] + h_dsc(oscils[1], oscils[0])) * dt
    for i in range(1, len(oscils) - 1):
        oscils[i] += (freqs[i] + h_asc(oscils[i-1], oscils[i]) + h_dsc(oscils[i+1], oscils[i])) * dt
    oscils[-1] += (freqs[-1] + h_asc(oscils[-2], oscils[-1])) * dt
    toplot.append([i for i in oscils])

#for i in range(10):
#    print([(float(j)) for j in toplot[i]])

for i in range(len(oscils)):
    plt.plot(ts, [toplot[j][i] for j in range(len(toplot))])
plt.show()

