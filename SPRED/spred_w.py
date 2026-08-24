import MDSplus
from numpy import *
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import numpy as np
import lbo_helpers
connection = MDSplus.Connection('localhost')
shots  = [190422, 190424, 190426, 190430,  190491,  190453, 190456, 190459, 190481, 190484]
lmin,lmax = 13.7, 16.2
from IPython import embed

for j, shot in enumerate(shots):
    print(shot)
    spectrum,time,  lam  = lbo_helpers.get_spred_data(shot,1,connection )
    l1 = argmin(abs(lam-lmin))
    l2 = argmin(abs(lam-lmax))
    l = l1-l2
    ind = (time > 1.5)&(time  < 6)
    spectrum = spectrum[ind].mean(0)
    for i in range(4):
        spectrum[l2+i:l1:4]-=spectrum[l2+i:l1:4].mean()
    
    
    
    ls = '-' if j < 5 else '--'
    
    
    plot(lam[l2:l1], spectrum[l2:l1]-median(spectrum[l2:l1]), label=shot, ls=ls)
legend()
xlabel('lambda (A)')
ylabel('SPRED spectrum')
show()
