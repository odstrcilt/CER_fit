from numpy import *
import MDSplus
from IPython import embed

shot = 190549
#mdsserver = 'localhost'

#MDSconn = MDSplus.Connection(mdsserver)
from matplotlib.pylab import *

#data  = load('spred_line_190549_2.06s_30.00nm.npz')
#data = load('spred_line_190551_2.00s_35.18nm.npz')
#shot = 190551
#data = load('spred_line_190550_1.94s_35.19nm.npz')
#shot = 190550

 
 

name  = 'spred_line_190549'

files = ['spred_line_190549_2.20s_19.30nm.npz',
         'spred_line_190549_2.15s_30.00nm.npz',
         'spred_line_190549_2.18s_34.26nm.npz',
         'spred_line_190549_2.20s_16.43nm.npz']

#name  = 'spred_line_192785_3.5s'
#files = ['spred_line_192785_3.47s_19.31nm.npz',
        #'spred_line_192785_3.47s_30.07nm.npz',
        #'spred_line_192785_3.47s_34.33nm.npz',
        #'spred_line_192785_3.49s_16.45nm.npz',]

name  = 'spred_line_193811_4.0s'
files = ['spred_line_193811_3.89s_19.32nm.npz',
        'spred_line_193811_3.97s_30.10nm.npz',
        'spred_line_193811_3.96s_34.38nm.npz',
        'spred_line_193811_3.93s_16.45nm.npz',]

name  = 'spred_line_193806_4.0s'
files = ['spred_line_193806_3.94s_19.32nm.npz',
        'spred_line_193806_3.98s_30.12nm.npz',
        'spred_line_193806_3.98s_34.34nm.npz',
        'spred_line_193806_3.98s_16.45nm.npz',]

 
#file:///C:/Users/odstrcil/Desktop/DIII-D/diagnostics/SPRED/spred_line_193806_3.94s_19.32nm.npz
#file:///C:/Users/odstrcil/Desktop/DIII-D/diagnostics/SPRED/spred_line_193806_3.98s_16.45nm.npz
#file:///C:/Users/odstrcil/Desktop/DIII-D/diagnostics/SPRED/spred_line_193806_3.98s_30.12nm.npz
#file:///C:/Users/odstrcil/Desktop/DIII-D/diagnostics/SPRED/spred_line_193806_3.98s_34.34nm.npz


#name  = 'spred_line_192785_2.0s'
#files = ['spred_line_192785_1.90s_19.31nm.npz',
         #'spred_line_192785_1.85s_30.06nm.npz',
         #'spred_line_192785_1.88s_34.33nm.npz',
         #'spred_line_192785_2.00s_16.44nm.npz']


#name  = 'spred_line_190549_3.7s'
#files = ['spred_line_190549_3.65s_19.30nm.npz',
         #'spred_line_190549_3.63s_30.00nm.npz',
         #'spred_line_190549_3.66s_34.26nm.npz',
         #'spred_line_190549_3.70s_16.44nm.npz',
        #]



#name  = 'spred_line_192784_3.5s'
#files = ['spred_line_192784_3.45s_19.32nm.npz',
         #'spred_line_192784_3.48s_30.07nm.npz',
         #'spred_line_192784_3.48s_34.34nm.npz',
         #'spred_line_192784_3.49s_16.45nm.npz']

#name  = 'spred_line_192784_2.0s'
#files = ['spred_line_192784_1.97s_19.31nm.npz',
         #'spred_line_192784_1.94s_30.06nm.npz',
        #'spred_line_192784_1.95s_34.33nm.npz',
        #'spred_line_192784_1.99s_16.45nm.npz',
        #]
        
#name  = 'spred_line_190549_2.2s_long'

#files = ['spred_line_190549_2.11s_19.30nm.npz',
        #'spred_line_190549_2.15s_30.00nm.npz',
        #'spred_line_190549_2.17s_34.26nm.npz',
        #'spred_line_190549_2.20s_16.43nm.npz',
        #]




#name  = 'spred_line_190549_3.7s_long'
#files = [
#'spred_line_190549_3.62s_19.30nm.npz',
#'spred_line_190549_3.64s_30.00nm.npz',
#'spred_line_190549_3.63s_34.26nm.npz',
#'spred_line_190549_3.70s_16.44nm.npz']




    #16        192.90     500       ca17
    #16        357.34     500       ca17
    #16        222.86     500       ca17
    #16        382.03     500       ca17
    #17        302.22     500       ca18
    #17        344.80     500       ca18
    #17        124.86     500       ca18
    #17        229.93     500       ca18
    #18        112.05     500       ca19
    #18        206.34     500       ca19
    #19        186.2      500       ca20
    #19        101.1      500       ca20
    #19        308.88     500       ca20
    
    
lines = ['CaXVII_193','CaXVIII_300','CaXVIII_343','CaXVI_164']


#name  = 'spred_line_190553'
#files = ['spred_line_190549_2.20s_19.30nm.npz',
         #'spred_line_190549_2.15s_30.00nm.npz',
         #'spred_line_190549_2.18s_34.26nm.npz',
         #'spred_line_190549_2.20s_16.43nm.npz']

#files = ['spred_line_190553_2.60s_97.54nm.npz',
#'spred_line_190553_2.60s_18.23nm.npz']


#lines = ['CIII_977','CVI_182']


#files =['spred_line_190551_1.62s_22.12nm.npz',
        #'spred_line_190551_1.49s_35.18nm.npz',
        #'spred_line_190551_1.58s_38.70nm.npz']
#lines = ['ArXV_221','ArXVI_354','ArXVI_387']
#name  = 'spred_line_190551'



#name  = 'spred_line_196511_1.4s_long'


#files = ['spred_line_196551_1.39s_54.89nm.npz',
         #'spred_line_196551_1.47s_56.69nm.npz',
         #'spred_line_196551_1.55s_33.13nm.npz']


#lines = ['AlXI_549','AlXI_567','AlX_331']


tvec =  []
data = []
error = []
for f in files:
    f = load(f)
    tvec.append(f['tvec'])
    data.append(f['data'])
    error.append(f['error'])
    

tmin = min([t[0] for t in tvec])
tmax = max([t[-1] for t in tvec])
dt = (tvec[0][-1]-tvec[0][0])/(tvec[0].size-1)

T = single(arange(tmin,tmax,dt))
 
data  = single([interp(T,t,d[:,0],0,0) for t,d in zip(tvec, data)]).T
error = single([interp(T,t,e[:,0],inf,inf) for t,e in zip(tvec, error)]).T
 
savez_compressed(name,data=data, error=error, tvec=T, lines=lines)



    
