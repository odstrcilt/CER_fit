from numpy import *
import MDSplus
from IPython import embed
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

shot = 190549
mdsserver = 'localhost'

MDSconn = MDSplus.Connection(mdsserver)
from matplotlib.pylab import *

data  = load('spred_line_190549_2.06s_30.00nm.npz')
data = load('spred_line_190551_2.00s_35.18nm.npz')
shot = 190551
#data = load('spred_line_190550_1.94s_35.19nm.npz')
#shot = 190550
#data = load('spred_line_192784_2.01s_34.33nm.npz')
#shot =  192784


MDSconn.openTree('NB',  shot)  

_load_beams = '30R','30L' ,'330L','330R', '210L','210R','150L','150R'
#_load_beams = '30R','30L'

paths = ['\\NB::TOP.NB{0}:'.format(b[:2]+b[-1]) for b in _load_beams] 

TDI = [p+'pinj_scalar' for p in paths] 
pinj_scal = MDSconn.get('['+','.join(TDI)+']')
fired = pinj_scal > 1e3

pinj_scal = np.array(pinj_scal)[fired]
_load_beams = np.array(_load_beams)[fired]
paths = np.array(paths)[fired]

TDI  = [p+'PINJ'+'F_'+p[-4:-1] for p in paths]
TDI += ['dim_of('+TDI[0]+')']

pow_data = list(MDSconn.get('['+','.join(TDI)+']').data())
tvec_nbi = pow_data.pop(-1)/1e3

integ_pow_data = cumtrapz(array(pow_data), tvec_nbi, axis =1, initial = 0)
integ_pow_data = interp1d(tvec_nbi, integ_pow_data)

pnbi  = (integ_pow_data(data['tvec'])-integ_pow_data(data['tvec']-0.002)).T
pnbi  /= 0.002*1e6

_load_beams = [b for b,p in zip(_load_beams,pnbi.T)  if sum(p > 0.1) > 10]
pnbi = array([p for p in pnbi.T if sum(p > 0.1) > 10]).T





#embed()
nt = len(data['tvec'])
X =  np.vstack((np.ones(nt),pnbi.T) ) 
D =  eye(nt)-eye(nt,k=-1)
lam  = 0.1
from scipy.signal import savgol_filter
from scipy.optimize import nnls
C = np.ones(len(X))*100
N_ = 1
for i in range(10):
    N = maximum(0.1, data['data'][:,0]-N_*dot(X[1:].T,C[1:]))
    lnN = log(N)
    N_ = exp(savgol_filter(log(N), 101, 2, mode='nearest'), )
    plot(data['data'][:,0]/N_ )
   # plot(X.T)
    C =  nnls(X.T,data['data'][:,0]/N_ )[0]
    plot(dot(X[1:].T,C[1:]))
    #print(C)
    #show()
    
#embed()
prof = []
for i in arange(2050, 3500, 50):
    j,k = data['tvec'].searchsorted([i/1e3,(i+50)/1e3])
    print(data['data'][j:k,0].shape)
    prof.append(data['data'][j:k,0])
    
step(data['tvec'],data['data']/1e6 ,'-')
xlabel('time [s]')
ylabel('SPRED [a.u.]')
ylim(0,None)
title('Passive SPRED Ca17+ line at 34.33nm in discharge 192784')
title('Passive SPRED Ar16+ line at 35.18nm in discharge 190551')

print(C)
f,ax = subplots(2,2,sharex='col')
f.suptitle('Passive Ca17+ line')
#ax[0].set_title('Passive Ar15+ line')

ax[0,0].step(data['tvec'],C[0]*N_/1e6 ,'-',label='offset')
ax[0,0].step(data['tvec'],N_*dot(X.T,C)/1e6 ,'-',label='fit')
p,=ax[0,0].step(data['tvec'],data['data']/1e6 ,'-')
ax[0,0].errorbar(data['tvec']-0.001,data['data']/1e6,data['error']/1e6,label='raw data',ls='none',c=p.get_color())
ax[0,0].legend(loc='best')
ax[0,0].set_ylim(0,None)
isort = argsort(C[1:])[::-1]
ax[1,0].step(data['tvec'],array(pnbi)[:,isort])
ax[1,0].legend( array(_load_beams)[isort])
ax[1,0].set_ylim(0,None)

ax[1,1].bar(_load_beams, C[1:])
ax[1,0].set_xlabel('time [s]')
ax[1,0].set_ylabel('NBI power [MW]')
ax[0,0].set_ylabel('SPRED + fit [a.u.]')
ax[1,1].set_ylabel('Beam contribution')
show()
    
    
    
figure()
plot(data['data'][:,0])
plot(N_)
plot(N_ +N*dot(X[1:].T,C[1:]) )

figure()

show()
embed()



    



dt = 0.1
step = int(dt/0.002)
A, B,C, T = [],[],[],[]
for i in range(len(data['tvec'])-step):
    X = np.vstack((np.ones(step), pnbi[i:i+step].T)).T
    x,r,rr,s = linalg.lstsq(X,data['data'][i:i+step],rcond=0.00001)
    A.append(x[0])
    C.append(x[1:])
    B.append(dot(X,x)[step//2])
    T.append(data['tvec'][i+step//2])

f,ax = subplots(3,1,sharex=True)
ax[0].set_title('Passive Ca17+ line')
#ax[0].set_title('Passive Ar15+ line')

ax[0].step(T,array(A)/1e6 ,'-',label='offset')
ax[0].step(T,array(B)/1e6 ,'-',label='fit')
p,=ax[0].step(data['tvec'],data['data']/1e6 ,'-')
ax[0].errorbar(data['tvec']-0.001,data['data']/1e6,data['error']/1e6,label='raw data',ls='none',c=p.get_color())
ax[0].legend(loc='best')
ax[0].set_ylim(0,None)
isort = argsort(array(C).mean(0)[:,0])[::-1]
ax[1].step(data['tvec'],array(pnbi)[:,isort])
#ax[1].plot(tvec_nbi, array(pow_data).T/1e6)


ax[2].step(T,array(C)[:,isort,0])
print(array(C).mean(0)[isort,0])
print(array(_load_beams)[isort])
ax[2].legend( _load_beams[isort])

ax[1].set_ylim(0,None)
ax[-1].set_xlabel('time [s]')
ax[1].set_ylabel('NBI power [MW]')
ax[0].set_ylabel('SPRED + fit [a.u.]')
show()
    


A = n*(A+B*P)


name  = 'spred_line_190549'

files = ['spred_line_190549_2.20s_19.30nm.npz',
         'spred_line_190549_2.15s_30.00nm.npz',
         'spred_line_190549_2.18s_34.26nm.npz',
         'spred_line_190549_2.20s_16.43nm.npz']

name  = 'spred_line_192785_3.5s'
files = ['spred_line_192785_3.47s_19.31nm.npz',
        'spred_line_192785_3.47s_30.07nm.npz',
        'spred_line_192785_3.47s_34.33nm.npz',
        'spred_line_192785_3.49s_16.45nm.npz',]



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



    
