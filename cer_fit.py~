from numpy import *
from matplotlib.pylab import *
#ion()
import time 
from IPython import embed
#from scipy.io import readsav
#shot = 189879
#shot = 190549
#beam = '30L'
#lbo = { 189880: [3798.88022396, 3898.86022795, 3998.84023195, 4098.82023595 ,4198.80023995, 4298.78024395 ,4398.76024795 ,4498.74025195 ,4598.72025595] ,
        #189879: [1999.4, 2999.2, 3499.1, 3899.1, 4199.0, 4399.0, 4498.9, 4598.9]}
#from matplotlib.widgets import MultiCursor
#import MDSplus
#from scipy.interpolate import interp1d
#TODO do background substraction right for the first phase
#try:
    ##ww

    #beams = load('beams_%d.npz'%shot)
    #pow_tvec = beams['tvec']
    ##pow_data = beams[beam]
    ##plot(pow_tvec, beams['30L'])
    ##plot(pow_tvec, beams['33L'])
    ##show()
    
#except:
    

        
    #mdsserver = 'localhost'
    #MDSconn = MDSplus.Connection(mdsserver)
            
    #MDSconn.openTree('NB',  shot)  

    #_load_beams = ['30L','33L']
    #paths = ['\\NB::TOP.NB{0}:'.format(b[:2]+b[-1]) for b in _load_beams] 
    #s = ''
    #TDI  = [p+'PINJ'+s+'_'+p[-4:-1] for p in paths]
    #TDI += ['dim_of('+TDI[0]+')']
    ##TDI  = [p+'PTDATA_CAL' for p in paths]
    ##TDI += ['dim_of('+TDI[0]+')']
    #beams = list(MDSconn.get('['+','.join(TDI)+']').data())
    #pow_tvec = beams.pop()
    #beams = {b:d for b,d in zip(_load_beams, beams)}
    
    #savez_compressed('beams_%d.npz'%shot, tvec = pow_tvec, **beams  )
    
    ##pow_data = pow_data[beam]
    


#plot(tvec, pow_data[0])
#plot(tvec, pow_data[1])

#show()
#lbo = lbo[shot]
#channels= ['V01', 'V02', 'V03', 'V04', 'V05', 'V06', 'V07', 'V08', 'V09', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31'] 
#channels= [ 'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'T32', 'T33', 'T34', 'T35', 'T36', 'T37', 'T38', 'T39', 'T40', 'T41', 'T42', 'T43', 'T44', 'T45', 'T46', 'T47']


#channels= [ 'T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07', 'T08', 'T09', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'T25', 'T26', 'T27', 'T28', 'T29', 'T30', 'T31', 'T32', 'T33', 'T34', 'T35', 'T36', 'T37', 'T38', 'T39', 'T40', 'T41', 'T42', 'T43', 'T44', 'T45', 'T46', 'T47']
#channels= ['T56','T55','T54','T53','T52','T51','T50','T49','T36','T35','T34','T33'][::-1]
from scipy.optimize import least_squares
import numexpr as ne

#G = A/(s*2*sqrt(2*pi))*exp(-(x-x0)**2/s**2/2)+B

def gauss(par,x,y=0,y_err=1):
    B,par = par[0], par[1:]
 
    G = np.zeros_like(x) + B
    for A,s,x0 in reshape(par,(-1,3)):
        G +=  (abs(A)/(s*sqrt(2*pi)))*exp(-((x-x0)/abs(s))**2/2)
 
    G-= y
    G/= y_err
    return G

def gauss2(par,x,y=0,y_err=1):
    A,B,s, z = par
    s = abs(s)
    A = abs(A)
    
    G =  (A/(s*sqrt(2*pi)))*exp(-((x-z)/s)**2/2)+B
    G-= y
    G/= y_err
    return G


def jacob(par,x,y=0,y_err=1):
    J = np.empty(( len(par), len(x)))
    
    B,par = par[0], par[1:]
    A,s,x0 = reshape(par,(-1,3)).T
 
    J[0] = 1
    for i,(A,s,x0) in enumerate(reshape(par,(-1,3))):
        s = abs(s)
        A = abs(A)
        E = exp(-((x-x0)/s)**2/2)/(s*sqrt(2*pi))
 
        J[i*3+1] = E
        J[i*3+2] = (-A/s**3)*(s**2-(x-x0)**2)*E
        J[i*3+3] = (A/s**2)*(x-x0)*E
    
    J/= y_err
    
    return J.T


def jacob2(par,x,y=0,y_err=1):
    B,A,s,x0 = par
    s = abs(s)
    A = abs(A)

    E = exp(-((x-x0)/s)**2/2)/(s*sqrt(2*pi))
    
    J = np.empty(( len(par), len(x)))
    J[0] = 1
    J[1] = E
    J[2] = (-A/s**3)*(s**2-(x-x0)**2)*E
    J[3] = (A/s**2)*(x-x0)*E
    
    J/= y_err
    
    return J.T




def calc_jac(f,x0,x):
    y0 = f(x0, x)
    dx = 1e-6
    J = zeros((len(y0), len(x0)))
    J[:,:] = -y0[:,None]
    
    for i in range(len(x0)):
        x0[i] += dx
        J[:,i] += f(x0, x)
        x0[i] -= dx
        
    return J/dx

def nnmf_fit(data,rank=2):

    V1 = data.mean(0)
    V1 -= V1.min()
    V = np.vstack((V1, ones_like(V1)))

    #from IPython import embed
    #embed()
    H_ = np.linalg.lstsq(V.T, data.T, rcond=None)[0]
    H_*= V.sum(1)[:,None]
 

    return H_

    import nimfa
    from IPython import embed
    embed()
    pcolormesh(data,vmax=20,vmin=0 );colorbar();show()


    nmf = nimfa.Snmf(data , seed="random_c", rank=rank, 
                 max_iter=20, update='euclidean',
                objective='fro', version='l',eta=0.1, beta=1e-5,
                i_conv=10, w_min_change=0,n_run=100)

    nmf_fit = nmf()


    H=nmf_fit.basis()
    W=nmf_fit.coef()

    plot(W.T)
    show()


    H_ = np.linalg.lstsq(W.T, data.T, rcond=None)[0].T
    plot(H_)
    plot(H)
    show()



def LocWidthFixedFit(data, bckg,wav_vec,p0=None,ph_per_count=1,readout_noise=0,
		lpix=0,upix=None,tsmin=0,tsmax=None):

    #embed()
    selected = slice(lpix,upix)
    data = (data[tsmin:tsmax,selected])
    bckg = bckg[tsmin:tsmax,selected]
    wav_vec = wav_vec[selected]
    if size(readout_noise) > 1:
        readout_noise = readout_noise[selected]
    if size(ph_per_count) > 1:
        ph_per_count = ph_per_count[selected]
    
    nw = data.shape[1]
    
    
    #initial guess
    if p0 is None:
        mdata = (data-bckg).mean(0)
    
        B = mdata.min()
        #use robust statistics??
      
        A = trapz(mdata-B, -wav_vec)
        x0 = trapz((mdata-B)*wav_vec, -wav_vec)/A
        s0 = sqrt(trapz((mdata-B)*(wav_vec-x0)**2, -wav_vec)/A)
        p0 = [B,A,s0,x0]
    else:
        [B,A,s0,x0] = p0

    def cost_fun(par,x, y, return_results=False):
        x0,s = par 

        gauss = exp(-((x-x0)/s)**2/2)/(s*sqrt(2*pi))
        background = np.ones_like(x)
        A = np.vstack((gauss,background )).T
        coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)
        if return_results:
            chi2_dof = r/(len(x)-rr)
            err = np.sqrt(np.linalg.inv(np.dot(A.T,A))[0,0]*chi2_dof)
            return coeff.T, np.dot(A, coeff).T, err
        return r
    
    nw = len(wav_vec)
    out = least_squares(cost_fun, (x0,s0),args=( wav_vec,data- bckg ), 
                        bounds=((wav_vec.min(), 0),(wav_vec.max(),  np.ptp(wav_vec))))    #hardcoded maximal and mininal line width 

    coeff,model,err = cost_fun(out.x,wav_vec, data- bckg ,return_results=True)
    return coeff.T,model,err.T
 

def fast_recursive_fit(data, bckg,wav_vec,p0=None,ph_per_count=1,readout_noise=0,
		lpix=0,upix=None,tsmin=0,tsmax=None):
    
    #embed()
    selected = slice(lpix,upix)
    data = (data[tsmin:tsmax,selected])
    bckg = bckg[tsmin:tsmax,selected]
    wav_vec = wav_vec[selected]
    if size(readout_noise) > 1:
        readout_noise = readout_noise[selected]
    if size(ph_per_count) > 1:
        ph_per_count = ph_per_count[selected]
    
    nw = data.shape[1]
    
    
    #initial guess
    if p0 is None:
        mdata = (data-bckg).mean(0)
    
        B = mdata.min()
        #use robust statistics??
        
        A = trapz(mdata-B, -wav_vec)
        x0 = trapz((mdata-B)*wav_vec, -wav_vec)/A
        s0 = sqrt(trapz((mdata-B)*(wav_vec-x0)**2, -wav_vec)/A)
        p0 = [B,A,s0,x0]
         
    #mdata = (data-bckg).mean(0) 
   # plot(wav_vec, mdata)
   # plot(wav_vec, gauss(p0, wav_vec))
   # show()

   
    p0 = atleast_2d(p0)
    nt = len(data)
    npar = len(p0.T)
    chi2n = zeros(nt)
    success = zeros(nt, dtype='bool')
    data_fit = np.zeros_like(data)
    solution = np.zeros((nt, npar))+p0
    solution_err = np.zeros((nt, npar)) 
    

    step = nt#//len(p0)
    while step > 0:
        d = data[:(nt//step)*step].reshape(-1, step, nw)
        b = bckg[:(nt//step)*step].reshape(-1, step, nw)

        
        d = d.mean(1)
        b = b.mean(1)
         
        outputs = zeros((nt//step, npar))
        _X0 = solution[::step]
        for i in range(nt//step):
            #TODO what to do when the fit fails?
            
            #not calculated correctly, calculate from fit function?
           # print(d.shape, ph_per_count, readout_noise)
            d_err = hypot(maximum(d[i]/ph_per_count,0)**.5, readout_noise)/sqrt(step)
            #d_err[38:41] = infty #BUG
            #d_err = d[i]*0+1om
            
   
            #try:
            out = least_squares(gauss,abs(_X0[i]),jac=jacob, 
                                args=[wav_vec,d[i]-b[i],d_err],max_nfev=50,method='lm' )
            #except:
                #embed()
                
            if out.success:
                outputs[i] = out.x
            else:
                #try again in next iteration
                outputs[i] = _X0[i]



 
  
            if step == 1: #final step
                data_fit[i] = out.fun*d_err+d[i]-b[i]
                chi2n[i] = out.cost/(len(wav_vec)-npar)
                success[i] = out.success
                try:
                    solution_err[i] = sqrt(diag(linalg.pinv(dot(out.jac.T, out.jac)))*chi2n[i])
                except:
                    embed()
            if False:
                print(step,nt,i)
                figure(figsize=(10,5))
                subplot(121)
                suptitle('step: %d'%step)
                data_fit[i] = out.fun*d_err+d[i]-b[i]
                plot(wav_vec,data_fit[i])
                errorbar(wav_vec, d[i]-b[i], d_err)
                title(chi2n[i])
                subplot(122)
                plot(wav_vec,(d[i]-b[i]-data_fit[i])/d_err)
                ylim(-4,4)
                axhline(-1)
                axhline(1)
                
                show()
                #embed()
                
        
        #prepare initial guess for next step
        solution[:(nt//step)*step].reshape(-1, step, npar)[:] = outputs[:,None]
        solution[(nt//step)*step:] = outputs[-1][None]
        step//=2
        
        
    return solution,solution_err,data_fit,chi2n,success

#x0 = [1, 10, 2, 3, 5, 2, -3,]
#x = linspace(-10,10)
#y = 2*gauss(x0,x)+1

#J1 = jacob(x0,x)
#J2 = calc_jac(gauss, x0,x)

#plot(J1)
#plot(J2,'x')
#show()
        
  
    


##channels = ['T04', 'T05', 'T06', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T24', 'V10', 'V12', 'V14']
##channels = ['T18']
#for ch in channels:
    #try:
        #data = readsav('shot%d'%shot+ch+'.sav')['chord_data']
        #spectrum = copy(data.item()[-1])
        #tvec = copy(data.item()[7])[:len(spectrum)]
    #except:
        ##raise
        #continue
    
    ##mdsserver = 'localhost'
    ##MDSconn = MDSplus.Connection(mdsserver)
            
    ##MDSconn.openTree('NB',  shot) 
    ###OMFIT['test']['CER']['CALIBRATION']['TANGENTIAL']['CHANNEL53']['DISPERSION']
    ###print(r'\\IONS::TOP.CER.CALIBRATION.TANGENTIAL.CHANNEL'+ch[1:]+':DISPERSION')
    ##dispersion = MDSconn.get(r'\\IONS::TOP.CER.CALIBRATION.TANGENTIAL.CHANNEL'+ch[1:]+':DISPERSION')

    ##MDSconn.closeTree('NB',  shot) 
    
    #dispersion = 0.24538
    
    #shot, chord, dark_shot, ftype, channels, idx, white, t_start, t_integ, tg, data_grp, bg_grp, wavelength0, comments, shot_time, timing_ref, timing_mode,bg_mode, camera_type, gain, bandwidth, binning, raw, data = data.item()
    ##print(data.shape[0])
    #lam = wavelength0-arange(-data.shape[1]//2,data.shape[1]//2)*dispersion
    
    ##print(ch, wavelength0, gain)
    ##if wavelength0  != 5290.5:
        ##continue
        
    #from scipy.integrate import cumtrapz
    ##plot(spectrum.T)
    ##show()
    
    ##embed()
    #cumpow = cumtrapz(beams['30L'], pow_tvec, initial=0)
    #nbi_30L = (interp(tvec+t_integ[0], pow_tvec,cumpow)-interp(tvec, pow_tvec,cumpow))/t_integ[0]
    #cumpow = cumtrapz(beams['33L'], pow_tvec, initial=0)
    #nbi_33L = (interp(tvec+t_integ[0], pow_tvec,cumpow)-interp(tvec, pow_tvec,cumpow))/t_integ[0]
    
        
    
    #beam_on = (nbi_30L > 1e6)|(nbi_33L > 1e6)
    #beam_off = (nbi_30L < 1e6)&(nbi_33L <  1e6)
    
    ##tvec = t_start 
    ##tind = nbi_power > 1e6
    ##wrange = slice(80,180)
    ##wrange = slice(163,210)
    
    
    ##data = data[tind]
    ##tvec = tvec[tind]
    
    #tvec, data, bckg,pow30L, pow33L = blip_average(spectrum, tvec, nbi_30L, nbi_33L,lam )
    
    
    #wrange = slice(60,155)

    
    #t = time.time()
    #ph_per_count=4.3
    #readout_noise=0.7
    #solution,solution_err,data_fit,chi2n,success = fast_recursive_fit(data[:, wrange],bckg[:, wrange] ,p0=None,ph_per_count=ph_per_count, readout_noise=readout_noise)
    #print(time.time()-t)
    
    ##continue
    ##TODO remove cosmic rays 
    ##embed()
    
    ##plot(data[tind, wrange])
    ##show()
    
    ##imshow(data[:, wrange])
    ##show()
    
    
    ##resid = (data[tind, wrange]-data_fit)[chi2n<10 ]
    ##val = data_fit[chi2n<10  ]
    
    ##plot(sqrt(val), resid,'o')
    ##show()
    ##x,y = [],[]
    ##for i in range(28):
        ##var = std(resid[(val**.5 > i)&(val**.5 < i+1)])
        ##if isfinite(var):
            ##x.append((i+.5)**2)
            ##y.append(var**2)
            
        
     
    ##poly = polyfit(x,y,1)
    ##ph_elect_per_cnt = poly[0]**-.5
    ##readout_noise = poly[1]**.5
    ##plot(linspace(0,1000), polyval(poly, linspace(0,1000)))
    ##title([ph_elect_per_cnt,readout_noise])
    ##plot(x,y)
    ##show()
    

    ##plot((data[tind, wrange]-data_fit)[chi2n > 10].T)
    


    
        
    #A,B,s,x0 = solution.T
    #Aerr,Berr,serr,x0err = solution_err.T
    #s = abs(s)

    #savez('F_fit_'+ch+'_'+str(shot),t=tvec, A = A, Aerr=Aerr )
    ##embed()
    #continue


    
    #f,ax = subplots(2,2,sharex=True)
    #ax[0,0].errorbar(tvec, A , Aerr)
    ##ax[0,0].errorbar(tvec, A/(pow33L/median(pow33L)), Aerr)

    ##ax[0,0].plot(tvec, pow30L/1e3,'--')
    ##ax[0,0].plot(tvec, pow33L/1e3)

    #ax[0,0].plot(tvec[~success], A[~success],'ro')

    ##success
    #ax[0,0].errorbar(tvec, B*10, Berr*10)
    
    #ax[0,1].semilogy(tvec, chi2n)
    ##ax[0,1].semilogy(tvec, optimality)

    #ax[1,0].errorbar(tvec, s, serr)
    #ax[1,1].errorbar(tvec, x0, x0err)
    #ax[0,0].set_ylim(0, nanmax(A))
    #ax[1,0].set_ylim(0, nanmax(s))
    #ax[1,1].set_ylim(nanmin(x0), nanmax(x0))
    
    #f.suptitle(ch)
    
    #multi0 = MultiCursor(f.canvas, ax.flatten(), color='r', lw=1)

    #show()
    
    
    #continue
        
    ##embed()

    ##e
    
    ##data[500, 80:180])
    
    
    ##least_squares()
    
    

    
    ##continue
    
    ##[(('shot', 'SHOT'), '>i4'), (('chord', 'CHORD'), 'O'), (('dark_shot', 'DARK_SHOT'), '>i4'), (('type', 'TYPE'), '>i2'), (('channels', 'CHANNELS'), '>i2'), (('idx', 'IDX'), '>i4'), (('white', 'WHITE'), '>i2'), (('t_start', 'T_START'), 'O'), (('t_integ', 'T_INTEG'), 'O'), (('tg', 'TG'), 'O'), (('data_grp', 'DATA_GRP'), 'O'), (('bg_grp', 'BG_GRP'), 'O'), (('wl', 'WL'), '>f4'), (('comments', 'COMMENTS'), 'O'), (('shot_time', 'SHOT_TIME'), 'O'), (('timing_def', 'TIMING_DEF'), 'O'), (('timing_mode', 'TIMING_MODE'), '>i2'), (('bg_mode', 'BG_MODE'), '>i2'), (('camera_type', 'CAMERA_TYPE'), '>i2'), (('gain', 'GAIN'), '>i2'), (('bandwidth', 'BANDWIDTH'), '>i2'), (('binning', 'BINNING'), '>i2'), (('raw', 'RAW'), 'O'), (('data', 'DATA'), 'O')])
    

    ##figure()
    ##imshow(spectrum, aspect='auto', vmin=0,vmax=10, origin='lower', extent=(0,spectrum.shape[1],tvec[0],tvec[-1]*2-tvec[-2]))

    ##for l in lbo:
        ##axhline(l,c='w')


    #lbo_spect = []
    #bckg_spect = []

    #for l in lbo:
        #i = tvec.searchsorted(l-5)
        #lbo_spect.append(spectrum[i:i+4].mean(0))
        #bckg_spect.append(spectrum[i-4: i].mean(0))

    #bckg_spect = mean(bckg_spect,0) 
    #lbo_spect = mean(lbo_spect,0) - bckg_spect
    
 
    #x0 = sum(spectrum*bckg_spect/linalg.norm(bckg_spect)**2, 1)

    #x1 = sum((spectrum.T - outer(bckg_spect, x0)).T*lbo_spect/linalg.norm(lbo_spect)**2, 1)
    ##print(data.item()[13])
    #f,ax = subplots(2,1)
    #f.suptitle(data.item()[1].decode()+'  '+data.item()[13][1].decode())
    #ax[0].plot(  bckg_spect+lbo_spect ,label='after LBO')
    #ax[0].plot(  bckg_spect ,label='before LBO')
    #ax[0].plot(lbo_spect ,label='difference')
    #ax[0].legend()
    #ax[0].set_ylabel('Counts')
    #ax[0].set_xlabel('pixel')
    
    #ind = slice(*tvec.searchsorted([3750, 4650]))
    #ind = slice(*tvec.searchsorted([lbo[0]-50, lbo[-1]+50]))

    
    #ax[1].plot(tvec[ind], x1[ind])
    #ax[1].set_xlabel('time [ms]')
    #for l in lbo:
        #ax[1].axvline(l,c='k',lw=.5,ls='--')
    ##ax[1].set_xlim(3750, 4650)
    #ax[1].set_ylim(max(-.5, x1[ind].min()),None)
    #ax[1].axhline(0,c='k',lw=.5,ls='-')
    
    #savefig(str(shot)+'_'+ch+'.png')
    #close()



#show()

#figure()

#imshow(spectrum.T - outer(bckg_spect, x0), aspect='auto', vmin=0,vmax=10, origin='lower', extent=(0,spectrum.shape[1],tvec[0],tvec[-1]*2-tvec[-2]))


#figure()
#plot(tvec, x[:,1])
#for l in lbo:
    #axvline(l,c='k')
    
    
    


#plot(tvec, spectrum[:,200:250].sum(1))
#for l in lbo:
    #axvline(l,c='k')


#figure()
#imshow(dot(X,x.T).T, aspect='auto', vmin=0,vmax=10, origin='lower', extent=(0,spectrum.shape[1],tvec[0],tvec[-1]*2-tvec[-2]))
#for l in lbo:
    #axhline(l,c='w')
