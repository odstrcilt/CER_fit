from interface_w_CER import *
from IPython import embed
import MDSplus 
from matplotlib.pylab import *
from scipy import *
from numpy import *
from cer_fit import fast_recursive_fit
from scipy.stats.mstats import mquantiles
from matplotlib.widgets import MultiCursor
from load_spectrum import *

 


shot  = 175674
tmin,tmax = 2300, 5000
blip_avg = False
remove_elms = False

chords = ['M%.2d'%i for i in range(17,33)]



beams = get_beams(shot)
 

cer_data = {}
#equ = None
for chord in chords:
    print('Loading : ', chord)
    
    try:
        wavelength, t_start, dt,raw_gain, spectra, pe, readout_noise  = load_chord(shot, chord)
    except:
      print(f'Loading of {chord} failed')
      continue

    #get calibration and wavelength
    calib = get_calib(shot, chord, wavelength)
    wav_vec, disp = get_wavelength(shot, chord, wavelength, spectra.shape[1])
   
    tsmin,tsmax = t_start.searchsorted([tmin,tmax])
 
    #offset substraction after 6350
    t_end = 6350
    if any(t_start > t_end):
    	spectra -= spectra[t_start > t_end].mean(0)
    
    spectra= spectra[tsmin:tsmax]
    tvec = t_start[tsmin:tsmax]

 
    #remove spikes from gamma and neutron radiation
    spectra = remove_spikes(spectra)

    #detect and remove ELMs detected as spikes in Dalpha line
    if remove_elms:
        wmin,wmax = (-wav_vec).searchsorted([-6560, -6550])
        H0 = spectra[:,wmin:wmax].mean(1)
        elms = H0/np.median(H0) > 1.3
        spectra= spectra[ ~elms ]
        tvec = tvec[ ~elms]
 

    nbi = '30' if int(chord[1:]) < 9 else '33'
    power = beams(nbi+'L',tvec,dt)+beams(nbi+'R',tvec,dt)
   

    if blip_avg:
   	 tvec, stime,spectra, bg_spectra, pow_avg  = blip_average(spectra, tvec, dt, power,skip_first =  0, passive=False)
    else:
         spectra= spectra[power > 5e5 ]
         tvec = tvec[power > 5e5]
         bg_spectra = spectra  * 0

 
 
    #equ, Psi_N_LOS, Psi_N_NBI, PLASMA_R, PLASMA_Z, R, Z, L = get_coord(shot, chord, nbi, tvec, equ=equ, diag='EFIT01')
   
    #BE intesnity and its uncertainty
    A_tot = np.zeros_like(tvec)
    Ae_tot = np.zeros_like(tvec)
   
    #wavelength range of the BE spectra
    wmin,wmax,c = [6589-6, 6610-4 ,'r'] 

    lpix,upix = (-wav_vec).searchsorted([-wmax,-wmin])
    
    #try to find a maximum of BE spectra
    imax =np.argmax(np.mean(spectra[:,lpix:upix]-bg_spectra[:,lpix:upix],0))
    wmid = wav_vec[lpix:upix][imax]
    

    p0 = [0,60,1,wmid-3.3, 120,1,wmid,60,1,wmid+3.3]

    wmin = wmid - 5
    wmax = wmid + 15
    lpix,upix = (-wav_vec).searchsorted([-wmax,-wmin])
    

 
    #fit BE spectra
    from scipy.optimize import least_squares

    def cost_fun(par,x, y, return_results=False):
        x0, w, s, A1, A2  = par
        spectrum = np.zeros_like(x)
        spectrum += A1*np.exp(-(x-x0-s)**2/(2*w**2))/np.sqrt(2*np.pi)/w
        spectrum += np.exp(-(x-x0)**2/(2*w**2))/np.sqrt(2*np.pi)/w
        spectrum += A2*np.exp(-(x-x0+s)**2/(2*w**2))/np.sqrt(2*np.pi)/w

        A = np.vstack((spectrum, np.ones_like(x))).T

        coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)

        if return_results:
            chi2_dof = r/(len(x)-rr)
            err = np.sqrt(np.linalg.pinv(np.dot(A.T,A))[0,0]*chi2_dof)
            return coeff.T, np.dot(A, coeff).T, err
        return r
        
    active_spect = spectra[:,lpix:upix]-bg_spectra[:,lpix:upix]
    
    out = least_squares(cost_fun, (wmid, 1,3.3,0.6,0.6),args=(wav_vec[lpix:upix], active_spect))
                #hardcoded maximal and mininal line width 
    
    coeff,data_fit,Ae = cost_fun(out.x,wav_vec[lpix:upix], active_spect ,return_results=True)
    A,B = coeff.T
    data_fit += bg_spectra[:,lpix:upix]

 
    A /= dt*1e-3  # convert data to [counts/s]
    A /= calib#*disp; # convert to [photons/m^2/s/ster]
    Ae /= dt*1e-3  # convert data to [counts/s]
    Ae /= calib#*disp; # convert to [photons/m^2/s/ster]


    cer_data[chord] = { 't_start': tvec ,'dt': dt,'full':np.single(A),
            'full_err': np.single(Ae),  }
    
    #continue
    
    f,ax = subplots(4,1,figsize=(5,12),sharex=True)
    f.suptitle(chord)
    sca(ax[0])
    errorbar(tvec, A, Ae ) 
    
    ylim(0,A.max())
    sca(ax[1])
    
    vmax=mquantiles(spectra[:,lpix:upix],0.99)
    pcolormesh(tvec,wav_vec[lpix:upix], spectra[:,lpix:upix].T ,vmin=0,vmax=vmax )
    sca(ax[2])
    pcolormesh(tvec,wav_vec[lpix:upix], data_fit.T ,vmin=0,vmax=vmax  )
    sca(ax[3])


    resid = spectra[:,lpix:upix]-data_fit
    vmax = mquantiles(abs(resid),0.999)
    pcolormesh(tvec,wav_vec[lpix:upix],resid.T ,cmap='seismic',vmin=-vmax,vmax=vmax )
    multi = MultiCursor(f.canvas, ax, color='r', lw=1)
    show()

 

 

 

savez_compressed('BE_signal_%d'%shot,**cer_data)
 
figure()
for chord in chords:
    plot(cer_data[chord]['t_start'], cer_data[chord]['full'],'.')
    text(cer_data[chord]['t_start'].mean(), cer_data[chord]['full'].mean()*1.1, chord)



show()
 
