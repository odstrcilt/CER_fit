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

 


shot  = 199111
tmin,tmax = 2000, 6000
shot  = 182660
tmin,tmax = 2350, 2850



blip_avg = True
remove_elms = False

chords = ['M%.2d'%i for i in np.r_[17:33]]
#chords = ['M31']
#chords = ['M%.2d'%i for i in range(29,33)]
from map_equ import equ_map

equ = equ_map(MDSplus.Connection(mdsserver))
equ.Open(shot,diag='EFIT01')

beams = get_beams(shot)

noelms = slice(None,None)
if remove_elms:
    chord = 'M25'
    wavelength, t_start, dt,raw_gain, spectra, pe, readout_noise  = load_chord(shot, chord)
    wav_vec, disp = get_wavelength(shot, chord, wavelength, spectra.shape[1])
    tsmin,tsmax = t_start.searchsorted([tmin,tmax])
    mspectra = spectra[tsmin:tsmax].mean(0)
    imax = np.argmax(mspectra)
    w0 = np.sum(mspectra[imax-20:imax+20] * wav_vec[imax-20:imax+20])/np.sum(mspectra[imax-20:imax+20])
    wav_vec += 6561 - w0
    wmin,wmax = (-wav_vec).searchsorted([ -6565,-6555])


    H0 = spectra[tsmin:tsmax,wmin:wmax].mean(1)
    #super simple estimate of ELMS!!
    elms = H0/np.median(H0) > 1.3
    noelms = ~elms


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
    
    spectra= spectra[tsmin:tsmax][noelms]
    tvec = t_start[tsmin:tsmax][noelms]
    dt = dt[tsmin:tsmax][noelms]
    

    #estimate wavelength from Dalpha peak at 656.1nm
    mspectra = spectra.mean(0)
    imax = np.argmax(mspectra)
    w0 = np.sum(mspectra[imax-20:imax+20] * wav_vec[imax-20:imax+20])/np.sum(mspectra[imax-20:imax+20])
    wav_vec += 6561 - w0
 
    #remove spikes from gamma and neutron radiation
    spectra = remove_spikes(spectra)

    #detect and remove ELMs detected as spikes in Dalpha line

 

    nbi = '30' if int(chord[1:]) < 9 else '33'
    power = beams(nbi+'L',tvec,dt)+beams(nbi+'R',tvec,dt)

    tssub, ts = get_tssub(beams, tvec,dt , nbi+'R')
    try:
        G = get_beamgeom(shot, chord, '33R' )
    except:
        G = 1

    if blip_avg:
   	 tvec, dt,spectra, bg_spectra, pow_avg  = blip_average(spectra, tvec, dt, power,skip_first =  0, passive=False)
    else:
         #print(tvec.shape,spectra.shape, )
         bg_spectra = spectra[tssub]
         spectra= spectra[ts ]
         tvec = tvec[ts]

    """
    plt.figure(chord)
    plt.plot(wav_vec,spectra.T-bg_spectra.T,'b')
    plt.plot(wav_vec,spectra.T ,'r')
    plt.plot(wav_vec, bg_spectra.T,'g')
    plt.xlim(6575,6595)
    plt.ylim(0, 150)
    #plt.show()


    continue
    """
    
 
    equ, Psi_N_LOS, Psi_N_NBI, PLASMA_R, PLASMA_Z, R, Z, L = get_coord(shot, chord, '33R', tvec, equ=equ, diag='EFIT01')
   
    #BE intesnity and its uncertainty
    A_tot = np.zeros_like(tvec)
    Ae_tot = np.zeros_like(tvec)
   
    #wavelength range of the BE spectra
    wmin,wmax,c = [6583,6605 ,'r'] 
    #wmin,wmax,c = [6570,6600 ,'r'] 
    lpix,upix = (-wav_vec).searchsorted([-wmax,-wmin])
    
    #try to find a maximum of BE spectra
    imax = np.argmax(np.mean(spectra[:,lpix:upix]-bg_spectra[:,lpix:upix],0))
    wmid = wav_vec[lpix:upix][imax]
   # wmid = 6587
    

   # p0 = [0,60,1,wmid-3.3, 120,1,wmid,60,1,wmid+3.3]

    wmin = wmid - 6
    wmax = wmid + 15
    lpix,upix = (-wav_vec).searchsorted([-wmax,-wmin])
    
    #plt.plot(wav_vec[lpix:upix], spectra[:,lpix:upix].T-bg_spectra[:,lpix:upix].T)
    #plt.plot(wav_vec[lpix:upix], bg_spectra[:,lpix:upix].T,'r')
    #plt.axvline(6587)
    #plt.show()
    #continue

    background = bg_spectra[:,lpix:upix].mean(0)

 
    #fit BE spectra
    from scipy.optimize import least_squares

    def cost_fun(par, x, y, return_results=False):
        x0, w, s, A1, A2  = par
        spectrum = np.zeros_like(x)
        spectrum += A1*np.exp(-(x-x0-s)**2/(2*w**2))/np.sqrt(2*np.pi)/w
        spectrum += np.exp(-(x-x0)**2/(2*w**2))/np.sqrt(2*np.pi)/w
        spectrum += A2*np.exp(-(x-x0+s)**2/(2*w**2))/np.sqrt(2*np.pi)/w

        A = np.vstack((spectrum, np.ones_like(x), background)).T

        coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)

        if return_results:
            chi2_dof = r/(len(x)-rr)
            err = np.sqrt(np.linalg.pinv(np.dot(A.T,A))[0,0]*chi2_dof)
            return coeff.T, np.dot(A, coeff).T, err
        return r
        
    active_spect = spectra[:,lpix:upix]-bg_spectra[:,lpix:upix]
    
    out = least_squares(cost_fun, (wmid, 1.2,3.3,0.6,0.6),
                         bounds=((wmid-5,1.0,0, 0, 0), (wmid+5, 1.5, np.inf,np.inf,  np.inf)),
                         args=(wav_vec[lpix:upix], active_spect))
                #hardcoded maximal and mininal line width 
    
    coeff,data_fit,Ae = cost_fun(out.x,wav_vec[lpix:upix],   active_spect  ,return_results=True)
    x0, w, s, A1, A2 = out.x
    print(w)
   # plt.plot(wav_vec[lpix:upix], active_spect.T,'r')
   # plt.plot(wav_vec[lpix:upix], data_fit.T,'b')
   # plt.axvline(wmid)
   ## plt.show()
    #continue
    A,B,C = coeff.T
    
    data_fit -= C[:,None]*bg_spectra[:,lpix:upix]
  

    
    A /= dt*1e-3  # convert data to [counts/s]
    A /= calib*disp; # convert to [photons/m^2/s/ster]
    Ae /= dt*1e-3  # convert data to [counts/s]
    Ae /= calib*disp; # convert to [photons/m^2/s/ster]


    cer_data[chord] = { 't_start': tvec ,'dt': dt,'full':np.single(A),
            'full_err': np.single(Ae), 'R':  PLASMA_R }
    
    
    #plt.figure()

    #plt.plot(wav_vec[lpix:upix], spectra[:,lpix:upix].T - bg_spectra[:,lpix:upix].T * (1+C), 'b')
    #plt.plot(wav_vec[lpix:upix], data_fit.T ,  'r')

   # plt.show()
    continue 


    f,ax = subplots(4,1,figsize=(5,9),sharex=True)
    f.suptitle(chord)
    sca(ax[0])
    errorbar(tvec, A, Ae ) 
    plt.title('BE signal')
    ylim(0,A.max())
    sca(ax[1])
    plt.title('Data')
    vmax=mquantiles( data_fit ,0.99)
    pcolormesh(tvec,wav_vec[lpix:upix], spectra[:,lpix:upix].T - bg_spectra[:,lpix:upix].T * (1+C) ,vmin=0,vmax=vmax )
    sca(ax[2])
    plt.title('Fit')
    pcolormesh(tvec,wav_vec[lpix:upix], data_fit.T ,vmin=0,vmax=vmax  )
    sca(ax[3])
    plt.title('Difference')
    plt.tight_layout()


    resid = spectra[:,lpix:upix]-data_fit
    vmax = mquantiles(abs(resid),0.999)
    pcolormesh(tvec,wav_vec[lpix:upix],resid.T ,cmap='seismic',vmin=-vmax,vmax=vmax )
    multi = MultiCursor(f.canvas, ax, color='r', lw=1)
    show()

 
#plt.show()
#


 

savez_compressed('BE_signal_%d'%shot,**cer_data)


#embed()
 
#figure()
#for chord in chords:
 #   plot(cer_data[chord]['t_start'], cer_data[chord]['full'],'.')
 #   text(cer_data[chord]['t_start'].mean(), cer_data[chord]['full'].mean()*1.1, chord)
figure()
for chord in chords:
    plot(cer_data[chord]['R'], cer_data[chord]['full'].mean(),'.')
    text(cer_data[chord]['R'], cer_data[chord]['full'].mean()*1.1, chord)

show()



 
