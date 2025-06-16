try:
    import MDSplus 
    from cer_fit import fast_recursive_fit
    import interface_w_CER  
    import load_spectrum  
except:
    print("""Try first:
        module purge
        module load omfit
        module load ceromfit""")
    exit()
    
import numpy as np
from matplotlib.widgets import MultiCursor
from IPython import embed
import matplotlib.pylab as plt 
from scipy.stats.mstats import mquantiles

mdsserver = 'atlas.gat.com'
MDSconn = MDSplus.Connection(mdsserver)


#some core LOS observing 30L and 30R beams
chord = 'T19' #try T17-T20
shot  = 175860
tmin,tmax = 1800,6000
average_over_beam_blip = False


#fitted wavelength range in Angstrom (1e-10m) units 
lam_min = 4076
lam_max = 4095 #

beams = load_spectrum.get_beams(shot)
 
 
cer_data = {}
 
#average over spectra 
all_spectra = []
readout_noise_tot = 0
 
print('Loading : ', chord)


wavelength, t_start, dt,raw_gain, spectra, pe, readout_noise  = load_spectrum.load_chord(shot, chord)

    
calib = load_spectrum.get_calib(shot, chord, wavelength)
wav_vec, dispersion = load_spectrum.get_wavelength(shot, chord, wavelength, spectra.shape[1])
tsmin,tsmax = t_start.searchsorted([tmin,tmax])


spectra = spectra[tsmin:tsmax]
tvec = t_start[tsmin:tsmax]

#remove spikes from gamma radiation
spectra = load_spectrum.remove_spikes(spectra)
 
#time indexes of ative beam 'ts' and times when beams was off for background substraction 'tssub' for 30L and 30R
tssub, ts = load_spectrum.get_tssub(beams, tvec ,dt, '30L')

#NBI power at these times
pownbi = beams('30L',tvec,dt) + beams('30R',tvec,dt)

 
 

if average_over_beam_blip:
    tvec, stime, spectra, bg_spectra, pow_avg  = load_spectrum.blip_average(spectra, tvec, dt, pownbi)
else: 
    bg_spectra = spectra[tssub]
    spectra = spectra[ts]
    tvec = tvec[ts]
    pow_avg = pownbi[ts] - pownbi[tssub] 


plt.figure()
plt.title('Average spectrum and selected wavelength region')
plt.plot(wav_vec, spectra.mean(0), label='Total')
plt.plot(wav_vec, bg_spectra.mean(0), label='Passive spectrum')
plt.plot(wav_vec, spectra.mean(0) - bg_spectra.mean(0), label='Active spectrum')
plt.xlabel('Wavelength [$\AA$]')
plt.ylabel('Amplitude [photons/m$^2$/s/ster]')
plt.axvline(lam_min)
plt.axvline(lam_max)
plt.legend(loc='best')
#plt.show()
 

lpix,upix = (-wav_vec).searchsorted([-lam_max,-lam_min])
  
#initial guess
p0 = None

solution,solution_err,data_fit,chi2n,success =  fast_recursive_fit(spectra, bg_spectra,wav_vec,p0=p0,
                                                                  ph_per_count=1/pe,readout_noise=readout_noise_tot, lpix=lpix,upix=upix)

#background
B,Berr  = solution[:,0], solution_err[:,0]
#Aplitude, width and shift of Gaussian
[A,s0,x0] = solution[:,1:].reshape(len(B), -1, 3).T
[Ae,s0e,x0e] = solution_err[:,1:].reshape(len(B), -1, 3).T
 

#apply calibration
A /= dt*1e-3  # convert data to [counts/s]
A /= calib*dispersion; # convert to [photons/m^2/s/ster]
Ae /= dt*1e-3  # convert data to [counts/s]
Ae /= calib*dispersion; # convert to [photons/m^2/s/ster]
 
 

f,ax = plt.subplots(4,1, sharex=True)
ax[0].set_yscale('log')
     
ax[0].set_ylim((A/pow_avg).max()/100, (A/pow_avg).max())
ax[0].errorbar(tvec, A[0] / pow_avg, Ae[0] / pow_avg)
ax[0].set_ylabel('Amplitude/power')

 
vmax=mquantiles(spectra[:,lpix:upix]-bg_spectra[:,lpix:upix],0.99)
ax[1].pcolormesh(tvec,wav_vec[lpix:upix], spectra[:,lpix:upix].T-bg_spectra[:,lpix:upix].T ,vmin=0,vmax=vmax )
ax[1].set_ylabel('Active spectrum')
ax[2].pcolormesh(tvec,wav_vec[lpix:upix], data_fit.T ,vmin=0,vmax=vmax  )
ax[2].set_ylabel('Fitted spectrum')



resid = spectra[:,lpix:upix]-data_fit-bg_spectra[:,lpix:upix]
vmax = mquantiles(abs(resid),0.999)
ax[3].pcolormesh(tvec,wav_vec[lpix:upix],resid.T ,cmap='seismic',vmin=-vmax,vmax=vmax )
ax[3].set_ylabel('Difference')

multi = MultiCursor(f.canvas, ax, color='r', lw=1)
plt.show()
 
