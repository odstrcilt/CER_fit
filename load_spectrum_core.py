
    
import numpy as np
from matplotlib.widgets import RectangleSelector,SpanSelector,MultiCursor
from IPython import embed
import matplotlib.pylab as plt 
from scipy.stats.mstats import mquantiles
from draggableColorbar import DraggableColorbar

mdsserver = 'atlas.gat.com'
MDSconn = MDSplus.Connection(mdsserver)


#some core LOS observing 30L and 30R beams
chord = 'T19' #try T17-T20
shot  = 175860
average_over_beam_blip = False


#fitted wavelength range in Angstrom (1e-10m) units 
lam_min = 4076
lam_max = 4095 #

beams = load_spectrum.get_beams(shot)
 

wavelength, t_start, dt,raw_gain, spectra, pe, readout_noise  = load_spectrum.load_chord(shot, chord)

    
calib = load_spectrum.get_calib(shot, chord, wavelength)
wav_vec, dispersion = load_spectrum.get_wavelength(shot, chord, wavelength, spectra.shape[1])

#remove spikes from gamma radiation
spectra = load_spectrum.remove_spikes(spectra)
 
#time indexes of ative beam 'ts' and times when beams was off for background substraction 'tssub' for 30L and 30R
tssub, ts = load_spectrum.get_tssub(beams, t_start ,dt, '30L')

#NBI power at these times
pownbi = beams('30L',t_start,dt) + beams('30R',t_start,dt)

bg_spectra = spectra[tssub]
spectra = spectra[ts]
tvec = t_start[ts]
pow_avg = pownbi[ts] - pownbi[tssub] 



def add_exp_fit( ax, time, sig, err):
    
    plt_offset, = ax.plot([],[],'k--',lw=1, zorder=100)
    fit, = ax.plot([],[],'k',lw=2, zorder=100)

    result= ax.text(0.5,0.8,'',transform=ax.transAxes,zorder=100)

    offset = True

    fun = lambda x,a,b,t: a*np.exp(-x/t)+b*offset

    def onselect_time(xmin, xmax):
        print('onselect')
        indmin, indmax = time.searchsorted((xmin, xmax))
        x = time[indmin:indmax]-time[indmin]
        y = sig[indmin:indmax]
        e = err[indmin:indmax]
        if len(x) == 0: return
        pos = abs(y.mean())+0.1
        p0 = pos ,0.,0.5

        popt,pcov = curve_fit(fun, x, y,jac='cs', p0=p0,sigma=e,
                            bounds=((0, -np.inf,0),(np.inf, np.inf, 1)),
                            x_scale=(pos,pos,0.1))
        
   
        print( 'chi2:', sum(((fun(x,*popt)-y)/e)**2)/len(x))
        x_fit = np.linspace(xmin, xmax,1000)-time[indmin]
        y_fit = fun(x_fit,*popt)
        fit.set_data(x_fit+time[indmin], y_fit)
        if offset:
            plt_offset.set_data(x_fit+time[indmin], y_fit*0+popt[1])
        fit_err = np.sqrt(np.diag(pcov))
        result.set_text(r'$\tau_p$ = %.1f+/-%.1f ms'%(popt[2]*1e3, fit_err[2]*1e3))
        ax.figure.canvas.draw()


    # set useblit True on gtkagg for enhanced performance
    fit_span = SpanSelector(ax, onselect_time, 'horizontal', useblit=True,
                        props=dict(alpha=0.5, facecolor='red'))
    print('selector')
    return fit_span
    
    

# Callback function
def fit_cer_spectra(tmin, tmax, wmin, wmax):
    print(f"Selected time range: {tmin:.2f} - {tmax:.2f}")
    print(f"Selected wavelength range: {wmin:.2f} - {wmax:.2f}")

    wind = (wav_vec < wmax)&(wav_vec > wmin)
    tind = (tvec > tmin)&(tvec < tmax)
    
    select_spectra = spectra[tind][:,wind]
    select_bg_spectra = bg_spectra[tind][:,wind]
    select_tvec = tvec[tind]
    select_pow_avg = pow_avg[tind]
    select_wav = wav_vec[wind]
    
      
                                                                      
    A, Ae, data_fit =  cer_fit.fit_fixed_shape_gauss(select_wav, select_spectra, 
    select_bg_spectra)
                                                                     

    #apply calibration
    A /= dt*1e-3  # convert data to [counts/s]
    A /= calib*dispersion; # convert to [photons/m^2/s/ster]
    Ae /= dt*1e-3  # convert data to [counts/s]
    Ae /= calib*dispersion; # convert to [photons/m^2/s/ster]
     

    active_spectrum = select_spectra - select_bg_spectra

    f,ax = plt.subplots(4,1, sharex=True)
    ax[0].set_yscale('log')
         
    ax[0].set_ylim((A/select_pow_avg).max()/100, (A/select_pow_avg).max()*1.5)
    ax[0].errorbar(select_tvec, A / select_pow_avg, Ae / select_pow_avg)
    ax[0].set_ylabel('Amplitude/power')

    offset = mquantiles(active_spectrum,0.05,axis=1).T
    vmax=mquantiles(active_spectrum,0.99)
    ax[1].pcolormesh(select_tvec,select_wav, active_spectrum.T-offset ,vmin=0,vmax=vmax )
    ax[1].set_ylabel('Active spectrum')
    ax[2].pcolormesh(select_tvec,select_wav, data_fit.T-offset ,vmin=0,vmax=vmax  )
    ax[2].set_ylabel('Fitted spectrum')



    resid = active_spectrum-data_fit
    vmax = mquantiles(abs(resid),0.999)
    ax[3].pcolormesh(select_tvec,select_wav,resid.T ,cmap='seismic',vmin=-vmax,vmax=vmax )
    ax[3].set_ylabel('Difference')

    multi = MultiCursor(f.canvas, ax, color='r', lw=1)
    
    fit_span = add_exp_fit( ax[0], select_tvec, A / select_pow_avg, Ae / select_pow_avg)
    plt.show()
     
     
     
     
     
from matplotlib.widgets import RectangleSelector

# Internal callback for RectangleSelector
def onselect(eclick, erelease):
    tmin, tmax = sorted([eclick.ydata, erelease.ydata])
    wmin, wmax = sorted([eclick.xdata, erelease.xdata])
    fit_cer_spectra(tmin, tmax, wmin, wmax)

active =  (spectra-bg_spectra).T
cmax = np.percentile(active, 99)
# Plotting
fig, ax = plt.subplots()
pc = ax.pcolormesh(wav_vec,tvec,spectra-bg_spectra, shading='auto', vmin=0, vmax=cmax)
ax.set_ylabel("Time [ms]")
ax.set_xlabel("Wavelength [$\AA$]")
ax.set_xlim(wav_vec[0],wav_vec[-1])
ax.set_ylim(tvec[0],tvec[-1])


cbar = plt.colorbar(pc, format='%.2g', ax=ax, label="Counts")
cbar = DraggableColorbar(cbar, pc)
cid = cbar.connect()




# Enable rectangle selector
toggle_selector = RectangleSelector(ax, onselect,  useblit=True,
                                    button=[1],  # Left mouse button
                                    minspanx=5, minspany=5,
                                    spancoords='data',
                                    interactive=True)

plt.title("Drag to select region")
plt.show()

