'''
Script to analyze CER data on DIII-D

'''

try:
    import MDSplus 
    import cer_fit 
    import interface_w_CER  
    import load_spectrum  
except:
  
    print("""Try first:
module purge
module load omfit
module load ceromfit
""")
    raise
    exit()
    

import matplotlib.pyplot as plt
import numpy as np
from draggableColorbar import DraggableColorbar
import lbo_helpers
from matplotlib.widgets import RectangleSelector,SpanSelector, MultiCursor
from scipy.optimize import least_squares,curve_fit
from IPython import embed
import os, socket, sys
from functools import partial
from os import path
 
hostname=socket.gethostname()
username = os.path.expanduser('~')[6:]


import argparse
parser = argparse.ArgumentParser( usage='Plot and fit SPRED data')

parser.add_argument('--shot', metavar='S',type=int, help='shot number',default=204397)
parser.add_argument('--channel', metavar='C',type=str, help='CER channel', default=None)
parser.add_argument('--downsample',type=int, help='Reduce time resolution N times', default=1)
parser.add_argument('--include_passive',action='store_true', help="don't remove passive", default=False)



args = parser.parse_args()
channel = args.channel
shot = args.shot

if shot is None:
    print('Set shot number')
    exit(1)


data_dir = ''
cam = 'C1'
from dict_hdf5 import load_dict_from_hdf5
    
    
def load_spectra(shot):
    #cal_filename = path.join(cal_dir, 'ProEM 512 '+cam+'_Neon_Wavelength_Calibration.h5')
    #cal = load_dict_from_hdf5(cal_filename)
    spec_filename = path.join(data_dir, str(shot) + '_DIIID_' + cam)
    
    
    spec = load_dict_from_hdf5(spec_filename)
    #spec['cam_name'] = cal['cam_name']
    #spec['wavelength_arr'] = cal['wavelength_arr']  
     
    #BUG this is just temporary fix
    spec['wavelength_arr'] = np.arange(spec['SPEC_DATA']['SIGNAL'].shape[1])

    return spec



def separate_active(shot, data, include_passive):

    spectra = np.single(data['SPEC_DATA']['SIGNAL'][:,:,0])
    t_start = data['SPEC_DATA']['T']
    dt = data['EXPOS_TIME']
    wav_vec = data['wavelength_arr']
    


   

    #remove spikes from gamma radiation
    spectra = load_spectrum.remove_spikes(spectra)

    if include_passive:
        bg_spectra = np.median(spectra, axis=1)[:,None]
        bg_spectra = bg_spectra + np.median(spectra-bg_spectra, axis=0)[None,:]
        return t_start, wav_vec, spectra, bg_spectra
        
        

    beams = load_spectrum.get_beams(shot, load_beams = ['15L','15R'])    
    
     
    #NBI power at these times
    pownbi = beams('15L',t_start,dt) + beams('15R',t_start,dt)
    
    
    #time indexes of ative beam 'ts' and times when beams was off for background substraction 
    _, ts = load_spectrum.get_tssub(beams, t_start, dt, '15_', 200)
    

    time = t_start[ts]
    #this will average passive spectra over the whole notch region 
    
    time_passive,_, spectra_passive, _, pow_avg_bg = load_spectrum.blip_average(spectra, t_start, dt, pownbi, passive=True)
    from scipy.interpolate import interp1d
    bg_spectra = interp1d(time_passive, spectra_passive, axis=0)(np.clip(time, time_passive[0], time_passive[-1]))
    pow_avg_bg = interp1d(time_passive, pow_avg_bg)(np.clip(time, time_passive[0], time_passive[-1]))
 

    spectra = spectra[ts]
    pow_avg = pownbi[ts] - pow_avg_bg
     

    #correction for varying NBI power
    pow_avg /= pow_avg.mean()
    bg_spectra/=pow_avg[:,None]
    spectra/=pow_avg[:,None]
 

    return time, wav_vec, spectra, bg_spectra

class fit_interactive:
  
    def __init__(self,shot, channel, args):


        self.shot = shot
        self.channel = channel
        
        
        self.connection = MDSplus.Connection('atlas.gat.com')
  
        self.background_data = None
        
        
        data = load_spectra(shot)
        
        
        
        self.time, self.lam, self.spectrum, self.bg_spectrum = separate_active(shot, data, args.include_passive)
        
        
        
        
        if args.downsample > 1:
        
            n = len(self.time)
            d = args.downsample
            self.time = self.time[:n//d*d].reshape(-1, d).mean(1)
            self.spectrum = self.spectrum[:n//d*d].reshape(-1, d, self.spectrum.shape[1]).mean(1)
            self.bg_spectrum = self.bg_spectrum[:n//d*d].reshape(-1, d, self.bg_spectrum.shape[1]).mean(1)
       

            
        # get LBO times and SPRED data from external routines
        try:
            lbo_times = lbo_helpers.get_lbo_times(shot, self.connection) 
        except:
            lbo_times = [0]
            
        #time when the integration has ended. 
        self.time /= 1000 #s
        self.lbo_times = np.array(lbo_times)/1000 #s
        
        #NOTE time is defined at the end of the frame exposure
        self.click_event = None
        
        self.initialise_figure()
        self.plot_spectra()

    def initialise_figure(self):

        self.fig, self.ax = plt.subplots(1,1, sharey=True,figsize=(12,8), 
                                num=f'CHANNEL: {channel}    SHOT:{self.shot}')
        
        self.ax.set_xlabel(r'$\lambda$ [nm]')        
                
        self.ax.set_ylabel('time [s]')
        # self.ax.set_ylim(self.time[0],self.time[-1])
        self.ax.set_ylim(self.time[0], 6)
        self.ax.set_xlim(self.lam.min(),self.lam.max())
        self.ax.set_title('Select background by right mouse button (optional), and fit by left button')

        
        
        props = dict(facecolor='red', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)
  
        select_kwg = dict(#drawtype='box', 
                          useblit=True,
                        button=[1],  # don't use middle button
                        minspanx=5, minspany=5,  props= props,
                        spancoords='pixels', interactive=True)
        
        self.select = RectangleSelector(self.ax, self.line_select_callback, **select_kwg)
        

        props_bckg = dict(facecolor='blue', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)
        
        select_kwg_bckg = dict(#drawtype='box', 
                          useblit=True,
                        button=[3],  # don't use middle button
                        minspanx=5, minspany=5,  props= props_bckg ,
                        spancoords='pixels', interactive=True)
                        
                        
        self.select_bckg = RectangleSelector(self.ax, self.line_select_callback, **select_kwg_bckg)

        
        
        self.fig.tight_layout()

    def plot_spectra(self):
      
        
        active_spectra =  self.spectrum - self.bg_spectrum
        self.scale =  np.percentile(active_spectra.max(1), 95)

        dt = np.mean(np.diff(self.time))
        dl = np.mean(np.diff(self.lam))

        img_kwars = dict(vmin=0, cmap='gray_r')

        #NOTE time is defined at the middle of the frame exposure
        
        img = self.ax.pcolorfast(np.r_[self.lam-dl/2,self.lam[-1]+dl/2],
         np.r_[self.time-dt,self.time[-1]],active_spectra,
         vmax=self.scale,**img_kwars)
      
       
        [self.ax.axhline(lbo, c='b', lw=.5) for lbo in self.lbo_times]
 
        self.cbar = plt.colorbar(img, format='%.2g', ax=self.ax)      
        self.cbar = DraggableColorbar(self.cbar, img)
        self.cid = self.cbar.connect()




    def line_select_callback(self,eclick, erelease):
        if self.click_event is None or self.click_event.xdata!=eclick.xdata :
            if eclick.button in [1,3]:
 
                self.click_event = eclick
                # Internal callback for RectangleSelector

                tmin, tmax = sorted([eclick.ydata, erelease.ydata])
                wmin, wmax = sorted([eclick.xdata, erelease.xdata])
                if tmin != tmax and wmin != wmax:
                    self.fit_spectra(tmin, tmax, wmin, wmax)
           
                
            
       
    def add_exp_fit(self, ax, time, sig, err):
        
        plt_offset, = ax.plot([],[],'k--',lw=1, zorder=100)
        fit, = ax.plot([],[],'k',lw=2, zorder=100)

        result= ax.text(0.5,0.95,'',transform=ax.transAxes,zorder=100, backgroundcolor='w')

        

        def onselect(xmin, xmax, add_offset=False):
        
            fun = lambda x,a,b,t: a*np.exp(-x/t)+b*add_offset
            indmin, indmax = time.searchsorted((xmin, xmax))
            x = time[indmin:indmax]-time[indmin]
            y = sig[indmin:indmax]
            e = err[indmin:indmax]
            if len(x) == 0: return
            pos = abs(y.mean())+0.1
            p0 = pos ,0.,0.5

            popt,pcov = curve_fit(fun, x, y,jac='cs', p0=p0,sigma=e, absolute_sigma=True,
                                bounds=((0, -np.inf,0),(np.inf, np.inf, 10)),
                                x_scale=(pos,pos,0.1))
            
            
            resid = (fun(x,*popt)-y)/e
            chi2 = sum(resid**2/len(x))
           
            #uncertanty correction for chi/n=1 and correlated errors
            err_scale = len(x) / (2.0 * np.sum(np.diff(resid > 0))) * np.sqrt(chi2)
            fit_err = np.sqrt(np.diag(pcov)) * err_scale
            
            print( 'chi2: %.2f'% chi2 )
            x_fit = np.linspace(xmin, xmax,1000)-time[indmin]
            y_fit = fun(x_fit,*popt)
            fit.set_data(x_fit+time[indmin], y_fit)
            if add_offset:
                plt_offset.set_data(x_fit+time[indmin], y_fit*0+popt[1])
                plt_offset.set_visible(True)
            else:
                plt_offset.set_visible(False)
                
   
            result.set_text(r'$\tau_p$ = %.0f+/-%.0f ms'%(popt[2]*1e3, fit_err[2]*1e3))
            ax.figure.canvas.draw()


        # set useblit True on gtkagg for enhanced performance
        self.fit_span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='red'), button = 1)
                            
        self.fit_span_offset = SpanSelector(ax, partial(onselect, add_offset=True),
                             'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='blue'), button = 3)
                            
                                           


        # Callback function
    def fit_spectra(self, tmin, tmax, wmin, wmax):
        print(f"Selected time range: {tmin:.1f} - {tmax:.1f}")
        print(f"Selected wavelength range: {wmin:.2f} - {wmax:.2f}")

        wind = (self.lam  < wmax)&(self.lam  > wmin)
        tind = (self.time > tmin)&(self.time < tmax)
        
        select_spectra = self.spectrum[tind][:,wind]
        select_bg_spectra = self.bg_spectrum[tind][:,wind]
        select_time = self.time[tind]
        select_wav = self.lam[wind]
        
        #save background or substratct, based on the used mouse button
         
        if self.click_event.button == 3:
            A, Ae, data_fit =  cer_fit.fit_fixed_shape_gauss(select_wav, select_spectra, 
            select_bg_spectra)
            isort = np.argsort(select_wav)
            self.background_data = (select_wav[isort],  data_fit.mean(0)[isort])
            return 
            
        if self.background_data is not None:
          
            offset = np.interp(select_wav, *self.background_data)
           
            select_bg_spectra = select_bg_spectra + offset
          
                                                  
        A, Ae, data_fit =  cer_fit.fit_fixed_shape_gauss(select_wav, select_spectra, 
            select_bg_spectra)
        
      
        active_spectrum = select_spectra - select_bg_spectra
        
        self.plot_spectra_fit( A, Ae, data_fit, active_spectrum, select_wav, select_time )
        
        
    def plot_spectra_fit(self,  A, Ae, data_fit, active_spectrum, wav, time):
   
        #plot  selected spectral region and the amplitude
        import matplotlib.gridspec as gridspec  

        fig = plt.figure(figsize=(10, 6), num=f'Line fit')
        fig.clf()
        fig.subplots_adjust(wspace = 0.1)
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1], hspace=0.05, wspace=0.05)

        # Large left plot spanning all rows
        ax_left = fig.add_subplot(gs[:, 0])  # All rows, first column

        # Right column: 3 subplots
        ax_r1 = fig.add_subplot(gs[0, 1], sharex=ax_left)
        ax_r2 = fig.add_subplot(gs[1, 1], sharex=ax_left, sharey=ax_r1)
        ax_r3 = fig.add_subplot(gs[2, 1], sharex=ax_left, sharey=ax_r1)

        # Optional: hide x labels except bottom
        for ax in [ax_r1, ax_r2]:
            ax.tick_params(labelbottom=False)

        # Optional: hide y labels on right subplots
        for ax in [ax_r1, ax_r2, ax_r3]:
            ax.tick_params(labelleft=False)
            
        ax_left.set_title('Drag to fit by exponential (right mouse button to add offset)')


        #f,ax = plt.subplots(4,1, sharex=True)
        ax_left.set_yscale('log')
             
        ax_left.set_ylim((A).max()/100, (A).max()*1.5)
        ax_left.errorbar(time, A, Ae)
        ax_left.set_ylabel('Amplitude')

        offset = np.percentile(active_spectrum,5,axis=1)
        vmax=np.percentile(active_spectrum,99)
        ax_r1.pcolormesh(time,wav, active_spectrum.T-offset ,vmin=0,vmax=vmax )
        ax_r1.set_ylabel('Active spectrum')
        
        ax_r2.pcolormesh(time,wav, data_fit.T-offset ,vmin=0,vmax=vmax  )
        ax_r2.set_ylabel('Fitted spectrum')
        
        resid = active_spectrum-data_fit
        vmax = np.percentile(abs(resid),99)
        ax_r3.pcolormesh(time, wav,resid.T ,cmap='seismic',vmin=-vmax,vmax=vmax )
        ax_r3.set_ylabel('Difference')
        ax_r3.set_xlabel('Time [s]')
        ax_left.set_xlabel('Time [s]')
        
                      
        self.add_exp_fit(ax_left, time,A, Ae)
               
        
        all_ax =  [ax_left, ax_r1, ax_r2, ax_r3]
        multi = MultiCursor(fig.canvas, all_ax, color='r', lw=1, useblit=True)

        #plt.show()
        fig.show()
        fig.canvas.draw()
 
       

    def fit_line(self,eclick,x,t):
        
        x1,x2 = min(x), max(x)
        tmin,tmax = min(t), max(t)

         
        if eclick.inaxes == self.ax[0]:
            lam = self.lam1
            spectrum = self.spectrum1
        elif eclick.inaxes == self.ax[1]:
            lam = self.lam2
            spectrum = self.spectrum2
        else:
            return


        imin,imax = self.time.searchsorted([tmin,tmax])
        ind = (lam>x1)&(lam<x2)  
        spec_slice = spectrum[imin:imax,ind].astype('double')
        time_slice = self.time[imin:imax]
        dt_slice = self.dt[imin:imax]
        lam_slice  = lam[ind].astype('double')


        def cost_fun(par,x, y, return_results=False):
            x0,s = par 

            A = np.vstack((np.exp(-(x-x0)**2/(2*s**2)), np.ones_like(x))).T

            coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)
            if return_results:
                chi2_dof = r/(len(x)-rr)
                err = np.sqrt(np.linalg.inv(np.dot(A.T,A))[0,0]*chi2_dof)
                return coeff.T, np.dot(A, coeff).T, err
            return r
            
        w0 = 0.1/2.355
        try:
            out = least_squares(cost_fun, ((x1+x2)/2,(x2-x1)/4),args=(lam_slice, spec_slice), 
                            bounds=((x1,w0),(x2,max(w0*2, (x2-x1)/3))))    
        except:
            return 
            
            
        coeff,model,err = cost_fun(out.x,lam_slice, spec_slice ,return_results=True)
        try:
            x0,s = out.x
        except:
            print( 'Fitting was unsuccesful')
            return
        
        
        
        f = plt.figure(num='Line fit', figsize=(10,5))
        f.clf()
        
        
            
        ax = f.add_subplot(131)
        ax.errorbar(time_slice-dt_slice/2,coeff[:,0], err,color='b', ls='none')
        ax.step(time_slice,coeff[:,0], color='b',  where='pre')

        ax.step(time_slice ,coeff[:,1], where='pre',c='r')
        ax.set_ylim(min(coeff[:,0].min(),0), (coeff[:,0]+err).max())
        ax.set_xlim(time_slice[[0,-1]])
        
        for l in self.lbo_times:
            ax.axvline(l/1e3,c='k')
        

        ax.legend(('Aplitude','background', ),loc='best')
        ax.set_title('lam0 = %.3f nm  FWHM = %.3f'%(x0,2.355*abs(s)))
        ax.set_xlabel('time [s]')
        
        
     
        self.add_exp_fit(ax, time_slice, coeff[:,0], err)

        
        dt = np.mean(np.diff(time_slice))
        dl = np.mean(np.diff(lam_slice))
        
        img_kwars = dict(aspect='auto',interpolation='nearest', origin='lower', 
                     extent=(lam_slice[0]-dl/2, lam_slice[-1]+dl/2,time_slice[0]-dt,time_slice[-1]))
        
        vmax =  np.percentile(spec_slice, 99.9)

        ax = f.add_subplot(132)
        ax.imshow(spec_slice, vmin=0,vmax=vmax, **img_kwars)
        ax.set_title('spectrum')
        ax.set_xlabel(r'$\lambda$ [nm]')

        resid = spec_slice-model
        vmax = abs(resid).max()
        
        ax = f.add_subplot(133, sharex=ax, sharey=ax)
        ax.imshow(spec_slice-model, cmap='seismic', vmin=-vmax, vmax = vmax, **img_kwars)
        ax.set_title('residuum')
        ax.set_xlabel(r'$\lambda$ [nm]')
        plt.setp(ax.get_yticklabels(), visible=False)
             
        f.show()
        f.canvas.draw()

        



spred_plot = fit_interactive(shot, channel, args)


 


plt.show()
