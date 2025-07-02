'''
Script to analyze CER data on DIII-D

'''

import MDSplus 

    

import matplotlib.pyplot as plt
import numpy as np
from draggableColorbar import DraggableColorbar
import lbo_helpers
from matplotlib.widgets import RectangleSelector,SpanSelector, MultiCursor
from scipy.optimize import least_squares,curve_fit
from IPython import embed
import os, socket, sys
from functools import partial
from IPython import embed
from time import time as tt

 
sys.path.append('/fusion/projects/codes/pytomo/geometry/DIIID/SXR')
from load_SXR import loader_SXR

hostname=socket.gethostname()
username = os.path.expanduser('~')[6:]


import argparse
parser = argparse.ArgumentParser( usage='Plot and fit SPRED data')
parser.add_argument('--shot', metavar='S',type=int, help='shot number')
parser.add_argument('--downsample',type=int, default=10, 
                    help='Downsample (dfault 10)')




args = parser.parse_args()
shot = args.shot

if shot is None:
    print('Set shot number')
    exit(1)

 
 
 
def signal_filter(tvec, signal,  cutoff = 1e3, nsvd = 5 ):
# Low-pass filter cutoff frequency in Hz
    
     
    from scipy.signal import butter, sosfilt, sosfiltfilt

    # Parameters
    fs = (tvec.size-1)/(tvec[-1]-tvec[0])         # Sampling frequency in Hz

    order = 4          # Filter order

    # Design IIR filter (Butterworth)
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')

    # Apply filter along time axis (axis=1)
    noise_signal = signal-sosfiltfilt(sos, signal, axis=0)

    
    u,s,v = np.linalg.svd(noise_signal, full_matrices = False)
    svd_filtered = np.dot(u[:, :nsvd] * s[ :nsvd], v[:nsvd])

    return signal - svd_filtered
 
 

 
 

class SXR_interactive:
    background_data = {}
  
    def __init__(self,shot, downsample):

        MDSconn = MDSplus.Connection('atlas.gat.com')
        self.cbar = {}
        self.cid = {}

        geometry_path = f'/local-scratch/{username}/sxr/'
        os.makedirs(geometry_path, exist_ok = True)
    
        self.SXRpol = loader_SXR(shot, geometry_path, MDSconn, False, False)
        self.SXRtor = loader_SXR(shot, geometry_path, MDSconn, False, True)

        self.shot = shot
        
        MDSconn.openTree('BCI', shot)
        
        #load density for normalisation of SXR signal
        dens,time_dens = MDSconn.get('_x=\\BCI::DENV1;[_x,dim_of(_x) ]').data()
        time_dens /= 1e3
        tind = (time_dens > 1) & (time_dens < 6)
        time_dens = time_dens[tind]
        dens = dens[tind]
        dens /= np.median(dens)
        
        tvec_pol, data_pol, _ = self.SXRpol.get_data(1, 6)
        tvec_tor, data_tor, _ = self.SXRtor.get_data(1, 6)
        
        #use only the last camera
        data_tor = data_tor[:,-12:]
        
          
        data_pol = signal_filter(tvec_pol, data_pol)
        data_tor = signal_filter(tvec_pol, data_tor)
        
        #normalise by density
 
        data_pol /= np.interp(tvec_pol, time_dens, dens)[:,None]
        data_tor /= np.interp(tvec_tor, time_dens, dens)[:,None]
        
        #simple downsample
  
        d = downsample
        nt, nch = data_pol.shape
       
        data_pol = data_pol[:nt//d*d].reshape(nt//d, d, -1).mean(1)
        
        self.data = {}
        self.los = {}
        self.tvec = {}
        
        self.data['90RP1'] = data_pol[:,:nch // 2] 
        self.data['90RM1'] = data_pol[:,nch // 2:] 
        self.los['90RP1'] = self.SXRpol.all_los[:nch // 2]
        self.los['90RM1'] = self.SXRpol.all_los[nch // 2:]
        self.tvec['90RM1'] = self.tvec['90RP1'] = tvec_pol[:nt//d*d].reshape(nt//d, d).mean(1)
        
        nt = len(tvec_tor)
        self.data['195R1'] = data_tor[:nt//d*d].reshape(nt//d, d, -1).mean(1)
        self.tvec['195R1'] = tvec_tor[:nt//d*d].reshape(nt//d, d).mean(1)
        self.los['195R1'] = self.SXRtor.all_los[-12:]
        
        
        self.cams = ['90RP1','90RM1','195R1']
        
        
        

        #plt.plot(self.tvec_pol, self.data_pol ,lw=.2)
        #denst = np.interp(self.tvec_pol, self.time_dens, self.dens)[:,None]
        #plt.plot(self.tvec_pol, self.data_pol / denst)
        #plt.show()
 
        # get LBO times and SPRED data from external routines
        try:
            lbo_times = lbo_helpers.get_lbo_times(shot, connection) 
        except:
            lbo_times = [0]
            
 
        self.lbo_times = np.array(lbo_times)/1000 #s
        
        #NOTE time is defined at the end of the frame exposure
        self.click_event = None
        
        self.initialise_figure()
        
        for cam in self.cams :
            self.plot_data(cam)

    def initialise_figure(self):

        self.fig, self.ax = plt.subplots(1,3, sharey=True,figsize=(12,8), 
                                num=f'     SHOT:{self.shot}  ')
        self.fig.suptitle('Select background by right mouse button (optional), and fit by left button')
        
        self.ax[0].set_ylabel('time [s]')

        
        
        for i, cam in enumerate(self.cams):
            self.ax[i].set_xlabel(r'Channels '+cam)        
            self.ax[i].set_xlim(0,  len(self.los[cam]))           
            self.ax[0].set_ylim(self.tvec[cam][0], self.tvec[cam][-1])
        
        props = dict(facecolor='red', edgecolor = 'red', alpha=0.5, fill=True, zorder=99)
  
        select_kwg = dict(#drawtype='box', 
                          useblit=True,
                        button=[1],  # don't use middle button
                        minspanx=5, minspany=5,  props= props,
                        spancoords='pixels', interactive=True)
        

        

        props_bckg = dict(facecolor='blue', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)
        
        select_kwg_bckg = dict(#drawtype='box', 
                          useblit=True,
                        button=[3],  # don't use middle button
                        minspanx=5, minspany=5,  props= props_bckg ,
                        spancoords='pixels', interactive=True)
        
        self.select = {}
        self.select_bckg = {}
        for i, cam in enumerate(self.cams):
            self.select[cam] = RectangleSelector(self.ax[i], self.line_select_callback,
             **select_kwg)     
            self.select_bckg[cam] = RectangleSelector(self.ax[i], self.line_select_callback,
             **select_kwg_bckg)

        
        
        self.fig.tight_layout()

    def plot_data(self, cam):
      
        
     
        scale = np.percentile(self.data[cam].max(1), 95)

        dt = np.mean(np.diff(self.tvec[cam]))
     
        channel = np.arange(len(self.los[cam]))

        img_kwars = dict(vmin=0, cmap='jet')

        #NOTE time is defined at the middle of the frame exposure
        iaxis = self.cams.index(cam)
     
        img = self.ax[iaxis].pcolorfast(np.r_[channel,channel[-1]+1],
         np.r_[self.tvec[cam]-dt/2,self.tvec[cam][-1]+dt/2],self.data[cam],
         vmax=scale,**img_kwars)
      
       
        [self.ax[iaxis].axhline(lbo, c='b', lw=.5) for lbo in self.lbo_times]
        
        self.cbar[cam] = plt.colorbar(img, format='%.1g', ax=self.ax[iaxis])      
        self.cbar[cam] = DraggableColorbar(self.cbar[cam], img)
        self.cid[cam] = self.cbar[cam].connect()




    def line_select_callback(self,eclick, erelease):
        if self.click_event is None or self.click_event.xdata!=eclick.xdata :
            if eclick.button in [1,3]:
 
                self.click_event = eclick
                # Internal callback for RectangleSelector

                tmin, tmax = sorted([eclick.ydata, erelease.ydata])
                chmin, chmax = sorted([eclick.xdata, erelease.xdata])
                if tmin != tmax and chmin != chmax:
                    self.fit_sxr(eclick, tmin, tmax, chmin, chmax)
           
                
            
       
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
            
            print(2.0 * np.sum(np.diff(resid > 0)) / len(x))
            err_scale = len(x) / (2.0 * np.sum(np.diff(resid > 0))) * np.sqrt(chi2)
            
            fit_err = np.sqrt(np.diag(pcov)) * err_scale


            x_fit = np.linspace(xmin, xmax,1000)-time[indmin]
            y_fit = fun(x_fit,*popt)
            fit.set_data(x_fit+time[indmin], y_fit)
            if add_offset:
                plt_offset.set_data(x_fit+time[indmin], y_fit*0+popt[1])
                plt_offset.set_visible(True)
            else:
                plt_offset.set_visible(False)
                
            #uncertainty corrected for chi2/n!=1
            fit_err = np.sqrt(np.diag(pcov)* chi2)
            #result.set_text(r'$\tau_p$ = %.0f  ms'%(popt[2]*1e3 ))
            result.set_text(r'$\tau_p$ = %.0f+/-%.0f ms'%(popt[2]*1e3, fit_err[2]*1e3))
            ax.figure.canvas.draw()


        # set useblit True on gtkagg for enhanced performance
        self.fit_span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='red'), button = 1)
                            
        self.fit_span_offset = SpanSelector(ax, partial(onselect, add_offset=True),
                             'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='blue'), button = 3)
                            
                                           

       
    def fit_sxr(self, eclick, tmin, tmax, chmin, chmax):
        print(f"Selected time range: {tmin:.1f} - {tmax:.1f}")
        print(f"Selected channel range: {chmin:.0f} - {chmax:.0f}")
        
        data = None
        for icam, cam in enumerate(self.cams):
            if eclick.inaxes == self.ax[icam]:
                los = self.los[cam]
                tvec = self.tvec[cam]
                data = self.data[cam]
                break
         

        if data is None:
            return
        
        channel = np.arange(len(los))

        chind = (channel  < chmax)&(channel  > chmin)
        tind = (tvec > tmin)&(tvec < tmax)
        
        data = data[tind]
        tvec = tvec[tind]
         
        #save background or substratct, based on the used mouse button
         
        if self.click_event.button == 3:
            self.background_data[cam] = data.mean(0)
            #print(self.background_data[cam])
            return 
            
            
        data = data[:,chind].mean(1)
            
        if cam  in  self.background_data :
            #print(self.background_data[cam])
            data = data - self.background_data[cam][chind].mean()
 
 
        f = plt.figure(num='Line fit', figsize=(10,5))
        f.clf()
      
        ax = f.add_subplot(111)
        ax.plot(tvec, data,color='b' )
        ax.set_xlim(tvec[[0,-1]])
        
        for l in self.lbo_times:
            ax.axvline(l/1e3,c='k')
        
 
        ax.set_xlabel('time [s]')
        
        self.add_exp_fit(ax, tvec, data, np.ones_like(data))

             
        f.show()
        f.canvas.draw()

        



sxr_plot = SXR_interactive(shot, args.downsample)


 


plt.show()
