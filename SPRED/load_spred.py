'''
Script to analyze SPRED data from DIII-D

sciortinof, 6/24/19
'''

from gadata import gadata
import matplotlib.pyplot as plt
import numpy as np
import mynormalize
from draggableColorbar import DraggableColorbar
from get_SPRED_wl import get_spred_wl  # routine to get SPRED wavelengths
import MDSplus
import itertools
from scipy.stats.mstats import mquantiles
import lbo_helpers
from matplotlib.widgets import RectangleSelector,SpanSelector
from scipy.optimize import least_squares,curve_fit
from IPython import embed
import os, socket, sys 
from functools import partial
 
hostname=socket.gethostname()
username = os.path.expanduser('~')[6:]


import argparse
parser = argparse.ArgumentParser( usage='Plot and fit SPRED data')

parser.add_argument('--shot', metavar='S',type=int, help='shot number')
parser.add_argument('--time_off_min',type=float, help='optional offset tmin [s]', default=None)
parser.add_argument('--time_off_max',type=float,  help='optional offset tmax [s]', default=None)
parser.add_argument('--downsample',type=int, help='Redice time resolution N times', default=1)



args = parser.parse_args()

shot = args.shot

if shot is None:
    print('Set shot number')
    exit(1)


class SPRED_interactive:
    data = []
    error = []
    lines = []
    def __init__(self,shot, offmin=None,offmax=None,downsample=1, connection=None):
        if connection is None:
           # Open MDSplus connection and get data
            try:
                connection = MDSplus.Connection('atlas.gat.com')
            except:            
                connection = MDSplus.Connection('localhost')

 
        self.shot = shot
       
        self.click_event = None
        self.background_data = None
        self.spectrum2, self.time, self.lam2 = lbo_helpers.get_spred_data(shot,2,connection)
        if len(self.time)<=1:
            print( 'SPRED data are not avalible')
            exit()
            
        n = len(self.time)
                
                
        if  172934  < shot < 177062: #2018 campaign - broken short grating
            self.spectrum1, self.lam1 = self.spectrum2*0, self.lam2
 
        else:
            self.spectrum1, _, self.lam1 = lbo_helpers.get_spred_data(shot,1,connection,self.time)

        if downsample > 1:
            self.time = self.time[:n//downsample*downsample].reshape(-1, downsample).mean(1)
            self.spectrum1 = self.spectrum1[:n//downsample*downsample].reshape(-1, downsample, len(self.lam1)).mean(1)
            self.spectrum2 = self.spectrum2[:n//downsample*downsample].reshape(-1, downsample, len(self.lam1)).mean(1)
            
            
        # get LBO times and SPRED data from external routines
        try:
            lbo_times = lbo_helpers.get_lbo_times(shot, connection) 
        except:
            lbo_times = [0]
            
        #time when the integration has ended. 
        #self.time /= 1000 #s
        self.lbo_times = np.array(lbo_times)/1000 #s
        self.dt = np.gradient(self.time)
        #import IPython
        #IPython.embed()
        
        if not (offmin is None or offmax is None):
            ind = self.time.searchsorted([offmin,offmax])
            if ind[0] < ind[1]:
                self.spectrum1 -= self.spectrum1[ind[0]:ind[1]].mean(0)[None]
                self.spectrum2 -= self.spectrum2[ind[0]:ind[1]].mean(0)[None]
        
        

        
        #NOTE time is defined at the end of the frame exposure
        self.click_event = None
        
        self.initialise_figure()
        self.plot_spectra()

    def initialise_figure(self):
     
        self.fig,self.ax = plt.subplots(1,2, sharey=True,figsize=(15,8), num=self.shot)
       

        self.ax[-2].set_xlabel(r'$\lambda$ [nm]')
        self.ax[-1].set_xlabel(r'$\lambda$ [nm]')        
                
        self.ax[0].set_ylabel('time [s]')
        self.ax[0].set_ylim(1,7)
        self.ax[0].set_xlim(self.lam1.min(),self.lam1.max())
        self.ax[1].set_xlim(self.lam2.min(),self.lam2.max())
        
        
        props = dict(facecolor='red', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)
        select_kwg = dict(#drawtype='box', 
                          useblit=True,
                        button=[1],  # don't use middle button
                        minspanx=5, minspany=5,  props= props,
                        spancoords='pixels', interactive=True)
        
        self.select1 = RectangleSelector(self.ax[0], self.line_select_callback, **select_kwg)
        self.select2 = RectangleSelector(self.ax[1], self.line_select_callback, **select_kwg)
        
        props_bckg = dict(facecolor='blue', edgecolor = 'red',alpha=0.5, fill=True,zorder=99)
        select_kwg_bckg = dict(#drawtype='box', 
                  useblit=True,
                button=[3],  # don't use middle button
                minspanx=5, minspany=5,  props= props_bckg ,
                spancoords='pixels', interactive=True)
                        
                        
        self.select1_bckg = RectangleSelector(self.ax[0], self.line_select_callback, **select_kwg_bckg)
        self.select2_bckg = RectangleSelector(self.ax[1], self.line_select_callback, **select_kwg_bckg)
        
        
        
        self.fig.tight_layout()

    def plot_spectra(self):
        
        self.scale1 =  mquantiles(self.spectrum1, 0.999)
        self.scale2 =  mquantiles(self.spectrum2, 0.999)

        dt = np.mean(np.diff(self.time))
        dl1 = np.mean(np.diff(self.lam1))
        dl2 = np.mean(np.diff(self.lam2))

        img_kwars = dict(vmin=0, cmap='gray_r')

        #NOTE time is defined at the middle of the frame exposure
        
        img0 = self.ax[0].pcolorfast(np.r_[self.lam1-dl1/2,self.lam1[-1]+dl1/2], np.r_[self.time-dt,self.time[-1] ],self.spectrum1,vmax=self.scale1,**img_kwars)
        
        
   
        img1 = self.ax[1].pcolorfast(np.r_[self.lam2-dl2/2,self.lam2[-1]+dl2/2], np.r_[self.time-dt,self.time[-1] ],self.spectrum2,vmax=self.scale2,**img_kwars)
    

        [self.ax[0].axhline(lbo, c='b', lw=.5) for lbo in self.lbo_times]
        [self.ax[1].axhline(lbo, c='b', lw=.5) for lbo in self.lbo_times]
  


        for l in self.lbo_times:
            self.ax[0].axhline(l/1e3,c='w')
            self.ax[1].axhline(l/1e3,c='w')

        # core SPRED - short grating #1: 10-29 nm 
        self.cbar0 = plt.colorbar(img0, format='%.2g', ax=self.ax[0])  #comment out ticks=[] to have numbers on colorbar
        
     
        self.cbar0 = DraggableColorbar(self.cbar0,img0)
        self.cid1 = self.cbar0.connect()

        # core SPRED - short grating #1: 15 nm to 125 nm 
        self.cbar1 = plt.colorbar(img1, format='%.2g', ax=self.ax[1])
        self.cbar1 = DraggableColorbar(self.cbar1,img1)
        self.cid2 = self.cbar1.connect()




    def line_select_callback(self,eclick, erelease):
    
        if self.click_event is None or self.click_event.xdata!=eclick.xdata :
            if eclick.button in [1,3]:
 
                self.click_event = eclick
                # Internal callback for RectangleSelector

                tmin, tmax = sorted([eclick.ydata, erelease.ydata])
                wmin, wmax = sorted([eclick.xdata, erelease.xdata])
                if tmin != tmax and wmin != wmax:
                    self.fit_line( eclick, (tmin, tmax), (wmin, wmax))
           
                 
       
    def add_exp_fit(self, ax, time, sig, err):
        
        plt_offset, = ax.plot([],[],'k--',lw=1, zorder=100)
        fit, = ax.plot([],[],'k',lw=2, zorder=100)

        result= ax.text(0.5,0.8,'',transform=ax.transAxes,zorder=100)

    
        

        def onselect(xmin, xmax, add_offset=False):
        
            fun = lambda x,a,b,t: a*np.exp(-x/t)+b*add_offset
          
            indmin, indmax = time.searchsorted((xmin, xmax))
            x = time[indmin:indmax]-time[indmin]
            y = sig[indmin:indmax]
            e = err[indmin:indmax]
            if len(x) == 0: return
            pos = abs(y.mean())+0.1
            p0 = pos ,0.,0.5

            popt,pcov = curve_fit(fun, x, y,jac='cs', p0=p0,sigma=e,
                                bounds=((0, -np.inf,0),(np.inf, np.inf, 5)),
                                x_scale=(pos,pos,0.1))
            
       
            print( 'chi2:', sum(((fun(x,*popt)-y)/e)**2)/len(x))
            x_fit = np.linspace(xmin, xmax,1000)-time[indmin]
            y_fit = fun(x_fit,*popt)
            fit.set_data(x_fit+time[indmin], y_fit)
            if add_offset:
                plt_offset.set_data(x_fit+time[indmin], y_fit*0+popt[1])
                plt_offset.set_visible(True)
            else:
                plt_offset.set_visible(False)
            fit_err = np.sqrt(np.diag(pcov))
            result.set_text(r'$\tau_p$ = %.1f+/-%.1f ms'%(popt[2]*1e3, fit_err[2]*1e3))
            ax.figure.canvas.draw()


        # set useblit True on gtkagg for enhanced performance
        self.fit_span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='red'))
                            
        self.fit_span_offset = SpanSelector(ax, partial(onselect, add_offset=True),
                             'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='blue'), button = 3)
                            
                               

    def fit_line(self,eclick,T,W):
 
        (wmin,wmax) = W
        (tmin,tmax) = T
        
        if eclick.inaxes == self.ax[0]:
            lam = self.lam1
            spectrum = self.spectrum1
        elif eclick.inaxes == self.ax[1]:
            lam = self.lam2
            spectrum = self.spectrum2
        else:
            return



        wind = (lam < wmax)&(lam > wmin)
        tind = (self.time > tmin)&(self.time < tmax)
        
        spec_slice = spectrum[tind][:,wind].astype('double')
        time_slice = self.time[tind].astype('double')
        dt_slice = self.dt[tind].astype('double')
        lam_slice  = lam[wind].astype('double')
   
        def cost_fun(par,x, y, return_results=False):
            x0,s = par 

            A = np.vstack((np.exp(-(x-x0)**2/(2*s**2)), np.ones_like(x))).T

            coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)
            if return_results:
                chi2_dof = r/(len(x)-rr)
                err = np.sqrt(np.linalg.inv(np.dot(A.T,A))[0,0]*chi2_dof)
                return coeff.T, np.dot(A, coeff).T, err
            return r
        
        #embed()
        w0 = 0.1/2.355
        out = least_squares(cost_fun, (sum(W)/2,(W[1]-W[0])/4),args=(lam_slice, spec_slice), 
                            bounds=((W[0],w0),(W[1],max(w0*2, (W[1]-W[0])/3))))    #hardcoded maximal and mininal line width 
        coeff,model,err = cost_fun(out.x,lam_slice, spec_slice ,return_results=True)
        try:
            x0,s = out.x
        except:
            print( 'Fitting was unsuccesful')
            return
            
        A, B = coeff[:,[0,1]].T
        if self.click_event.button == 3:
            self.background_data = ( lam_slice,  model.mean(0), A.mean(0))
        else:

            self.plot_spectra_fit( x0, s, A, B, err, model, spec_slice, lam_slice,time_slice, dt_slice)
        
        
        
    def plot_spectra_fit(self,  x0, s, A, B,  Ae, model, spec_slice, lam_slice,time_slice, dt_slice):
        
        
        #remove background if it was selected by the user
        if self.background_data is not None:
            A = A - self.background_data[2]
            offset = np.interp(lam_slice, self.background_data[0], self.background_data[1])
            model = model - offset
            spec_slice = spec_slice - offset
            
            
            
        f = plt.figure(num='Line fit', figsize=(10,5))
        f.clf()
        
         
            
        ax = f.add_subplot(131)
        ax.errorbar(time_slice-dt_slice/2, A, Ae,color='b', ls='none')
        ax.step(time_slice, A, color='b',  where='pre')

        ax.step(time_slice, B, where='pre',c='r')
        ax.set_ylim(min(A.min(),0), (A+Ae).max())
        ax.set_xlim(time_slice[[0,-1]])
        
        for l in self.lbo_times:
            ax.axvline(l/1e3,c='k')
        

        ax.legend(('Aplitude','background', ),loc='best')
        ax.set_title('lam0 = %.3f nm  FWHM = %.3f'%(x0,2.355*abs(s)))
        ax.set_xlabel('time [s]')
        
        self.add_exp_fit(ax, time_slice, A, Ae)

        
        dt = np.mean(np.diff(time_slice))
        dl = np.mean(np.diff(lam_slice))
        
        img_kwars = dict(aspect='auto',interpolation='nearest', origin='lower', 
                     extent=(lam_slice[0]-dl/2, lam_slice[-1]+dl/2,time_slice[0]-dt,time_slice[-1]))
        
        vmax =  mquantiles(spec_slice, 0.999)

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

        



spred_plot = SPRED_interactive(shot,  args.time_off_min,args.time_off_max, args.downsample)


 

try:
    # save in personal directory for SPRED data, if it exists
    fig.savefig('~/'+username+'/SPRED_data/SPRED_spec_'+ _imp + '_'+str(shot))
    print( " ***** Saved SPRED spectrum *****")
except:
    pass

 

##############################
'''
Find lines from SPRED database

'''
try:
    ax = spred_plot.ax

    # read database of known SPRED lines
    import pandas as pd
    db = pd.read_csv('/fusion/projects/diagnostics/lbo/spred/spred_lines.csv')

    columns = db.columns #db.to_dict().keys()
    index = db.index

    names = db['New Calibration']
    Z = db['Atomic No. Z']
    description = db['Ionic Charge State']
    elem = db['Element']

    # get wavelength
    lam=np.zeros((len(names),1))
    charge_states= [] #np.chararray((len(names),1))
    for jj in range(len(names)):
        lam[jj] = names[jj].split('_')[1]
        charge_states.append(names[jj].split('_')[0])

    charge_states=np.asarray(charge_states)

    if imp!='':
        # check if impurity is present in database (require unicode-to-unicode comparison)
        lam_lines = lam[elem[elem==imp].keys()[:]]/10 # nm
        cs_lines = charge_states[elem[elem==imp].keys()[:]]
    else:
        lam_lines=[]; cs_lines=[]

    xlims1 = ax[0].get_xlim()
    xlims2 = ax[1].get_xlim()

    for l in lam_lines:
        if xlims1[0]<l<xlims1[1]:
            ax[0].axvline(l, c='w',ls='--',lw=2)
        if xlims2[0]<l<xlims2[1]:
            ax[1].axvline(l, c='w',ls='--', lw=2)


    # Plot all other known lines in different style
    lam_other_lines = lam[elem[elem!=imp].keys()[:]]/10
    cs_other_lines = charge_states[elem[elem!=imp].keys()[:]]

    ####
    color_vals = ['b','g','r','c','m','k']
    style_vals = ['-', '--', '-.', ':']
    ls_vals = []
    for s in style_vals:
        for c in color_vals:
            ls_vals.append(c+s)
    ls_cycle=itertools.cycle(ls_vals)

    #c_cycle=itertools.cycle(color_vals)
    #ls_cycle=itertools.cycle(style_vals)
    ####

            
    for l in range(len(lam_other_lines)):
        line = lam_other_lines[l]
        cs = cs_other_lines[l]
         
        ls = ls_cycle.next()
        if xlims1[0]<l<xlims1[1]:
            ax[0].axvline(line,c='k',ls='--',lw=2, label=cs)
            ax[2].axvline(line,c=ls[0],ls=ls[1],lw=2, label=cs)
        if xlims2[0]<l<xlims2[1]:
            ax[1].axvline(line,c='k',ls='--',lw=2, label=cs)
            ax[3].axvline(line,c=ls[0],ls=ls[1],lw=2, label=cs)

    ax[2].legend().draggable()
    ax[3].legend().draggable()
except:
    pass



plt.show()
