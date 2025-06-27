from numpy import *
from scipy.optimize import least_squares
import numpy as np



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
        background0 = np.ones_like(x)
        background1 = x-x0
        A = np.vstack((gauss,background0, background1 )).T
        coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)
        coeff[0] = np.abs(coeff[0])
      
        r = np.sum((np.dot(A, coeff) - y.T)**2,0)
      
        if return_results:
            chi2_dof = r/(len(x)-rr)
            try:
                err = np.sqrt(np.linalg.inv(np.dot(A.T,A))[0,0]*chi2_dof)
            except: err =  np.inf  * coeff[0]
           
            return coeff.T, np.dot(A, coeff).T, err
        return r
    
    nw = len(wav_vec)
    out = least_squares(cost_fun, (x0,s0),args=( wav_vec,data- bckg ), 
                        bounds=((wav_vec.min(), 0),(wav_vec.max(),  np.ptp(wav_vec))))    #hardcoded maximal and mininal line width


    

    coeff,model,err = cost_fun(out.x,wav_vec, data- bckg ,return_results=True)
    #plt.plot(model.T,'--')
    #plt.plot(data.T,'-')
    #plt.show()

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

        
        d = np.median(d, 1)
        b = np.median(b, 1)
         
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
                    solution_err[i] = sqrt(diag(linalg.pinv(dot(out.jac.T, out.jac)))*max(1,chi2n[i]))
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



def fit_fixed_shape_gauss(wav_vec, spectra, bg_spectra):

    active = spectra-bg_spectra
 
 
    background = bg_spectra.mean(0)

 
    #fit BE spectra
    from scipy.optimize import least_squares

    def cost_fun(par,x, y, return_results=False):
        x0, w  = par
        spectrum = np.exp(-(x-x0)**2/(2*w**2))/np.sqrt(2*np.pi)/np.abs(w)

        A = np.vstack((spectrum, np.ones_like(x), background)).T

        coeff,r,rr,s = np.linalg.lstsq(A, y.T, rcond=None)

        if return_results:
            chi2_dof = r/(len(x)-rr)
            err = np.sqrt(np.linalg.pinv(np.dot(A.T,A))[0,0]*chi2_dof)
            return coeff.T, np.dot(A, coeff).T, err
        return r
        
   # initial guess for gaussian width is one thord of the selected region
    width_max = (wav_vec.max()-wav_vec.min())
    width0 =  width_max/ 3
    wmid0 =  wav_vec[np.argmax(active.mean(0))]

 
    out = least_squares(cost_fun, (wmid0, width0),
                bounds = [[wav_vec.min(), width_max/10, ], [wav_vec.max(), width_max]],
                args=(wav_vec, active))
                #hardcoded maximal and mininal line width 
    
    coeff,data_fit,Ae = cost_fun(out.x, wav_vec, active,return_results=True)


    A,B1,B2 = coeff.T

    
    return A, Ae, data_fit




