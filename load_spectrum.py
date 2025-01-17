
try:
    from interface_w_CER import *
except OSError:
    print('error, try "module load cer"')
    exit()
    
from IPython import embed
import MDSplus 
from matplotlib.pylab import *
from scipy import *
from numpy import *
from scipy.stats.mstats import mquantiles
from matplotlib.widgets import MultiCursor
import os 
mdsserver = 'atlas.gat.com'
MDSconn = MDSplus.Connection(mdsserver)




def load_chord(shot, chord, white_corr=1):
    """
    Load raw CER spectra, trigger time, timestep, and readout noise. 
    
    Parameters:
    ------------------------
    shot: int
            Shot number to be loaded
    chord: str 
            Three char long chord name (V01, T01, M01,...)
    white_corr: int
            0 or 1, turn off/on while correction to fix the vignetting
    Outputs:
        wavelength: float
            The requested wavelength in A, it is not exact!
        t_start: float:
            the time when the light integration started ms
        dt: float
            Integration time  in ms 
        raw_gain: int
            Not used
        spectra: float 2D array (wavelength, time)
            Raw spectra in counts, after the white correction and offset subtraction
        pe: float
            counts pr one photoelectron 
        readout_noise: float array
            readout out noise estimated from each pixel
       
    """
    
    chord_num = chord_no(chord) # get chord number corresponding to channel
    ierr = load_all_ccd_data(shot,chord)
    t_start,t_integ,n_os,group = get_ccd_chord_timing() # get timing vectors [ms]
    spectra = make_ccd_image(white_corr=white_corr) # get data array, counts(wavelength,time)
    
    pe = get_ccd_cpe(shot,chord)
    #use last 32 slices. It is already white corrected
    readout_noise = spectra[-32:].std(0)
   
    raw_gain = get_ccd_gain() # get gain: 0 for low gain, 1 for high gain
    wavelength,slit_width,order,grating_pitch = get_ccd_spectrometer() # get central wavelength setting [A]
    #tV = t_start + 0.5*t_integ# % central measurement time [ms]
    dt = mean(t_integ)# % integration time [ms]

    return wavelength, t_start, t_integ ,raw_gain, spectra, pe, readout_noise
 

def beam_chord(chord):
    nbi = '30L'
    if chord in ['T%.2d'%i for i in range(8,17)]+['T23','T24']+['T%.2d'%i for i in range(41,49,2)]:
       nbi = '33L'
    return nbi

def load_pe(chord, shot):
    path = '/fusion/projects/results/cer/pe/ccd/'
    size = 0
    pe = None
    with open(path+chord.lower()+'.pe') as file:
        for l in file:
            if len(l) <= 1 or l[0] in '!*':
                continue
            elif l.startswith('SHOT'):
                if shot > int(l.split()[1]):
                    pe = []
                    
            elif pe is None:
                continue
            elif l.startswith('SIZE'):
                size = int(l.split()[1])	
            elif size  > len(pe):
                pe += [float(p) for p in l.split()]
            else:
                break
    
    return np.array(pe)   
	

import scipy.constants as consts
def doppler_broadening_sigma(Ti, mass, lam):
    return lam * np.sqrt(2*Ti * consts.e / (consts.m_p * mass * consts.c**2))
	

def gauss_fun(lam_vec, lam, disp,  sigma_doppler, inst_fun):
    W = np.array(inst_fun['WIDTH'])*disp
    L = np.array(inst_fun['LOCATION'])*disp
    A = inst_fun['AMPLITUDE']  * np.sqrt( np.pi* W**2 )
    fun = np.zeros_like(lam_vec)
    for i in range(len(A)):
        sigma2 = sigma_doppler **2 + W[i] **2 /2
        mu =  lam + L[i]
        fun +=   A[i]/np.sqrt(2*np.pi*sigma2)  * np.exp( - (lam_vec - mu) **2 / (2*sigma2))
    return fun


 
def load_inst_fun(chord, shot, lam  ):

    # The equation for a gaussian for this data file is:
    #   y = A e^(-1(x-L)^2/W^2)
    #
    #   A = AMPLITDUDE
    #   L = LOCATION
    #   W = WIDTH
    #
    # The gaussians put here should be normalized to an area of 1.0. 


    path = '/fusion/projects/results/cer/cerdata/profiles/'
    files = os.listdir(path)
    files = [f for f in files if f.startswith('inst') and f[-1] != '~']
     
    wavelengths = np.array([int(f.split('_')[2].split('.')[0]) for f in files])
    ifile = np.where(wavelengths  < lam )[0]
    ifile = ifile[np.argmax(wavelengths[ifile])]
    file_name = files[ifile]
    fun = None
    with open(path+ file_name) as file:
       for l in file:
           
           if len(l) <= 1 or l[0] in '!*':
               continue

           elif l.startswith('SHOT'):
                if fun is not None and len(fun):
                   break
                elif shot > int(l.split()[1]):
                    fun = { }
            
           elif fun is None:
               continue
           elif l.startswith(chord):
               l = l.split()
               fun[l[1]] = [float(f) for f in l[2:]]
 
    return fun    
 

def cer_sys_id(chord):
    chord = chord.upper()
    return {'M':0,'T':1,'V':2}[chord[0]]

def get_calib(shot, chord, lam0):
    """
    Get the calibration factor for given shot, chord and wavelength

    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded
    chord: str 
        Three char long chord name (V01, T01, M01,...)
    lam0: float 
        Line wavelength in A

    Outputs:
    calib: float
        calibration factor in [counts/ s / photons/m^2/s/ster]
    """

    ichord = int(chord[1:])
    
    isys = cer_sys_id(chord)
    
    calib_path ='/fusion/projects/results/cer/intensity/' 
    calib_file = calib_path+['main','tang','vert'][isys]+f'{ichord}_abs.calib'  
    cshot = None
    with open(calib_file) as file:
        for l in file:
            if len(l) <= 1 or l[0] in '!*':
                continue
            #calib header
            if l[:3] != '   ':
                dtype={'names': ('shot', 'nwav', 'voltperbit'),'formats': ('i4', 'i4', 'f4')}

                cshot, ncalib, volt_per_bit = loadtxt([l], 
                        dtype=dtype,usecols=[0,1,2],
                        comments='!',unpack=True, encoding='utf8' )
                
                if int(cshot) > shot:
                    break
                calib_data = []
            
            elif cshot is not None:
                # calibration wavelength [A], 
                # calibration factor [1e-7 counts/s / photons/cm^2/s/ster]
                # gain factor of spectrometer (always appears to be 1)
                try:
                    wvln, calib, gain = loadtxt([l],dtype=float,
                                usecols=[0,1,2],comments='!',
                                encoding='utf8')
                except Exception as e:
                    print(e)
                    embed()
                calib_data.append([wvln, calib, gain])

    calib_data = array(calib_data)

    # calibration factor [counts/s / photons/m^2/s/ster]
    calib = interp(lam0, calib_data[:,0], calib_data[:,1])/1e7/1e4
  
    return calib

def get_beamgeom(shot, chord, beam ):
    """
    Get beam geometry factor for active CX lines

    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded
    chord: str 
        Three char long chord name (V01, T01, M01,...)
    beam: str 
        Beam name i.e. 30L, 33L, ...

    Outputs:
    G: float
        beam geometry factor in 1/m
    """
 
    isys = cer_sys_id(chord)
 
    sys = ['TANGENTIAL','TANGENTIAL','VERTICAL'][isys]
    subtree = 'CER'+['MAIN','',''][isys]

    calib = f'\\IONS::TOP.{subtree}.CALIBRATION.{sys}.CHANNEL{chord[1:]}:'
 
 
    MDSconn.openTree('IONS', shot)
  
    BEAM_ORDER = MDSconn.get(f'\\IONS::TOP.{subtree}.CALIBRATION:BEAM_ORDER').data()
 
    BEAM_ORDER = [b.decode()[:2]+b.decode().strip()[-2] for b in BEAM_ORDER]

    ibeam = BEAM_ORDER.index(beam)
  
    
    if isys!= 0:
       G = MDSconn.get(calib+'BEAMGEOMETRY').data()[ibeam]
    else: 
       #No geomtry factros for MICER system, estimate from tangential impurity system

        #simple estimate of G from nearest tangential CER system
        if int(chord[1:]) < 9: #core MICER
            cer_ch = r_[1:8, 17: 23]
            return 3.7 #better to use a fixed value 
        else:
            cer_ch = r_[9:17, 41: 49]
 
        R  = MDSconn.get(f'\\IONS::TOP.{subtree}.CALIBRATION.TANGENTIAL.CHANNEL{chord[1:]}:PLASMA_R').data()[ibeam]
        R_cer  = [MDSconn.get(f'\\IONS::TOP.CER.CALIBRATION.TANGENTIAL.CHANNEL%.2d:PLASMA_R'%j).data()[ibeam] 
for j in cer_ch]
        G_cer = [MDSconn.get(f'\\IONS::TOP.CER.CALIBRATION.TANGENTIAL.CHANNEL%.2d:BEAMGEOMETRY'%j).data()[ibeam] 
for j in cer_ch]
    
    G = np.interp(R, np.sort(R_cer), np.array(G_cer)[np.argsort(R_cer)])

   
    return G


def get_coord(shot, chord, beam, tvec, equ=None,  coord_out='Psi_N' ,diag='EFIT01'):
    """
    Get beam geometry factor for active CX lines

    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded
    chord: str 
        Three char long chord name (V01, T01, M01,...)
    beam: str 
        Beam name i.e. 30L, 33L, ...
    tvec: 1d array
        time for the equilibrium mapping, in ms
    equ: equilibrium mapping object
    
    coord_out: radial coordinate used for equilibrium apping  
        default 'Psi_N', other options are 'rho_pol', 'rho_tor', 'r_V', 'rho_vol','r_a', 'Rmaj'
    
    Outputs:
    equ: initialized equilibrium mapping object
    Psi_N_LOS: 2d array float (time, n)
        poloidal flux (or another coordinate when selected) as function of time along the LOS
    Psi_N_NBI : 1d array float (time )
       poloidal flux (or another coordinate when selected) as a function of time at NBI LOS crosssection
    PLASMA_R :  float  
        R coordinate of NBI LOS crosssection [m]
    PLASMA_Z :  float 
        Z coordinate of NBI LOS crosssection [m]
    R : 1d array float (n )  R along the LOS [m]
    Z : 1d array float (n )  Z along the LOS [m]
    L : 1d array float (n )  distance along the LOS  [m]
    """


    if equ is None:
        from map_equ import equ_map
        equ = equ_map(MDSplus.Connection(mdsserver))

    
    equ.Open(shot,diag=diag)

    isys = cer_sys_id(chord)
 
    sys = ['TANGENTIAL','TANGENTIAL','VERTICAL'][isys]
    subtree = 'CER'+['MAIN','',''][isys]

    calib = f'\\IONS::TOP.{subtree}.CALIBRATION.{sys}.CHANNEL{chord[1:]}:'
 
 
    MDSconn.openTree('IONS', shot)
 
    LENS_R = MDSconn.get(calib+'LENS_R').data()
    LENS_Z = MDSconn.get(calib+'LENS_Z').data()
    LENS_PHI = MDSconn.get(calib+'LENS_PHI').data()

    BEAM_ORDER = MDSconn.get(f'\\IONS::TOP.{subtree}.CALIBRATION:BEAM_ORDER').data()

    BEAM_ORDER = [b.decode()[:2]+b.decode().strip()[-2] for b in BEAM_ORDER]

    ibeam = BEAM_ORDER.index(beam)
    
 
    PLASMA_R = MDSconn.get(calib+'PLASMA_R').data()[ibeam]
    PLASMA_Z = MDSconn.get(calib+'PLASMA_Z').data()[ibeam]
    PLASMA_PHI = MDSconn.get(calib+'PLASMA_PHI').data()[ibeam]


 

    X0 = LENS_R*cos( deg2rad(LENS_PHI)) 
    Y0 = LENS_R*sin( deg2rad(LENS_PHI)) 
    Z0 = LENS_Z 
    LENS = array((X0,Y0,Z0))

    X1 = PLASMA_R*cos( deg2rad(PLASMA_PHI)) 
    Y1 = PLASMA_R*sin( deg2rad(PLASMA_PHI)) 
    Z1 = PLASMA_Z 
    PLASMA = array((X1,Y1,Z1))


    t = linspace(0,2.5,1000)
    LOS = LENS + t[:,None]*(PLASMA-LENS)
 
    L = linalg.norm(LOS-LENS,axis=1)

    R = hypot(LOS[:,0],LOS[:,1])
    Z = LOS[:,2]
 
    Psi_N_LOS =equ.rz2rho(R,Z,tvec/1e3, coord_out='Psi_N')
    
    Psi_N_NBI =equ.rz2rho(PLASMA_R,PLASMA_Z,tvec/1e3, coord_out=coord_out)
    
    if Psi_N_NBI is None or any(Psi_N_NBI > 1.5): 
         print('get_coord error')
         embed()

 
    return  equ, Psi_N_LOS, Psi_N_NBI, PLASMA_R, PLASMA_Z, R, Z, L 

def get_wavelength(shot, chord, lam0, npix):
    """
    Get wavelength vector 
 
    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded
    chord: str 
        Three char long chord name (V01, T01, M01,...)
    lam0: float 
        Requested central line wavelength in A (can be wrong by few nm)
    npix: int
        number of pixels of the CCD
        
    Outputs:
    wavelength: 1d array (npix)
        Wavelength pf each pixel
    DISP: float
        dispersion [A/pix]
    """
 
  
    isys = cer_sys_id(chord)
 
    sys = ['TANGENTIAL','TANGENTIAL','VERTICAL'][isys]
    subtree = 'CER'+['MAIN','',''][isys]

    calib = f'\\IONS::TOP.{subtree}.CALIBRATION.{sys}.CHANNEL{chord[1:]}:'

    ##load dispersion and fiduciable 
 
    MDSconn.openTree('IONS', shot)
 
    try:
        DISP = MDSconn.get(calib+'DISPERSION').data()
    except:
        DISP = 0.2261 #for U4
   
    try:
        FIDU = MDSconn.get(calib+'FIDUCUAL').data()
    except:
        FIDU = nan
 

    if not isfinite(FIDU) or FIDU == -1:
        print('Invalid FIDUCIAL')
        FIDU = npix/2

    #hardcoded corrections for our experiment
    if shot in [190553,190552]:
        if chord in ['V%.2d'%i for i in range(25,33,2)]:
            FIDU += 80
        if chord in ['V%.2d'%i for i in range(26,33,2)]:
            FIDU -= 80
        if chord in  ['T%.2d'%i for i in range(41,49,2)]:
             lam0+=10
        if chord in  ['T%.2d'%i for i in range(42,49,2)]:
             lam0-=6
        if chord in  ['V%.2d'%i for i in range(9,17,2)]:
             lam0 +=10          
        if chord in  ['V%.2d'%i for i in [10,12,14,16]]:
             lam0 +=-6          

    if shot in range(199100, 199112):
  
        if chord in  ['T%.2d'%i for i in range(5, 9)]:
             lam0-=+13
    if shot in [199102, 199103, 199111]:
        if chord in  ['T10','T12', 'T08']:
             lam0+=+1

    if shot in [190652,190653,190654]:
  
        if chord in  ['T%.2d'%i for i in range(41,49,2)]:
             lam0-=6471-6482
        if chord in  ['T%.2d'%i for i in range(42,49,2)]:
             lam0-=6490-6482
 
    if shot in [192786,192787]:
        if chord in  ['T%.2d'%i for i in range(10,17,2)]:
             lam0 -=18          
        if chord in  ['T%.2d'%i for i in range(17,25)]:
             lam0 -=6  

    if shot in [190552, 190553]:

        if chord in  ['M17','M18']:
             lam0 -=2 
        if chord in  ['M19','M20']:
             lam0 -=2

        if chord in  ['M31','M32']:
             lam0 +=2     

        if chord in  ['T%.2d'%i for i in range(17,25)]:
             lam0 -=6  
  
    
    wavelength = lam0 + -(arange(1,npix+1)-FIDU)*DISP; # wavelength vector [A], a negative sigh is just CER convection on DIII-D
    return  wavelength, DISP


def load_all_channels(shot):
    """
    Print line and wavelength for all channels
 
    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded
    """
 
    subtree = 'CER'
 
    MDSconn.openTree('IONS', shot)

    for sys,nch in [('TANGENTIAL', 56),('VERTICAL', 32)]:
       for i in range(1,nch+1):
           calib = '\\IONS::TOP.CER.CALIBRATION.'+sys+'.CHANNEL%.2d:'%i
           try:
              WAV = MDSconn.get(calib+'WAVELENGTH').data()
              LINEID = MDSconn.get(calib+'LINEID').data()
              print(sys[0], i, LINEID.decode(), WAV)
           except:
              continue




def get_beams(shot, fast_data = True, load_beams = ['30L','33L', '30R','33R',]):
    """
    Get a dictionary of functions that returns beam power integrated over the integration time of CCD
 
    Parameters:
    ------------------------
    shot: int
        Shot number to be loaded

    Outputs:
    beams_power:  power interpolation functions
        example: nbi_pow_30L = beams_power('30L',t_start, dt)
        
    
    """
    MDSconn.openTree('NB',  shot)  

 
    paths = ['\\NB::TOP.NB{0}:'.format(b[:2]+b[-1]) for b in load_beams] 
    s = 'F' if fast_data else ''
    TDI  = [p+'PINJ'+s+'_'+p[-4:-1] for p in paths]
    TDI += ['dim_of('+TDI[0]+')']
    
    beams_data = list(MDSconn.get('['+','.join(TDI)+']').data())
    pow_tvec = beams_data.pop()
  
    beams = {}
    from scipy.integrate import cumtrapz
    from scipy.interpolate import interp1d
 
    for b,power in zip(load_beams, beams_data):
        cpow = cumtrapz(power, pow_tvec, initial=0)
        beams[b] = interp1d(pow_tvec, cpow, bounds_error=False,assume_sorted=True,
                              fill_value=(0, cpow[-1])) 
 

    beams_power = lambda b, t, dt:  (beams[b](t+dt)-beams[b](t))/dt
     
    return beams_power


def get_tssub(beams, t_start, dt, beam):
    """
    Get wavelength vector get_tssub
 
    Parameters:
    ------------------------
    beams:   power interpolation functions   from get_beams function
    t_start: array of floats
        time when integration started [ms]
    dt: float
        integration time [ms]
    beam: str
        beam name, it will count with both L and R beams when they are used in the discharge
        
    Outputs:
         tssub: array of ints
             indexes of nearest timeslices used for background subtraction
         ts_: array of ints
             indexes of timeslice where the beam is on and a appropriate subtraction was found
   
    """
 
    #calculate average beam power over integrated window
    nbi_pow_L = beams(beam[:-1]+'L', t_start,dt)
    nbi_pow_R = beams(beam[:-1]+'R', t_start,dt)   

    beam_on = (nbi_pow_L > 1e6)|(nbi_pow_R > 1e6)
    beam_off = (nbi_pow_L < 1e6)&(nbi_pow_R <  1e6)
 
    tssub = []
    #maximum distance to find the background subtraction timeslices
    ts_range = 20
    
    ts  = where(beam_on)[0]

    ts_ = []
    for it in ts:
        for i in range(1,ts_range):
            if it+i < len(t_start) and beam_off[it+i]:
                tssub.append( it+i)
                ts_.append(it)
                break
            if it-i >= 0 and beam_off[it-i]:
                tssub.append(it-i)
                ts_.append(it)
                break

    return tssub,  ts_

def remove_spikes(spect,n=9, m = 3, sigma=5):
    """
    Remove gamma spikes from the spectra
 
    Parameters:
    ------------------------
    spect: 2D array of float
        raw CCD spectrum
    sigma: float
        How many sigmas threshold should be used to remove the spikes

    Outputs:
        spect: 2D array of float
           corrected raw CCD spectrum
    """
    
    from scipy.signal import order_filter

    #order filter over 9 channels, for each timeslice independently
    filt_spect = order_filter(spect, ones((1,n)), m)

    noise = spect-filt_spect
    noise[:,[0,-1]] = 0 #issues with order filter at the edge 

    invalid = noise > sigma*noise.std(0)[None]
  
    #replace invalid pixels by the estimate from the order filter
    spect[invalid] = filt_spect[invalid]
    
    return  spect

 
  
def blip_average(spectrum, tvec,dt, beam_pow,skip_first = 1, passive=False):
    """
    Average emission over the whole beam blip
 
    Parameters:
    ------------------------
    spectrum: 2D array of float
        raw CCD spectrum
    tvec: array of floats
        times when the integration started
    dt: float
        integration time
    beam_pow: array of floats
        NBI power for each timeslice
    skip_first: int
        how many timeslices should be skipped from the beam blip (the first is often different from others)
    passive: bool
        return passive spectrum (else active CX spectrum will be returned)
      
    Outputs:
        tvec_: array of floats
            time when the beam blips started
        stime array of floats
            time integration over the entire beam blip
        data 2D array of float
            spectrum averaged over the entire beam blip
        bckg 2D array of float
            background estimated from off time before and after the beam blip
        pow_avg
            average beam power over the beam blip
    """
    
    beam_on = (beam_pow > 0.5e6)
    beam_off = (beam_pow < 0.5e6)
    
    if np.size(dt) == 1:
        dt = dt + np.zeros_like(tvec)
    
    dbeam = diff(float_(beam_on))
    start = where(dbeam == 1)[0]+1+skip_first
    end = where(dbeam == -1)[0]+1

    if not any(beam_on) and not passive :
        raise Exception('Beam is always off')
    else:
        #embed()
        if end[0] < start[0]:
            start  = hstack((0,start))
    
   
    dbeam = diff(float_(beam_off))
    off_start = where(dbeam == 1)[0]+1
    off_end = where(dbeam == -1)[0]+1

    if off_end[0] < off_start[0]:
        off_start  = hstack((0,off_start))
 

    data = []
    tvec_ = []
    bckg = []
    pow_avg = []
    stime = []
    
    for i in range(0,len(start)-2):
        ioff = off_start.searchsorted( start[i])
        bckg1 = spectrum[off_start[ioff]:off_end[ioff]]
        tbckg1 = tvec[off_start[ioff]:off_end[ioff]]
        bckg2 = spectrum[off_start[ioff+1]:off_end[ioff+1]]
        if passive:
            data.append(bckg1.mean(0))
            tvec_.append(tbckg1[0])
            stime.append(dt[off_start[ioff]:off_end[ioff]].sum())
            beam_bg = beam_pow[off_start[ioff]:off_end[ioff]].mean()
            pow_avg.append(beam_bg)
            #substract  any contribution from the time when the beam was still on. 
            beam_on = beam_pow[start[i+1]:end[i+1]].mean()
            spect_on = spectrum[start[i+1]:end[i+1]].mean(0)
            bckg.append(beam_bg/beam_on * (spect_on - data[-1]))


        else:
             sig = spectrum[start[i+1]:end[i+1]]
             tvec_.append(tvec[start[i+1]])
             stime.append(dt[start[i+1]:end[i+1]].sum())
             bckg.append((bckg1.mean(0)+bckg2.mean(0))/2)
             data.append(sig.mean(0))
             pow_avg.append(beam_pow[start[i+1]:end[i+1]].mean())

 
    tvec_ = hstack(tvec_)
    stime = hstack(stime)
    data = vstack(data)
    bckg = vstack(bckg)
    pow_avg = hstack(pow_avg)
       
    return tvec_, stime, data, bckg, pow_avg
    
  
    
