''' Get wavelengths for SPRED. 
Adapted from  /fusion/usc/src/4dlib/bin/lib/get_spred.pro

06/2019
'''
import numpy as np

#def get_spred_wl(shot,grating):
    #''' Use coefficients measured by SPRED developers. 
    #wavelengths are reported in nm
    #'''
    #pixel = np.arange(1024)+1.

    #if grating==1:  #short wavelength grating
            
        ## pixel to short wavelength coefficients
            
        #if shot< 76845:
            #A0= 292.12
            #A1= -0.22401
            #A2= 3.05567E-6
            #A3= 8.89045E-9
        #elif shot<81050:
            #A0= 292.191
            #A1= -0.220969
            #A2= 8.78252E-6
            #A3= 6.10158E-9
        #elif shot<178600:
            #A0= 291.205
            #A1= -0.21918
            #A2= 0.473785E-05
            #A3= 0.862097E-08
        #else:# error due to SPRED reparation
            #A0= 291.205
            #A1= -0.21918
            #A2= 0.473785E-05
            #A3= 0.862097E-08
            #pixel += 3.5
        
      
    #elif grating==2:   #long wavelength grating
        
            ## updated 5/19/94 by R. Wood
            
        #if shot<81050:
            #A0= 150.97
            #A1= 0.92507
            #A2= 1.58894E-4
            #A3= -4.26530E-8
        #else:
            #A0= 173.450
            #A1= 0.938225
            #A2= 0.146646E-03
            #A3= -0.378789E-07   
    
    ##calculate wavelength and time vector
    #wavelength = np.polyval([A3,A2,A1,A0], pixel)

    #return wavelength*0.1  #reported in nm


def get_spred_wl(shot,grating):
    ''' Obtain the wavelength vector over coreSPRED detectors using coefficients 
    estimated by Adam McLean, updated Nov 2020.

    Grating=1: short
    Grating=2: long
    
    This is equivalent to what is done in 
    /fusion/projects/codes/cer/spredview/get_spred_2010.pro 
    on Iris. 

    Wavelengths are reported in nm.
    '''
    pixels = np.arange(1024)+1.
    
    # polynomial coefficients for short grating:
    if shot<177976:
        cwps = [291.455,-0.21918,4.73785e-6,8.62097e-9]
    else:
        # Applies after broken camera was replaced. Starts 177976 on 5/17/2019.    
        cwps = [290.925,-0.220424,7.43441e-6,6.87753e-9] 
        
    if shot<152192:
        cwpl = [172.7,0.938225,1.46646e-4,-3.78789e-8] 
    else:
        cwpl = [171.077,0.933639,1.60769E-4,-4.86485E-8] # Long. New 11/3/2020.
        
    # form wavelength vector from 3rd order polynomial fit (from Adam McLean, Nov 2020)
    if int(grating)==1:  # short wavelength grating
        wavelength = np.polyval(cwps[::-1], pixels)
    elif int(grating)==2:  # long wavelength grating
        wavelength = np.polyval(cwpl[::-1], pixels)
    else:
        raise ValueError('Unrecognized grating number!')

    return wavelength*0.1  #reported in nm
