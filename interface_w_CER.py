#-*-Python-*-
#
#
# Copy of /fusion/projects/codes/atom/omfit_v3.x/atom/OMFIT-SOURCE_master/modules/MainIonData/SCRIPTS which Sterling Smith
# says I should use, seems to work
#
# /fusion/projects/codes/atom/dev/atom_SATURN_GCC/OMFIT-source/modules/MainIonData/SCRIPTS/interface_w_CER_routines.py
# which Shaun Haskey says to use doesn't work for me
#
# Source for shared library venus:/u/kaplan/src/ccd_routines64_v10.2/ccd_routines.c
# iris:/fusion/projects/codes/cer/cerfit/core/source/lib/cerdata/ccd/ccd_routines.c
# Created by shaskey at 25 Apr 2016  15:24
# /  Routine	: DONE load_all_ccd_data
# / 		  load_ccd_header_data
# / 		  DONE make_ccd_spectrum
# / 		  get_ccd_signal
# / 		  get_ccd_theory
# / 		  DONE make_ccd_image
# / 		  DONE get_ccd_chord_size
# / 		  DONE get_ccd_chord_timing
# / 		  PARTIAL get_ccd_spectrometer
# / 		  DONE get_ccd_camera
# / 		  get_ccd_shot_time
# / 		  get_ccd_v2b
# / 		  get_ccd_cpe
# / 		  get_ccd_config
# / 		  get_ccd_background
# / 		  DONE get_ccd_gain
# / 		  get_ccd_white_source
# / 		  set_ccd_white_source
# / 		  cleanup_ccd_spectrum
# / 		  DONE reset_ccd_cache
# / 


import pickle, socket
import numpy as np
import ctypes

import scipy.io


host_name = socket.gethostname()

library_loc = '/fusion/projects/codes/cer/acq_in_c/cerview/cerview_external64.so'


try:
    cer_lib = ctypes.cdll.LoadLibrary(library_loc)
except:
    print('Try "module load cer"')
    raise

def get_ccd_camera():
    ty = ctypes.c_short(100)
    gain = ctypes.c_short(100)
    bandwidth = ctypes.c_short(100)
    binning = ctypes.c_short(100)
    ret_val = cer_lib.get_ccd_camera(ctypes.byref(ty), ctypes.byref(gain), ctypes.byref(bandwidth), ctypes.byref(binning))
    if ret_val == 1:
        return int(ty.value), int(gain.value), int(bandwidth.value), int(binning.value)
    else:
        raise ValueError("Error obtaining type, gain, bandwidth, binning from get_ccd_camera. Return error:{}".format(ret_val))


def get_ccd_gain():
    ierr = ctypes.c_int(100)
    cer_lib.get_ccd_gain.restype = ctypes.c_float
    gain = cer_lib.get_ccd_gain(ctypes.byref(ierr))
    if ierr.value == 1:
        return gain
    else:
        raise ValueError("Error obtaining ccd gain, return error:{}".format(ierr.value))


def chord_no(chord, verbose=False):
    ''' Fetch the fortran chord number using the chord name, i.e t09, m21, etc...'''
    # Check to see if the chord is an integer, if so, then don't need to do any lookup
    # assume it is the chord number
    try:
        chord_num = int(chord)
        if verbose:
            print(' Chord_no passed an integer, chord num for CER is (fortran) {}'.format(chord_num))
        return chord_num
    except Exception:
        pass
    # Make it lower case
    chord = chord.lower()
    if chord[0] not in ['m', 'v', 't']:
        raise (OMFITException("Format of chord not understood: {}, should be m/v/tXX".format(chord)))
    chord_num = cer_lib.chord_no(ctypes.c_char_p(chord.encode()))
    if chord_num == 0:
        raise (OMFITException("Error chord number returned is 0 for chord:{}".format(chord_name)))
    if verbose:
        print(' Chord number for {} is (fortran) {}'.format(chord, chord_num))
    return chord_num


def get_ccd_config(shot, chord):
    chord = chord_no(chord, verbose=True)
    print(chord)
    camera_type = -1
    spec_name_tmp = 'a' * 500
    cam_name_tmp = 'a' * 500
    host_name_tmp = 'a' * 500
    spectr_name = ctypes.c_char_p(spec_name_tmp.encode())
    camera_name = ctypes.c_char_p(cam_name_tmp.encode())
    host_name = ctypes.c_char_p(host_name_tmp.encode())
    ret_val = cer_lib.get_ccd_config(
        ctypes.byref(ctypes.c_int(shot)),
        ctypes.byref(ctypes.c_int(chord)),
        ctypes.byref(ctypes.c_int(camera_type)),
        spectr_name,
        camera_name,
        host_name,
    )
    if ret_val != 1:
        printe("Error getting CCD config for chord:{}, retval:{}".format(chord, ret_val))
        print(spectr_name.value.strip(), camera_name.value.strip(), host_name.value.strip())
    return spectr_name.value.strip().decode(), camera_name.value.strip().decode(), host_name.value.strip().decode()


def get_ccd_cpe(shot, chord):
    """
    The GET_CCD_CPE routines returns the photoelectron to digitizer bit conversion factor.
    """

    ierr = ctypes.c_int(100)
    cpe = ctypes.c_float(0.0)
    chord = chord_no(chord)

    cer_lib.get_ccd_cpe(
        ctypes.byref(ctypes.c_int(shot)),
        ctypes.byref(ctypes.c_int(chord)),
        ctypes.byref(cpe),
        ctypes.byref(ierr)
    )

 
    return cpe.value 

def chord_id(num, verbose=False):
    ''' Fetch the chord string representation'''
    # declare the functions we use
    cer_lib.chord_id.restype = ctypes.c_char_p
    p = cer_lib.chord_id(ctypes.c_int(num))
    # print p, len(p)
    # q = ctypes.c_char_p(p)
    # if verbose:print("{}:Fortran number:{}".format(chord_name, chord_num))
    # if chord_num==0:
    #    printe("Error chord number returned is 0 for chord:{}".format(chord_name))
    return p


def load_all_ccd_data(shot, chord, wavecal=0, raise_exception_error=True, verbose=False):
    """
	The LOAD_ALL_CCD_DATA routine is called to read all the point data
	required for a particular shot and chord's tokamak or wavecal data,
	and archive the data in the global area.

    chord number: main ion starts at 80
    wavecal: wavecal flag
      == 0 : get tokamak data
      != 0 : get wavecal data
    """
    chord = chord_no(chord)
    if verbose:
        print(' Chord number for CER is (fortran) {}'.format(chord))
    ierr = ctypes.c_int(100)
    cer_lib.load_all_ccd_data(
        ctypes.byref(ctypes.c_int(shot)), ctypes.byref(ctypes.c_int(chord)), ctypes.byref(ctypes.c_int(wavecal)), ctypes.byref(ierr)
    )
    if verbose:
        print(' Finished load_all_ccd_data, shot:{}, chord num:{}, ierr:{}'.format(shot, chord, ierr.value))
    if raise_exception_error:
        if ierr.value != 0:
            raise RuntimeError('load_all_ccd_data was not successful')
    return ierr.value


def reset_ccd_cache():
    cer_lib.reset_ccd_cache()
    print(' Finished reseting ccd cache')


def get_ccd_chord_timing(lslice=1, uslice=None, verbose=False):
    if lslice is None:
        lslice = 1
    if uslice is None:
        n_pixels, n_data_slices, n_bkgnd_slices, n_data_groups, n_bkgnd_groups = get_ccd_chord_size(verbose=verbose)
        uslice = n_data_slices
    nslice = uslice - lslice + 1
    print("nslice:{}".format(nslice))
    npfloat = np.float32
    t_start = np.arange(nslice, dtype='double').astype('double')  # double
    t_integ = np.arange(nslice, dtype='double').astype('double')  # double
    n_os = np.arange(nslice, dtype='short').astype('short')  # short
    group = np.arange(nslice, dtype='short').astype('short')  # short

    t_start_ptr = ctypes.c_void_p(t_start.ctypes.data)
    t_integ_ptr = ctypes.c_void_p(t_integ.ctypes.data)
    n_os_ptr = ctypes.c_void_p(n_os.ctypes.data)
    group_ptr = ctypes.c_void_p(group.ctypes.data)
    timing_mode = 0
    bg_mode = 1
    cer_lib.get_ccd_chord_timing(
        ctypes.byref(ctypes.c_int(lslice)),
        ctypes.byref(ctypes.c_int(uslice)),
        t_start_ptr,
        t_integ_ptr,
        n_os_ptr,
        group_ptr,
        ctypes.byref(ctypes.c_short(timing_mode)),
        ctypes.byref(ctypes.c_short(bg_mode)),
    )
    return t_start, t_integ, n_os, group


def get_ccd_chord_timing_str(lslice=1, uslice=None, verbose=False):
    """Returns the start times, integration times, number of oversamples, string representation
    all length(n_groups)
    """
    t_start, t_integ, n_os, group = get_ccd_chord_timing(lslice=1, uslice=None, verbose=False)
    masks = [group == i for i in np.unique(group)]
    start_list = np.array([np.min(t_start[mask]) for mask in masks], dtype=float)
    integ_list = np.array([t_integ[mask][0] for mask in masks], dtype=float)
    n_list = np.array([np.sum(mask) for mask in masks], dtype=int)
    n_os_list = np.array([n_os[mask][0] for mask in masks], dtype=int)
    str_rep = ["{:.1f}:{}@{}/{}".format(ss, nn, ii, nno) for ss, nn, ii, nno in zip(start_list, n_list, integ_list, n_os_list)]
    return start_list, integ_list, n_os_list, str_rep


def get_ccd_spectrometer():
    wavelength = ctypes.c_float(0.0)
    slit_width = ctypes.c_short(0)
    order = ctypes.c_short(0)
    grating_pitch = ctypes.c_short(0)
    # Currently set these to null pointers, need to improve this
    comments_ptr = ctypes.POINTER(ctypes.c_int)()
    fiber_change = ctypes.POINTER(ctypes.c_int)()
    cer_lib.get_ccd_spectrometer(
        ctypes.byref(wavelength),
        ctypes.byref(slit_width),
        ctypes.byref(order),
        ctypes.byref(grating_pitch),
        comments_ptr,
        fiber_change,
    )
    return wavelength.value, slit_width.value, order.value, grating_pitch.value


# int get_ccd_spectrometer (wavelength, slit_width, order, grating_pitch,
#               comments, fiber_change)

# float        *wavelength;        /* wavelength setting           */
# short        *slit_width;        /* slit width in microns        */
# short        *order;             /* spectral order               */
# short        *grating_pitch;     /* grating pitch in grooves/mm  */
# char        comments[][41];      /* comment strings              */
# char        fiber_change[];      /* most recent fiber change date*/


def make_ccd_spectrum_image(
    ts_slices,
    ts_sub=0,
    lpix=1,
    upix=None,
    bkgnd=1,
    white_corr=1,
    num_sub=0,
    ts_avg=1,
    num_grps=1,
    grp_step=1,
    min_cut=0,
    max_cut=0,
    raise_exception_error=True,
    verbose=False,
):
    """
    lpix : lower pixel
    upix : upper pixel
    bkgnd: Background subtraction 1,-1,2
    white_corr: White light correction 1
    num_sub:
    ts_avg:
    num_grps: cerfit
    grp_step: cerfit
    min_cut:
    max_cut:
    """
    if upix is None or ts_slices is None:
        n_pixels, n_data_slices, n_bkgnd_slices, n_data_groups, n_bkgnd_groups = get_ccd_chord_size(verbose=verbose)
        if upix is None:
            upix = n_pixels
            if verbose:
                print("Set upix to {}".format(upix))
        if ts_slices is None:
            ts_slices = np.atleast_1d(n_data_slices)
            if verbose:
                print("Set ts_slices to {}".format(ts_slices))
    npix = upix - lpix + 1
    npfloat = np.float32
    y_accum = np.zeros((len(ts_slices), npix), dtype=npfloat)
    weight_accum = y_accum * 0
    sigma_accum = y_accum * 0

    y = np.arange(npix, dtype=npfloat)
    weight = np.arange(npix, dtype=npfloat)
    sigma = np.arange(npix, dtype=npfloat)

    y = y.astype(npfloat)
    weight = weight.astype(npfloat)
    sigma = sigma.astype(npfloat)

    y_ptr = ctypes.c_void_p(y.ctypes.data)
    weight_ptr = ctypes.c_void_p(weight.ctypes.data)
    sigma_ptr = ctypes.c_void_p(sigma.ctypes.data)

    maxsig = ctypes.c_int(100)
    ierr = ctypes.c_int(100)
    for ind, ts_slice in enumerate(ts_slices):
        cer_lib.make_ccd_spectrum(
            ctypes.byref(ctypes.c_int(ts_slice)),
            ctypes.byref(ctypes.c_int(lpix)),
            ctypes.byref(ctypes.c_int(upix)),
            ctypes.byref(ctypes.c_int(bkgnd)),
            ctypes.byref(ctypes.c_int(white_corr)),
            ctypes.byref(ctypes.c_int(ts_sub)),
            ctypes.byref(ctypes.c_int(num_sub)),
            ctypes.byref(ctypes.c_int(ts_avg)),
            ctypes.byref(ctypes.c_int(num_grps)),
            ctypes.byref(ctypes.c_int(grp_step)),
            ctypes.byref(ctypes.c_int(min_cut)),
            ctypes.byref(ctypes.c_int(max_cut)),
            y_ptr,
            weight_ptr,
            sigma_ptr,
            ctypes.byref(maxsig),
            ctypes.byref(ierr),
        )

        y_accum[ind, :] = +y
        weight_accum[ind, :] = +weight
        sigma_accum[ind, :] = +sigma
    if raise_exception_error:
        if ierr.value != 0:
            raise RuntimeError('make_ccd_spectrum was not successful')

    if verbose:
        print(' Finished make_ccd_spectrum, ierr:{}'.format(ierr.value))
    return y_accum, weight_accum, sigma_accum


def make_ccd_spectrum2(
    ts_slices,
    ts_sub=0,
    lpix=1,
    upix=768,
    bkgnd=1,
    white_corr=1,
    num_sub=0,
    ts_avg=1,
    num_grps=1,
    grp_step=1,
    min_cut=0,
    max_cut=0,
    raise_exception_error=True,
):
    """
    lpix : lower pixel
    upix : upper pixel
    bkgnd: Background subtraction 1,-1,2
    white_corr: White light correction 1
    num_sub:
    ts_avg:
    num_grps: cerfit
    grp_step: cerfit
    min_cut:
    max_cut:
    """
    print("Currently cannot get this to work")
    return
    npint = np.int
    npfloat = np.float32

    ts_slices = np.atleast_1d(ts_slices)
    num_ts_slices = ctypes.c_int(len(ts_slices))
    # num_ts_slices = len(ts_slices)
    print(num_ts_slices)
    # All subtractions to zero
    ts_subs = ts_slices * 0 + 5
    num_ts_sub = ctypes.c_int(len(ts_subs))
    num_ts_sub = ctypes.c_int(0)  # len(ts_subs))

    ts_slices = ts_slices.astype(npint)
    ts_subs = ts_subs.astype(npint)

    npix = upix - lpix + 1
    arr_len = npix * len(ts_subs)
    arr_len = npix * 5000
    print(arr_len)
    y = np.arange(arr_len, dtype=npfloat)
    weight = y * 0
    sigma = y * 0

    y = y.astype(npfloat)
    weight = weight.astype(npfloat)
    sigma = sigma.astype(npfloat)

    y_ptr = ctypes.c_void_p(y.ctypes.data)
    weight_ptr = ctypes.c_void_p(weight.ctypes.data)
    sigma_ptr = ctypes.c_void_p(sigma.ctypes.data)

    ts_slices_ptr = ctypes.c_void_p(ts_slices.ctypes.data)
    ts_sub_ptr = ctypes.c_void_p(ts_subs.ctypes.data)

    cut_list = np.zeros(npix, dtype=npint)
    cut_list = cut_list.astype(npint)
    cut_list_ptr = ctypes.c_void_p(cut_list.ctypes.data)

    num_cut = ctypes.c_int(0)
    maxsig = ctypes.c_int(100)
    ierr = ctypes.c_int(100)

    cer_lib.make_ccd_spectrum2(
        ts_slices_ptr,
        ctypes.byref(num_ts_slices),
        ctypes.byref(ctypes.c_int(lpix)),
        ctypes.byref(ctypes.c_int(upix)),
        ctypes.byref(ctypes.c_int(bkgnd)),
        ctypes.byref(ctypes.c_int(white_corr)),
        ts_sub_ptr,
        ctypes.byref(num_ts_sub),
        cut_list_ptr,
        ctypes.byref(num_cut),
        y_ptr,
        weight_ptr,
        sigma_ptr,
        ctypes.byref(maxsig),
        ctypes.byref(ierr),
    )
    if raise_exception_error:
        if ierr.value != 0:
            raise RuntimeError('make_ccd_spectrum was not successful')

    print(' Finished make_ccd_spectrum, ierr:{}'.format(ierr.value))
    return y, weight, sigma


def make_ccd_spectrum(
    ts_slice,
    ts_sub=0,
    lpix=1,
    upix=None,
    bkgnd=1,
    white_corr=1,
    num_sub=0,
    ts_avg=1,
    num_grps=1,
    grp_step=1,
    min_cut=0,
    max_cut=0,
    raise_exception_error=True,
    verbose=False,
):
    """

/ 		  The MAKE_CCD_SPECTRUM routine is called to extract a spectrum at a
/ 		  particular time slice, with/without background subtraction and/or
/ 		  other corrections, from the data.


    lpix : lower pixel
    upix : upper pixel
    bkgnd: Background subtraction 1,-1,2
    white_corr: White light correction 1
    num_sub:
    ts_avg:
    num_grps: cerfit
    grp_step: cerfit
    min_cut:
    max_cut:
    """
    if upix is None or ts_slice is None:
        n_pixels, n_data_slices, n_bkgnd_slices, n_data_groups, n_bkgnd_groups = get_ccd_chord_size(verbose=verbose)
        if verbose:
            print("n_pixels:{},n_data_slices:{}".format(n_pixels, n_data_slices))
        if upix is None:
            upix = n_pixels
            if verbose:
                print("Set upix to {}".format(upix))
        if ts_slice is None:
            ts_slice = n_data_slices
            if verbose:
                print("Set ts_slices to {}".format(ts_slice))
    npix = upix - lpix + 1
    npfloat = np.float32
    y = np.arange(npix, dtype=npfloat)
    weight = np.arange(npix, dtype=npfloat)
    sigma = np.arange(npix, dtype=npfloat)

    y = y.astype(npfloat)
    weight = weight.astype(npfloat)
    sigma = sigma.astype(npfloat)

    y_ptr = ctypes.c_void_p(y.ctypes.data)
    weight_ptr = ctypes.c_void_p(weight.ctypes.data)
    sigma_ptr = ctypes.c_void_p(sigma.ctypes.data)

    maxsig = ctypes.c_int(100)
    ierr = ctypes.c_int(100)

    cer_lib.make_ccd_spectrum(
        ctypes.byref(ctypes.c_int(ts_slice)),
        ctypes.byref(ctypes.c_int(lpix)),
        ctypes.byref(ctypes.c_int(upix)),
        ctypes.byref(ctypes.c_int(bkgnd)),
        ctypes.byref(ctypes.c_int(white_corr)),
        ctypes.byref(ctypes.c_int(ts_sub)),
        ctypes.byref(ctypes.c_int(num_sub)),
        ctypes.byref(ctypes.c_int(ts_avg)),
        ctypes.byref(ctypes.c_int(num_grps)),
        ctypes.byref(ctypes.c_int(grp_step)),
        ctypes.byref(ctypes.c_int(min_cut)),
        ctypes.byref(ctypes.c_int(max_cut)),
        y_ptr,
        weight_ptr,
        sigma_ptr,
        ctypes.byref(maxsig),
        ctypes.byref(ierr),
    )

    if raise_exception_error:
        if ierr.value != 0:
            raise RuntimeError('make_ccd_spectrum was not successful')

    if verbose:
        print(' Finished make_ccd_spectrum, ierr:{}'.format(ierr.value))
    return y.astype(np.float64), weight.astype(np.float64), sigma.astype(np.float64)


def get_ccd_chord_size(verbose=False):
    """Fetch the number of pixels, data slices, bkgnd slices, data_groups and bgnd groups"""
    n_pixels = ctypes.c_int(0)
    n_data_slices = ctypes.c_int(0)
    n_bkgnd_slices = ctypes.c_int(0)
    n_data_groups = ctypes.c_int(0)
    n_bkgnd_groups = ctypes.c_int(0)
    cer_lib.get_ccd_chord_size(
        ctypes.byref(n_pixels),
        ctypes.byref(n_data_slices),
        ctypes.byref(n_bkgnd_slices),
        ctypes.byref(n_data_groups),
        ctypes.byref(n_bkgnd_groups),
    )
    if verbose:
        print(
            ' Pixels:{},data_slices:{},bkgnd_slices:{},data_groups:{},bkgnd_groups:{}'.format(
                n_pixels.value, n_data_slices.value, n_bkgnd_slices.value, n_data_groups.value, n_bkgnd_groups.value
            )
        )
    return n_pixels.value, n_data_slices.value, n_bkgnd_slices.value, n_data_groups.value, n_bkgnd_groups.value


def make_ccd_image(lslice=1, uslice=None, bkgnd=1, white_corr=1, lpix=1, upix=None, raise_exception_error=True, verbose=False):
    """This is to get multiple time slices in one go including
    background subtraction and white light correction

/ 		  The MAKE_CCD_IMAGE routine is called to retrieve a 2-dimensional
/ 		  subarray of CCD data over a range of timeslices and pixels,
/ 		  with/without background subtraction.

    """
    if upix is None or uslice is None:
        n_pixels, n_data_slices, n_bkgnd_slices, n_data_groups, n_bkgnd_groups = get_ccd_chord_size()
    if uslice is None:
        uslice = n_data_slices
    if verbose:
        print(' Lower slice: {}, Upper slice:{}'.format(lslice, uslice))
    if upix is None:
        upix = n_pixels  # 768 #upper pixel
    if verbose:
        print(' Lower pixel:{}, Upper pixel:{}'.format(upix, lpix))
    if verbose:
        print(' Bkgnd:{}, white_corr:{}'.format(bkgnd, white_corr))
    npfloat = np.float32
    y = np.zeros((uslice - lslice + 1, upix), dtype=npfloat)
    y = y.astype(npfloat)
    y_ptr = ctypes.c_void_p(y.ctypes.data)
    ierr = ctypes.c_int(100)
    cer_lib.make_ccd_image(
        ctypes.byref(ctypes.c_int(lslice)),
        ctypes.byref(ctypes.c_int(uslice)),
        ctypes.byref(ctypes.c_int(lpix)),
        ctypes.byref(ctypes.c_int(upix)),
        ctypes.byref(ctypes.c_int(bkgnd)),
        ctypes.byref(ctypes.c_int(white_corr)),
        y_ptr,
        ctypes.byref(ierr),
    )
    if verbose:
        print(' Finished make_ccd_image, array min:{}, max:{}, ierr:{}'.format(np.min(y), np.max(y), ierr.value))
    if raise_exception_error:
        if ierr.value != 0:
            raise RuntimeError('make_ccd_image was not successful')
    return y


def read_pe(shot, chord):
    import os
    pe = open(os.path.expandvars('$CERBL/pe/ccd/%s.pe'%chord.lower()))
    pe = pe.readlines()

    for il,line in enumerate(pe):
        if line.startswith('SHOT'):
            _,num = line.split()
            if shot < int(num):
                continue
            pe_size = int(pe[il+1].split()[1])
            pe_coeff =[]
            il += 5
            while len(pe_coeff) < pe_size:
                pe_coeff+= [float(l) for l in pe[il].split()]
                il += 1
            break
    
 
    assert  len(pe_coeff) == pe_size
    return np.array(pe_coeff)
 


def read_tssub(shot, chord_number, beam):
    tssub_file = open('~/cerfit/{}/m{:02d}/{}/tssub.dat'.format(shot, chord_number, beam))
    lines = tssub_file.readlines() 
    ts = []
    tssub = []

    # ts = []; tssub = []; timesub = []

    for i in lines:
        if i.find('ts=') == 0:
            ts.append(int(i.replace('ts=', '').rstrip('\n')))
        if i.find('tssub=') == 0:
            tssub.append(int(i.replace('tssub=', '').rstrip('\n')))
    return np.array(ts), np.array(tssub)
