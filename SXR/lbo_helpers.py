''' Helper functions for LBO analysis

sciortinof & odstrcilt, 2019
'''

import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
#from gadata import gadata

from IPython import embed
from matplotlib.pylab import *

def load_spred_lines_database():

    ff = open("/fusion/projects/diagnostics/lbo/spred/fs_spred_lines_database.dat", "r")
    contents=ff.readlines()

    database={}
    next=0
    for line in contents:
        if line.strip()=='':  # only '\n' in line
            next=0
            pass
        
        if next: # add atomic line to database
            database[imp_name].append(line.split())

        if line.startswith('Z'):  # create new dictionary element for new ion
            imp_name = line.split(',')[1].split('\n')[0].split(';')[0].strip()
            database[imp_name]=[]
            next=1
            pass
        
    for ion in database.keys():  # conversions to int & float
        for i,atomline in enumerate(database[ion]):
            database[ion][i][0] = int(database[ion][i][0])
            database[ion][i][1] = float(database[ion][i][1])
            database[ion][i][2] = int(database[ion][i][2])

    return database


def get_spred_data(shot, grating, connection, time=None):
    ''' Load SPRED data for the chosen shot and grating. 
    This function expects the MDS+ connection to be given as input, 
    e.g. after running
    connection = MDSplus.Connection('atlas.gat.com')

    If a time array is given as input, it is assumed that this is 
    the correct one and the function does not reload it (it returns the 
    input time array).
    '''
    from get_SPRED_wl import get_spred_wl  # routine to get SPRED wavelengths
    assert grating in [1,2]

    if time is None:
        time = connection.get(f'ptdata2("frosttimes",{shot})').data()[1:]
        if len(time) <= 1:
            connection.openTree('SPECTROSCOPY', shot)
            time = connection.get('\\spred_time').data()/1e3
            
        time = time[10:]

    #embed()
    dt = gradient(time)
        
    data = connection.get(f'ptdata2("frostgrat{grating}",{shot}, 0)').data()

    #reshape and downsample
    spect = np.float32(data.reshape( -1, 1024))
    
    
    #substract offset
    spect = spect[10:]
    #offset = spect[tvec < 0].mean(0)
    offset = spect[-70:].mean(0)
    spect-= offset
    
    
    if grating == 1:
        #  subtract "scattered light" and neutron/gamma background using channels
        #  which are beyond the edge of the microchannel plate
        #spect-= mean(spect[:,949:1019], 1)[:,None]

        #  correct for odd/even pixel gain difference
        spect[:,::2] *= 1.0945
                    
        #set data eq 0 for wavelengths <  90 A, where SHORT grating has no sensitivity
        spect[:,977:] = 0
    if grating == 2:
        # set data eq 0 for wavelengths > 1180 A, where LONG grating has no sensitivity  
        spect[:, 965:] = 0
        
    #spect = spect.T
        
    spect /= dt[:,None]
    #embed()
        
   
    
    # get wavelengths for SPRED 
    lam = get_spred_wl(shot, grating)
    
    return spect, time, lam



def get_lbo_times(shot, MDSconn=None):
    '''
    Get times of LBO injections in a given shot. 
    '''
    
    #plot( MDSconn.get(r'dim_of(\lbo::TOP.ACQAO:OUTPUT_04)').data(), MDSconn.get(r'\lbo::TOP.ACQAO:OUTPUT_04').data(),'--',label='Xaxis')
    #plot( MDSconn.get(r'dim_of(\lbo::TOP.ACQAO:OUTPUT_03)').data()+.1, MDSconn.get(r'\lbo::TOP.ACQAO:OUTPUT_03').data()+.1,'--',label='Yaxis')
    
    #show()
    #LBOSHUTT = MDSconn.get(r'\lbo::TOP.ACQ:INPUT_06').data()
    #LBOQSWCH = MDSconn.get(r'_x = \lbo::TOP.ACQ:INPUT_05').data()
    #tvec = MDSconn.get('dim_of(_x)').data()*1000 #ms 
    lbo_times = []
    #try:
    
    #data seems to be lost!!!
    if shot<180289:   # suggested by Tomas, 9/24/19
        
        # old storage, PTDATA
        LBOSHUTT = MDSconn.get('PTDATA("LBOSHUTT", %d)'%shot).data()
        LBOQSWCH = MDSconn.get('_x = PTDATA("LBOQSWCH", %d)'%shot).data()    
        tvec = MDSconn.get('dim_of(_x)').data()  #ms

    else:
        MDSconn.openTree('lbo', shot)

        # new storage, MDS+ tree
        MDSconn.openTree('lbo', shot)
        LBOSHUTT = MDSconn.get(r'\lbo::TOP.ACQ:INPUT_06').data()
        LBOQSWCH = MDSconn.get(r'_x = \lbo::TOP.ACQ:INPUT_05').data()
        tvec = MDSconn.get('dim_of(_x)').data()*1000 #ms  
        ##figure(shot)
        #plot(tvec/1000, MDSconn.get(r'_x = \lbo::TOP.ACQ:INPUT_%.2d'%7).data(),label='Xaxis output')
        #plot(tvec/1000, MDSconn.get(r'_x = \lbo::TOP.ACQ:INPUT_%.2d'%8).data(),label='Yaxis output')
        ##legend(loc='best')
        #for i in range(1,10):
            #title(i)
            #plot(tvec, MDSconn.get(r'_x = \lbo::TOP.ACQ:INPUT_%.2d'%i).data())
            #show()
        #show()
        
        #tvec = MDSconn.get('dim_of(_x)').data()*1000 #ms  
    #except:
        #return lbo_times

    if  len(LBOSHUTT) > 1:
        LBOQSWCH  = LBOQSWCH >= 1
        LBOSHUTT  = LBOSHUTT >= 1 
        if any(LBOQSWCH&LBOSHUTT):
            lbo_times = tvec[np.where((np.diff(np.double(LBOQSWCH&LBOSHUTT)) > 0 ))]
            print( 'LBO times: '+', '.join(['%.1f'%l for l in  lbo_times]))
                
    #else:
        
      
        ## new storage on MDS+
        #LBOSHUTT = MDSconn.get(r'\lbo::TOP.ACQAO:OUTPUT_02').data()
        #LBOQSWCH = MDSconn.get(r'\lbo::TOP.ACQAO:OUTPUT_05').data()
        
        #tvec_shut = MDSconn.get(r'dim_of(\lbo::TOP.ACQAO:OUTPUT_02,0)').data()
        #tvec_q = MDSconn.get(r'dim_of(\lbo::TOP.ACQAO:OUTPUT_05,0)').data()

        #t_open_shutter = tvec_shut[LBOSHUTT!=0]
        #t_pairs = [[t_open_shutter[i], t_open_shutter[i + 1]] for i in range(0,len(t_open_shutter) - 1,2)] 
        #for tt, t_interval in enumerate(t_pairs):
            #tind0 = np.argmin(np.abs(tvec_q-t_pairs[tt][0]))
            #tind1 = np.argmin(np.abs(tvec_q-t_pairs[tt][1]))
            
            ## Round to 3 decimal digits, assuming operator didn't set strangely precise times 
            #lbo_times.append(round(tvec_q[tind0:tind1][LBOQSWCH[tind0:tind1]!=0][0],3))
        ## return lbo times in ms, as with the old fetching option with ptdata
        #lbo_times = np.array(lbo_times)*1000
  
    return lbo_times

class LambdaScale(mscale.ScaleBase):
    name = 'lambdascale'
    def __init__(self, axis, lam, **kwargs):
        mscale.ScaleBase.__init__(self)       
        self.lam = lam
        
    def get_transform(self):
        return self.LambdaTransform(self.lam)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, self.lam.min()), min(vmax, self.lam.max())
    
    def set_default_locators_and_formatters(self, axis):
        pass
    
    class LambdaTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, lam):
            mtransforms.Transform.__init__(self)
            self.lam = lam
            self.x = np.arange(len(self.lam))
            self.ind = np.argsort(self.lam)
        def transform_non_affine(self, a):
            
            return np.interp(a, self.lam[self.ind], self.x[self.ind])

        def inverted(self):
            return LambdaScale.InvertedLambdaTransform(self.lam)

    class InvertedLambdaTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, lam):
            mtransforms.Transform.__init__(self)
            self.lam = lam
            self.x = np.arange(len(self.lam))

        def transform_non_affine(self, a):
            return np.interp(a, self.x,self.lam)

        def inverted(self):
            return LambdaScale.LambdaTransform(self.lam)


#mscale.register_scale(LambdaScale)

def main():
    
    import MDSplus
    
    connection = MDSplus.Connection('atlas.gat.com')
    get_lbo_times(183484 , MDSconn=connection)

    #for i in range(182792, 182800, 1):
        #print(i)
        #try:
#x        except:
            #pass

if __name__ == "__main__":
  main()
