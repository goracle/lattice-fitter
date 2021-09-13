#!/usr/bin/python3
"""get K->K 2pt from Masaaki's data"""

import sys
import h5py
import numpy as np
from latfit.utilities.postprod.kkbar import addt, LT, mcomplex

def traj(fil):
    """get trajectory from file name"""
    return int(fil.split('.')[0])

def main():
    """main"""
    for fil in sys.argv[1:]:
        fn1 = h5py.File(fil, 'r')
        base = 'kaon000wlvs_y0_kaon000wsvl_x0_dt_'
        conf = traj(fil)
        print("processing conf #", conf)
        arr = np.zeros((LT,LT), dtype=np.complex128)
        for tsrc in range(LT):
            for tdis in range(LT):
                dname = base+str(tsrc)
                idx = tdis
                toadd = mcomplex(fn1[dname][idx])
                arr[tsrc, tdis] = toadd
        fn1.close()
        out = h5py.File('traj_'+str(conf)+'_5566.hdf5', 'w')
        #out['traj_'+str(conf)+'_FigureHbub_kaon_mom000'] = arr
        out['traj_'+str(conf)+'_kaoncorrChk_mom000'] = arr
        out.close()


if __name__ == '__main__':
    main()
