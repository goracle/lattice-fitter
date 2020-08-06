"""Get processing params"""
import sys
import pickle
import numpy as np

def look_for_file():
    """Look for the proc_params.p file"""
    fn1 = None
    try:
        fn1 = open('proc_params.p', 'rb')
    except FileNotFoundError:
        print("process params file not found")
    return fn1

def array_from_file(fn1):
    """Get the pickled array"""
    ret = pickle.load(fn1)
    ret = np.asarray(ret)
    return ret

def check_params(matsub, prgrdonly, pionratio, strongcuts):
    """Check config file vs. process param file found in the working dir"""
    fn1 = look_for_file()
    if fn1 is not None:
        darr = array_from_file(fn1)
        assert darr['matsub'] == matsub, (darr, matsub)
        assert darr['pr ground only'] == prgrdonly, (darr, prgrdonly)
        assert darr['pionratio'] == pionratio, (darr, pionratio)
        assert darr['strong cuts'] == strongcuts, (darr, strongcut)
    else:
        darr = {'matsub': matsub, 'pr ground only': prgrdonly,
                'pionratio':, pionratio, 'strong cuts': strongcuts}
        #darr = [matsub, prgrdonly, pionratio, strongcuts]
        #darr = np.asarray(arr)
        #fn1 = open('proc_params.p', 'wb')
        #print("writing process param file to lock this directory's config.")
        #print("writing: proc_params.p")
        #pickle.dump(arr, fn1)
    return darr

def check_params2(irrep, ens, dim, darr=None):
    """Check config file vs. process param file found in the working dir (2nd)"""
    fn1 = look_for_file()
    if fn1 is not None:
        darr = array_from_file(fn1)
        assert darr['irrep'] == matsub, (darr, irrep)
        assert darr['lattice ensemble'] == prgrdonly, (darr, ens)
        assert darr['dim'] == dim, (darr, dim)
    else:
        assert darr is not None, darr
        darr2 = {'irrep': irrep, 'lattice ensemble': ens, 'dim': dim}
        ret = {**darr, **darr2}
        fn1 = open('proc_params.p', 'wb')
        print("writing process param file to lock this directory's config.")
        print("writing: proc_params.p")
        pickle.dump(darr, fn1)
